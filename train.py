from collections import defaultdict
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch
import torchvision
import torchvision.transforms as transforms
import tqdm
from flax.training import train_state
from torch.utils.data import DataLoader

from logger import Logger


class CifarResnet(nn.Module):
    n: int

    @nn.compact
    def __call__(self, x, train: bool):
        x = nn.Conv(16, kernel_size=(3, 3))(x)
        x = nn.BatchNorm(use_running_average=not train)(x)

        for _ in range(self.n):
            out = nn.Conv(16, kernel_size=(3, 3))(x)
            out = nn.relu(out)
            out = nn.BatchNorm(use_running_average=not train)(out)
            out = nn.Conv(16, kernel_size=(3, 3))(out)
            out = nn.BatchNorm(use_running_average=not train)(out)
            x = nn.relu(out + x)

        for _ in range(self.n):
            out = nn.Conv(32, kernel_size=(3, 3), strides=(2, 2))(x)
            out = nn.relu(out)
            out = nn.BatchNorm(use_running_average=not train)(out)
            out = nn.Conv(32, kernel_size=(3, 3))(out)
            out = nn.BatchNorm(use_running_average=not train)(out)
            residual = nn.Conv(32, kernel_size=(3, 3), strides=(2, 2))(x)
            x = nn.relu(out + residual)

        for _ in range(self.n):
            out = nn.Conv(64, kernel_size=(3, 3), strides=(2, 2))(x)
            out = nn.relu(out)
            out = nn.BatchNorm(use_running_average=not train)(out)
            out = nn.Conv(64, kernel_size=(3, 3))(out)
            out = nn.BatchNorm(use_running_average=not train)(out)
            residual = nn.Conv(64, kernel_size=(3, 3), strides=(2, 2))(x)
            x = nn.relu(out + residual)

        x = x.mean(axis=(-2, -3))
        x = nn.Dense(10)(x)
        return x


class TrainState(train_state.TrainState):
    batch_stats: Any


def loss_fn(params, ts, images, labels):
    logits, updates = ts.apply_fn(
        {"params": params, "batch_stats": ts.batch_stats},
        images,
        train=True,
        mutable=["batch_stats"],
    )
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
    return loss, (logits, updates)


def inner_step(ts, images, labels):
    grad_fn = jax.value_and_grad(loss_fn, argnums=1, has_aux=True)
    (loss, (logits, updates)), grads = grad_fn(ts.params, images, labels)
    ts = ts.apply_gradients(
        grads=grads,
        batch_stats=updates["batch_stats"],
    )
    metrics = {
        "train_loss": loss,
        "train_accuracy": jnp.mean(jnp.argmax(logits, -1) == labels),
    }
    return ts, metrics


def eval_step(ts: TrainState, images, labels):
    logits = ts.apply_fn(
        {"params": ts.params, "batch_stats": ts.batch_stats}, images, train=False
    )
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
    metrics = {
        "test_loss": loss,
        "test_accuracy": jnp.mean(jnp.argmax(logits, -1) == labels),
    }
    return metrics


class Trainer:
    def __init__(self, rng, lr, minibatch_size, weight_decay):
        self.lr = lr
        self.minibatch_size = minibatch_size
        self.logger = Logger()

        train_dataset = torchvision.datasets.CIFAR10(
            "data/", transform=torchvision.transforms.ToTensor(), download=True
        )
        test_dataset = torchvision.datasets.CIFAR10(
            "data/",
            transform=torchvision.transforms.ToTensor(),
            train=False,
        )

        rng, key1, key2 = jax.random.split(rng, num=3)
        g1 = torch.Generator()
        g1.manual_seed(jax.random.randint(key1, (), 0, 2**31).item())
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=minibatch_size,
            shuffle=True,
            drop_last=True,
            num_workers=1,
            pin_memory=True,
            generator=g1,
        )
        g2 = torch.Generator()
        g2.manual_seed(jax.random.randint(key2, (), 0, 2**31).item())
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=minibatch_size,
            shuffle=False,
            drop_last=False,
            generator=g2,
        )

        model = CifarResnet(n=3)
        variables = model.init(rng, np.zeros((32, 32, 3)), train=True)
        self.ts = TrainState.create(
            apply_fn=model.apply,
            params=variables["params"],
            batch_stats=variables["batch_stats"],
            tx=optax.chain(
                optax.add_decayed_weights(
                    weight_decay=weight_decay,
                    # exclude biases and batchnorm parameters
                    mask=jax.tree_map(lambda x: x.ndim != 1, variables["params"]),
                ),
                optax.sgd(lr, momentum=0.9),
            ),
        )

    def run_epoch(self, verbose=False):
        for images, labels in (
            tqdm.tqdm(self.train_loader) if verbose else self.train_loader
        ):
            images = transforms.RandomCrop((32, 32), padding=4)(images)
            images = np.asarray(images).transpose((0, 2, 3, 1))
            labels = np.asarray(labels)

            self.ts, train_metrics = inner_step(self.ts, images, labels)
            self.logger.push(train_metrics)
        self.logger.step()

    def evaluate(self, verbose=False):
        metrics = defaultdict(list)
        weights = []
        for images, labels in (
            tqdm.tqdm(self.test_loader) if verbose else self.test_loader
        ):
            images = np.asarray(images).transpose((0, 2, 3, 1))
            labels = np.asarray(labels)

            new_metrics = eval_step(self.ts, images, labels)
            for k, v in new_metrics.items():
                if hasattr(v, "shape"):
                    v = v.item()
                metrics[k].append(v)
                weights.append(images.shape[0])
        self.logger.log({k: np.average(v, weights=weights) for k, v in metrics.items()})


# def outer_loss(weight_decay, ts, images, labels, valid_images, valid_labels):
#     assert "weight_decay" in ts.opt_state.hyperparams
#     ts.opt_state.hyperparams["weight_decay"] = weight_decay
#
#     ts = inner_step(ts, images, labels)
#
#     loss = loss_fn(ts.params, ts, valid_images, valid_labels)
#     return loss, ts
#
#
# @jax.jit
# def outer_step(
#     weight_decay, ts: TrainState, images, labels, valid_images, valid_labels
# ):
#     (loss, ts), grad = jax.value_and_grad(outer_loss, has_aux=True)(
#         weight_decay, ts, images, labels, valid_images, valid_labels
#     )
#     weight_decay = weight_decay - 0.01 * grad  # lol hardcoded lr whatever
#     return weight_decay, ts
#
#
# def meta_reg_trial(rng, train_loader, test_loader, config):
#     model = CifarResnet(n=3)
#     variables = model.init(rng, np.zeros((32, 32, 3)), train=True)
#     ts = TrainState.create(
#         apply_fn=model.apply,
#         params=variables["params"],
#         batch_stats=variables["batch_stats"],
#         tx=optax.chain(
#             optax.add_decayed_weights(
#                 weight_decay=config["weight_decay"],
#                 mask=jax.tree_map(lambda x: x.ndim != 1, variables["params"]),
#             ),
#             optax.sgd(config["lr"], momentum=0.9),
#         ),
#     )
#     logger = Logger()
#     weight_decay = 1e-4
#
#     for _ in range(100):
#         for images, labels in test_loader:
#             images = np.asarray(images).transpose((0, 2, 3, 1))
#             labels = np.asarray(labels)
#
#             test_metrics = eval_step(ts, images, labels)
#             test_metrics = jax.tree_map(lambda x: x.item(), test_metrics)
#             logger.push({"test_" + k: v for k, v in test_metrics.items()})
#
#         logger.step()
#         logger.generate_plots()


if __name__ == "__main__":
    trainer = Trainer(
        jax.random.PRNGKey(42), 1e-2, minibatch_size=128, weight_decay=1e-4
    )
    trainer.run_epoch()

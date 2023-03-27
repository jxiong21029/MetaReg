import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch
import torchvision
import tqdm.auto as tqdm
from flax.training.train_state import TrainState as MetaTrainState
from torch.utils.data import DataLoader
from torchvision import transforms

from trainer import CifarResnet, Trainer, TrainState, inner_step, loss_fn


@jax.jit
def outer_step(
    outer_ts: MetaTrainState, ts: TrainState, images, labels, valid_images, valid_labels
):
    def outer_loss(meta_params):
        new_opt_state = ts.opt_state
        new_opt_state[0].hyperparams["weight_decay"] = jnp.exp(
            meta_params["log_weight_decay"]
        )
        new_ts = ts.replace(opt_state=new_opt_state)
        new_ts, inner_metrics = inner_step(new_ts, images, labels)
        loss, _ = loss_fn(new_ts.params, new_ts, valid_images, valid_labels)
        return loss, (new_ts, inner_metrics)

    (loss, (new_ts, inner_metrics)), grads = jax.value_and_grad(
        outer_loss, has_aux=True
    )(outer_ts.params)
    outer_ts = outer_ts.apply_gradients(grads=grads)
    metrics = {
        "outer_loss": loss,
        "log_weight_decay": outer_ts.params["log_weight_decay"],
        "weight_decay": jnp.exp(outer_ts.params["log_weight_decay"]),
    }
    metrics.update(inner_metrics)
    return outer_ts, new_ts, metrics


class MetaTrainer(Trainer):
    def __init__(self, rng, inner_lr, outer_lr, minibatch_size, initial_weight_decay):
        rng, key = jax.random.split(rng)
        super().__init__(
            rng=key,
            lr=inner_lr,
            minibatch_size=minibatch_size,
            weight_decay=initial_weight_decay,
        )

        rng, key = jax.random.split(rng)
        model = CifarResnet(n=3)
        variables = model.init(key, np.zeros((32, 32, 3)), train=True)

        train_dataset = torchvision.datasets.CIFAR10(
            "data/", transform=torchvision.transforms.ToTensor()
        )

        rng, key = jax.random.split(rng)
        g = torch.Generator()
        g.manual_seed(jax.random.randint(key, (), 0, 2**31 - 1).item())
        self.valid_loader = DataLoader(
            train_dataset,
            batch_size=minibatch_size,
            shuffle=True,
            drop_last=True,
            generator=g,
        )

        self.outer_ts = MetaTrainState.create(
            apply_fn=None,
            params={"log_weight_decay": jnp.log(jnp.array(initial_weight_decay))},
            tx=optax.sgd(outer_lr, momentum=0.9),
        )

        self.ts = TrainState.create(
            apply_fn=model.apply,
            params=variables["params"],
            batch_stats=variables["batch_stats"],
            tx=optax.chain(
                optax.inject_hyperparams(optax.add_decayed_weights)(
                    weight_decay=jnp.array(initial_weight_decay),
                    # exclude biases
                    mask=jax.tree_map(lambda x: x.ndim != 1, variables["params"]),
                ),
                optax.sgd(inner_lr, momentum=0.9),
            ),
        )

    def run_epoch(self, verbose=False):
        for (images, labels), (valid_images, valid_labels) in zip(
            tqdm.tqdm(self.train_loader) if verbose else self.train_loader,
            self.valid_loader,
        ):
            images = transforms.RandomCrop((32, 32), padding=4)(images)
            images = np.asarray(images).transpose((0, 2, 3, 1))
            labels = np.asarray(labels)

            valid_images = np.asarray(valid_images).transpose((0, 2, 3, 1))
            valid_labels = np.asarray(valid_labels)

            self.outer_ts, self.ts, train_metrics = outer_step(
                self.outer_ts, self.ts, images, labels, valid_images, valid_labels
            )
            self.logger.push(train_metrics)
        self.logger.step()


if __name__ == "__main__":
    trainer = MetaTrainer(
        jax.random.PRNGKey(42),
        inner_lr=1e-2,
        outer_lr=1,
        minibatch_size=128,
        initial_weight_decay=1e-2,
    )
    for _ in range(100):
        trainer.run_epoch(verbose=True)
        trainer.evaluate()

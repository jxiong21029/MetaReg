import jax

from meta_learning import MetaTrainer


def trial(seed, initial_weight_decay):
    rng = jax.random.PRNGKey(seed)
    trainer = MetaTrainer(rng, 1e-2, 10, 128, initial_weight_decay)

    trainer.evaluate()
    for i in range(500):
        trainer.run_epoch()
        if i % 10 == 9:
            trainer.evaluate()
            # TODO: report results

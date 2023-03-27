import time

import numpy as np

from logger import Logger
from tuning import Tuner, config_name


def trial(config):
    logger = Logger()
    for t in range(10):
        time.sleep(1)
        logger.log(loss=config["lr"] * (2 - t / 10) * np.random.normal(1, 0.2))
        logger.generate_plots("plotgen/" + config_name(config))

        yield logger  # report by yielding the logger


# defining experiment config
tuner = Tuner(
    {"dataset": "task", "weight_decay": "science", "lr": "nuisance", "trial_idx": "id"},
    trial_fn=trial,
    metric="loss",
    mode="min",
)

# adding trials
for dataset in ("cifar", "imagenet"):
    for weight_decay in (1e-5, 1e-4):
        for lr in (10**-3, 10**-2.5, 10**-2):
            for trial_idx in range(3):
                tuner.add(
                    {
                        "dataset": dataset,
                        "weight_decay": weight_decay,
                        "lr": lr,
                        "trial_idx": trial_idx,
                    }
                )

tuner.run()
tuner.save_results("tuner.pickle")

# example of loading existing tuner
tuner_2 = Tuner.load_results("tuner.pickle")

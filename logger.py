import math
import os
import pickle
from collections import defaultdict

import numpy as np


class RunningMoments:
    def __init__(self):
        self.n = 0
        self.m = 0
        self.s = 0

    def push(self, x):
        assert isinstance(x, float) or isinstance(x, int)
        self.n += 1
        if self.n == 1:
            self.m = x
        else:
            old_m = self.m
            self.m = old_m + (x - old_m) / self.n
            self.s = self.s + (x - old_m) * (x - self.m)

    def mean(self):
        return self.m

    def std(self):
        if self.n > 1:
            return math.sqrt(self.s / (self.n - 1))
        else:
            return self.m


class Logger:
    def __init__(self):
        self._buffer_data = defaultdict(RunningMoments)
        self.cumulative_data = defaultdict(list)
        self.seen_plot_directories = set()

    # log metrics reported once per epoch
    def log(self, metrics=None, **kwargs):
        metrics = {} if metrics is None else metrics
        for k, v in (metrics | kwargs).items():
            if hasattr(v, "shape"):
                v = v.item()
            self.cumulative_data[k].append(v)

    # push metrics logged many times per epoch, to aggregate later
    def push(self, metrics=None, **kwargs):
        metrics = {} if metrics is None else metrics
        for k, v in (metrics | kwargs).items():
            if hasattr(v, "shape"):
                v = v.item()
            self._buffer_data[k].push(v)

    # computes mean and std of metrics pushed many times per epoch
    def step(self):
        for k, v in self._buffer_data.items():
            self.cumulative_data[k + "_mean"].append(v.mean())
            self.cumulative_data[k + "_std"].append(v.std())
        self._buffer_data.clear()

    def tune_report(self):
        from ray import tune

        tune.report(**{k: v[-1] for k, v in self.cumulative_data.items()})

    def air_report(self, **kwargs):
        from ray.air import session

        session.report({k: v[-1] for k, v in self.cumulative_data.items()}, **kwargs)

    def save(self, filename):
        if not filename.endswith(".pickle"):
            filename = filename + ".pickle"
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    def generate_plots(self, dirname="plotgen"):
        import matplotlib
        import matplotlib.pyplot as plt
        import seaborn as sns

        matplotlib.use("Agg")
        sns.set_theme()

        if dirname not in self.seen_plot_directories:
            os.makedirs(dirname, exist_ok=True)

            for filename in os.listdir(dirname):
                file_path = os.path.join(dirname, filename)
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
            self.seen_plot_directories.add(dirname)

        for k, v in self.cumulative_data.items():
            if k.endswith("_std"):
                continue

            fig, ax = plt.subplots()

            x = np.arange(len(self.cumulative_data[k]))
            v = np.array(v)
            if k.endswith("_mean"):
                name = k[:-5]

                (line,) = ax.plot(x, v, label=k)
                stds = np.array(self.cumulative_data[name + "_std"])
                ax.fill_between(
                    x, v - stds, v + stds, color=line.get_color(), alpha=0.15
                )
            else:
                name = k
                (line,) = ax.plot(x, v)
            ax.scatter(x, v, color=line.get_color())

            fig.suptitle(name)
            fig.savefig(os.path.join(dirname, name))
            plt.close(fig)

    def convergence(self, key, p=0.98):
        """Estimates the degree to which some metric has converged.

        A custom metric created by Jerry Xiong. Close to zero when the metric is clearly
        trending upwards or downwards, and close to one when changes in the metric seem
        to be dominated by noise. Intended for debugging.

        Higher p corresponds to higher weight placed on the last few measurements.
        """
        assert key in self.cumulative_data

        data = self.cumulative_data[key]
        if len(data) <= 1:
            return 0

        diffs = np.array([data[i + 1] - data[i] for i in range(len(data) - 1)])
        w = np.power((1 - p), (1 - np.linspace(0, 1, num=len(diffs))))
        w = w / np.sum(w)

        m = np.sum(w * diffs)
        v = np.sum(w * diffs * diffs)

        return 1 - abs(m) / math.sqrt(v + 1e-8)

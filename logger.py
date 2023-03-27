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
        self.buffer = defaultdict(RunningMoments)

        self.data = defaultdict(list)
        self.std_data = defaultdict(list)

        self.seen_plot_directories = set()

    # log metrics reported once per epoch
    def log(self, metrics=None, **kwargs):
        metrics = {} if metrics is None else metrics
        for k, v in {**metrics, **kwargs}.items():
            if hasattr(v, "shape"):
                v = v.item()
            self.data[k].append(v)

    # push metrics logged many times per epoch, to aggregate later
    def push(self, metrics=None, **kwargs):
        metrics = {} if metrics is None else metrics
        for k, v in {**metrics, **kwargs}.items():
            if hasattr(v, "shape"):
                v = v.item()
            self.buffer[k].push(v)

    # computes mean and std of metrics pushed many times per epoch
    def step(self):
        for k, v in self.buffer.items():
            self.data[k].append(v.mean())
            self.std_data[k].append(v.std())
        self.buffer.clear()

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
            self.seen_plot_directories.add(dirname)
            os.makedirs(dirname, exist_ok=True)

            for filename in os.listdir(dirname):
                file_path = os.path.join(dirname, filename)
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)

        for name, values in self.data.items():
            fig, ax = plt.subplots()
            fig: plt.Figure
            ax: plt.Axes

            x = np.arange(len(self.data[name]))
            values = np.array(values)

            (line,) = ax.plot(x, values)
            if name in self.std_data:
                stds = np.array(self.std_data[name])
                ax.fill_between(
                    x, values - stds, values + stds, color=line.get_color(), alpha=0.3
                )

            if len(values) <= 100:  # add thick circles for clarity
                ax.scatter(x, values, color=line.get_color())

            ax.set_title(name.replace("_", " "))
            ax.set_xlabel("epochs")

            fig.savefig(os.path.join(dirname, name))
            plt.close(fig)

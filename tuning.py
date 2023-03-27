import inspect
import os
import pickle
import shutil
import warnings
from collections import defaultdict
from typing import Callable, Dict, List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import ray

from logger import Logger


def iqm(scores: List[float]):
    scores = np.asarray(scores).ravel()

    n_obs = scores.shape[0]
    lowercut = int(0.25 * n_obs)
    uppercut = n_obs - lowercut

    scores = np.partition(scores, (lowercut, uppercut - 1))
    return np.mean(scores[lowercut : uppercut])


# https://arxiv.org/abs/2108.13264 IQM with bootstrapped confidence intervals, with
# support for NaN results (e.g. incomplete trials)
def bootstrapped_iqm(runs: np.ndarray, iters=1000, alpha=0.9, seed=42):
    assert 0 < alpha < 1

    rng = np.random.default_rng(seed)

    idx = rng.integers(runs.shape[0], size=(iters, runs.shape[0]))
    bootstraps: np.ndarray = runs[idx]  # (iters, trials, time)

    lowercut = runs.shape[0] // 4
    uppercut = runs.shape[0] - lowercut

    bootstraps.partition(uppercut - 1, axis=1)
    bootstraps[:, uppercut:] = np.inf

    bootstraps[np.isnan(bootstraps)] = -np.inf
    bootstraps.partition(lowercut, axis=1)
    bootstraps[:, :lowercut] = np.nan

    bootstraps[~np.isfinite(bootstraps)] = np.nan

    # ignore numpy warning: mean of empty slice
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = np.nanmean(bootstraps, axis=1)

        lo = np.nanquantile(results, 0.5 - alpha / 2, axis=0)
        mid = np.nanmean(results, axis=0)
        hi = np.nanquantile(results, 0.5 + alpha / 2, axis=0)
    return lo, mid, hi


def config_name(config):
    ret = []
    for k, v in config.items():
        if isinstance(v, float):
            ret.append(f"{k}={v:.2g}")
        else:
            ret.append(f"{k}={v}")
    return "_".join(ret)


class TrialConfig:
    def __init__(self, spec, cfg_dict):
        vals_by_type = defaultdict(list)
        for key in cfg_dict:
            assert key in spec
        for key in spec:
            assert key in cfg_dict, f"{key} not found in config={cfg_dict}"
            assert spec[key] in ("task", "science", "nuisance", "id")
            vals_by_type[spec[key]].append(cfg_dict[key])

        self.cfg_dict = cfg_dict

        self.task_key = tuple(vals_by_type["task"])
        self.science_key = tuple(vals_by_type["science"])
        self.nuisance_key = tuple(vals_by_type["nuisance"])
        self.id_key = tuple(vals_by_type["id"])

    def __hash__(self):
        return hash((self.task_key, self.science_key, self.nuisance_key, self.id_key))

    def __eq__(self, other):
        if not isinstance(other, TrialConfig):
            return False

        return (
            self.task_key == other.task_key
            and self.science_key == other.science_key
            and self.nuisance_key == other.nuisance_key
            and self.id_key == other.id_key
        )


class ResultReporter:
    def __init__(self, spec, metric, mode):
        self.spec = spec
        self.comparison_metric = metric
        self.mode = mode
        self.results: Dict[TrialConfig, Logger] = {}
        self.finished = set()

        self.task_metrics = sorted(
            [k for k, v in self.spec.items() if v == "task"],
        )
        self.science_metrics = sorted(
            [k for k, v in self.spec.items() if v == "science"],
        )
        self.nuisance_metrics = sorted(
            [k for k, v in self.spec.items() if v == "nuisance"],
        )

        self.cleared_plots = False

    def add_result(self, config: TrialConfig, result: Logger):
        self.results[config] = result
        self.plot_results()

    def set_done(self, config: TrialConfig):
        self.finished.add(config)
        self.plot_results()

    def plot_results(self, directory_name="tuner_plots"):
        matplotlib.use("Agg")

        metrics = list(self.results.values())[0].data.keys()
        if not self.cleared_plots and os.path.exists(directory_name):
            self.cleared_plots = True
            # print("DELETING ALL")
            shutil.rmtree(directory_name)
        os.makedirs(directory_name, exist_ok=True)
        # print(
        #     "BEGIN PLOTTING IN:",
        #     directory_name,
        #     "CURRENT FILES:",
        #     os.listdir(directory_name),
        # )

        for metric in metrics:
            for task in set(config.task_key for config in self.results.keys()):
                fig, ax = plt.subplots(constrained_layout=True)
                fig: plt.Figure
                ax: plt.Axes

                task_title = ", ".join(
                    f"{k}={v:.2g}" if isinstance(v, float) else f"{k}={v}"
                    for k, v in zip(self.task_metrics, task)
                )
                if task_title:
                    title = f"{task_title} :: {metric}"
                else:
                    title = metric

                for config in self.results.keys():
                    # print(config, self.finished)
                    if config.task_key == task and config not in self.finished:
                        title += " (in progress)"
                        break

                ax.set_title(title.replace("_", " "))
                ax.set_xlabel("epochs")

                science_keys = set(
                    config.science_key
                    for config in self.results.keys()
                    if config.task_key == task
                )
                for science_key in science_keys:
                    nuisance_scores = defaultdict(list)

                    for config, result in self.results.items():
                        if config.task_key != task or config.science_key != science_key:
                            continue
                        nuisance_scores[config.nuisance_key].append(
                            result.data[self.comparison_metric][-1]
                        )

                    nuisance_scores = {k: iqm(v) for k, v in nuisance_scores.items()}
                    best_nuisance = (max if self.mode == "max" else min)(
                        nuisance_scores.keys(), key=lambda k: nuisance_scores[k]
                    )

                    best_nuisance_results = []
                    for config, result in self.results.items():
                        if (
                            config.task_key == task
                            and config.science_key == science_key
                            and config.nuisance_key == best_nuisance
                        ):
                            best_nuisance_results.append(result.data[metric])

                    max_len = max(len(run) for run in best_nuisance_results)
                    best_nuisance_results = np.array(
                        [
                            run + [np.nan] * (max_len - len(run))
                            for run in best_nuisance_results
                        ]
                    )
                    lo, mid, hi = bootstrapped_iqm(best_nuisance_results)

                    science_label = ", ".join(
                        f"{k}={v:.2g}" if isinstance(v, float) else f"{k}={v}"
                        for k, v in zip(self.science_metrics, science_key)
                    )
                    nuisance_label = ", ".join(
                        f"{k}={v:.2g}" if isinstance(v, float) else f"{k}={v}"
                        for k, v in zip(self.nuisance_metrics, best_nuisance)
                    )
                    if science_label and nuisance_label:
                        label = f"{science_label} [{nuisance_label}]"
                    elif science_label:
                        label = science_label
                    elif nuisance_label:
                        label = nuisance_label
                    else:
                        label = None
                    (line,) = ax.plot(
                        mid,
                        label=label,
                    )
                    ax.fill_between(
                        np.arange(len(mid)), lo, hi, color=line.get_color(), alpha=0.3
                    )

                    if len(mid) <= 100:  # add thick circles for clarity
                        ax.scatter(np.arange(len(mid)), mid, color=line.get_color())

                if self.science_metrics or self.nuisance_metrics:
                    fig.legend(bbox_to_anchor=(1, 1), loc="upper left")

                path = os.path.join(
                    directory_name, *[s.replace(" ", "_") for s in title.split(" :: ")]
                )
                dirname, filename = os.path.split(path)
                os.makedirs(dirname, exist_ok=True)
                # print("SAVING", path)
                fig.savefig(path, bbox_inches="tight")
                # print("SUCCESSFULLY SAVED", path)
                # print(path, f"ATTEMPTING DELETE", path + '_(in_progress).png')
                if not path.endswith("(in_progress)") and os.path.exists(
                    path + "_(in_progress).png"
                ):
                    os.remove(path + "_(in_progress).png")
                    # print(
                    #     "SUCCESSFULLY DELETED", path[: len(" (in progress)")] + ".png"
                    # )
                plt.close(fig)
                # print(
                #     "AFTERMATH, DIRECTORY:",
                #     directory_name,
                #     "WITH FILES:",
                #     os.listdir(directory_name),
                # )

    def serializable(self):
        return {
            "spec": self.spec,
            "metric": self.comparison_metric,
            "mode": self.mode,
            "results": self.results,
            "finished": self.finished,
        }


remote_reporter = ray.remote(ResultReporter)


class Tuner:
    def __init__(self, spec, trial_fn: Callable, metric: str, mode="max"):
        for k, v in spec.items():
            assert v in ("task", "science", "nuisance", "id")
        assert inspect.isgeneratorfunction(trial_fn)
        assert mode in ("min", "max")

        self.spec = spec
        self.metric = metric
        self.mode = mode

        def run_fn(config, reporter):
            try:
                handles = []
                for result in trial_fn(config.cfg_dict):
                    # print("ADDING RESULT", result)
                    handles.append(reporter.add_result.remote(config, result))
                ray.get(handles)
            except Exception as e:
                print(f"Trial config={config_name(config)} failed with exception {e}")
            finally:
                # print("SETTING DONE")
                ray.get(reporter.set_done.remote(config))

        self.reporter = remote_reporter.remote(self.spec, self.metric, self.mode)
        self.run_fn = ray.remote(run_fn)
        self.remote_args = []

    def add(self, cfg_dict):
        self.remote_args.append((TrialConfig(self.spec, cfg_dict), self.reporter))

    def run(self):
        ray.get([self.run_fn.remote(*args) for args in self.remote_args])

    def save_results(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(ray.get(self.reporter.serializable.remote()), f)

    @staticmethod
    def load_results(filename) -> ResultReporter:
        with open(filename, "rb") as f:
            data = pickle.load(f)
            ret = ResultReporter(
                spec=data["spec"], metric=data["metric"], mode=data["mode"]
            )
            ret.results = data["results"]
            ret.finished = data["finished"]
            return ret

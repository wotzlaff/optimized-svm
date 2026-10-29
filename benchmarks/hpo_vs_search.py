"""Gradient-based hyperparameter optimization vs. grid, random and successive-halving search.

For four data sets the regularization and the RBF width are chosen by

* ``OptimizedSVC``/``OptimizedSVR`` (exact CV gradients, L-BFGS-B),
* ``GridSearchCV`` over a 13 x 11 grid of ``(C, gamma)``,
* ``RandomizedSearchCV`` with 30 draws and
* ``HalvingRandomSearchCV``

with scikit-learn's ``SVC``/``SVR`` (LIBSVM) and 5-fold cross-validation. The selected model is
evaluated on a held-out test set. Times are wall-clock seconds of the search (single process).

Usage::

    python benchmarks/hpo_vs_search.py --seeds 3 > hpo_vs_search.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
from scipy.stats import loguniform
from sklearn.datasets import load_diabetes, make_classification, make_friedman1
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import (
    GridSearchCV,
    HalvingRandomSearchCV,
    RandomizedSearchCV,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

from optimized_svm import OptimizedSVC, OptimizedSVR


def checkerboard(n: int, seed: int):
    rng = np.random.default_rng(seed)
    x = rng.random((n, 2))
    y = (np.sin(4 * np.pi * x[:, 0]) * np.sin(4 * np.pi * x[:, 1]) > 0).astype(int)
    return x, np.where(rng.random(n) < 0.05, 1 - y, y)


def datasets(seed: int):
    yield "checkerboard (n=800, 2 features)", "cls", *checkerboard(800, seed)
    x, y = make_classification(1500, 10, n_informative=5, flip_y=0.1, random_state=seed)
    yield "noisy classification (n=1500, 10 features)", "cls", x, y
    x, y = make_friedman1(1000, noise=1.0, random_state=seed)
    yield "Friedman #1 (n=1000, 10 features)", "reg", x, y
    d = load_diabetes()
    yield "diabetes (n=442, 10 features)", "reg", d.data, d.target


def run_search(search, xtr, ytr):
    t = time.perf_counter()
    search.fit(xtr, ytr)
    return time.perf_counter() - t


def evaluate(task, model, xte, yte):
    if task == "cls":
        return float(np.mean(model.predict(xte) == yte))  # accuracy
    return float(np.mean((model.predict(xte) - yte) ** 2))  # MSE


def main(seeds: int) -> None:
    results = []
    for seed in range(seeds):
        for name, task, x, y in datasets(seed):
            xtr, xte, ytr, yte = train_test_split(x, y, test_size=0.4, random_state=seed)
            sx = StandardScaler().fit(xtr)
            xtr, xte = sx.transform(xtr), sx.transform(xte)
            if task == "reg":  # standardise targets so that the default epsilon is sensible
                mu, sd = ytr.mean(), ytr.std()
                ytr, yte = (ytr - mu) / sd, (yte - mu) / sd
            base = SVC() if task == "cls" else SVR()
            grid = {"C": np.logspace(-2, 4, 13), "gamma": np.logspace(-4, 1, 11)}
            dist = {"C": loguniform(1e-2, 1e4), "gamma": loguniform(1e-4, 1e1)}
            methods = {
                "grid search (13 x 11)": (GridSearchCV(base, grid, cv=5), 13 * 11),
                "random search (30)": (
                    RandomizedSearchCV(base, dist, n_iter=30, cv=5, random_state=seed),
                    30,
                ),
                "successive halving": (
                    HalvingRandomSearchCV(base, dist, cv=5, random_state=seed, factor=3),
                    None,
                ),
            }
            for method, (search, evals) in methods.items():
                dt = run_search(search, xtr, ytr)
                if evals is None:  # candidates evaluated over all halving iterations
                    evals = int(len(search.cv_results_["params"]))
                results.append(dict(dataset=name, task=task, seed=seed, method=method,
                                    seconds=dt, evaluations=evals,
                                    score=evaluate(task, search, xte, yte)))  # fmt: skip
                print(f"{name} seed {seed} {method}: {dt:.1f}s", file=sys.stderr, flush=True)
            for label, blocks in (("gradient method", None), ("gradient method, ARD", "features")):
                if blocks and xtr.shape[1] < 5:
                    continue
                est = (OptimizedSVC if task == "cls" else OptimizedSVR)(blocks=blocks, cv=5)
                dt = run_search(est, xtr, ytr)
                results.append(dict(dataset=name, task=task, seed=seed, method=label,
                                    seconds=dt, evaluations=len(est.history_),
                                    score=evaluate(task, est, xte, yte)))  # fmt: skip
                print(f"{name} seed {seed} {label}: {dt:.1f}s", file=sys.stderr, flush=True)
    json.dump(results, sys.stdout, indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    main(ap.parse_args().seeds)

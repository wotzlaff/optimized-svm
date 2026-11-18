r"""Compare a fixed holdout with five-fold hyperparameter optimization.

Run from the repository root with a single thread per library::

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=1 \
      PYTHONPATH=src python benchmarks/hpo_holdout.py --samples 800 --seeds 0 1 2 3 4 \
      --csv benchmarks/results/hpo_holdout.csv > benchmarks/results/hpo_holdout.txt

Each pair uses the same outer training/test split and initial hyperparameters.
The holdout is exactly the first of the five fixed inner folds, so its per-fold
training size matches that fold. Both estimators then refit on the full outer
training set. Test accuracy and RMSE are computed on untouched outer test data.
Fit time includes tuning and the estimator's final refit. The fold-solve count
counts one training and hypergradient computation per fold per objective
evaluation, excluding the final refit. The two tuning objectives are different,
so only the outer test metrics are compared. Runs are sequential and alternate
which protocol goes first.

The two synthetic datasets and the selected seeds illustrate the time/quality
trade-off. The experiment does not estimate speedups for parallel CV or
generalization to other datasets.
"""

from __future__ import annotations

import argparse
import csv
import os
import platform
import time
from pathlib import Path

import numpy as np
import scipy
import sklearn
import smo_solver
from sklearn.datasets import make_friedman1, make_moons
from sklearn.metrics import accuracy_score, root_mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from threadpoolctl import threadpool_info, threadpool_limits

import optimized_svm
from optimized_svm import OptimizedSVC, OptimizedSVR


def run_pair(task: str, n: int, seed: int, max_iter: int, holdout_first: bool) -> list[dict]:
    classification = task == "svc"
    if classification:
        x, y = make_moons(n_samples=n, noise=0.27, random_state=seed)
    else:
        x, y = make_friedman1(n_samples=n, noise=0.5, random_state=seed)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=seed, stratify=y if classification else None
    )
    splitter = StratifiedKFold if classification else KFold
    folds = list(splitter(n_splits=5, shuffle=True, random_state=seed).split(x_train, y_train))
    protocols = [("cv5", folds), ("holdout", folds[:1])]
    if holdout_first:
        protocols.reverse()
    rows = []
    for order, (protocol, cv) in enumerate(protocols, start=1):
        estimator = OptimizedSVC if classification else OptimizedSVR
        model = estimator(cv=cv, max_iter=max_iter, random_state=seed, n_jobs=1)
        start = time.perf_counter()
        model.fit(x_train, y_train)
        seconds = time.perf_counter() - start
        prediction = model.predict(x_test)
        metric = accuracy_score if classification else root_mean_squared_error
        result = model.optimize_result_
        evaluations = len(model.history_)
        row = {
            "task": task,
            "samples": n,
            "seed": seed,
            "protocol": protocol,
            "order": order,
            "outer_train": len(x_train),
            "outer_test": len(x_test),
            "folds": len(cv),
            "first_fold_train": len(folds[0][0]),
            "fit_seconds": seconds,
            "evaluations": evaluations,
            "fold_solves": evaluations * len(cv),
            "iterations": model.n_iter_,
            "converged": bool(result.success),
            "optimizer_message": str(result.message),
            "lambda": model.lam_,
            "scaling": float(np.asarray(model.scaling_opt_).item()),
            "metric": "accuracy" if classification else "rmse",
            "test_score": float(metric(y_test, prediction)),
        }
        rows.append(row)
        print(
            f"{task} seed={seed} {protocol:7s} {seconds:7.3f}s "
            f"{evaluations:3d} evaluations / {row['fold_solves']:3d} fold solves "
            f"{row['metric']}={row['test_score']:.5f}, converged={row['converged']}",
            flush=True,
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=800)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--max-iter", type=int, default=40)
    parser.add_argument("--task", choices=["svc", "svr", "both"], default="both")
    parser.add_argument("--csv", type=Path, required=True)
    args = parser.parse_args()
    if args.samples < 50 or args.max_iter < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error("use at least 50 samples, a positive iteration limit and distinct seeds")

    print(f"Platform: {platform.platform()}, Python {platform.python_version()}")
    print(
        f"optimized-svm {optimized_svm.__version__}, smo-solver {smo_solver.__version__}, "
        f"NumPy {np.__version__}, SciPy {scipy.__version__}, scikit-learn {sklearn.__version__}"
    )
    print(
        f"Samples={args.samples}, seeds={args.seeds}, max_iter={args.max_iter}, task={args.task}. "
        f"RAYON_NUM_THREADS={os.environ.get('RAYON_NUM_THREADS', 'unset')}"
    )
    tasks = ["svc", "svr"] if args.task == "both" else [args.task]
    rows = []
    with threadpool_limits(limits=1):
        print(
            "Thread pools: "
            + ", ".join(f"{p['internal_api']}={p['num_threads']}" for p in threadpool_info()),
            flush=True,
        )
        for task in tasks:
            for i, seed in enumerate(args.seeds):
                rows.extend(
                    run_pair(task, args.samples, seed, args.max_iter, holdout_first=i % 2 == 1)
                )

    with args.csv.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    for task in tasks:
        cv = {r["seed"]: r for r in rows if r["task"] == task and r["protocol"] == "cv5"}
        holdout = {r["seed"]: r for r in rows if r["task"] == task and r["protocol"] == "holdout"}
        speedups = [cv[s]["fit_seconds"] / holdout[s]["fit_seconds"] for s in args.seeds]
        differences = [holdout[s]["test_score"] - cv[s]["test_score"] for s in args.seeds]
        print(
            f"{task}: paired speedup median {np.median(speedups):.2f}x "
            f"(range {min(speedups):.2f}–{max(speedups):.2f}x); "
            f"holdout minus CV test {next(iter(cv.values()))['metric']} median "
            f"{np.median(differences):+.5f} (range {min(differences):+.5f}–{max(differences):+.5f})"
        )


if __name__ == "__main__":
    main()

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

Add `--subset-fraction 0.5 --prescreen-iter 10 --refine-iter 5` and write to
`benchmarks/results/hpo_subset.csv` to add three protocols: short CV from the
original initialization and two fixed-subset prescreens followed by short CV.
The prescreen takes a subset of the first fold's training indices and uses its
unchanged validation indices. Both stages fit on the same development data
with different CV index sets. Their fit times include both estimator refits.
Only hyperparameters cross the stage boundary. In the scaled protocol the
training-size ratio scales the prescreen lambda initialization and bounds, and
its reciprocal rescales the learned lambda for the full CV stage. The direct
protocol uses lambda without either adjustment. The timing excludes generating
the shared splits and scoring the external test set. Peak RSS is not measured.

Use `--datasets moons friedman1 breast_cancer diabetes` to select datasets and
include a dataset ID in the CSV. Breast Cancer Wisconsin (Diagnostic) and
Diabetes use their complete scikit-learn datasets. Their features are
standardized once from the outer development features before making the
inner splits. The inner protocols share this preprocessing, and the outer
test data remains independent. Synthetic datasets use their original feature
scales.
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
from sklearn.datasets import load_breast_cancer, load_diabetes, make_friedman1, make_moons
from sklearn.metrics import accuracy_score, root_mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_info, threadpool_limits

import optimized_svm
from optimized_svm import OptimizedSVC, OptimizedSVR
from optimized_svm.estimators import auto_scaling

DATASET_TASKS = {
    "moons": "svc",
    "friedman1": "svr",
    "breast_cancer": "svc",
    "diabetes": "svr",
}


def run_pair(
    dataset: str,
    n: int,
    seed: int,
    max_iter: int,
    order_index: int,
    subset_fraction: float | None,
    prescreen_iter: int,
    refine_iter: int,
    selected_protocol: str | None,
    include_dataset: bool,
) -> list[dict]:
    task = DATASET_TASKS[dataset]
    classification = task == "svc"
    if dataset == "moons":
        x, y = make_moons(n_samples=n, noise=0.27, random_state=seed)
    elif dataset == "friedman1":
        x, y = make_friedman1(n_samples=n, noise=0.5, random_state=seed)
    elif dataset == "breast_cancer":
        x, y = load_breast_cancer(return_X_y=True)
    else:
        x, y = load_diabetes(return_X_y=True, scaled=False)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=seed, stratify=y if classification else None
    )
    if dataset in {"breast_cancer", "diabetes"}:
        scaler = StandardScaler().fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)
    splitter = StratifiedKFold if classification else KFold
    folds = list(splitter(n_splits=5, shuffle=True, random_state=seed).split(x_train, y_train))
    protocols = ["cv5", "holdout"]
    subset_fold = None
    subset_ratio = 1.0
    if subset_fraction is not None:
        reference, validation = folds[0]
        count = round(len(reference) * subset_fraction)
        if not 2 <= count <= len(reference) - 2:
            raise ValueError("subset needs at least two included and two excluded training rows")
        subset, _ = train_test_split(
            reference,
            train_size=count,
            random_state=seed,
            stratify=y_train[reference] if classification else None,
        )
        subset_fold = [(np.sort(subset), validation)]
        subset_ratio = len(subset) / len(reference)
        protocols.extend(["short_cv", "subset_scaled", "subset_direct"])
    shift = order_index % len(protocols)
    protocols = protocols[shift:] + protocols[:shift]
    if selected_protocol is not None:
        protocols = [selected_protocol]

    rows = []
    estimator = OptimizedSVC if classification else OptimizedSVR
    for order, protocol in enumerate(protocols, start=1):
        cv = folds[:1] if protocol == "holdout" else folds
        budget = max_iter if protocol in {"cv5", "holdout"} else refine_iter
        model = estimator(cv=cv, max_iter=budget, random_state=seed, n_jobs=1)
        prescreen = None
        prescreen_seconds = 0.0
        start = time.perf_counter()
        if subset_fold is not None:
            model.set_params(scaling_init=auto_scaling(x_train))
            if protocol in {"subset_scaled", "subset_direct"}:
                ratio = subset_ratio if protocol == "subset_scaled" else 1.0
                prescreen = estimator(
                    cv=subset_fold,
                    max_iter=prescreen_iter,
                    random_state=seed,
                    n_jobs=1,
                    scaling_init=model.scaling_init,
                    lam_init=model.lam_init * ratio,
                    lam_bounds=tuple(bound * ratio for bound in model.lam_bounds),
                )
                prescreen.fit(x_train, y_train)
                prescreen_seconds = time.perf_counter() - start
                model.set_params(
                    lam_init=float(np.clip(prescreen.lam_ / ratio, *model.lam_bounds)),
                    scaling_init=float(prescreen.scaling_opt_.item()),
                )
        model.fit(x_train, y_train)
        seconds = time.perf_counter() - start
        prediction = model.predict(x_test)
        metric = accuracy_score if classification else root_mean_squared_error
        result = model.optimize_result_
        final_evaluations = len(model.history_)
        prescreen_evaluations = len(prescreen.history_) if prescreen is not None else 0
        row = {
            "task": task,
            "samples": len(x),
            "seed": seed,
            "protocol": protocol,
            "order": order,
            "outer_train": len(x_train),
            "outer_test": len(x_test),
            "folds": len(cv),
            "first_fold_train": len(folds[0][0]),
            "fit_seconds": seconds,
            "evaluations": final_evaluations + prescreen_evaluations,
            "fold_solves": final_evaluations * len(cv) + prescreen_evaluations,
            "iterations": model.n_iter_,
            "converged": bool(result.success),
            "optimizer_message": str(result.message),
            "lambda": model.lam_,
            "scaling": float(np.asarray(model.scaling_opt_).item()),
            "metric": "accuracy" if classification else "rmse",
            "test_score": float(metric(y_test, prediction)),
        }
        if include_dataset:
            row["dataset"] = dataset
        if subset_fold is not None:
            row.update(
                subset_train=len(subset_fold[0][0]) if prescreen is not None else 0,
                prescreen_seconds=prescreen_seconds,
                prescreen_optimize_seconds=prescreen.time_ if prescreen is not None else 0.0,
                prescreen_evaluations=prescreen_evaluations,
                prescreen_iterations=prescreen.n_iter_ if prescreen is not None else 0,
                prescreen_message=(
                    str(prescreen.optimize_result_.message) if prescreen is not None else ""
                ),
                prescreen_lambda=prescreen.lam_ if prescreen is not None else "",
                final_evaluations=final_evaluations,
                final_cv_error=model.cv_error_,
                final_max_iter=budget,
                final_lam_init=model.lam_init,
                final_scaling_init=model.scaling_init,
                refits=2 if prescreen is not None else 1,
            )
        rows.append(row)
        print(
            f"{dataset if include_dataset else task} seed={seed} {protocol:13s} {seconds:7.3f}s "
            f"{row['evaluations']:3d} evaluations / {row['fold_solves']:3d} fold solves "
            f"{row['metric']}={row['test_score']:.5f}, converged={row['converged']}",
            flush=True,
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=800)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--max-iter", type=int, default=40)
    parser.add_argument("--subset-fraction", type=float)
    parser.add_argument("--prescreen-iter", type=int, default=10)
    parser.add_argument("--refine-iter", type=int, default=5)
    parser.add_argument(
        "--protocol", choices=["cv5", "holdout", "short_cv", "subset_scaled", "subset_direct"]
    )
    parser.add_argument("--task", choices=["svc", "svr", "both"], default="both")
    parser.add_argument("--datasets", nargs="+", choices=DATASET_TASKS)
    parser.add_argument("--csv", type=Path, required=True)
    args = parser.parse_args()
    if args.samples < 50 or args.max_iter < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error("use at least 50 samples, a positive iteration limit and distinct seeds")
    if args.subset_fraction is not None and (
        not 0 < args.subset_fraction < 1 or min(args.prescreen_iter, args.refine_iter) < 1
    ):
        parser.error(
            "subset fraction must be in (0, 1) and stage iteration limits must be positive"
        )
    if args.protocol is not None and args.subset_fraction is None:
        parser.error("protocol filtering requires --subset-fraction")

    if args.datasets is not None:
        if args.task != "both":
            parser.error(
                "--datasets selects tasks automatically and cannot be combined with --task"
            )
        if len(set(args.datasets)) != len(args.datasets):
            parser.error("use distinct dataset IDs")

    print(f"Platform: {platform.platform()}, Python {platform.python_version()}")
    print(
        f"optimized-svm {optimized_svm.__version__}, smo-solver {smo_solver.__version__}, "
        f"NumPy {np.__version__}, SciPy {scipy.__version__}, scikit-learn {sklearn.__version__}"
    )
    print(
        f"Samples={args.samples}, seeds={args.seeds}, max_iter={args.max_iter}, task={args.task}. "
        f"RAYON_NUM_THREADS={os.environ.get('RAYON_NUM_THREADS', 'unset')}"
    )
    if args.subset_fraction is not None:
        print(
            f"Subset fraction={args.subset_fraction}, prescreen_iter={args.prescreen_iter}, "
            f"refine_iter={args.refine_iter}, protocol={args.protocol or 'all'}. "
            "Timings include every refit. Peak RSS not measured."
        )
    datasets = args.datasets or [
        dataset
        for dataset in ("moons", "friedman1")
        if args.task in {"both", DATASET_TASKS[dataset]}
    ]
    if args.datasets is not None:
        print(f"Datasets={datasets}. --samples applies to synthetic data only.")
    rows = []
    per_dataset = {}
    with threadpool_limits(limits=1):
        print(
            "Thread pools: "
            + ", ".join(f"{p['internal_api']}={p['num_threads']}" for p in threadpool_info()),
            flush=True,
        )
        for dataset in datasets:
            dataset_rows = []
            for i, seed in enumerate(args.seeds):
                dataset_rows.extend(
                    run_pair(
                        dataset,
                        args.samples,
                        seed,
                        args.max_iter,
                        i,
                        args.subset_fraction,
                        args.prescreen_iter,
                        args.refine_iter,
                        args.protocol,
                        args.datasets is not None,
                    )
                )
            per_dataset[dataset] = dataset_rows
            rows.extend(dataset_rows)

    with args.csv.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    if args.protocol is not None:
        return

    for dataset, dataset_rows in per_dataset.items():
        label = dataset if args.datasets is not None else DATASET_TASKS[dataset]
        cv = {r["seed"]: r for r in dataset_rows if r["protocol"] == "cv5"}
        if args.subset_fraction is not None:
            for protocol in ("holdout", "short_cv", "subset_scaled", "subset_direct"):
                other = {r["seed"]: r for r in dataset_rows if r["protocol"] == protocol}
                speedups = [cv[s]["fit_seconds"] / other[s]["fit_seconds"] for s in args.seeds]
                deltas = [other[s]["test_score"] - cv[s]["test_score"] for s in args.seeds]
                metric = next(iter(cv.values()))["metric"]
                print(
                    f"{label} {protocol}: paired speedup median {np.median(speedups):.2f}x "
                    f"(range {min(speedups):.2f}–{max(speedups):.2f}x); test {metric} delta "
                    f"median {np.median(deltas):+.5f} (range {min(deltas):+.5f}–{max(deltas):+.5f})"
                )
            continue
        holdout = {r["seed"]: r for r in dataset_rows if r["protocol"] == "holdout"}
        speedups = [cv[s]["fit_seconds"] / holdout[s]["fit_seconds"] for s in args.seeds]
        differences = [holdout[s]["test_score"] - cv[s]["test_score"] for s in args.seeds]
        print(
            f"{label}: paired speedup median {np.median(speedups):.2f}x "
            f"(range {min(speedups):.2f}–{max(speedups):.2f}x); "
            f"holdout minus CV test {next(iter(cv.values()))['metric']} median "
            f"{np.median(differences):+.5f} (range {min(differences):+.5f}–{max(differences):+.5f})"
        )


if __name__ == "__main__":
    main()

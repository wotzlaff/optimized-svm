"""Wall-clock time of complete hyperparameter optimizations (for comparing releases).

Run the same script in environments with different versions of ``optimized-svm`` and
``smo-solver`` and compare the output::

    python benchmarks/hpo_timing.py > timing-new.json
    OPENBLAS_NUM_THREADS=1 python benchmarks/hpo_timing.py > timing-new-1thread.json

Each case is run ``--reps`` times and the minimum is reported.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import smo_solver
from sklearn.datasets import make_classification, make_friedman1
from sklearn.preprocessing import StandardScaler

import optimized_svm
from optimized_svm import OptimizedSVC, OptimizedSVR


def data(task: str, n: int):
    if task == "svc":
        x, y = make_classification(n, 8, n_informative=5, flip_y=0.1, random_state=0)
    else:
        x, y = make_friedman1(n, noise=0.5, random_state=0)
    return StandardScaler().fit_transform(x), y


CASES = [
    ("classification, n = 600", lambda: OptimizedSVC(max_iter=15), "svc", 600),
    ("classification, n = 1500", lambda: OptimizedSVC(max_iter=12), "svc", 1500),
    ("ε-SVR, n = 450", lambda: OptimizedSVR(max_iter=15), "svr", 450),
    ("ε-SVR, n = 1200", lambda: OptimizedSVR(max_iter=10), "svr", 1200),
    ("ε-SVR, one scaling per feature, n = 450",
     lambda: OptimizedSVR(max_iter=8, blocks="features"), "svr", 450),
    ("ε-SVR with ν-bound, n = 300", lambda: OptimizedSVR(max_iter=8, max_asum=40.0), "svr", 300),
]  # fmt: skip


def main(reps: int) -> None:
    # `__version__` instead of the package metadata, which is stale in editable installs
    out = {
        "optimized_svm": optimized_svm.__version__,
        "smo_solver": smo_solver.__version__,
        "cases": [],
    }
    for name, make, task, n in CASES:
        x, y = data(task, n)
        best, err = float("inf"), float("nan")
        for _ in range(reps):
            t = time.perf_counter()
            est = make().fit(x, y)
            best = min(best, time.perf_counter() - t)
            err = min(h["error"] for h in est.history_)
        out["cases"].append({"case": name, "seconds": best, "cv_error": err})
        print(f"{name}: {best:.1f}s", file=sys.stderr, flush=True)
    json.dump(out, sys.stdout, indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=2)
    main(ap.parse_args().reps)

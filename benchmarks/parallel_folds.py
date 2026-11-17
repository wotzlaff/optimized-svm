"""Wall time of OptimizedSVR / OptimizedSVC with serial vs. parallel cross-validation folds.

One case per process (clean peak RSS)::

    python benchmarks/parallel_folds.py svr 2000 --n-jobs 5 [--max-iter 100]

Prints a JSON line; run it for ``--n-jobs 1`` and ``--n-jobs 5`` and compare.
"""

from __future__ import annotations

import argparse
import json
import resource
import time

import numpy as np
from sklearn.datasets import make_classification, make_friedman1
from sklearn.preprocessing import StandardScaler

from optimized_svm import OptimizedSVC, OptimizedSVR


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["svr", "svc"])
    ap.add_argument("n", type=int)
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--max-iter", type=int, default=100)
    a = ap.parse_args()
    if a.task == "svc":
        x, y = make_classification(a.n, 8, n_informative=5, flip_y=0.1, random_state=0)
        est = OptimizedSVC(max_iter=a.max_iter, n_jobs=None if a.n_jobs == 1 else a.n_jobs)
    else:
        x, y = make_friedman1(a.n, noise=1.0, random_state=0)
        est = OptimizedSVR(max_iter=a.max_iter, n_jobs=None if a.n_jobs == 1 else a.n_jobs)
    x = StandardScaler().fit_transform(x)
    t0 = time.perf_counter()
    est.fit(x, y)
    wall = time.perf_counter() - t0
    print(
        json.dumps(
            {
                "task": a.task,
                "n": a.n,
                "n_jobs": a.n_jobs,
                "max_iter": a.max_iter,
                "wall_s": round(wall, 2),
                "n_evals": len(est.history_),
                "cv_error": min(float(h["error"]) for h in est.history_),
                "lam": float(est.lam_),
                "scaling": np.atleast_1d(est.scaling_opt_).tolist(),
                "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
            }
        )
    )


if __name__ == "__main__":
    main()

"""Hyperparameter optimization with the full model and with the Subspace SVM (``n_basis="auto"``).

OptimizedSVR on Friedman #1 (standardized features), 5-fold CV, default settings; reports
wall-clock time, CV error, test error and the optimized hyperparameters.

Usage::

    python benchmarks/subspace_hpo.py results.json
"""

import json
import sys
import time
import warnings

from sklearn.datasets import make_friedman1
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from optimized_svm import OptimizedSVR


def main(out: str) -> None:
    rows = []
    for n in (5000, 10000):
        x, t = make_friedman1(n + 2000, noise=1.0, random_state=0)
        x = StandardScaler().fit_transform(x)
        xtr, ttr, xte, tte = x[:n], t[:n], x[n:], t[n:]
        for label, kw in (
            ("subspace tol 1", {"n_basis": "auto", "basis_tol": 1.0}),
            ("subspace tol 10", {"n_basis": "auto", "basis_tol": 10.0}),
            ("full", {}),
        ):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                t0 = time.perf_counter()
                est = OptimizedSVR(max_iter=30, **kw).fit(xtr, ttr)
                elapsed = time.perf_counter() - t0
            row = {
                "n": n, "variant": label, "time": elapsed, "evaluations": len(est.history_),
                "cv_error": float(est.cv_error_), "lam": float(est.lam_),
                "scaling": [float(s) for s in est.scaling_opt_],
                "test_mse": float(mean_squared_error(tte, est.predict(xte))),
                "n_basis": [h.get("n_basis") for h in est.history_][:1],
                "final_basis": int(len(est.basis_)) if hasattr(est, "basis_") else None,
                "warnings": sorted({str(w.message)[:120] for w in caught}),
            }  # fmt: skip
            rows.append(row)
            print(row, flush=True)
            json.dump(rows, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])

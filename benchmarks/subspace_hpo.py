"""Hyperparameter optimization with the full model and with the Subspace SVM (``n_basis="auto"``).

OptimizedSVR on Friedman #1 (standardized features), 5-fold CV, default settings; reports
wall-clock time, CV error, test error and the optimized hyperparameters.

Usage::

    python benchmarks/subspace_hpo.py results.json [--n 5000 10000] [--noise 1.0] [--lam-init 0.1]
        [--scaling-init 0.05] [--variants tol1 tol10 full]
"""

import argparse
import json
import time
import warnings

from sklearn.datasets import make_friedman1
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from optimized_svm import OptimizedSVR

VARIANTS = {
    "tol1": ("subspace tol 1", {"n_basis": "auto", "basis_tol": 1.0}),
    "tol10": ("subspace tol 10", {"n_basis": "auto", "basis_tol": 10.0}),
    "full": ("full", {}),
}


def main(
    out: str,
    ns: list[int],
    noise: float,
    lam_init: float | None,
    scaling_init: float | None,
    variants: list[str],
) -> None:
    rows = []
    init = {}
    if lam_init is not None:
        init["lam_init"] = lam_init
    if scaling_init is not None:
        init["scaling_init"] = scaling_init
    for n in ns:
        x, t = make_friedman1(n + 2000, noise=noise, random_state=0)
        x = StandardScaler().fit_transform(x)
        xtr, ttr, xte, tte = x[:n], t[:n], x[n:], t[n:]
        for label, kw in (VARIANTS[v] for v in variants):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                t0 = time.perf_counter()
                est = OptimizedSVR(max_iter=30, **init, **kw).fit(xtr, ttr)
                elapsed = time.perf_counter() - t0
            row = {
                "n": n, "noise": noise, "variant": label,
                "lam_init": lam_init, "scaling_init": scaling_init,
                "time": elapsed, "evaluations": len(est.history_),
                "cv_error": float(est.cv_error_), "lam": float(est.lam_),
                "scaling": [float(s) for s in est.scaling_opt_],
                "test_mse": float(mean_squared_error(tte, est.predict(xte))),
                "n_basis": [h.get("n_basis") for h in est.history_][:1],
                "n_basis_last": est.history_[-1].get("n_basis"),
                "basis_rounds": max(h.get("basis_round", 0) for h in est.history_) + 1,
                "final_basis": int(len(est.basis_)) if hasattr(est, "basis_") else None,
                "warnings": sorted({str(w.message)[:120] for w in caught}),
            }  # fmt: skip
            rows.append(row)
            print(row, flush=True)
            json.dump(rows, open(out, "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--n", type=int, nargs="+", default=[5000, 10000])
    ap.add_argument("--noise", type=float, default=1.0)
    ap.add_argument("--lam-init", type=float, default=None)
    ap.add_argument("--scaling-init", type=float, default=None)
    ap.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=list(VARIANTS))
    a = ap.parse_args()
    main(a.out, a.n, a.noise, a.lam_init, a.scaling_init, a.variants)

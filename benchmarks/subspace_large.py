"""Subspace SVM against the full model for larger n (ε-SVR, Friedman #1, basis: select_basis).

Usage::

    python benchmarks/subspace_large.py results.json
"""

import json
import sys
import time

import numpy as np
from sklearn.datasets import make_friedman1
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from optimized_svm import EpsilonInsensitiveLoss, cv_error_and_gradient
from optimized_svm.subspace import select_basis
from optimized_svm.validation import SmoothAbsoluteValidation


def main(out: str) -> None:
    rows = []
    loss, val = EpsilonInsensitiveLoss(0.1, 0.3), SmoothAbsoluteValidation()
    for n in (5000, 10000):
        x, t = make_friedman1(n, noise=1.0, random_state=0)
        t = (t - t.mean()) / t.std()
        x = StandardScaler().fit_transform(x)
        folds = list(KFold(5, shuffle=True, random_state=0).split(x))
        w = np.ones(n)
        for scale, lam in ((0.05, 0.1), (0.05, 0.01), (0.1, 0.1)):
            theta = np.array([np.log(lam), np.log(scale)])
            t0 = time.perf_counter()
            full = cv_error_and_gradient(theta, x, t, w, folds, loss, val)
            row = {"n": n, "scale": scale, "lam": lam, "full_error": full.error,
                   "full_time": time.perf_counter() - t0, "runs": []}  # fmt: skip
            print(
                n, scale, lam, f"full {row['full_error']:.4f} {row['full_time']:.1f}s", flush=True
            )
            for tol in (1.0, 10.0):
                t0 = time.perf_counter()
                sel = [select_basis(x[tr] * scale, lam, tol, max_size=n // 4) for tr, _ in folds]
                basis = [tr[s[0]] for (tr, _), s in zip(folds, sel, strict=True)]
                t_sel = time.perf_counter() - t0
                t0 = time.perf_counter()
                r = cv_error_and_gradient(theta, x, t, w, folds, loss, val, basis=basis)
                run = {"tol": tol, "m": float(np.mean([len(b) for b in basis])),
                       "ratio": max(s[1] for s in sel), "error": r.error,
                       "select_time": t_sel, "time": time.perf_counter() - t0}  # fmt: skip
                row["runs"].append(run)
                print("  ", run, flush=True)
            rows.append(row)
            json.dump(rows, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])

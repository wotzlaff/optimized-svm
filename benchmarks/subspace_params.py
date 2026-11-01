"""Choosing the basis of a Subspace SVM: size and selection rule (exploratory study).

For each setting (task, n, kernel scale, lambda) the cross-validation error of the full model is
compared with Subspace SVMs whose basis is drawn at random or chosen by greedy pivoted Cholesky
(largest remaining kernel variance first). The pivoted Cholesky also gives the relative trace of the
Nystroem residual ``tr(K - K_s) / tr(K)``, which can serve as a stopping rule for the basis size.

Usage::

    python benchmarks/subspace_params.py results.json
"""

import json
import sys
import time

import numpy as np
from sklearn.datasets import make_classification, make_friedman1
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from optimized_svm import EpsilonInsensitiveLoss, HingeLoss, cv_error_and_gradient
from optimized_svm.kernels import rbf
from optimized_svm.subspace import choose_basis
from optimized_svm.validation import SmoothAbsoluteValidation, SmoothHingeValidation

SIZES = (25, 50, 100, 200, 400, 800)


def pivoted_cholesky(x: np.ndarray, m: int) -> tuple[np.ndarray, np.ndarray]:
    """Greedy pivots for the RBF kernel on scaled ``x`` and the relative residual trace after each
    pivot (``exp(-|x - z|^2)``, so the trace of K is ``n``)."""
    n = len(x)
    diag = np.ones(n)
    chol = np.zeros((m, n))
    pivots: list[int] = []
    trace = [1.0]
    for j in range(m):
        p = int(np.argmax(diag))
        pivots.append(p)
        col = rbf(x, x[p : p + 1])[:, 0] - chol[:j, p] @ chol[:j]
        chol[j] = col / np.sqrt(max(diag[p], 1e-300))
        diag = np.maximum(diag - chol[j] ** 2, 0.0)
        diag[pivots] = 0.0
        trace.append(float(diag.sum() / n))
    return np.array(pivots), np.array(trace)


def settings():
    for n in (1000, 2500):
        x, t = make_friedman1(n, noise=1.0, random_state=0)
        t = (t - t.mean()) / t.std()
        yield "regression", n, StandardScaler().fit_transform(x), t
        x, y = make_classification(
            n, 10, n_informative=5, n_redundant=2, flip_y=0.05, class_sep=1.0, random_state=0
        )
        yield "classification", n, StandardScaler().fit_transform(x), 2.0 * y - 1.0


def main(out: str) -> None:
    rows = []
    for task, n, x, y in settings():
        loss = (
            EpsilonInsensitiveLoss(0.1, 0.3) if task == "regression" else HingeLoss(smoothing=0.3)
        )
        val = SmoothAbsoluteValidation() if task == "regression" else SmoothHingeValidation()
        folds = list(KFold(5, shuffle=True, random_state=0).split(x))
        w = np.ones(n)
        for scale in (0.05, 0.15, 0.5):
            for lam in (1e-3, 1e-1):
                theta = np.array([np.log(lam), np.log(scale)])
                t0 = time.perf_counter()
                full = cv_error_and_gradient(theta, x, y, w, folds, loss, val)
                row = {"task": task, "n": n, "scale": scale, "lam": lam, "full_error": full.error,
                       "full_time": time.perf_counter() - t0, "runs": []}  # fmt: skip
                xs = x * scale
                piv = [pivoted_cholesky(xs[tr], max(SIZES)) for tr, _ in folds]
                for m in SIZES:
                    if m > 0.8 * n * 0.8:
                        continue
                    for rule in ("random", "pivoted"):
                        if rule == "random":
                            basis = [
                                tr[choose_basis(len(tr), m, seed=k)]
                                for k, (tr, _) in enumerate(folds)
                            ]
                        else:
                            basis = [tr[p[0][:m]] for (tr, _), p in zip(folds, piv, strict=True)]
                        t0 = time.perf_counter()
                        r = cv_error_and_gradient(theta, x, y, w, folds, loss, val, basis=basis)
                        row["runs"].append({
                            "m": m, "rule": rule, "error": r.error,
                            "time": time.perf_counter() - t0,
                            "trace": float(np.mean([p[1][m] for p in piv])),
                            "grad_err": float(np.linalg.norm(r.gradient - full.gradient)
                                              / max(np.linalg.norm(full.gradient), 1e-12)),
                        })  # fmt: skip
                rows.append(row)
                print(
                    task,
                    n,
                    scale,
                    lam,
                    f"full {row['full_error']:.4f} {row['full_time']:.1f}s",
                    flush=True,
                )
                json.dump(rows, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])

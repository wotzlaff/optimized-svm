"""Cross-validation error and its gradient w.r.t. the hyperparameters (Strasdat, Sec. 9.2).

Hyperparameters are ``theta = (log lam, log s_1, ..., log s_K)`` (regularization and feature
block scalings). For each fold the training problem is solved and the derivative of the
solution is obtained from the implicit function theorem applied to the (semismooth) optimality
conditions ``a_l + phi_l'(ka_l + b + s_l c) = 0``, ``sum(a) = 0`` (and ``s^T a = A``). Only the
terms with positive curvature ``h_l = phi_l''`` react to parameter changes. Instead of the
``K`` forward sensitivities, a single *adjoint* system is solved per fold:

``[[K_PP / lam + diag(1 / h_P), 1, s_P], [1^T, 0, 0], [s_P^T, 0, 0]] [u; .]``
``= [K_VP^T dE / lam; sum dE; 0]``

after which ``dE/dtheta = -u^T (d ka_P / dtheta) + dE^T (d f_V / dtheta)`` with ``a`` fixed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .kernels import Blocks, contract_gradient, rbf, rbf_parts, scale_features
from .linalg import solve_bordered
from .losses import Loss
from .problem import TrainingProblem
from .training import Solution, train
from .validation import ValidationLoss

Array = NDArray[np.float64]
Fold = tuple[NDArray[np.intp], NDArray[np.intp]]


@dataclass
class FoldResult:
    error: float
    gradient: Array
    solution: Solution
    n_support: int
    time: float


def fold_error_and_gradient(
    x_train: Array,
    y_train: Array,
    w_train: Array,
    x_val: Array,
    y_val: Array,
    loss: Loss,
    validation: ValidationLoss,
    lam: float,
    blocks: Blocks,
    max_asum: float | None = None,
    smo_options: dict[str, Any] | None = None,
    newton_options: dict[str, Any] | None = None,
    alpha0: Array | None = None,
) -> FoldResult:
    """Validation error of one fold and its gradient w.r.t. ``(log lam, log s_k)``.

    ``x_train``/``x_val`` must already be scaled.
    """
    prob = TrainingProblem(x_train, y_train, loss, lam, w_train, max_asum)
    sol = train(prob, smo_options, newton_options, alpha0)
    a = sol.a
    sv = np.flatnonzero(a)
    a_sv = a[sv] / lam
    x_terms = prob.x[prob.terms.kernel_index]

    # validation predictions and explicit derivatives (a fixed)
    k_val_sv, d_val_sv = rbf_parts(x_val, x_terms[sv], blocks)
    ka_val = k_val_sv @ a_sv
    f_val = ka_val + sol.b
    err, d_err = validation(f_val, y_val)
    grad = np.zeros(1 + len(d_val_sv))
    grad[0] = -d_err @ ka_val
    grad[1:] = contract_gradient(k_val_sv, d_val_sv, d_err, a_sv)

    # implicit part: adjoint system on terms with positive curvature
    dec = prob.decision(sol.b, sol.c, sol.ka)
    h = prob.loss.d2(dec, prob.terms)
    p = np.flatnonzero(h > 0.0)
    if p.size > 0:
        k_p_sv, d_p_sv = rbf_parts(x_terms[p], x_terms[sv], blocks)
        q = prob.kernel(p, p) / lam + np.diag(1.0 / h[p])
        cols = [np.ones(p.size)]
        rhs0 = [d_err.sum()]
        if max_asum is not None and sol.c > 0.0:
            cols.append(prob.sign[p])
            rhs0.append(0.0)
        k_val_p = rbf(x_val, x_terms[p])
        u, _ = solve_bordered(q, np.stack(cols, axis=1), k_val_p.T @ d_err / lam, np.array(rhs0))
        # d ka_P / d log lam = -ka_P,  d ka_P / d log s_k = dK_{P,SV} a_SV / lam
        grad[0] += u @ sol.ka[p]
        grad[1:] -= contract_gradient(k_p_sv, d_p_sv, u, a_sv)
    return FoldResult(err, grad, sol, int(sv.size), sol.time_smo + sol.time_newton)


@dataclass
class CVResult:
    error: float
    gradient: Array
    folds: list[FoldResult] = field(default_factory=list)


def cv_error_and_gradient(
    theta: Array,
    x: Array,
    y: Array,
    w: Array,
    folds: Sequence[Fold],
    loss: Loss,
    validation: ValidationLoss,
    blocks: Blocks = None,
    max_asum: float | None = None,
    smo_options: dict[str, Any] | None = None,
    newton_options: dict[str, Any] | None = None,
    warm_start: list[Array | None] | None = None,
) -> CVResult:
    """Mean validation error over all folds and its gradient w.r.t. ``theta``.

    ``warm_start`` is a list with one (possibly ``None``) start vector per fold; it is updated
    in place with the new solutions.
    """
    lam = float(np.exp(theta[0]))
    xs = scale_features(x, np.exp(theta[1:]), blocks)
    total, grad, results = 0.0, np.zeros_like(theta, dtype=np.float64), []
    n_val = 0
    for i, (train_idx, val_idx) in enumerate(folds):
        start = warm_start[i] if warm_start is not None else None
        res = fold_error_and_gradient(
            xs[train_idx],
            y[train_idx],
            w[train_idx],
            xs[val_idx],
            y[val_idx],
            loss,
            validation,
            lam,
            blocks,
            max_asum,
            smo_options,
            newton_options,
            start,
        )
        if warm_start is not None:
            warm_start[i] = res.solution.a
        total += res.error
        grad += res.gradient
        n_val += len(val_idx)
        results.append(res)
    return CVResult(total / n_val, grad / n_val, results)

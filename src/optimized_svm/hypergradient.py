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

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from .kernels import Blocks, contract_gradient, n_blocks, rbf, scale_features
from .linalg import solve_bordered
from .losses import Loss
from .problem import TrainingProblem
from .training import Solution, train
from .validation import ValidationLoss

Array = NDArray[np.float64]
Fold = tuple[NDArray[np.intp], NDArray[np.intp]]


def _kernel_columns(
    x_val: Array, x: Array, k_val_sv: Array, sv: NDArray[np.intp], cols: NDArray[np.intp]
) -> Array:
    """Kernel matrix between ``x_val`` and the samples ``cols``, reusing the columns of
    ``k_val_sv`` (kernel to the samples ``sv``, sorted) where they overlap."""
    pos = np.minimum(np.searchsorted(sv, cols), sv.size - 1)
    known = sv[pos] == cols if sv.size else np.zeros(cols.size, dtype=bool)
    if known.all():
        return k_val_sv[:, pos]
    out = np.empty((x_val.shape[0], cols.size))
    out[:, known] = k_val_sv[:, pos[known]]
    out[:, ~known] = rbf(x_val, x[cols[~known]])
    return out


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
    loss_params: Sequence[str] = (),
) -> FoldResult:
    """Validation error of one fold and its gradient w.r.t. ``(log lam, log s_k, log p_j)``.

    ``p_j`` are the loss parameters named in ``loss_params`` (e.g. ``"epsilon"``, ``"smoothing"``),
    taken from ``loss``. ``x_train``/``x_val`` must already be scaled.
    """
    prob = TrainingProblem(x_train, y_train, loss, lam, w_train, max_asum)
    sol = train(prob, smo_options, newton_options, alpha0)
    a = sol.a
    lam_inv = 1.0 / lam
    # everything below works on samples (the two terms of a sample share their kernel row)
    beta = prob.sample_coefficients(a) * lam_inv
    sv = np.flatnonzero(beta)
    x_sv = prob.x[sv]

    # validation predictions and explicit derivatives (a fixed)
    k_val_sv = rbf(x_val, x_sv)
    ka_val = k_val_sv @ beta[sv]
    f_val = ka_val + sol.b
    err, d_err = validation(f_val, y_val)
    grad = np.zeros(1 + n_blocks(blocks) + len(loss_params))
    grad[0] = -d_err @ ka_val
    grad[1 : 1 + n_blocks(blocks)] = contract_gradient(
        x_val, x_sv, k_val_sv, d_err, beta[sv], blocks
    )

    # implicit part: adjoint system on terms with positive curvature
    dec = prob.decision(sol.b, sol.c, sol.ka)
    h = prob.loss.d2(dec, prob.terms)
    p = np.flatnonzero(h > 0.0)
    if p.size > 0:
        p_samples, p_inv = np.unique(prob.terms.kernel_index[p], return_inverse=True)
        factor = sol.factor
        if factor is not None and factor.matches(p, h[p]):
            # same curvature pattern as in the last Newton system: reuse its factorisation
            q, cho = None, factor.cho
        else:
            q, cho = prob.kernel(p, p) * lam_inv + np.diag(1.0 / h[p]), None
        cols: list[Array] = [np.ones(p.size)]
        rhs0 = [d_err.sum()]
        if max_asum is not None and sol.c > 0.0:
            cols.append(prob.sign[p])
            rhs0.append(0.0)
        k_val_p = _kernel_columns(x_val, prob.x, k_val_sv, sv, p_samples)
        rhs = (k_val_p.T @ d_err)[p_inv] * lam_inv
        u, _ = solve_bordered(q, np.stack(cols, axis=1), rhs, np.array(rhs0), cho)
        # d ka_P / d log lam = -ka_P,  d ka_P / d log s_k = dK_{P,SV} a_SV / lam
        grad[0] += u @ sol.ka[p]
        u_samples = np.bincount(p_inv, weights=u, minlength=p_samples.size).astype(np.float64)
        k_p_sv = prob.sample_kernel(p_samples, sv)
        nb = n_blocks(blocks)
        grad[1 : 1 + nb] -= contract_gradient(
            prob.x[p_samples], x_sv, k_p_sv, u_samples, beta[sv], blocks
        )
        # loss parameters enter the optimality condition a + phi'(t; p) = 0 directly:
        # dE/dp = -u^T (d phi'/dp / phi'') on the terms with positive curvature
        for j, name in enumerate(loss_params):
            ratio = loss.param_ratio(name, dec, prob.terms)[p]
            grad[1 + nb + j] = -float(u @ ratio) * float(getattr(loss, name))
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
    loss_params: Sequence[str] = (),
) -> CVResult:
    """Mean validation error over all folds and its gradient w.r.t. ``theta``.

    ``theta = (log lam, log s_1, ..., log s_K, log p_1, ...)``: regularization, kernel scalings and
    the values of the loss parameters named in ``loss_params`` (fields of ``loss`` such as
    ``"epsilon"`` or ``"smoothing"``; they replace the values stored in ``loss``).

    ``warm_start`` is a list with one (possibly ``None``) start vector per fold; it is updated
    in place with the new solutions.
    """
    nb = 1 if blocks is None else len(blocks)
    lam = float(np.exp(theta[0]))
    xs = scale_features(x, np.exp(theta[1 : 1 + nb]), blocks)
    if loss_params:
        probe = loss.terms(np.ones(1), np.ones(1))
        for name in loss_params:  # raises ValueError for parameters that cannot be tuned
            loss.param_ratio(name, np.zeros(probe.size), probe)
        loss = dataclasses.replace(
            cast(Any, loss),
            **{n: float(np.exp(v)) for n, v in zip(loss_params, theta[1 + nb :], strict=True)},
        )
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
            loss_params,
        )
        if warm_start is not None:
            warm_start[i] = res.solution.a
        total += res.error
        grad += res.gradient
        n_val += len(val_idx)
        results.append(res)
    return CVResult(total / n_val, grad / n_val, results)

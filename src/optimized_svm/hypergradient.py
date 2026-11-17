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
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeVar, cast

import numpy as np
from joblib import Parallel, delayed, effective_n_jobs
from numpy.typing import NDArray

from .kernels import Blocks, contract_gradient, n_blocks, rbf, scale_features
from .linalg import blas_threads, parallel_section, solve_bordered
from .losses import HardEpsilonInsensitiveLoss, Loss
from .problem import TrainingProblem
from .subspace import SubspaceFoldResult, fold_error_and_gradient_subspace
from .training import Solution, train
from .validation import ValidationLoss

Array = NDArray[np.float64]
Fold = tuple[NDArray[np.intp], NDArray[np.intp]]
_T = TypeVar("_T")


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
    offset_train: Array | None = None,
    w_val: Array | None = None,
    offset_val: Array | None = None,
) -> FoldResult:
    """Validation error of one fold and its gradient w.r.t. ``(log lam, log s_k, log p_j)``.

    ``p_j`` are the loss parameters named in ``loss_params`` (e.g. ``"epsilon"``, ``"smoothing"``),
    taken from ``loss``. ``x_train``/``x_val`` must already be scaled. ``offset_train`` and
    ``offset_val`` are per-sample offsets added to the decision values (``None``: zero); ``w_val``
    are validation weights, passed to ``validation`` only if it is ``weighted``.
    """
    prob = TrainingProblem(x_train, y_train, loss, lam, w_train, max_asum, offset_train)
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
    if offset_val is not None:
        f_val = f_val + offset_val
    err, d_err = (
        validation(f_val, y_val, w_val) if validation.weighted else validation(f_val, y_val)
    )
    grad = np.zeros(1 + n_blocks(blocks) + len(loss_params))
    grad[0] = -d_err @ ka_val
    grad[1 : 1 + n_blocks(blocks)] = contract_gradient(
        x_val, x_sv, k_val_sv, d_err, beta[sv], blocks
    )

    # implicit part: adjoint system on terms with positive curvature
    dec = prob.decision(sol.b, sol.c, sol.ka)
    hard = prob.loss.hard
    h = np.zeros(prob.size) if hard else prob.loss.d2(dec, prob.terms)
    p = sol.info["active"] if hard else np.flatnonzero(h > 0.0)
    if p.size > 0:
        p_samples, p_inv = np.unique(prob.terms.kernel_index[p], return_inverse=True)
        factor = sol.factor
        if factor is not None and factor.matches(p, h[p]):
            # same curvature pattern as in the last Newton system: reuse its factorisation
            q, cho = None, factor.cho
        else:
            q = prob.kernel(p, p) * lam_inv
            if not hard:
                q = q + np.diag(1.0 / h[p])
            cho = None
        cols: list[Array] = [np.ones(p.size)]
        rhs0 = [d_err.sum()]
        if max_asum is not None and sol.c > 0.0:
            cols.append(prob.sign[p])
            rhs0.append(0.0)
        k_val_p = _kernel_columns(x_val, prob.x, k_val_sv, sv, p_samples)
        rhs = (k_val_p.T @ d_err)[p_inv] * lam_inv
        if hard:
            # A hard active set can contain duplicate/dependent constraints. A singular
            # bordered system has no unique local sensitivity: report this explicitly.
            assert q is not None
            system = np.block(
                [
                    [prob.kernel(p, p), np.ones((p.size, 1))],
                    [np.ones((1, p.size)), np.zeros((1, 1))],
                ]
            )
            if np.linalg.cond(system) > 1e12:
                raise ValueError("hard active set is degenerate; hypergradient is not unique")
            adjoint = np.linalg.solve(system, np.r_[rhs * lam, rhs0])
            u = adjoint[:-1]
        else:
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
            ratio = (
                np.sign(sol.a[p])
                if hard and name == "epsilon"
                else (loss.param_ratio(name, dec, prob.terms)[p])
            )
            grad[1 + nb + j] = -float(u @ ratio) * float(getattr(loss, name))
    if hard:
        grad[0] = 0.0  # lambda rescales the norm objective only; the predictor is invariant
    # loss parameters that also enter the validation loss (e.g. the negative binomial
    # likelihood): direct term p dE/dp at fixed predictions
    for j, name in enumerate(loss_params):
        if name in validation.params:
            wv = (w_val,) if validation.weighted else ()
            grad[1 + n_blocks(blocks) + j] += float(getattr(loss, name)) * validation.param_grad(
                name, f_val, y_val, *wv
            )
    return FoldResult(err, grad, sol, int(sv.size), sol.time_smo + sol.time_newton)


@dataclass
class CVResult:
    error: float
    gradient: Array
    folds: list[FoldResult | SubspaceFoldResult] = field(default_factory=list)


def run_folds(run_fold: Callable[[int], _T], n_folds: int, n_jobs: int | None = None) -> list[_T]:
    """Runs ``run_fold(i)`` for ``i = 0, ..., n_folds - 1``, serially or in parallel threads.

    ``n_jobs`` has scikit-learn semantics (``None`` or 1: serial, -1: all cores). The folds run in
    threads (the SMO solver and BLAS release the GIL, nothing is copied). The BLAS threads are
    divided among the workers; the limit is global to the process, so it is set once here and the
    nested limits in :mod:`optimized_svm.linalg` are inactive meanwhile. ``run_fold`` must not
    touch shared state. The results are returned in fold order.
    """
    if n_jobs is None or n_jobs == 1 or n_folds < 2:
        return [run_fold(i) for i in range(n_folds)]
    n_workers = max(1, min(n_folds, effective_n_jobs(n_jobs)))
    with parallel_section(max(1, blas_threads() // n_workers)):
        return list(
            Parallel(n_jobs=n_workers, prefer="threads")(
                delayed(run_fold)(i) for i in range(n_folds)
            )
        )


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
    basis: Sequence[NDArray[np.intp]] | None = None,
    offset: Array | None = None,
    n_jobs: int | None = None,
) -> CVResult:
    """Mean validation error over all folds and its gradient w.r.t. ``theta``.

    ``theta = (log lam, log s_1, ..., log s_K, log p_1, ...)``: regularization, kernel scalings and
    the values of the loss parameters named in ``loss_params`` (fields of ``loss`` such as
    ``"epsilon"`` or ``"smoothing"``; they replace the values stored in ``loss``).

    ``warm_start`` is a list with one (possibly ``None``) start vector per fold; it is updated
    in place with the new solutions.

    With ``basis`` (one array of sample indices per fold, indices into ``x``) every fold trains a
    *Subspace SVM* whose model is spanned by these basis points (:mod:`optimized_svm.subspace`);
    ``warm_start`` then holds expansion coefficients instead of dual variables. Loss parameters and
    the 1-norm constraint are not supported in that case.

    ``n_jobs`` runs the folds in parallel threads (scikit-learn semantics: ``None`` or 1 is serial,
    -1 uses all cores); the BLAS threads are divided among the workers. The result equals the
    serial one up to floating-point rounding (the BLAS thread count differs; two parallel runs are
    bit-identical). Parallel folds pay off most when SMO dominates (classification); for large
    Newton-dominated problems the serial run already uses multithreaded BLAS, and the memory grows
    with the number of parallel folds (each holds its kernel and factor matrices).

    ``offset`` is an optional per-sample offset (like ``w``): it is added to the decision values
    of the training problem and of the validation predictions. If ``validation.weighted`` is set,
    ``w[val_idx]`` is also passed to the validation loss.
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
        tuned = {
            n: float(np.exp(v))
            for n, v in zip(loss_params, theta[1 + nb :], strict=True)
            if n in validation.params
        }
        if tuned:  # validation losses that share the tuned parameter use its current value
            validation = dataclasses.replace(cast(Any, validation), **tuned)
    weighted = validation.weighted
    if basis is not None and loss.hard:
        raise ValueError("hard constraints do not support a subspace basis")
    if basis is not None and (loss_params or max_asum is not None):
        raise ValueError("loss_params and max_asum are not supported with a subspace basis")

    def run_fold(i: int) -> FoldResult | SubspaceFoldResult:
        """Trains and validates fold ``i``; touches no shared state (safe in threads)."""
        train_idx, val_idx = folds[i]
        start = warm_start[i] if warm_start is not None else None
        if basis is not None:
            return fold_error_and_gradient_subspace(
                xs[train_idx], y[train_idx], w[train_idx], xs[basis[i]], xs[val_idx], y[val_idx],
                loss, validation, lam, blocks, newton_options, start,
                None if offset is None else offset[train_idx],
                w[val_idx] if weighted else None,
                None if offset is None else offset[val_idx],
            )  # fmt: skip
        fold_loss = loss
        if isinstance(loss, HardEpsilonInsensitiveLoss) and np.ndim(loss.epsilon) == 1:
            eps = loss.tolerances(y.size)
            fold_loss = dataclasses.replace(loss, epsilon=eps[train_idx])
        return fold_error_and_gradient(
            xs[train_idx],
            y[train_idx],
            w[train_idx],
            xs[val_idx],
            y[val_idx],
            fold_loss,
            validation,
            lam,
            blocks,
            max_asum,
            smo_options,
            newton_options,
            start,
            loss_params,
            None if offset is None else offset[train_idx],
            w[val_idx] if weighted else None,
            None if offset is None else offset[val_idx],
        )

    results = run_folds(run_fold, len(folds), n_jobs)
    # summed and stored in fold order, so that the result does not depend on ``n_jobs``
    total, grad = 0.0, np.zeros_like(theta, dtype=np.float64)
    n_val = 0
    for i, res in enumerate(results):
        if warm_start is not None:
            if isinstance(res, SubspaceFoldResult):
                warm_start[i] = res.solution.beta
            else:
                warm_start[i] = res.solution.a
        total += res.error
        grad += res.gradient
        n_val += len(folds[i][1])
    return CVResult(total / n_val, grad / n_val, results)

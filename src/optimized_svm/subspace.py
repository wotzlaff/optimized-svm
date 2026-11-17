"""Subspace SVM: training and hypergradient in a low-dimensional subspace (Strasdat, Sec. 5.3).

The model is restricted to the subspace spanned by ``m`` basis points ``z_1, ..., z_m``::

    w = sum_j beta_j phi(z_j),    f(x) = sum_j beta_j k(x, z_j) + b.

By Corollary 5.9 the dual of this problem is the standard dual with the *projected* kernel
``K_s = K_XZ K_ZZ^{-1} K_ZX`` (a Nystroem approximation), so a Subspace SVM is an SVM with the
explicit finite-dimensional feature map ``psi(x) = L^{-1} k_Z(x)`` where ``K_ZZ = L L^T``. The
primal problem has only ``m + 1`` unknowns,

    min_{u, b}  lam/2 |u|^2 + sum_l phi_l(psi_{k(l)}^T u + b),        beta = L^{-T} u,

and is solved by a damped Newton method with systems of size ``m + 1`` instead of ``n``: cost
``O(n m^2 + m^3)`` per step. Losses must have positive curvature somewhere (smoothed hinge or
epsilon-insensitive loss with ``smoothing > 0``, squared, logistic, Poisson); SMO is not used.

The hypergradient of the cross-validation error w.r.t. ``log lam`` and the log kernel scalings
follows from the implicit function theorem for the stationarity conditions in ``(beta, b)``; one
adjoint system of size ``m + 1`` is solved per fold (derivation in ``docs/theory/subspace.md``).
"""

from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import cho_factor, cho_solve, cholesky, solve_triangular

from .kernels import Blocks, contract_gradient, n_blocks, rbf
from .losses import Loss
from .training import ConvergenceError
from .validation import ValidationLoss

Array = NDArray[np.float64]

#: relative jitter added to the diagonal of the basis kernel matrix ``K_ZZ``
JITTER = 1e-10


def choose_basis(n: int, m: int, seed: int | None = 0) -> NDArray[np.intp]:
    """Random subset of ``m`` out of ``n`` sample indices (sorted)."""
    if not 1 <= m <= n:
        raise ValueError(f"n_basis must be between 1 and the number of samples ({n}), got {m}")
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=m, replace=False))


def select_basis(
    x: Array, lam: float, tol: float = 1.0, max_size: int | None = None
) -> tuple[NDArray[np.intp], float]:
    """Greedy pivoted Cholesky of the RBF kernel on scaled features ``x``.

    Adds the sample with the largest remaining kernel variance until the absolute trace of the
    Nystroem residual ``tr(K - K_s)`` is at most ``tol * lam`` or ``max_size`` points are chosen
    (default ``max(10, n // 10)``). Returns the sorted indices and the final ``tr(K - K_s) / lam``;
    a value above ``tol`` means that the size limit was hit and the subspace model is not a close
    approximation (the study in ``benchmarks/subspace_params.py`` found a cross-validation error
    at most 1 % above the full model for ``tol = 1`` and at most 14 % for ``tol = 10``).
    """
    if lam <= 0 or tol <= 0:
        raise ValueError("lam and tol must be positive")
    x = np.ascontiguousarray(x, dtype=np.float64)
    n = x.shape[0]
    limit = min(n, max(10, n // 10)) if max_size is None else min(n, int(max_size))
    if limit < 1:
        raise ValueError("max_size must be at least 1")
    diag = np.ones(n)
    chol = np.zeros((limit, n))
    pivots: list[int] = []
    while diag.sum() > tol * lam and len(pivots) < limit:
        j = len(pivots)
        p = int(np.argmax(diag))
        pivots.append(p)
        col = rbf(x, x[p : p + 1])[:, 0] - chol[:j, p] @ chol[:j]
        chol[j] = col / np.sqrt(max(diag[p], 1e-300))
        diag = np.maximum(diag - chol[j] ** 2, 0.0)
        diag[pivots] = 0.0
    return np.sort(np.array(pivots, dtype=np.intp)), float(diag.sum() / lam)


def nystroem_residual(x: Array, idx: NDArray[np.intp]) -> float:
    """Trace of the Nystroem residual ``tr(K - K_s)`` of the RBF kernel on scaled features ``x``
    for the basis ``x[idx]``, i.e. ``n - ||psi||_F^2`` with ``psi = K_XZ L^-T`` and ``L`` the
    Cholesky factor of the jittered ``K_ZZ`` (as in :class:`SubspaceProblem`). Clipped at 0."""
    x = np.ascontiguousarray(x, dtype=np.float64)
    z = x[np.asarray(idx, dtype=np.intp)]
    k_zz = rbf(z, z)
    chol = cholesky(k_zz + JITTER * np.eye(z.shape[0]), lower=True)
    psi = solve_triangular(chol, rbf(x, z).T, lower=True)
    return max(float(x.shape[0] - np.sum(psi**2)), 0.0)


class SubspaceProblem:
    """Training problem on scaled features ``x`` restricted to the span of the basis points ``z``
    (``z`` are scaled, too)."""

    def __init__(
        self,
        x: Array,
        z: Array,
        y: Array,
        loss: Loss,
        lam: float,
        w: Array | None = None,
        offset: Array | None = None,
    ):
        if lam <= 0:
            raise ValueError(f"lam must be positive, got {lam}")
        if getattr(loss, "smoothing", 1.0) <= 0.0:
            raise ValueError("subspace training needs a smooth loss (smoothing > 0)")
        self.x = np.ascontiguousarray(x, dtype=np.float64)
        self.z = np.ascontiguousarray(z, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.float64)
        self.w = np.ones_like(self.y) if w is None else np.asarray(w, dtype=np.float64)
        self.loss = loss
        self.lam = float(lam)
        self.terms = loss.terms(self.y, self.w)
        if offset is not None:
            if loss.smo_kind is not None:
                raise ValueError(f"offset is not supported for {type(loss).__name__} (SMO dual)")
            off = np.asarray(offset, dtype=np.float64)
            if off.shape != self.y.shape:
                raise ValueError(f"offset must have shape {self.y.shape}, got {off.shape}")
            self.terms = dataclasses.replace(self.terms, offset=off[self.terms.kernel_index])
        self.k_xz = rbf(self.x, self.z)
        self.k_zz = rbf(self.z, self.z)
        m = self.z.shape[0]
        # factor of the (jittered) basis kernel and the explicit feature map psi
        self.chol = cholesky(self.k_zz + JITTER * np.eye(m), lower=True)
        self.psi = solve_triangular(self.chol, self.k_xz.T, lower=True).T  # (n, m)

    @property
    def n(self) -> int:
        return self.x.shape[0]

    @property
    def m(self) -> int:
        return self.z.shape[0]

    def per_sample(self, f: Array) -> tuple[Array, Array, float]:
        """Derivatives of the loss w.r.t. the decision values ``f`` (summed over the terms of a
        sample) and the loss value."""
        t = self.terms
        dec = f[t.kernel_index]
        if t.offset is not None:
            dec = dec + t.offset
        g = np.bincount(t.kernel_index, weights=self.loss.d1(dec, t), minlength=self.n)
        h = np.bincount(t.kernel_index, weights=self.loss.d2(dec, t), minlength=self.n)
        return g.astype(np.float64), h.astype(np.float64), float(self.loss.value(dec, t).sum())


@dataclass
class SubspaceSolution:
    """Solution of a subspace problem: expansion coefficients ``beta`` w.r.t. the basis points."""

    beta: Array
    b: float
    u: Array
    f: Array  # decision values at the training samples
    residual: float = np.nan
    newton_steps: int = 0
    time_newton: float = 0.0
    info: dict[str, Any] = field(default_factory=dict)

    # names shared with `training.Solution` (used by the cross-validation code)
    @property
    def a(self) -> Array:
        return self.beta

    time_smo: float = 0.0
    c: float = 0.0


def newton_subspace(
    prob: SubspaceProblem,
    beta0: Array | None = None,
    b0: float | None = None,
    **options: Any,
) -> SubspaceSolution:
    """Damped Newton method for the subspace problem (``m + 1`` unknowns).

    A warm start ``(beta0, b0)`` may lie in a region where the loss is linear in all samples (tiny
    ``lam`` gives huge coefficients), so the Hessian is just ``lam I`` and no step is accepted. The
    solve is then repeated from the cold start ``u = 0, b = loss.initial_bias``, which is always
    well-posed; a :class:`ConvergenceError` of the cold start is raised.
    """
    cold = prob.loss.initial_bias(prob.terms)
    if beta0 is None:
        return _newton_subspace(prob, None, cold, **options)
    try:
        return _newton_subspace(prob, beta0, cold if b0 is None else b0, **options)
    except ConvergenceError:
        return _newton_subspace(prob, None, cold, **options)


def _newton_subspace(
    prob: SubspaceProblem,
    beta0: Array | None,
    b0: float,
    *,
    tol: float = 1e-12,
    max_steps: int = 100,
    max_backtracks: int = 30,
    sigma: float = 1e-4,
) -> SubspaceSolution:
    t0 = time.perf_counter()
    lam, psi, m = prob.lam, prob.psi, prob.m
    u = prob.chol.T @ beta0 if beta0 is not None else np.zeros(m)
    b = float(b0)
    f = psi @ u + b
    g, h, loss_value = prob.per_sample(f)
    obj = 0.5 * lam * float(u @ u) + loss_value
    steps = 0
    for steps in range(max_steps + 1):
        grad_u = lam * u + psi.T @ g
        grad_b = float(g.sum())
        res = (float(grad_u @ grad_u) + grad_b**2) / (m + 1)
        if res < tol or steps == max_steps:
            break
        # Newton system [[lam I + Psi^T D Psi, Psi^T h], [h^T Psi, sum h]]. The loss may be linear
        # in all samples (sum h = 0, e.g. at the start), which leaves the bias direction without
        # curvature; its diagonal entry is then increased until a descent step is found.
        hpsi = h[:, None] * psi
        system = np.empty((m + 1, m + 1))
        system[:m, :m] = lam * np.eye(m) + psi.T @ hpsi
        system[:m, m] = system[m, :m] = hpsi.sum(axis=0)
        rhs = -np.append(grad_u, grad_b)
        accepted = False
        for damping in (1e-12, 1e-4, 1e-2, 1.0, 1e2):
            system[m, m] = h.sum() + damping
            try:
                d = cho_solve(cho_factor(system), rhs)
            except np.linalg.LinAlgError:
                continue
            slope = float(-rhs @ d)
            if not slope < 0.0:
                continue
            t = 1.0
            for _ in range(max_backtracks if damping >= 1.0 else 6):
                u_new, b_new = u + t * d[:m], b + t * d[m]
                f_new = psi @ u_new + b_new
                g_new, h_new, loss_new = prob.per_sample(f_new)
                obj_new = 0.5 * lam * float(u_new @ u_new) + loss_new
                if obj_new <= obj + sigma * t * slope + 1e-15 * abs(obj):
                    accepted = True
                    break
                t *= 0.5
            if accepted:
                break
        if not accepted:
            if res < 1e3 * tol:
                break
            raise ConvergenceError(f"no descent step found (residual {res:.3e})")
        u, b, f, g, h, obj = u_new, b_new, f_new, g_new, h_new, obj_new
    beta = solve_triangular(prob.chol.T, u, lower=False)
    return SubspaceSolution(beta, b, u, f, res, steps, time.perf_counter() - t0)


@dataclass
class SubspaceFoldResult:
    error: float
    gradient: Array
    solution: SubspaceSolution
    n_support: int
    time: float


def fold_error_and_gradient_subspace(
    x_train: Array,
    y_train: Array,
    w_train: Array,
    z: Array,
    x_val: Array,
    y_val: Array,
    loss: Loss,
    validation: ValidationLoss,
    lam: float,
    blocks: Blocks,
    newton_options: dict[str, Any] | None = None,
    beta0: Array | None = None,
    offset_train: Array | None = None,
    w_val: Array | None = None,
    offset_val: Array | None = None,
) -> SubspaceFoldResult:
    """Validation error of one fold and its gradient w.r.t. ``(log lam, log s_k)`` (subspace model).

    ``x_train``, ``z`` and ``x_val`` must already be scaled. ``offset_train`` and
    ``offset_val`` are per-sample offsets added to the decision values; ``w_val`` are validation
    weights, passed to ``validation`` only if it is ``weighted``.
    """
    prob = SubspaceProblem(x_train, z, y_train, loss, lam, w_train, offset_train)
    sol = newton_subspace(prob, beta0, **(newton_options or {}))
    beta, b, f = sol.beta, sol.b, sol.f
    g, h, _ = prob.per_sample(f)
    # validation: error and its gradient w.r.t. the predictions
    k_vz = rbf(x_val, prob.z)
    f_val = k_vz @ beta + b
    if offset_val is not None:
        f_val = f_val + offset_val
    err, e = validation(f_val, y_val, w_val) if validation.weighted else validation(f_val, y_val)
    nb = n_blocks(blocks)
    grad = np.zeros(1 + nb)
    # explicit part: the kernel between validation points and basis points depends on the scalings
    grad[1:] = contract_gradient(x_val, prob.z, k_vz, e, beta, blocks)
    # adjoint system H [p; q] = [K_ZV e; sum e], solved in the coordinates u = L^T beta:
    # H_u z = [L^{-1} K_ZV e; sum e], [p; q] = [L^{-T} z_u; z_b]
    m = prob.m
    psi = prob.psi
    hpsi = h[:, None] * psi
    system = np.empty((m + 1, m + 1))
    system[:m, :m] = lam * np.eye(m) + psi.T @ hpsi
    system[:m, m] = system[m, :m] = hpsi.sum(axis=0)
    system[m, m] = h.sum() + 1e-12
    rhs = np.append(solve_triangular(prob.chol, k_vz.T @ e, lower=True), e.sum())
    zz = cho_solve(cho_factor(system), rhs)
    p = solve_triangular(prob.chol.T, zz[:m], lower=False)
    q = float(zz[m])
    # implicit part: -[p; q]^T d(grad J)/d theta
    # d/dlog(lam) of grad_beta J = lam (K_ZZ + JITTER I) beta; p^T L L^T beta = z_u^T u (stable)
    grad[0] = -lam * float(zz[:m] @ sol.u)
    left = h * (prob.k_xz @ p) + q * h
    grad[1:] -= (
        lam * contract_gradient(prob.z, prob.z, prob.k_zz, p, beta, blocks)
        + contract_gradient(prob.x, prob.z, prob.k_xz, g, p, blocks)
        + contract_gradient(prob.x, prob.z, prob.k_xz, left, beta, blocks)
    )
    return SubspaceFoldResult(err, grad, sol, m, sol.time_newton)

"""Multi-class soft-max SVM: training and hypergradient (Strasdat, Sec. 5.1, Prop. 5.1).

Labels are integers ``0, ..., r - 1``. The model has one decision function per class,

    T = K A / lam + 1 b^T      (n x r),      A in R^{n x r}, b in R^r,

the multi-class analogue of the binary ``t = K a / lam + b``, and minimises

    sum_k a_k^T K a_k / (2 lam) + sum_i w_i (logsumexp(T_i) - T_{i, y_i}).

Identifiability. The soft-max loss is invariant under ``T_i -> T_i + c_i 1``. Stationarity in
``A`` gives ``A = -G`` with ``G_ik = w_i (p_ik - [k = y_i])``, whose rows sum to zero, so
``sum_k a_k = 0`` holds at every solution without a constraint (and exactly after one full Newton
step). The unregularised biases are determined only up to a common constant (Hessian singular
along ``(0, 1)``); it is fixed by ``sum_k b_k = 0``, enforced by the penalty
``rho/2 (1^T b)^2`` (``rho = mean(w)``), which vanishes at the solution. This keeps the
Hessian of the penalised objective positive definite, leaves ``A`` and the decision differences
untouched and has no dependence on the hyperparameters, so the hypergradient needs no extra
term. Adding a function ``c`` of the RKHS to all classes changes the regulariser, so the ``A``
block has no further redundancy (``K`` is positive definite for distinct points).

Newton method. Newton is applied to the stationarity equations ``F(A, b) = 0``,

    F_a = A + G(T),            F_b = sum_i G_i + rho (1^T b) 1,

(the gradient is ``(M F_a, F_b)`` with ``M = K / lam``; the Newton step is the same). With the
per-sample Hessian ``H_i = w_i (diag p_i - p_i p_i^T)`` the Jacobian is

    J = [[ I + H M ,  H E ], [ E^T H M ,  E^T H E + rho 1 1^T ]],
    (H M)_{(ik),(jl)} = H_i[k,l] M_ij.

``J`` is nonsymmetric but nonsingular, a dense system of size ``r (n + 1)``; one step costs
``O((r n)^3)``, so the module is meant for ``r n`` up to a few thousand.

Hypergradient. ``dE/dtheta`` for the validation loss ``E(T_val)`` with ``e = dE/dT_val``:
the explicit part uses ``T_val = K_vt A / lam + 1 b^T`` at fixed ``(A, b)``; the implicit part
solves the adjoint system ``J^T lam_adj = [K_vt^T e / lam; e^T 1]`` once per fold and gives
``-sum_i (lam_a,i + lam_b)^T H_i dT_i/dtheta`` with ``dT/dlog lam = -K A / lam`` and
``dT/dlog s_k = (dK_k A) / lam`` (all at fixed ``A``).
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import lu_factor, lu_solve
from scipy.special import logsumexp, softmax

from .hypergradient import Fold
from .kernels import Blocks, contract_gradient, n_blocks, rbf, scale_features
from .training import ConvergenceError

Array = NDArray[np.float64]


def initial_bias(y: NDArray[np.intp], w: Array, n_classes: int) -> Array:
    """Centred log class frequencies (the exact bias for ``A = 0`` up to centring)."""
    freq = np.bincount(y, weights=w, minlength=n_classes) + 1e-12
    b = np.log(freq / freq.sum())
    return b - b.mean()


class MulticlassProblem:
    """Soft-max training problem on (already scaled) features ``x`` with an RBF kernel."""

    def __init__(
        self, x: Array, y: Array, n_classes: int, lam: float, w: Array | None = None
    ) -> None:
        if lam <= 0:
            raise ValueError(f"lam must be positive, got {lam}")
        self.x = np.ascontiguousarray(x, dtype=np.float64)
        self.y = np.asarray(y).astype(np.intp)
        if n_classes < 2 or self.y.size == 0 or np.min(self.y) < 0 or np.max(self.y) >= n_classes:
            raise ValueError("labels must be integers in 0, ..., n_classes - 1 (n_classes >= 2)")
        self.r = int(n_classes)
        self.lam = float(lam)
        self.w = np.ones(self.y.size) if w is None else np.asarray(w, dtype=np.float64)
        self.k = rbf(self.x, self.x)
        self.rho = float(self.w.mean())

    @property
    def n(self) -> int:
        return self.x.shape[0]

    def decision(self, a: Array, b: Array) -> tuple[Array, Array]:
        """``(K A / lam, K A / lam + 1 b^T)``."""
        ka = self.k @ a / self.lam
        return ka, ka + b

    def loss(self, t: Array) -> float:
        return float(self.w @ (logsumexp(t, axis=1) - t[np.arange(self.n), self.y]))

    def gradient(self, t: Array) -> Array:
        """``G_ik = w_i (p_ik - [k = y_i])`` (derivative of the loss w.r.t. ``T``)."""
        g = softmax(t, axis=1)
        g[np.arange(self.n), self.y] -= 1.0
        return self.w[:, None] * g

    def curvature(self, t: Array) -> Array:
        """Per-sample Hessians ``H_i = w_i (diag p_i - p_i p_i^T)``, shape ``(n, r, r)``."""
        p = softmax(t, axis=1)
        h = -p[:, :, None] * p[:, None, :]
        idx = np.arange(self.r)
        h[:, idx, idx] += p
        return self.w[:, None, None] * h

    def objective(self, a: Array, b: Array, ka: Array) -> float:
        """Training objective plus the gauge penalty ``rho/2 (1^T b)^2``."""
        t = ka + b
        return 0.5 * float(np.sum(a * ka)) + self.loss(t) + 0.5 * self.rho * float(b.sum()) ** 2

    def stationarity(self, a: Array, b: Array, t: Array) -> tuple[Array, Array]:
        g = self.gradient(t)
        return a + g, g.sum(axis=0) + self.rho * b.sum()

    def jacobian(self, h: Array) -> Array:
        """Dense Jacobian ``J`` of ``F`` w.r.t. ``(A.ravel(), b)`` (sample-major ordering)."""
        n, r = self.n, self.r
        m = self.k / self.lam
        jac = np.zeros((n * r + r, n * r + r))
        jac[: n * r, : n * r] = np.einsum("ikl,ij->ikjl", h, m).reshape(n * r, n * r)
        jac[: n * r, : n * r] += np.eye(n * r)
        jac[: n * r, n * r :] = h.reshape(n * r, r)
        jac[n * r :, : n * r] = np.einsum("iml,ij->mjl", h, m).reshape(r, n * r)
        jac[n * r :, n * r :] = h.sum(axis=0) + self.rho
        return jac


@dataclass
class MulticlassSolution:
    """Solution ``(A, b)`` of the soft-max problem."""

    a: Array
    b: Array
    ka: Array
    t: Array
    residual: float = np.nan
    newton_steps: int = 0
    time_newton: float = 0.0
    info: dict[str, Any] = field(default_factory=dict)


def newton_multiclass(
    prob: MulticlassProblem,
    a0: Array | None = None,
    b0: Array | None = None,
    **options: Any,
) -> MulticlassSolution:
    """Damped Newton method for the soft-max problem with warm start ``(a0, b0)``.

    A failed warm start is repeated from the cold start ``A = 0``, ``b = initial_bias``; a
    :class:`ConvergenceError` of the cold start is raised. Classes without any training sample
    have no finite optimum (``b_k -> -inf``) and are not supported.
    """
    cold = initial_bias(prob.y, prob.w, prob.r)
    if a0 is None:
        return _newton(prob, None, cold, **options)
    try:
        return _newton(prob, a0, cold if b0 is None else b0, **options)
    except ConvergenceError:
        return _newton(prob, None, cold, **options)


def _newton(
    prob: MulticlassProblem,
    a0: Array | None,
    b0: Array,
    *,
    tol: float = 1e-12,
    max_steps: int = 100,
    max_backtracks: int = 30,
    sigma: float = 1e-4,
) -> MulticlassSolution:
    t0 = time.perf_counter()
    n, r, lam = prob.n, prob.r, prob.lam
    a = np.zeros((n, r)) if a0 is None else np.array(a0, dtype=np.float64)
    b = np.asarray(b0, dtype=np.float64) - np.mean(b0)
    ka, t = prob.decision(a, b)
    obj = prob.objective(a, b, ka)
    res, steps = np.nan, 0
    for steps in range(max_steps + 1):
        f_a, f_b = prob.stationarity(a, b, t)
        res = float(np.mean(f_a**2) + np.sum(f_b**2))
        if res < tol or steps == max_steps:
            break
        grad_a, grad_b = prob.k @ f_a / lam, f_b
        directions: list[tuple[Array, Array]] = []
        try:
            d = lu_solve(lu_factor(prob.jacobian(prob.curvature(t))), -np.append(f_a, f_b))
            if np.all(np.isfinite(d)):
                directions.append((d[: n * r].reshape(n, r), d[n * r :]))
        except (np.linalg.LinAlgError, ValueError):
            pass
        directions.append((-f_a, -lam * f_b))  # preconditioned gradient as fallback
        for da, db in directions:
            kda = prob.k @ da / lam
            slope = float(np.sum(grad_a * da) + grad_b @ db)
            if not slope < 0.0:
                continue
            step = 1.0
            for _ in range(max_backtracks):
                a_new, b_new = a + step * da, b + step * db
                ka_new = ka + step * kda
                obj_new = prob.objective(a_new, b_new, ka_new)
                if obj_new <= obj + sigma * step * slope + 1e-15 * abs(obj):
                    break
                step *= 0.5
            else:
                continue
            a, b, ka, obj = a_new, b_new, ka_new, obj_new
            t = ka + b
            break
        else:
            if res < 1e3 * tol:
                break
            raise ConvergenceError(f"no descent step found (residual {res:.3e})")
    ka, t = prob.decision(a, b)  # remove accumulated rounding errors
    return MulticlassSolution(a, b, ka, t, res, steps, time.perf_counter() - t0)


# ----------------------------------------------------------------------------------------------
# validation loss and prediction
# ----------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class MultinomialNLL:
    """Multinomial negative log-likelihood ``sum_v w_v (logsumexp(T_v) - T_{v, y_v})`` (summed, not
    averaged), returning its gradient w.r.t. the decision values ``T`` (``n_val x r``).

    A smooth surrogate of the misclassification rate is not provided (use this loss).
    """

    weighted: bool = False

    def __call__(self, t: Array, y: Array, w: Array | None = None) -> tuple[float, Array]:
        idx = np.arange(t.shape[0])
        yi = np.asarray(y).astype(np.intp)
        w = np.ones(t.shape[0]) if w is None else w
        e = softmax(t, axis=1)
        e[idx, yi] -= 1.0
        return float(w @ (logsumexp(t, axis=1) - t[idx, yi])), w[:, None] * e


def decision_function(x_new: Array, x_train: Array, a: Array, b: Array, lam: float) -> Array:
    """Decision values ``K(x_new, x_train) A / lam + 1 b^T`` for *scaled* features."""
    return rbf(x_new, x_train) @ a / lam + b


def predict_proba(x_new: Array, x_train: Array, a: Array, b: Array, lam: float) -> Array:
    """Class probabilities ``softmax(T)`` for *scaled* features."""
    return softmax(decision_function(x_new, x_train, a, b, lam), axis=1)


# ----------------------------------------------------------------------------------------------
# hypergradient
# ----------------------------------------------------------------------------------------------


@dataclass
class MulticlassFoldResult:
    error: float
    gradient: Array
    solution: MulticlassSolution
    time: float


def fold_error_and_gradient_multiclass(
    x_train: Array,
    y_train: Array,
    w_train: Array,
    x_val: Array,
    y_val: Array,
    n_classes: int,
    validation: MultinomialNLL,
    lam: float,
    blocks: Blocks,
    newton_options: dict[str, Any] | None = None,
    warm_start: tuple[Array, Array] | None = None,
    w_val: Array | None = None,
) -> MulticlassFoldResult:
    """Validation error of one fold and its gradient w.r.t. ``(log lam, log s_k)``.

    ``x_train`` and ``x_val`` must already be scaled; ``y`` are integer labels in
    ``0..n_classes-1``. ``warm_start`` is ``(A0, b0)``; ``w_val`` is passed to ``validation`` only
    if it is ``weighted``.
    """
    prob = MulticlassProblem(x_train, y_train, n_classes, lam, w_train)
    a0, b0 = warm_start if warm_start is not None else (None, None)
    sol = newton_multiclass(prob, a0, b0, **(newton_options or {}))
    n, r = prob.n, prob.r
    a, ka, t = sol.a, sol.ka, sol.t
    k_val = rbf(x_val, prob.x)
    ka_val = k_val @ a / lam
    t_val = ka_val + sol.b
    err, e = validation(t_val, y_val, w_val) if validation.weighted else validation(t_val, y_val)
    nb = n_blocks(blocks)
    grad = np.zeros(1 + nb)
    # explicit part (A, b fixed)
    grad[0] = -float(np.sum(e * ka_val))
    for k in range(r):
        grad[1:] += contract_gradient(x_val, prob.x, k_val, e[:, k], a[:, k], blocks) / lam
    # implicit part via the adjoint system J^T z = [K_vt^T e / lam; e^T 1]
    h = prob.curvature(t)
    rhs = np.append(k_val.T @ e / lam, e.sum(axis=0))
    z = lu_solve(lu_factor(prob.jacobian(h)), rhs, trans=1)
    z_a, z_b = z[: n * r].reshape(n, r), z[n * r :]
    u = np.einsum("ikl,il->ik", h, z_a + z_b)
    grad[0] += float(np.sum(u * ka))
    for k in range(r):
        grad[1:] -= contract_gradient(prob.x, prob.x, prob.k, u[:, k], a[:, k], blocks) / lam
    return MulticlassFoldResult(err, grad, sol, sol.time_newton)


@dataclass
class MulticlassCVResult:
    error: float
    gradient: Array
    folds: list[MulticlassFoldResult] = field(default_factory=list)


def cv_error_and_gradient_multiclass(
    theta: Array,
    x: Array,
    y: Array,
    w: Array,
    folds: Sequence[Fold],
    n_classes: int | None = None,
    validation: MultinomialNLL | None = None,
    blocks: Blocks = None,
    newton_options: dict[str, Any] | None = None,
    warm_start: list[tuple[Array, Array] | None] | None = None,
) -> MulticlassCVResult:
    """Mean validation error over all folds and its gradient w.r.t.
    ``theta = (log lam, log s_1, ..., log s_K)`` (serial over the folds).

    ``warm_start`` holds one (possibly ``None``) pair ``(A, b)`` per fold and is updated in place.
    ``n_classes`` defaults to ``max(y) + 1``.
    """
    validation = MultinomialNLL() if validation is None else validation
    y = np.asarray(y).astype(np.intp)
    n_classes = int(np.max(y)) + 1 if n_classes is None else n_classes
    nb = n_blocks(blocks)
    lam = float(np.exp(theta[0]))
    xs = scale_features(x, np.exp(theta[1 : 1 + nb]), blocks)
    total, grad, n_val = 0.0, np.zeros(1 + nb), 0
    results: list[MulticlassFoldResult] = []
    for i, (train_idx, val_idx) in enumerate(folds):
        res = fold_error_and_gradient_multiclass(
            xs[train_idx], y[train_idx], w[train_idx], xs[val_idx], y[val_idx], n_classes,
            validation, lam, blocks, newton_options,
            warm_start[i] if warm_start is not None else None,
            w[val_idx] if validation.weighted else None,
        )  # fmt: skip
        if warm_start is not None:
            warm_start[i] = (res.solution.a, res.solution.b)
        total += res.error
        grad += res.gradient
        n_val += len(val_idx)
        results.append(res)
    return MulticlassCVResult(total / n_val, grad / n_val, results)

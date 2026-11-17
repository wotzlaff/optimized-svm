"""Multi-class soft-max SVM: training and hypergradient (Strasdat, Sec. 5.1, Prop. 5.1).

Labels are integers ``0, ..., r - 1``. The model has one decision function per class,

    T = K A / lam + 1 b^T      (n x r),      A in R^{n x r}, b in R^r,

the multi-class analogue of the binary ``t = K a / lam + b``, and minimises

    sum_k a_k^T K a_k / (2 lam) + sum_i w_i (logsumexp(T_i) - T_{i, y_i}).

Identifiability. The common class direction is removed with orthonormal Helmert contrasts.
Biases are centred; the equivalent hyperparameter-independent gauge penalty
``rho/2 (1^T b)^2`` (``rho = mean(w)``) is retained in the objective/stationarity API.
The canonical representer coefficients satisfy ``A = -G`` and hence have zero row sums.

Newton and hypergradient. The stationarity Jacobian is the nonsymmetric matrix

    J = [[ I + H M , H E ], [ E^T H M , E^T H E + rho 1 1^T ]],

with ``M = K / lam`` and block-diagonal categorical covariances ``H``. It is not
factorized directly: use covariance roots ``H = R R^T`` in the class contrasts and
factor the symmetric positive definite ``S = I + R^T M R`` (smallest eigenvalue >= 1),
then the small centred-bias Schur complement. The same factors solve the adjoint and
produce ``H (z_a + E z_b)`` directly. No inverse, eigenvalue truncation or jitter of K
is needed; duplicate samples and zero individual sample weights remain supported.
The dense system has size ``n (r - 1)`` and still costs cubic time.

The factor can be reused for the adjoint if its curvature exactly matches the final
iterate. Linear solves check backward residuals in the original stationarity equations.
Validation hypergradients are contracted with kernel derivatives as before.

"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import cho_factor, cho_solve
from scipy.special import logsumexp, softmax

from .hypergradient import Fold, run_folds
from .kernels import Blocks, contract_gradient, n_blocks, rbf, scale_features
from .linalg import _limit
from .training import ConvergenceError

Array = NDArray[np.float64]


def _kernel_product(k: Array, a: Array, lam: float) -> Array:
    """Accumulate before dividing, with extra precision for tiny regularization.

    Oppositely labelled duplicate samples can have O(1) coefficients that cancel
    to O(lambda). Ordinary BLAS summation otherwise magnifies its rounding error
    by 1/lambda. Extended accumulation changes no kernel or objective.
    """
    if lam < 1e-5:
        return np.asarray(
            np.asarray(k, dtype=np.longdouble)
            @ np.asarray(a, dtype=np.longdouble)
            / np.longdouble(lam),
            dtype=np.float64,
        )
    return k @ a / lam


def _nll(t: Array, y: NDArray[np.intp]) -> Array:
    """Log loss without cancellation when the target probability is almost one."""
    differences = t - t[np.arange(t.shape[0]), y, None]
    differences[np.arange(t.shape[0]), y] = -np.inf
    return np.logaddexp(0.0, logsumexp(differences, axis=1))


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
        if not np.isfinite(lam) or lam <= 0:
            raise ValueError(f"lam must be positive, got {lam}")
        self.x = np.ascontiguousarray(x, dtype=np.float64)
        self.y = np.asarray(y).astype(np.intp)
        if n_classes < 2 or self.y.size == 0 or np.min(self.y) < 0 or np.max(self.y) >= n_classes:
            raise ValueError("labels must be integers in 0, ..., n_classes - 1 (n_classes >= 2)")
        self.r = int(n_classes)
        self.lam = float(lam)
        self.w = np.ones(self.y.size) if w is None else np.asarray(w, dtype=np.float64)
        if (
            self.x.ndim != 2
            or self.y.shape != (self.x.shape[0],)
            or self.w.shape != self.y.shape
            or not np.all(np.isfinite(self.x))
            or not np.all(np.isfinite(self.w))
            or np.any(self.w < 0)
        ):
            raise ValueError("soft-max needs finite features and nonnegative matching weights")
        if np.any(np.bincount(self.y, weights=self.w, minlength=self.r) == 0):
            raise ValueError("every soft-max class needs positive total training weight")
        self.k = rbf(self.x, self.x)
        self.rho = float(self.w.mean())

    @property
    def n(self) -> int:
        return self.x.shape[0]

    def decision(self, a: Array, b: Array) -> tuple[Array, Array]:
        """``(K A / lam, K A / lam + 1 b^T)``."""
        ka = _kernel_product(self.k, a, self.lam)
        return ka, ka + b

    def loss(self, t: Array) -> float:
        return float(self.w @ _nll(t, self.y))

    def gradient(self, t: Array) -> Array:
        """``G_ik = w_i (p_ik - [k = y_i])`` (derivative of the loss w.r.t. ``T``)."""
        g = softmax(t, axis=1)
        idx = np.arange(self.n)
        g[idx, self.y] = 0.0
        g[idx, self.y] = -g.sum(axis=1)
        return self.w[:, None] * g

    def curvature(self, t: Array) -> Array:
        """Per-sample Hessians ``H_i = w_i (diag p_i - p_i p_i^T)``, shape ``(n, r, r)``."""
        p = softmax(t, axis=1)
        h = -p[:, :, None] * p[:, None, :]
        idx = np.arange(self.r)
        h[:, idx, idx] = 0.0
        h[:, idx, idx] = -h.sum(axis=2)
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


def _contrasts(r: int) -> Array:
    """Orthonormal Helmert contrasts, perpendicular to the common class direction."""
    c = np.zeros((r, r - 1))
    for j in range(r - 1):
        scale = np.sqrt((j + 1) * (j + 2))
        c[: j + 1, j] = 1.0 / scale
        c[j + 1, j] = -(j + 1) / scale
    return c


@dataclass
class SoftmaxFactorization:
    """Covariance-root SPD reduction of the stationarity Jacobian.

    With block-diagonal ``R`` such that ``H = R R.T``, factor
    ``S = I + R.T M R`` and its small centred-bias Schur complement. No inverse
    or jitter of the kernel is needed, including with duplicate samples.
    """

    h: Array
    m: Array
    c: Array
    root: Array
    cho: Any
    bias_columns: Array
    bias_solutions: Array
    bias_cho: Any
    rho: float

    @classmethod
    def build(cls, prob: MulticlassProblem, h: Array) -> SoftmaxFactorization:
        n, r = prob.n, prob.r
        c = _contrasts(r)
        reduced = np.einsum("ka,ikl,lb->iab", c, h, c)
        values, vectors = np.linalg.eigh(reduced)
        # Only discard roundoff-negative covariance eigenvalues, never kernel directions.
        root = np.einsum("ka,iab->ikb", c, vectors * np.sqrt(np.maximum(values, 0))[:, None, :])
        m = prob.k / prob.lam
        size = n * (r - 1)
        system = np.einsum("ika,jkb,ij->iajb", root, root, m).reshape(size, size)
        system.flat[:: size + 1] += 1.0
        columns = np.einsum("ika,kb->iab", root, c).reshape(size, r - 1)
        with _limit(size):
            cho = cho_factor(system, lower=True)
            solutions = cho_solve(cho, columns)
            schur = columns.T @ solutions
            bias_cho = cho_factor((schur + schur.T) * 0.5, lower=True)
        return cls(h.copy(), m, c, root, cho, columns, solutions, bias_cho, prob.rho)

    def _root_t(self, a: Array) -> Array:
        return np.einsum("ika,ik->ia", self.root, a).ravel()

    def _root(self, z: Array) -> Array:
        return np.einsum("ika,ia->ik", self.root, z.reshape(self.root.shape[0], -1))

    def solve(self, rhs_a: Array, rhs_b: Array) -> tuple[Array, Array]:
        """Solve the original, nonsymmetric ``J d = rhs`` through SPD systems."""
        with _limit(self.bias_columns.shape[0]):
            z = cho_solve(self.cho, self._root_t(self.m @ rhs_a))
            beta = cho_solve(self.bias_cho, self.c.T @ rhs_b - self.bias_columns.T @ z)
        z += self.bias_solutions @ beta
        db = self.c @ beta + rhs_b.sum() / (self.rho * rhs_b.size**2)
        da = rhs_a - self._root(z)
        self._check(da, db, rhs_a, rhs_b)
        return da, db

    def adjoint_curvature(self, rhs_a: Array, rhs_b: Array) -> Array:
        """Return ``H (z_a + E z_b)`` for ``J.T z = rhs`` without forming J."""
        with _limit(self.bias_columns.shape[0]):
            v = cho_solve(self.cho, self._root_t(rhs_a))
            beta = cho_solve(self.bias_cho, self.c.T @ rhs_b - self.bias_columns.T @ v)
        v += self.bias_solutions @ beta
        u = self._root(v)
        za = rhs_a - self.m @ u
        zb = self.c @ beta + rhs_b.sum() / (self.rho * rhs_b.size**2)
        # Check the original adjoint equations, rather than trusting finite factors alone.
        hu = np.einsum("ikl,il->ik", self.h, za + zb)
        ra = za + self.m @ hu - rhs_a
        rb = hu.sum(axis=0) + self.rho * zb.sum() - rhs_b
        self._check_residual(ra, rb, za, zb, rhs_a, rhs_b)
        return u

    def _check(self, da: Array, db: Array, rhs_a: Array, rhs_b: Array) -> None:
        hd = np.einsum("ikl,il->ik", self.h, self.m @ da + db)
        ra = da + hd - rhs_a
        rb = hd.sum(axis=0) + self.rho * db.sum() - rhs_b
        self._check_residual(ra, rb, da, db, rhs_a, rhs_b)

    def _check_residual(
        self, ra: Array, rb: Array, da: Array, db: Array, rhs_a: Array, rhs_b: Array
    ) -> None:
        error = max(float(np.max(np.abs(ra))), float(np.max(np.abs(rb))))
        scale = max(float(np.max(np.abs(rhs_a))), float(np.max(np.abs(rhs_b))), 1.0)
        # A conservative Jacobian norm bound makes this a backward-error test even when
        # M is large. The factorization itself never forms the ill-scaled product H M.
        hnorm = float(np.max(np.sum(np.abs(self.h), axis=2)))
        mnorm = float(np.max(np.sum(np.abs(self.m), axis=1)))
        size = self.h.shape[0]
        jnorm = 1.0 + size * hnorm * (mnorm + 1.0) + self.rho * self.h.shape[1]
        scale += jnorm * max(float(np.max(np.abs(da))), float(np.max(np.abs(db))))
        if not np.isfinite(error) or not error <= 1e-9 * scale:
            raise np.linalg.LinAlgError("soft-max linear solve failed its backward-error check")


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
    factor: SoftmaxFactorization | None = field(default=None, repr=False)


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
    a -= a.mean(axis=1, keepdims=True)
    b = np.asarray(b0, dtype=np.float64) - np.mean(b0)
    ka, t = prob.decision(a, b)
    obj = prob.objective(a, b, ka)
    res, steps = np.nan, 0
    factor: SoftmaxFactorization | None = None
    for steps in range(max_steps + 1):
        f_a, f_b = prob.stationarity(a, b, t)
        res = float(np.mean(f_a**2) + np.sum(f_b**2))
        if res < tol or steps == max_steps:
            break
        grad_a, grad_b = _kernel_product(prob.k, f_a, lam), f_b
        directions: list[tuple[Array, Array]] = []
        try:
            factor = SoftmaxFactorization.build(prob, prob.curvature(t))
            da, db = factor.solve(-f_a, -f_b)
            directions.append((da, db))
        except (np.linalg.LinAlgError, ValueError):
            pass
        directions.append((-f_a, -lam * f_b))  # preconditioned gradient as fallback
        for da, db in directions:
            slope = float(np.sum(grad_a * da) + grad_b @ db)
            if not slope < 0.0:
                continue
            step = 1.0
            for _ in range(max_backtracks):
                a_new, b_new = a + step * da, b + step * db
                ka_new, _ = prob.decision(a_new, b_new)
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
    f_a, f_b = prob.stationarity(a, b, t)
    res = float(np.mean(f_a**2) + np.sum(f_b**2))
    if factor is not None and not np.array_equal(factor.h, prob.curvature(t)):
        factor = None
    return MulticlassSolution(a, b, ka, t, res, steps, time.perf_counter() - t0, factor=factor)


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
        e[idx, yi] = 0.0
        e[idx, yi] = -e.sum(axis=1)
        return float(w @ _nll(t, yi)), w[:, None] * e


def decision_function(x_new: Array, x_train: Array, a: Array, b: Array, lam: float) -> Array:
    """Decision values ``K(x_new, x_train) A / lam + 1 b^T`` for *scaled* features."""
    return _kernel_product(rbf(x_new, x_train), a, lam) + b


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
    r = prob.r
    a, ka, t = sol.a, sol.ka, sol.t
    k_val = rbf(x_val, prob.x)
    ka_val = _kernel_product(k_val, a, lam)
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
    factor = sol.factor
    if factor is None:
        factor = SoftmaxFactorization.build(prob, h)
        sol.factor = factor
    u = factor.adjoint_curvature(k_val.T @ e / lam, e.sum(axis=0))
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
    n_jobs: int | None = None,
) -> MulticlassCVResult:
    """Mean validation error over all folds and its gradient w.r.t.
    ``theta = (log lam, log s_1, ..., log s_K)``.

    ``warm_start`` holds one (possibly ``None``) pair ``(A, b)`` per fold and is updated in place.
    ``n_classes`` defaults to ``max(y) + 1``. ``n_jobs`` runs the folds in parallel threads
    (see :func:`optimized_svm.hypergradient.run_folds`); the result is independent of it up to
    rounding.
    """
    validation = MultinomialNLL() if validation is None else validation
    y = np.asarray(y).astype(np.intp)
    n_classes = int(np.max(y)) + 1 if n_classes is None else n_classes
    nb = n_blocks(blocks)
    lam = float(np.exp(theta[0]))
    xs = scale_features(x, np.exp(theta[1 : 1 + nb]), blocks)

    def run_fold(i: int) -> MulticlassFoldResult:
        train_idx, val_idx = folds[i]
        return fold_error_and_gradient_multiclass(
            xs[train_idx], y[train_idx], w[train_idx], xs[val_idx], y[val_idx], n_classes,
            validation, lam, blocks, newton_options,
            warm_start[i] if warm_start is not None else None,
            w[val_idx] if validation.weighted else None,
        )  # fmt: skip

    results = run_folds(run_fold, len(folds), n_jobs)
    total, grad, n_val = 0.0, np.zeros(1 + nb), 0
    for i, res in enumerate(results):  # in fold order: independent of n_jobs
        if warm_start is not None:
            warm_start[i] = (res.solution.a, res.solution.b)
        total += res.error
        grad += res.gradient
        n_val += len(folds[i][1])
    return MulticlassCVResult(total / n_val, grad / n_val, results)

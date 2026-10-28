"""Two-step training: SMO warm start (``svm_smo``) and Newton refinement (Strasdat, Ch. 8.4)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import svm_smo
from numpy.typing import NDArray

from .linalg import solve_bordered
from .problem import TrainingProblem

Array = NDArray[np.float64]


class ConvergenceError(RuntimeError):
    """The Newton method could not find a descent step."""


@dataclass
class Solution:
    """Solution ``(a, b, c)`` of a training problem with ``ka = K a / lam``."""

    a: Array
    b: float
    c: float
    ka: Array
    residual: float = np.nan
    newton_steps: int = 0
    smo_steps: int = 0
    time_smo: float = 0.0
    time_newton: float = 0.0
    info: dict[str, Any] = field(default_factory=dict)


def residual(prob: TrainingProblem, a: Array, b: float, c: float, ka: Array, use_c: bool) -> float:
    """Mean squared violation of the optimality conditions ``a + phi'(dec) = 0``, ``sum(a) = 0``
    and (if the 1-norm constraint is active) ``s^T phi'(dec) + A = 0``."""
    g = prob.loss.d1(prob.decision(b, c, ka), prob.terms)
    res = float(np.mean((a + g) ** 2) + g.sum() ** 2)
    if use_c:
        res += float(g @ prob.sign + prob.max_asum) ** 2
    return res


def _newton_direction(
    prob: TrainingProblem, a: Array, g: Array, h: Array, use_c: bool
) -> tuple[Array, float, float]:
    """Semismooth Newton direction for ``a + phi'(K a / lam + b + s c) = 0`` and constraints."""
    lam = prob.lam
    p = h > 0.0
    z = ~p
    da = np.zeros_like(a)
    da[z] = a[z] + g[z]  # terms with vanishing curvature: a_Z = -phi'_Z after the step
    k_pz_dz = prob.kernel(p, z) @ da[z] / lam if z.any() and p.any() else np.zeros(p.sum())
    q = prob.kernel(p, p) / lam + np.diag(1.0 / h[p])
    rhs = (a[p] + g[p]) / h[p] - k_pz_dz
    cols = [np.ones(p.sum())]
    rhs0 = [a[p].sum() - g[z].sum()]
    if use_c:
        s = prob.sign
        cols.append(s[p])
        rhs0.append(a @ s - prob.max_asum - s[z] @ da[z])
    d_p, v = solve_bordered(q, np.stack(cols, axis=1), rhs, np.array(rhs0))
    da[p] = d_p
    return da, float(v[0]), float(v[1]) if use_c else 0.0


def newton(
    prob: TrainingProblem,
    a: Array,
    b: float,
    c: float,
    *,
    tol: float = 1e-12,
    max_steps: int = 100,
    max_backtracks: int = 20,
    sigma: float = 1e-4,
    beta: float = 0.5,
    verbose: int = 0,
) -> Solution:
    """Damped semismooth Newton method for the (smoothed) primal training problem.

    Requires a loss with (piecewise) second derivative. The 1-norm constraint is treated as
    active iff ``c > 0`` at the starting point.
    """
    use_c = prob.max_asum is not None and c > 0.0
    if not use_c:
        c = 0.0
    a = np.array(a, dtype=np.float64)
    ka = prob.ka(a)
    obj = prob.objective(a, b, c, ka)
    for step in range(max_steps + 1):
        dec = prob.decision(b, c, ka)
        g = prob.loss.d1(dec, prob.terms)
        r = a + g
        res = float(np.mean(r**2) + g.sum() ** 2)
        grad_c = float(g @ prob.sign + prob.max_asum) if use_c else 0.0
        res += grad_c**2
        if verbose:
            print(f"[newton {step:3d}] f = {obj:.10g}, residual = {res:.3e}")
        if res < tol or step == max_steps:
            break
        # gradient of the objective w.r.t. (a, b, c): (K r / lam, sum g, s^T g + A)
        grad_b = float(g.sum())
        h = prob.loss.d2(dec, prob.terms)
        directions = []
        try:
            directions.append(_newton_direction(prob, a, g, h, use_c))
        except np.linalg.LinAlgError:
            pass
        # preconditioned gradient as fallback
        directions.append((r, prob.lam * grad_b, prob.lam * grad_c))
        for da, db, dc in directions:
            kda = prob.ka(da)
            slope = float(r @ kda) + db * grad_b + dc * grad_c
            if slope <= 0.0:
                continue
            t = 1.0
            for _ in range(max_backtracks):
                a_new, b_new, c_new = a - t * da, b - t * db, c - t * dc
                ka_new = ka - t * kda
                obj_new = prob.objective(a_new, b_new, c_new, ka_new)
                if obj_new <= obj - sigma * t * slope + 1e-15 * abs(obj):
                    break
                t *= beta
            else:
                continue
            a, b, c, ka, obj = a_new, b_new, c_new, ka_new, obj_new
            break
        else:
            if res < 1e3 * tol:
                break
            raise ConvergenceError(f"no descent step found (residual {res:.3e})")
    # recompute ka from scratch to remove accumulated rounding errors
    ka = prob.ka(a)
    return Solution(a, float(b), float(c), ka, residual(prob, a, b, c, ka, use_c), newton_steps=step)


_SMO_OPTIONS = {"tol", "max_steps", "time_limit", "working_set", "two_sided", "shrinking",
                "shrinking_period", "cache_size_mb"}


def smo_warm_start(prob: TrainingProblem, **options: Any) -> tuple[Array, float, float, int]:
    """Solves the dual with quadratic smoothing by SMO (``svm_smo``); zeros if not applicable."""
    unknown = set(options) - _SMO_OPTIONS
    if unknown:
        raise ValueError(f"unknown SMO options: {sorted(unknown)}")
    loss = prob.loss
    if loss.smo_kind is None:
        return np.zeros(prob.size), 0.0, 0.0, 0
    common = dict(gamma=1.0, weights=prob.w, smoothing=loss.smoothing, max_asum=prob.max_asum)
    if loss.smo_kind == "classification":
        res = svm_smo.solve_classification(prob.x, prob.y, prob.lam, shift=loss.shift,
                                           **common, **options)
    else:
        res = svm_smo.solve_regression(prob.x, prob.y, prob.lam, epsilon=loss.epsilon,
                                       **common, **options)
    return np.asarray(res.alpha), res.b, res.c, res.steps


def train(
    prob: TrainingProblem,
    smo_options: dict[str, Any] | None = None,
    newton_options: dict[str, Any] | None = None,
) -> Solution:
    """Two-step training: SMO warm start followed by Newton refinement.

    For non-smooth losses (``smoothing = 0``) the SMO solution is already exact; for losses
    without SMO formulation (squared, Poisson) Newton starts at zero.
    """
    t0 = time.perf_counter()
    a, b, c, smo_steps = smo_warm_start(prob, **(smo_options or {}))
    t_smo = time.perf_counter() - t0
    t0 = time.perf_counter()
    if getattr(prob.loss, "smoothing", 1.0) == 0.0:
        sol = Solution(a, b, c, prob.ka(a))
    else:
        try:
            sol = newton(prob, a, b, c, **(newton_options or {}))
        except ConvergenceError:
            sol = Solution(a, b, c, prob.ka(a), info={"newton": "failed"})
        if prob.max_asum is not None and sol.c < 0.0:
            # the constraint is inactive at the solution: solve without it
            sol = newton(prob, sol.a, sol.b, 0.0, **(newton_options or {}))
    sol.smo_steps = smo_steps
    sol.time_smo = t_smo
    sol.time_newton = time.perf_counter() - t0
    return sol

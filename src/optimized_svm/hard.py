"""Exact hard constraints (thesis Sec. 4.1.2 and 4.2.3), for moderate data sets.

The QP is solved in an eigenbasis of the kernel without a regularization jitter. A linear
feasibility problem precedes SLSQP; primal feasibility and stationarity are checked afterwards.
Lambda is normalized to one during the solve, since it cannot change the primal minimizer.
"""

from __future__ import annotations

import time

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import LinearConstraint, linprog, minimize, nnls

from .linalg import _limit
from .losses import HardEpsilonInsensitiveLoss, MaximalMarginLoss
from .problem import TrainingProblem
from .training import ConvergenceError, Solution


def train_hard(prob: TrainingProblem) -> Solution:
    """Minimize the RKHS norm subject to the indicator loss constraints.

    Raises ``ValueError`` for infeasible constraints and ``ConvergenceError`` when a numerical
    solution cannot be certified. No finite soft-margin penalty is substituted. Tiny kernel
    eigenvalues below floating-point resolution are discarded; this can make very ill-conditioned
    problems numerically infeasible. Dense eigendecomposition/SLSQP cost cubic time in ``n``.
    """
    start = time.perf_counter()
    keep = np.flatnonzero(prob.w > 0)
    beta = np.zeros(prob.n_samples)
    if not keep.size:
        return Solution(
            beta,
            0.0,
            0.0,
            prob.ka(beta),
            residual=0.0,
            info={"hard": True, "active": np.zeros(0, dtype=np.intp)},
        )
    k = prob.sample_kernel(keep, keep)
    with _limit(keep.size):
        eig, vec = np.linalg.eigh(k)
        threshold = np.finfo(float).eps * max(1, keep.size) * max(1.0, float(eig[-1]))
        nonzero = eig > threshold
        basis = vec[:, nonzero] * np.sqrt(eig[nonzero])
    loss = prob.loss
    if isinstance(loss, MaximalMarginLoss):
        signs = prob.y[keep]
        rows = np.arange(keep.size)
        targets = np.full(keep.size, loss.shift)
    elif isinstance(loss, HardEpsilonInsensitiveLoss):
        eps = loss.tolerances(prob.n_samples)[keep]
        rows = np.tile(np.arange(keep.size), 2)
        signs = np.repeat([1.0, -1.0], keep.size)
        targets = np.concatenate((prob.y[keep] - eps, -prob.y[keep] - eps))
    else:
        raise ValueError("train_hard requires an indicator loss")
    c = np.column_stack((basis[rows] * signs[:, None], signs))
    # First handle a zero-norm solution: the feasible biases form a closed interval. Choosing
    # its midpoint is deterministic, independent of the kernel and the active-set multipliers.
    lower = float(np.max(targets[signs > 0], initial=-np.inf))
    upper = float(np.min(-targets[signs < 0], initial=np.inf))
    if lower <= upper:
        bias = (
            (lower + upper) / 2
            if np.isfinite(lower + upper)
            else (lower if np.isfinite(lower) else upper)
        )
        return Solution(
            beta,
            bias,
            0.0,
            prob.ka(beta),
            residual=0.0,
            info={"hard": True, "active": np.zeros(0, dtype=np.intp)},
        )
    feasibility = linprog(
        np.zeros(c.shape[1]),
        A_ub=-c,
        b_ub=-targets,
        bounds=[(None, None)] * c.shape[1],
        method="highs",
    )
    if feasibility.status == 2:
        raise ValueError("hard constraints are infeasible (conflicting labels or disjoint tubes)")
    if not feasibility.success:
        raise ConvergenceError(f"hard constraint feasibility check failed: {feasibility.message}")

    def objective(z: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        g = z.copy()
        g[-1] = 0.0
        return 0.5 * float(z[:-1] @ z[:-1]), g

    with _limit(keep.size):
        result = minimize(
            objective,
            feasibility.x,
            jac=True,
            method="SLSQP",
            constraints=[LinearConstraint(c, targets, np.inf)],
            options={"ftol": 1e-12, "maxiter": 2000},
        )
    z = result.x
    slack = c @ z - targets
    scale = max(1.0, float(np.abs(targets).max()))
    tol = 2e-7 * scale
    if not result.success or float(slack.min()) < -tol:
        raise ConvergenceError(
            f"hard constraint QP failed: {result.message}; "
            f"max violation={max(0.0, -float(slack.min())):.3g}"
        )
    tight = np.flatnonzero(slack <= tol)
    try:
        with _limit(keep.size):
            multipliers, stationarity = nnls(
                c[tight].T, np.r_[z[:-1], 0.0], maxiter=100 * c.shape[0]
            )
    except RuntimeError as exc:
        raise ConvergenceError("hard constraint multiplier recovery failed") from exc
    if stationarity > 1e-6 * max(1.0, float(np.linalg.norm(z[:-1]))):
        raise ConvergenceError("hard constraint QP did not satisfy stationarity")
    np.add.at(beta, keep[rows[tight]], signs[tight] * multipliers)
    # Certify the actual kernel expansion used by predict, not just the eigenbasis coordinates.
    ka = prob.ka(beta * prob.lam)
    dec = ka + z[-1]
    expansion_slack = signs * dec[keep[rows]] - targets
    if float(expansion_slack.min()) < -tol:
        raise ConvergenceError("hard constraint kernel expansion lost primal feasibility")
    active = np.flatnonzero(np.abs(beta) > 1e-9 * max(1.0, float(np.abs(beta).max())))
    beta[np.setdiff1d(np.arange(beta.size), active)] = 0.0
    # Polish the stable active set in the original kernel coordinates. This removes SLSQP
    # termination noise, which would otherwise be amplified by finite-difference gradients.
    if active.size:
        kp = prob.sample_kernel(active, active)
        system = np.block(
            [[kp, np.ones((active.size, 1))], [np.ones((1, active.size)), np.zeros((1, 1))]]
        )
        if isinstance(loss, MaximalMarginLoss):
            boundary = prob.y[active] * loss.shift
            direction = prob.y[active]
        else:
            assert isinstance(loss, HardEpsilonInsensitiveLoss)
            direction = np.sign(beta[active])
            boundary = prob.y[active] - loss.tolerances(prob.n_samples)[active] * direction
        polished = np.linalg.lstsq(system, np.r_[boundary, 0.0], rcond=None)[0]
        candidate = np.zeros_like(beta)
        candidate[active] = polished[:-1]
        candidate_dec = prob.ka(candidate * prob.lam) + polished[-1]
        candidate_slack = signs * candidate_dec[keep[rows]] - targets
        if (
            np.min(candidate[active] * direction) >= 0.0
            and float(candidate_slack.min()) >= -tol
            and np.linalg.norm(system @ polished - np.r_[boundary, 0.0]) <= tol
        ):
            beta = candidate
            z[-1] = polished[-1]
            expansion_slack = candidate_slack
            stationarity = abs(float(beta.sum()))
    ka = prob.ka(beta * prob.lam)
    final_slack = signs * (ka + z[-1])[keep[rows]] - targets
    equality_residual = abs(float(beta.sum()))
    if equality_residual > 1e-6 * max(1.0, float(np.linalg.norm(z[:-1]))):
        raise ConvergenceError("hard constraint support pruning lost dual equality feasibility")
    stationarity = max(stationarity, equality_residual)
    if float(final_slack.min()) < -tol:
        raise ConvergenceError("hard constraint support pruning lost primal feasibility")
    return Solution(
        beta * prob.lam,
        float(z[-1]),
        0.0,
        ka,
        residual=max(stationarity, max(0.0, -float(final_slack.min()))),
        time_newton=time.perf_counter() - start,
        info={"hard": True, "active": active, "qp_steps": result.nit},
    )

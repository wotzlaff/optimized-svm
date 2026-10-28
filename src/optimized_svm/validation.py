"""Validation error measures ``E(f) = sum_v L_v(f_v)`` for the upper level problem (Sec. 9.1.4).

All measures are differentiable so that the hyperparameter objective has a gradient.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.special import gammaln

from .smooth_max import Degree, smooth_max

Array = NDArray[np.float64]


class ValidationLoss:
    """Returns the summed loss and its gradient w.r.t. the predictions."""

    def __call__(self, f: Array, y: Array) -> tuple[float, Array]:
        raise NotImplementedError


@dataclass(frozen=True)
class SmoothHingeValidation(ValidationLoss):
    """``sum_v w_v m_delta(shift - y_v f_v)``; ``balanced`` weights both classes equally."""

    shift: float = 1.0
    smoothing: float = 0.5
    degree: Degree = "6"
    balanced: bool = False

    def __call__(self, f: Array, y: Array) -> tuple[float, Array]:
        m, dm, _ = smooth_max(self.smoothing, self.degree)
        w = np.ones_like(y)
        if self.balanced:
            pos = y > 0
            frac = pos.mean()
            if 0 < frac < 1:
                w = 0.5 / np.where(pos, frac, 1.0 - frac)
        arg = self.shift - y * f
        return float(w @ m(arg)), -w * y * dm(arg)


@dataclass(frozen=True)
class SmoothAbsoluteValidation(ValidationLoss):
    """Smoothed absolute error ``sum_v m_delta(f_v - y_v) + m_delta(y_v - f_v)`` (Huber-like)."""

    smoothing: float = 0.1
    degree: Degree = "2"

    def __call__(self, f: Array, y: Array) -> tuple[float, Array]:
        m, dm, _ = smooth_max(self.smoothing, self.degree)
        e = f - y
        return float(m(e).sum() + m(-e).sum()), dm(e) - dm(-e)


@dataclass(frozen=True)
class SquaredValidation(ValidationLoss):
    """``1/2 sum_v (f_v - y_v)^2``."""

    def __call__(self, f: Array, y: Array) -> tuple[float, Array]:
        e = f - y
        return 0.5 * float(e @ e), e


@dataclass(frozen=True)
class PoissonValidation(ValidationLoss):
    """Poisson deviance-type loss ``sum_v exp(f_v) - y_v f_v + log(y_v!)``."""

    def __call__(self, f: Array, y: Array) -> tuple[float, Array]:
        with np.errstate(over="ignore"):
            ef = np.exp(f)
        return float(ef.sum() - f @ y + gammaln(y + 1).sum()), ef - y

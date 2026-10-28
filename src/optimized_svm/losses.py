"""Loss functions of the training problem, expressed per *loss term*.

A training problem with ``n`` samples consists of ``m`` loss terms: one per sample, or two for
the epsilon-insensitive loss (one for each side of the tube). Term ``l`` belongs to sample
``kernel_index[l]``, has weight ``w_l``, label ``y_l`` and sign ``s_l`` (used by the optional
1-norm constraint, ``0`` if not applicable). The decision value of a term is
``f(x_l) + s_l c``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.special import gammaln

from .smooth_max import Degree, smooth_max

Array = NDArray[np.float64]


@dataclass(frozen=True)
class Terms:
    """Term structure of a training problem."""

    kernel_index: NDArray[np.intp]
    y: Array
    w: Array
    sign: Array

    @property
    def size(self) -> int:
        return self.y.size


class Loss:
    """Base class: per-term loss ``phi_l(t)`` and its derivatives."""

    #: whether the dual of this loss can be solved by ``smo_solver`` (as warm start)
    smo_kind: str | None = None
    #: whether the 1-norm constraint is meaningful (non-zero signs)
    signed: bool = False

    def terms(self, y: Array, w: Array) -> Terms:
        n = y.size
        return Terms(np.arange(n), y, w, np.zeros(n))

    def value(self, t: Array, terms: Terms) -> Array:
        raise NotImplementedError

    def d1(self, t: Array, terms: Terms) -> Array:
        raise NotImplementedError

    def d2(self, t: Array, terms: Terms) -> Array:
        raise NotImplementedError


@dataclass(frozen=True)
class HingeLoss(Loss):
    """Smoothed hinge loss ``w * m_delta(shift - y t)`` for labels ``y`` in ``{-1, +1}``."""

    shift: float = 1.0
    smoothing: float = 0.5
    degree: Degree = "2"
    smo_kind = "classification"
    signed = True

    def terms(self, y: Array, w: Array) -> Terms:
        return Terms(np.arange(y.size), y, w, y.astype(float))

    def value(self, t: Array, terms: Terms) -> Array:
        m = smooth_max(self.smoothing, self.degree)[0]
        return terms.w * m(self.shift - terms.y * t)

    def d1(self, t: Array, terms: Terms) -> Array:
        dm = smooth_max(self.smoothing, self.degree)[1]
        return -terms.w * terms.y * dm(self.shift - terms.y * t)

    def d2(self, t: Array, terms: Terms) -> Array:
        ddm = smooth_max(self.smoothing, self.degree)[2]
        return terms.w * ddm(self.shift - terms.y * t)


@dataclass(frozen=True)
class EpsilonInsensitiveLoss(Loss):
    """Smoothed epsilon-insensitive loss ``w * (m(y - t - eps) + m(t - y - eps))``.

    Uses two terms per sample: sign ``+1`` for ``m(y - t - eps)`` and ``-1`` for the other side.
    """

    epsilon: float = 0.1
    smoothing: float = 0.5
    degree: Degree = "2"
    smo_kind = "regression"
    signed = True

    def terms(self, y: Array, w: Array) -> Terms:
        n = y.size
        idx = np.arange(n)
        return Terms(
            np.concatenate((idx, idx)),
            np.concatenate((y, y)),
            np.concatenate((w, w)),
            np.concatenate((np.ones(n), -np.ones(n))),
        )

    def _arg(self, t: Array, terms: Terms) -> Array:
        return -terms.sign * (t - terms.y) - self.epsilon

    def value(self, t: Array, terms: Terms) -> Array:
        m = smooth_max(self.smoothing, self.degree)[0]
        return terms.w * m(self._arg(t, terms))

    def d1(self, t: Array, terms: Terms) -> Array:
        dm = smooth_max(self.smoothing, self.degree)[1]
        return -terms.sign * terms.w * dm(self._arg(t, terms))

    def d2(self, t: Array, terms: Terms) -> Array:
        ddm = smooth_max(self.smoothing, self.degree)[2]
        return terms.w * ddm(self._arg(t, terms))


@dataclass(frozen=True)
class SquaredLoss(Loss):
    """Least-squares loss ``w / 2 * (t - y)^2`` (LS-SVM)."""

    def value(self, t: Array, terms: Terms) -> Array:
        return 0.5 * terms.w * (t - terms.y) ** 2

    def d1(self, t: Array, terms: Terms) -> Array:
        return terms.w * (t - terms.y)

    def d2(self, t: Array, terms: Terms) -> Array:
        return terms.w * np.ones_like(t)


@dataclass(frozen=True)
class PoissonLoss(Loss):
    """Poisson negative log-likelihood ``w * (exp(t) - y t + log(y!))`` for counts ``y``."""

    def value(self, t: Array, terms: Terms) -> Array:
        with np.errstate(over="ignore"):  # inf is handled by the line search
            return terms.w * (np.exp(t) - terms.y * t + gammaln(terms.y + 1))

    def d1(self, t: Array, terms: Terms) -> Array:
        return terms.w * (np.exp(t) - terms.y)

    def d2(self, t: Array, terms: Terms) -> Array:
        return terms.w * np.exp(t)


@dataclass(frozen=True)
class ApproxPoissonLoss(Loss):
    """Poisson-type loss with a linear left tail (from the original experiments):

    ``w * (exp(t) - (y + 1) t - 1 + log(y!))`` for ``t > 0`` and ``w * (-y t + log(y!))`` else.

    It is convex and continuously differentiable; for ``t <= 0`` its curvature vanishes.
    """

    def value(self, t: Array, terms: Terms) -> Array:
        pos = t > 0.0
        with np.errstate(over="ignore"):
            r = np.where(
                pos, np.exp(np.where(pos, t, 0.0)) - (terms.y + 1.0) * t - 1.0, -terms.y * t
            )
        return terms.w * (r + gammaln(terms.y + 1))

    def d1(self, t: Array, terms: Terms) -> Array:
        pos = t > 0.0
        return terms.w * np.where(pos, np.exp(np.where(pos, t, 0.0)) - (terms.y + 1.0), -terms.y)

    def d2(self, t: Array, terms: Terms) -> Array:
        pos = t > 0.0
        return terms.w * np.where(pos, np.exp(np.where(pos, t, 0.0)), 0.0)

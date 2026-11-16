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
from scipy.special import expit, gammaln, xlogy

from .smooth_max import Degree, smooth_max

Array = NDArray[np.float64]


@dataclass(frozen=True)
class Terms:
    """Term structure of a training problem."""

    kernel_index: NDArray[np.intp]
    y: Array
    w: Array
    sign: Array
    offset: Array | None = None

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

    def param_ratio(self, name: str, t: Array, terms: Terms) -> Array:
        """``(d phi'_l / d p) / phi''_l`` at fixed ``t`` for the loss parameter ``p = name``.

        Only meaningful for terms with positive curvature; it is the sensitivity of the optimality
        condition ``a_l + phi'_l(t_l) = 0`` w.r.t. ``p`` in units of ``t`` and is used by the
        hypergradient with respect to loss parameters.
        """
        raise ValueError(f"{type(self).__name__} has no tunable parameter {name!r}")

    def initial_bias(self, terms: Terms) -> float:
        """Starting value of the bias for Newton without an SMO warm start (``a = 0``)."""
        return 0.0


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

    def param_ratio(self, name: str, t: Array, terms: Terms) -> Array:
        if name != "smoothing":
            return super().param_ratio(name, t, terms)
        # every smoothing is m_delta(x) = delta m_1(x / delta), hence
        # d m_delta'(x) / d delta = -(x / delta) m_delta''(x)
        return terms.y * (self.shift - terms.y * t) / self.smoothing


@dataclass(frozen=True)
class LogisticLoss(Loss):
    """Logistic loss ``w * log(1 + exp(-y t))`` for ``y`` in ``{-1, +1}`` (Strasdat, Sec. 4.1.3).

    Smooth without a smoothing parameter and strictly convex, but with positive curvature
    everywhere: the dual solution is dense and the dual is not a separable quadratic, so there is
    no SMO warm start and every Newton system has the full size ``n``.
    """

    def terms(self, y: Array, w: Array) -> Terms:
        return Terms(np.arange(y.size), y, w, np.zeros(y.size))

    def value(self, t: Array, terms: Terms) -> Array:
        return terms.w * np.logaddexp(0.0, -terms.y * t)

    def d1(self, t: Array, terms: Terms) -> Array:
        return -terms.w * terms.y * expit(-terms.y * t)

    def d2(self, t: Array, terms: Terms) -> Array:
        s = expit(-terms.y * t)
        return terms.w * s * (1.0 - s)


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

    def param_ratio(self, name: str, t: Array, terms: Terms) -> Array:
        if name == "epsilon":
            return terms.sign
        if name == "smoothing":  # see HingeLoss.param_ratio
            return terms.sign * self._arg(t, terms) / self.smoothing
        return super().param_ratio(name, t, terms)


@dataclass(frozen=True)
class SquaredLoss(Loss):
    """Least-squares loss ``w / 2 * (t - y)^2`` (LS-SVM)."""

    def initial_bias(self, terms: Terms) -> float:
        """Weighted mean of ``y - offset`` (the minimiser for ``a = 0``)."""
        off = 0.0 if terms.offset is None else terms.offset
        total = float(terms.w.sum())
        return float(terms.w @ (terms.y - off)) / total if total > 0.0 else 0.0

    def value(self, t: Array, terms: Terms) -> Array:
        return 0.5 * terms.w * (t - terms.y) ** 2

    def d1(self, t: Array, terms: Terms) -> Array:
        return terms.w * (t - terms.y)

    def d2(self, t: Array, terms: Terms) -> Array:
        return terms.w * np.ones_like(t)


@dataclass(frozen=True)
class PoissonLoss(Loss):
    """Poisson negative log-likelihood ``w * (exp(t) - y t + log(y!))`` for counts ``y``."""

    def initial_bias(self, terms: Terms) -> float:
        """``log(sum w y / sum w exp(offset))``, the minimiser for ``a = 0`` (for all-zero ``y``
        the numerator is replaced by a small positive mean ``1e-3 sum w``)."""
        off = np.zeros(terms.size) if terms.offset is None else terms.offset
        denom = float(terms.w @ np.exp(off))
        if not denom > 0.0 or not np.isfinite(denom):
            return 0.0
        num = max(float(terms.w @ terms.y), 1e-3 * float(terms.w.sum()))
        return float(np.log(num / denom))

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


def _mean_offset(terms: Terms) -> float:
    if terms.offset is None:
        return 0.0
    return float(terms.w @ terms.offset / terms.w.sum())


def _log_ratio_bias(terms: Terms) -> float:
    """``log(sum w y / sum w exp(offset))``, guarded for ``y == 0`` (log link families)."""
    sw = float(terms.w.sum())
    num = max(float(terms.w @ terms.y), 1e-6 * sw)
    den = sw if terms.offset is None else float(terms.w @ np.exp(terms.offset))
    return float(np.log(num / den))


def _check_terms(name: str, y: Array, w: Array) -> None:
    if y.shape != w.shape:
        raise ValueError(f"{name}: y and w must have the same shape")
    if not np.all(np.isfinite(y)):
        raise ValueError(f"{name}: targets must be finite")
    if np.any(w < 0):
        raise ValueError(f"{name}: weights must be non-negative")


@dataclass(frozen=True)
class BinomialLoss(Loss):
    """Binomial half unit deviance with logit link for proportions ``y`` in ``[0, 1]``.

    With ``mu = expit(t)`` and ``w`` the number of trials (times the sample weight)::

        phi = w * (y log(y/mu) + (1 - y) log((1 - y)/(1 - mu)))
            = w * (xlogy(y, y) + xlogy(1 - y, 1 - y) + log(1 + e^t) - y t)

    ``phi' = w (mu - y)``, ``phi'' = w mu (1 - mu)``; the minimum over ``t`` is 0 at ``mu = y``.
    """

    def terms(self, y: Array, w: Array) -> Terms:
        _check_terms("BinomialLoss", y, w)
        if np.any((y < 0) | (y > 1)):
            raise ValueError("BinomialLoss: targets must be proportions in [0, 1]")
        return super().terms(y, w)

    def value(self, t: Array, terms: Terms) -> Array:
        y = terms.y
        ent = xlogy(y, y) + xlogy(1.0 - y, 1.0 - y)
        return terms.w * (ent + np.logaddexp(0.0, t) - y * t)

    def d1(self, t: Array, terms: Terms) -> Array:
        return terms.w * (expit(t) - terms.y)

    def d2(self, t: Array, terms: Terms) -> Array:
        mu = expit(t)
        return terms.w * mu * (1.0 - mu)

    def initial_bias(self, terms: Terms) -> float:
        m = float(terms.w @ terms.y / terms.w.sum())
        m = min(max(m, 1e-6), 1.0 - 1e-6)
        return float(np.log(m / (1.0 - m))) - _mean_offset(terms)


@dataclass(frozen=True)
class NegativeBinomialLoss(Loss):
    """Negative binomial half unit deviance, log link ``mu = e^t``, dispersion ``r > 0``.

    ::

        phi = w * (y log(y/mu) - (y + r) log((y + r)/(mu + r)))
            = w * (xlogy(y, y) - y t - (y + r) log(y + r) + (y + r) logaddexp(t, log r))

    ``phi' = w r (mu - y)/(mu + r)``, ``phi'' = w (y + r) r mu/(mu + r)^2``; minimum 0 at
    ``mu = y``. Variance ``mu + mu^2 / r``; ``r -> inf`` approaches the Poisson loss.
    ``d phi'/d r = w mu (mu - y)/(mu + r)^2``, hence ``param_ratio = (mu - y)/(r (y + r))``.
    """

    dispersion: float = 1.0

    def terms(self, y: Array, w: Array) -> Terms:
        _check_terms("NegativeBinomialLoss", y, w)
        if self.dispersion <= 0 or not np.isfinite(self.dispersion):
            raise ValueError("NegativeBinomialLoss: dispersion must be positive")
        if np.any(y < 0):
            raise ValueError("NegativeBinomialLoss: targets must be non-negative")
        return super().terms(y, w)

    def value(self, t: Array, terms: Terms) -> Array:
        r, y = self.dispersion, terms.y
        yr = y + r
        return terms.w * (xlogy(y, y) - y * t - yr * np.log(yr) + yr * np.logaddexp(t, np.log(r)))

    def d1(self, t: Array, terms: Terms) -> Array:
        s = expit(t - np.log(self.dispersion))  # mu / (mu + r)
        return terms.w * (self.dispersion * s - terms.y * (1.0 - s))

    def d2(self, t: Array, terms: Terms) -> Array:
        r = self.dispersion
        s = expit(t - np.log(r))
        return terms.w * (terms.y + r) * s * (1.0 - s)

    def param_ratio(self, name: str, t: Array, terms: Terms) -> Array:
        if name != "dispersion":
            return super().param_ratio(name, t, terms)
        r = self.dispersion
        with np.errstate(over="ignore"):
            return (np.exp(t) - terms.y) / (r * (terms.y + r))

    def initial_bias(self, terms: Terms) -> float:
        return _log_ratio_bias(terms)


@dataclass(frozen=True)
class GammaLoss(Loss):
    """Gamma half unit deviance, log link ``mu = e^t``, targets ``y > 0``.

    ``phi = w (y e^{-t} + t - log y - 1)``, ``phi' = w (1 - y e^{-t})``, ``phi'' = w y e^{-t}``;
    minimum 0 at ``mu = y``.
    """

    def terms(self, y: Array, w: Array) -> Terms:
        _check_terms("GammaLoss", y, w)
        if np.any(y <= 0):
            raise ValueError("GammaLoss: targets must be positive")
        return super().terms(y, w)

    def value(self, t: Array, terms: Terms) -> Array:
        with np.errstate(over="ignore"):
            return terms.w * (terms.y * np.exp(-t) + t - np.log(terms.y) - 1.0)

    def d1(self, t: Array, terms: Terms) -> Array:
        with np.errstate(over="ignore"):
            return terms.w * (1.0 - terms.y * np.exp(-t))

    def d2(self, t: Array, terms: Terms) -> Array:
        with np.errstate(over="ignore"):
            return terms.w * terms.y * np.exp(-t)

    def initial_bias(self, terms: Terms) -> float:
        return _log_ratio_bias(terms)


@dataclass(frozen=True)
class TweedieLoss(Loss):
    """Tweedie half unit deviance, log link ``mu = e^t``, power ``1 < p < 2``, targets ``y >= 0``.

    ::

        phi = w * (y^(2-p)/((1-p)(2-p)) - y e^{(1-p)t}/(1-p) + e^{(2-p)t}/(2-p))

    ``phi' = w e^{(1-p)t} (e^t - y)``,
    ``phi'' = w ((p-1) y e^{(1-p)t} + (2-p) e^{(2-p)t})``; minimum 0 at ``mu = y``
    (compound Poisson-gamma; exact zeros allowed).
    """

    power: float = 1.5

    def _check_power(self) -> None:
        if not 1.0 < self.power < 2.0:
            raise ValueError("TweedieLoss: power must lie in (1, 2)")

    def terms(self, y: Array, w: Array) -> Terms:
        _check_terms("TweedieLoss", y, w)
        self._check_power()
        if np.any(y < 0):
            raise ValueError("TweedieLoss: targets must be non-negative")
        return super().terms(y, w)

    def value(self, t: Array, terms: Terms) -> Array:
        p, y = self.power, terms.y
        with np.errstate(over="ignore", invalid="ignore"):
            r = (
                y ** (2.0 - p) / ((1.0 - p) * (2.0 - p))
                - y * np.exp((1.0 - p) * t) / (1.0 - p)
                + np.exp((2.0 - p) * t) / (2.0 - p)
            )
            return np.where(np.isnan(r), np.inf, terms.w * r)

    def d1(self, t: Array, terms: Terms) -> Array:
        p = self.power
        with np.errstate(over="ignore"):
            return terms.w * (np.exp((2.0 - p) * t) - terms.y * np.exp((1.0 - p) * t))

    def d2(self, t: Array, terms: Terms) -> Array:
        p = self.power
        with np.errstate(over="ignore"):
            return terms.w * (
                (p - 1.0) * terms.y * np.exp((1.0 - p) * t) + (2.0 - p) * np.exp((2.0 - p) * t)
            )

    def initial_bias(self, terms: Terms) -> float:
        return _log_ratio_bias(terms)

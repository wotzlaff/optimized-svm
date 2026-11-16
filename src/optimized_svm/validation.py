"""Validation error measures ``E(f) = sum_v L_v(f_v)`` for the upper level problem (Sec. 9.1.4).

All measures are differentiable so that the hyperparameter objective has a gradient.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.special import digamma, expit, gammaln

from .losses import BinomialLoss, GammaLoss, Loss, NegativeBinomialLoss, TweedieLoss
from .smooth_max import Degree, smooth_max

Array = NDArray[np.float64]


class ValidationLoss:
    """Returns the summed loss and its gradient w.r.t. the predictions."""

    #: whether the loss uses per-sample validation weights (e.g. binomial trials)
    weighted: bool = False
    #: names of the parameters (dataclass fields) with a ``param_grad``; the hypergradient adds
    #: their direct derivative if they are tuned as loss parameters
    params: tuple[str, ...] = ()

    def __call__(self, f: Array, y: Array, w: Array | None = None) -> tuple[float, Array]:
        raise NotImplementedError

    def param_grad(self, name: str, f: Array, y: Array, w: Array | None = None) -> float:
        """``dE/dp`` of the summed value at fixed predictions ``f`` for the parameter ``name``."""
        raise ValueError(f"{type(self).__name__} has no tunable parameter {name!r}")


@dataclass(frozen=True)
class SmoothHingeValidation(ValidationLoss):
    """``sum_v w_v m_delta(shift - y_v f_v)``; ``balanced`` weights both classes equally."""

    shift: float = 1.0
    smoothing: float = 0.5
    degree: Degree = "6"
    balanced: bool = False

    def __call__(self, f: Array, y: Array, w: Array | None = None) -> tuple[float, Array]:
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
class LogisticValidation(ValidationLoss):
    """``sum_v w_v log(1 + exp(-y_v f_v))``; ``balanced`` weights both classes equally."""

    balanced: bool = False

    def __call__(self, f: Array, y: Array, w: Array | None = None) -> tuple[float, Array]:
        w = np.ones_like(y)
        if self.balanced:
            pos = y > 0
            frac = pos.mean()
            if 0 < frac < 1:
                w = 0.5 / np.where(pos, frac, 1.0 - frac)
        return float(w @ np.logaddexp(0.0, -y * f)), -w * y * expit(-y * f)


@dataclass(frozen=True)
class SmoothAbsoluteValidation(ValidationLoss):
    """Smoothed absolute error ``sum_v m_delta(f_v - y_v) + m_delta(y_v - f_v)`` (Huber-like)."""

    smoothing: float = 0.1
    degree: Degree = "2"

    def __call__(self, f: Array, y: Array, w: Array | None = None) -> tuple[float, Array]:
        m, dm, _ = smooth_max(self.smoothing, self.degree)
        e = f - y
        return float(m(e).sum() + m(-e).sum()), dm(e) - dm(-e)


@dataclass(frozen=True)
class SquaredValidation(ValidationLoss):
    """``1/2 sum_v (f_v - y_v)^2``."""

    def __call__(self, f: Array, y: Array, w: Array | None = None) -> tuple[float, Array]:
        e = f - y
        return 0.5 * float(e @ e), e


@dataclass(frozen=True)
class PoissonValidation(ValidationLoss):
    """Poisson deviance-type loss ``sum_v exp(f_v) - y_v f_v + log(y_v!)``."""

    def __call__(self, f: Array, y: Array, w: Array | None = None) -> tuple[float, Array]:
        with np.errstate(over="ignore"):
            ef = np.exp(f)
        return float(ef.sum() - f @ y + gammaln(y + 1).sum()), ef - y


def _deviance(loss: Loss, f: Array, y: Array, w: Array | None) -> tuple[float, Array]:
    """Summed half unit deviance of ``loss`` at predictions ``f`` (already on the link scale)."""
    terms = loss.terms(y, np.ones_like(y) if w is None else w)
    return float(loss.value(f, terms).sum()), loss.d1(f, terms)


@dataclass(frozen=True)
class BinomialDeviance(ValidationLoss):
    """Binomial half deviance ``sum_v w_v (y_v log(y_v/mu_v) + (1-y_v) log((1-y_v)/(1-mu_v)))``.

    ``mu = expit(f)``, ``y`` are proportions and ``w`` the numbers of trials.
    """

    weighted: bool = True

    def __call__(self, f: Array, y: Array, w: Array | None = None) -> tuple[float, Array]:
        return _deviance(BinomialLoss(), f, y, w)


@dataclass(frozen=True)
class NegativeBinomialDeviance(ValidationLoss):
    """Negative binomial half deviance with log link, see ``NegativeBinomialLoss``."""

    dispersion: float = 1.0

    def __call__(self, f: Array, y: Array, w: Array | None = None) -> tuple[float, Array]:
        return _deviance(NegativeBinomialLoss(self.dispersion), f, y, None)


@dataclass(frozen=True)
class GammaDeviance(ValidationLoss):
    """Gamma half deviance with log link, see ``GammaLoss``."""

    def __call__(self, f: Array, y: Array, w: Array | None = None) -> tuple[float, Array]:
        return _deviance(GammaLoss(), f, y, None)


@dataclass(frozen=True)
class TweedieDeviance(ValidationLoss):
    """Tweedie half deviance with log link, see ``TweedieLoss``."""

    power: float = 1.5

    def __call__(self, f: Array, y: Array, w: Array | None = None) -> tuple[float, Array]:
        return _deviance(TweedieLoss(self.power), f, y, None)


@dataclass(frozen=True)
class NegativeBinomialNLL(ValidationLoss):
    """Full negative binomial negative log-likelihood with log link ``mu = e^f``::

        E = sum_v -lgamma(y+r) + lgamma(r) + lgamma(y+1) - r log(r/(r+mu)) - y log(mu/(r+mu))

    Unlike the deviance it contains the terms that depend on the dispersion ``r`` and is
    therefore a proper criterion for tuning ``r``. ``dE/df = r (mu - y)/(mu + r)`` and
    ``dE/dr = sum -psi(y+r) + psi(r) - log(r/(r+mu)) - 1 + (y+r)/(r+mu)``.
    """

    dispersion: float = 1.0
    params: tuple[str, ...] = ("dispersion",)

    def __call__(self, f: Array, y: Array, w: Array | None = None) -> tuple[float, Array]:
        r = self.dispersion
        lr = np.log(r)
        with np.errstate(over="ignore"):
            lrm = np.logaddexp(lr, f)  # log(r + mu)
        val = (-gammaln(y + r) + gammaln(r) + gammaln(y + 1) - r * (lr - lrm) - y * (f - lrm)).sum()
        return float(val), r * (np.exp(f - lrm) - y * np.exp(lr - lrm) / r)

    def param_grad(self, name: str, f: Array, y: Array, w: Array | None = None) -> float:
        if name != "dispersion":
            return super().param_grad(name, f, y, w)
        r = self.dispersion
        lrm = np.logaddexp(np.log(r), f)
        return float(
            (-digamma(y + r) + digamma(r) + lrm - np.log(r) - 1.0 + (y + r) * np.exp(-lrm)).sum()
        )

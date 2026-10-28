"""Smooth approximations of the maximum function ``max(0, t)`` (Strasdat, Sec. 4.4).

Each approximation ``m_delta`` coincides with ``max(0, t)`` for ``|t| >= delta`` and is a
polynomial on ``(-delta, delta)``. The degree determines the smoothness:

* ``"2"``: ``C^1``, piecewise quadratic (its conjugate is quadratic, so SMO applies),
* ``"3"``: ``C^2``, piecewise cubic,
* ``"6"``: ``C^2``, a single sextic polynomial on ``(-delta, delta)``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
Degree = Literal["2", "3", "6"]
SmoothMax = tuple[Callable[[Array], Array], Callable[[Array], Array], Callable[[Array], Array]]


def _deg2(d: float) -> SmoothMax:
    def m(t: Array) -> Array:
        return np.where(t >= d, t, np.where(t <= -d, 0.0, 0.25 / d * (t + d) ** 2))

    def dm(t: Array) -> Array:
        return np.where(t >= d, 1.0, np.where(t <= -d, 0.0, 0.5 / d * (t + d)))

    def ddm(t: Array) -> Array:
        return np.where(np.abs(t) < d, 0.5 / d, 0.0)

    return m, dm, ddm


def _deg3(d: float) -> SmoothMax:
    def m(t: Array) -> Array:
        inner = np.where(t <= 0.0, (d + t) ** 3 / (6 * d**2), t + (d - t) ** 3 / (6 * d**2))
        return np.where(t >= d, t, np.where(t <= -d, 0.0, inner))

    def dm(t: Array) -> Array:
        inner = np.where(t <= 0.0, (d + t) ** 2 / (2 * d**2), 1.0 - (d - t) ** 2 / (2 * d**2))
        return np.where(t >= d, 1.0, np.where(t <= -d, 0.0, inner))

    def ddm(t: Array) -> Array:
        return np.where(np.abs(t) < d, (d - np.abs(t)) / d**2, 0.0)

    return m, dm, ddm


def _deg6(d: float) -> SmoothMax:
    def m(t: Array) -> Array:
        u = t / d
        inner = d * (5 + u * (16 + u * (15 + u * u * (-5 + u * u)))) / 32
        return np.where(t >= d, t, np.where(t <= -d, 0.0, inner))

    def dm(t: Array) -> Array:
        u = t / d
        inner = (8 + u * (15 + u * u * (-10 + 3 * u * u))) / 16
        return np.where(t >= d, 1.0, np.where(t <= -d, 0.0, inner))

    def ddm(t: Array) -> Array:
        u = t / d
        return np.where(np.abs(t) < d, (15 + u * u * (-30 + 15 * u * u)) / (16 * d), 0.0)

    return m, dm, ddm


def _exact() -> SmoothMax:
    return (
        lambda t: np.maximum(t, 0.0),
        lambda t: (t > 0.0).astype(float),
        lambda t: np.zeros_like(t, dtype=float),
    )


def smooth_max(delta: float, degree: Degree = "2") -> SmoothMax:
    """Returns ``(m, dm, ddm)``: the smoothed maximum and its first two derivatives.

    ``delta = 0`` yields the (non-smooth) maximum itself.
    """
    if delta < 0:
        raise ValueError(f"delta must be non-negative, got {delta}")
    if delta == 0:
        return _exact()
    try:
        return {"2": _deg2, "3": _deg3, "6": _deg6}[str(degree)](float(delta))
    except KeyError:
        raise ValueError(f"unknown degree {degree!r}, use '2', '3' or '6'") from None

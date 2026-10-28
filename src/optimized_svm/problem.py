"""Training problem in the dual parametrisation used by the Newton method.

For loss terms ``phi_l`` (see :mod:`optimized_svm.losses`), the primal training problem
``min lam/2 |w|^2 + sum_l phi_l(<w, x_l> + b + s_l c) + A c`` is written in terms of
``a`` with ``w = 1/lam sum_l a_l phi(x_l)`` (representer theorem). With
``ka = K a / lam`` the objective reads ``1/2 a^T ka + sum_l phi_l(ka_l + b + s_l c) + A c``
(Strasdat, Ch. 6 and 8).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .kernels import rbf
from .losses import Loss

Array = NDArray[np.float64]
Index = NDArray[np.intp] | NDArray[np.bool_] | slice

#: kernel matrices up to this number of samples are computed once and kept in memory
PRECOMPUTE_LIMIT = 8000


class TrainingProblem:
    """Training problem on (already scaled) features ``x`` with an RBF kernel (``gamma = 1``)."""

    def __init__(
        self,
        x: Array,
        y: Array,
        loss: Loss,
        lam: float,
        weights: Array | None = None,
        max_asum: float | None = None,
    ) -> None:
        if lam <= 0:
            raise ValueError(f"lam must be positive, got {lam}")
        if max_asum is not None and not loss.signed:
            raise ValueError(f"max_asum is not supported for {type(loss).__name__}")
        self.x = np.ascontiguousarray(x, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.float64)
        self.w = np.ones_like(self.y) if weights is None else np.asarray(weights, dtype=np.float64)
        self.loss = loss
        self.lam = float(lam)
        self.max_asum = max_asum
        self.terms = loss.terms(self.y, self.w)
        n = self.x.shape[0]
        # the matrix is stored per *sample*, not per loss term
        self._k = rbf(self.x, self.x) if n <= PRECOMPUTE_LIMIT else None

    @property
    def n_samples(self) -> int:
        return self.x.shape[0]

    @property
    def size(self) -> int:
        """Number of loss terms (= dual variables)."""
        return self.terms.size

    @property
    def sign(self) -> Array:
        return self.terms.sign

    def kernel(self, rows: Index = slice(None), cols: Index = slice(None)) -> Array:
        """Kernel matrix between loss terms ``rows`` and ``cols``."""
        ki = self.terms.kernel_index
        r, c = ki[rows], ki[cols]
        if self._k is not None:
            return self._k[np.ix_(r, c)]
        return rbf(self.x[r], self.x[c])

    def kernel_to(self, z: Array, rows: Index = slice(None)) -> Array:
        """Kernel matrix between loss terms ``rows`` and new (scaled) points ``z``."""
        return rbf(self.x[self.terms.kernel_index[rows]], z)

    def sample_coefficients(self, a: Array) -> Array:
        """Coefficients per sample ``beta_k = sum_{l: k(l) = k} a_l``."""
        return np.bincount(self.terms.kernel_index, weights=a, minlength=self.n_samples).astype(
            np.float64
        )

    def ka(self, a: Array) -> Array:
        """``K a / lam`` per loss term (computed via the sample coefficients)."""
        beta = self.sample_coefficients(a)
        sv = np.flatnonzero(beta)
        if self._k is not None:
            per_sample = self._k[:, sv] @ beta[sv]
        else:
            per_sample = rbf(self.x, self.x[sv]) @ beta[sv]
        return per_sample[self.terms.kernel_index] / self.lam

    def decision(self, b: float, c: float, ka: Array) -> Array:
        return ka + b + self.sign * c

    def objective(self, a: Array, b: float, c: float, ka: Array) -> float:
        dec = self.decision(b, c, ka)
        obj = 0.5 * float(a @ ka) + float(self.loss.value(dec, self.terms).sum())
        if self.max_asum is not None:
            obj += self.max_asum * c
        return obj

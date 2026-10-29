"""Bordered symmetric positive definite systems."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import cho_factor, cho_solve
from threadpoolctl import ThreadpoolController

Array = NDArray[np.float64]

#: systems up to this size are factorised with a single BLAS thread; multithreaded LAPACK is
#: several times slower for them (synchronisation, oversubscription on shared machines)
SINGLE_THREAD_LIMIT = 1000

#: the library scan of ``threadpool_limits`` is expensive (ms), so do it once
_blas = ThreadpoolController().select(user_api="blas")


def factorize(q: Array) -> Any:
    """Cholesky factorisation of an SPD matrix (for :func:`solve_bordered`)."""
    with _blas.limit(limits=1 if q.shape[0] <= SINGLE_THREAD_LIMIT else None):
        return cho_factor(q)


def solve_bordered(
    q: Array | None, a: Array, r: Array, r0: Array, factor: Any = None
) -> tuple[Array, Array]:
    """Solves ``[[Q, A], [A^T, 0]] [u; v] = [r; r0]`` for SPD ``Q`` (Schur complement).

    ``A`` has shape ``(n, k)`` with ``k`` small (one column per equality constraint). A
    factorisation of ``Q`` from :func:`factorize` can be passed instead of ``Q``.
    """
    k = a.shape[1]
    if a.shape[0] == 0:
        return np.zeros(0), np.zeros(k)
    if factor is None:
        assert q is not None
        factor = factorize(q)
    with _blas.limit(limits=1 if a.shape[0] <= SINGLE_THREAD_LIMIT else None):
        qia = cho_solve(factor, a)
        qir = cho_solve(factor, r)
    v = np.linalg.solve(a.T @ qia, a.T @ qir - r0)
    return qir - qia @ v, v

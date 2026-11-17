"""Bordered symmetric positive definite systems."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
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

#: >0 while folds run in parallel threads: the BLAS limit is then set once by
#: :func:`parallel_section`, and the nested limits below are skipped (they save and restore a
#: process-global value, so concurrent workers would interleave them)
_parallel_depth = 0


def blas_threads() -> int:
    """Current number of BLAS threads (of the largest pool)."""
    return max((int(lib["num_threads"]) for lib in _blas.info()), default=1)


@contextmanager
def parallel_section(limit: int) -> Iterator[None]:
    """Limits BLAS to ``limit`` threads for a section whose workers run concurrently; the
    single-thread limits of :func:`factorize` and :func:`solve_bordered` are inactive meanwhile."""
    global _parallel_depth
    with _blas.limit(limits=limit):
        _parallel_depth += 1
        try:
            yield
        finally:
            _parallel_depth -= 1


def _limit(n: int) -> Any:
    """Single-thread limit for systems of size ``n`` (no-op inside a parallel section)."""
    small = n <= SINGLE_THREAD_LIMIT and _parallel_depth == 0
    return _blas.limit(limits=1 if small else None)


def factorize(q: Array) -> Any:
    """Cholesky factorisation of an SPD matrix (for :func:`solve_bordered`)."""
    with _limit(q.shape[0]):
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
    with _limit(a.shape[0]):
        qia = cho_solve(factor, a)
        qir = cho_solve(factor, r)
    v = np.linalg.solve(a.T @ qia, a.T @ qir - r0)
    return qir - qia @ v, v

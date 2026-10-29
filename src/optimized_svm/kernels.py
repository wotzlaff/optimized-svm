"""Gaussian kernel with feature scaling and its derivatives w.r.t. log-scalings.

The hyperparameters of the kernel are scalings ``s_k > 0`` of feature blocks ``B_k``:
``k(x, z) = exp(-sum_k s_k^2 |x_{B_k} - z_{B_k}|^2)``. With a single block this is the RBF
kernel with ``gamma = s^2``; with one block per feature it is an ARD kernel. The training code
works on *scaled* features ``x_{B_k} * s_k`` and an RBF kernel with ``gamma = 1``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

Array = NDArray[np.float64]
Blocks = Sequence[Sequence[int]] | None


def squared_distances(x: Array, z: Array) -> Array:
    """Matrix of squared Euclidean distances ``|x_i - z_j|^2`` (clipped at zero)."""
    d = (x * x).sum(axis=1)[:, None] + (z * z).sum(axis=1)[None, :] - 2.0 * x @ z.T
    return np.maximum(d, 0.0)


def rbf(x: Array, z: Array) -> Array:
    """``exp(-|x - z|^2)`` for already scaled features."""
    return np.exp(-squared_distances(x, z))


def rbf_with_gradient(x: Array, z: Array, blocks: Blocks = None) -> tuple[Array, Array]:
    """Kernel matrix and its derivatives w.r.t. the log-scalings of all blocks.

    Returns ``K`` of shape ``(n, m)`` and ``dK`` of shape ``(n, m, n_blocks)`` with
    ``dK[..., k] = d K / d log s_k = -2 K * |x_{B_k} - z_{B_k}|^2`` (features already scaled).
    """
    if blocks is None:
        d = squared_distances(x, z)
        k = np.exp(-d)
        return k, (-2.0 * k * d)[:, :, None]
    parts = [squared_distances(x[:, list(b)], z[:, list(b)]) for b in blocks]
    k = np.exp(-sum(parts))
    return k, -2.0 * k[:, :, None] * np.stack(parts, axis=-1)


def contract_gradient(
    x: Array, z: Array, k: Array, left: Array, right: Array, blocks: Blocks = None
) -> Array:
    """Bilinear forms ``left^T (dK / d log s_k) right`` with ``dK_k = -2 K * |x_Bk - z_Bk|^2``.

    ``k`` is the kernel matrix between the (already scaled) points ``x`` and ``z``. Same as
    contracting the output of :func:`rbf_with_gradient`, but without forming the distance
    matrices of the blocks: with ``W = K * left right^T`` one has
    ``sum_ij W_ij |x_i - z_j|^2 = rowsum(W) . |x|^2 + colsum(W) . |z|^2 - 2 sum(x * (W z))``
    per feature, so a single matrix product serves all blocks.
    """
    w = k * left[:, None] * right[None, :]
    per_feature = (
        w.sum(axis=1) @ (x * x) + w.sum(axis=0) @ (z * z) - 2.0 * (x * (w @ z)).sum(axis=0)
    )
    if blocks is None:
        return np.array([-2.0 * float(per_feature.sum())])
    return np.array([-2.0 * float(per_feature[list(b)].sum()) for b in blocks])


def n_blocks(blocks: Blocks) -> int:
    return 1 if blocks is None else len(blocks)


def scale_features(x: ArrayLike, scaling: ArrayLike, blocks: Blocks = None) -> Array:
    """Multiplies the features of each block by its scaling."""
    x = np.asarray(x, dtype=np.float64)
    scaling = np.atleast_1d(np.asarray(scaling, dtype=np.float64))
    if blocks is None:
        if scaling.size != 1:
            raise ValueError("a single scaling is required without blocks")
        return x * scaling[0]
    if scaling.size != len(blocks):
        raise ValueError(f"expected {len(blocks)} scalings, got {scaling.size}")
    xs = x.copy()
    for s, block in zip(scaling, blocks, strict=True):
        xs[:, list(block)] *= s
    return xs

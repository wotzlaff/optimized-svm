import numpy as np


def diff(x1, x2, idx=slice(None)):
    x1sqr = (x1[:, idx] * x1[:, idx]).sum(axis=1)
    x2sqr = (x2[:, idx] * x2[:, idx]).sum(axis=1)
    d = x1sqr[:, None] + x2sqr[None, :] - 2.0 * x1[:, idx].dot(x2[:, idx].T)
    return d


def rbf_kernel(x1, x2, kernel_param=dict(), deriv=False):
    blocks = kernel_param.get('blocks', None)
    d = diff(x1, x2, slice(None))
    k = np.exp(-kernel_param['gamma'] * d)
    if not deriv:
        return k
    if blocks is not None:
        dk = -2.0 * k[:, :, None] * np.stack(
            [diff(x1, x2, block) for block in blocks],
            axis=-1,
        )
        return k, dk
    return k, -2.0 * k * d

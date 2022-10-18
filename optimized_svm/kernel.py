import numpy as np
import numpy.typing as npt
Arr = npt.NDArray[np.float64]


def diff(x1, x2, idx=slice(None)):
    x1sqr = (x1[:, idx] * x1[:, idx]).sum(axis=1)
    x2sqr = (x2[:, idx] * x2[:, idx]).sum(axis=1)
    d = x1sqr[:, None] + x2sqr[None, :] - 2.0 * x1[:, idx].dot(x2[:, idx].T)
    return d


def rbf_kernel(x1, x2, kernel_param=dict(), deriv=False):
    blocks = kernel_param.get('blocks', None)
    d = diff(x1, x2, slice(None))
    k = np.exp(-kernel_param.get('gamma', 1.0) * d)
    if not deriv:
        return k
    if blocks is not None:
        dk = -2.0 * k[:, :, None] * np.stack(
            [diff(x1, x2, block) for block in blocks],
            axis=-1,
        )
        return k, dk
    return k, -2.0 * (k * d)[:, :, np.newaxis]


class KernelMxin:
    _x: Arr

    def __init__(self, kernel_fun, kernel_param):
        self.kernel_fun = kernel_fun
        self.kernel_param = kernel_param

    def k(
        self,
        x: Arr | None = None,
        idx=None,
        idx0=None,
        deriv: bool = False,
    ):
        if idx0 is None:
            x0 = self._x
        else:
            x0 = self._x[idx0]
        if x is not None:
            x1 = x
        if idx is not None:
            x1 = self._x[idx]
        return self.kernel_fun(
            x0, x1,
            self.kernel_param,
            deriv=deriv,
        )

import numpy as np


def estimate_sensitivity(self, x):
    xs = self.scale(x, self.scaling)
    d = xs[:, np.newaxis, :] - self._xsv[np.newaxis, :, :]
    gamma = self.kernel_param['gamma']
    k = self.kernel_fun(
        xs, self._xsv,
        self.kernel_param,
    )
    dka = -2.0 * gamma * np.tensordot(
        k[:, :, np.newaxis] * d, self._asv,
        axes=(1, 0),
    )
    return dka

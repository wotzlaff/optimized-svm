import numpy as np


class DerChecker:
    def __init__(self, tau=1e-8):
        self._vals = []
        self.tau = tau

    def push(self, val):
        self._vals.append(val)

    def compare(self, dval):
        n = len(self._vals)
        if n == 0:
            print('no value there')
            return
        if n == 1:
            print('only one value there')
            return
        if n > 2:
            print('more than two value there')
            return
        val0, val1 = self._vals
        dval_approx = (val1 - val0) / self.tau
        if np.array(val0).size == 1:
            print([dval, dval_approx, dval - dval_approx, dval / dval_approx])
        else:
            print(np.hstack((
                dval[:, None],
                dval_approx[:, None],
                dval[:, None] - dval_approx[:, None],
                dval[:, None] / dval_approx[:, None],
            )))


def check_grad(f, x0, tau=1e-6):
    f0, df0 = f(x0)
    df_approx = np.zeros_like(x0)
    for i in range(x0.size):
        x1 = np.array(x0)
        x1.flat[i] += tau
        f1 = f(x1)[0]
        df_approx.flat[i] = (f1 - f0) / tau

    print(np.hstack((
        df0[:, None],
        df_approx[:, None],
        df0[:, None] - df_approx[:, None],
        df0[:, None] / df_approx[:, None],
    )))

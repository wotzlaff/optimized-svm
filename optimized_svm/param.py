from newton import solve_twostep

import numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import KFold


from smooth_max import maxfun
from deriv_check import check_grad
from rendered import solve_rendered, solve_rendered_transposed


def diff(x1, x2, idx=slice(None)):
    x1sqr = (x1[:, idx] * x1[:, idx]).sum(axis=1)
    x2sqr = (x2[:, idx] * x2[:, idx]).sum(axis=1)
    d = x1sqr[:, None] + x2sqr[None, :] - 2.0 * x1[:, idx].dot(x2[:, idx].T)
    return d


def kernel(x1, x2, gamma=1.0):
    return np.exp(-gamma * diff(x1, x2, slice(None)))


class SimpleSVM(BaseEstimator, RegressorMixin):

    def __init__(self, params):
        self._params = params

    def fit(self, x, y):
        z = solve_twostep(
            self._params['gamma'] * x, y,
            maxfun(self._params['smoothingParam']),
            self._params['lmbda'],
            1.0,
            self._params['smoothingParam'],
            self._params['shift'],
        )
        self._a, self._b, self._c = z
        idx_sv = self._a != 0.0
        self._asv = self._a[idx_sv]
        self._xsv = x[idx_sv]

    def predict(self, x):
        lmbda = self._params['lmbda']
        gamma = np.array(self._params['gamma'])
        xs = gamma[:, None] * x
        xs_sv = gamma[:, None] * self._xsv
        return kernel(xs, xs_sv).dot(self._asv) / lmbda + self._b


class OptimizedSVM(BaseEstimator, RegressorMixin):

    def __init__(self, params):
        self._params = params

    def optimize_params(self, x, y, splits):
        def get_scaling(scaling_params):
            assert scaling_params.size == 1
            return scaling_params

        def objective_function(param):
            log_lmbda = param[0]
            lmbda = np.exp(log_lmbda)
            log_gamma = get_scaling(param[1:])
            gamma = np.exp(log_gamma)
            mf = maxfun(self._params['smoothingParam'])
            shift = self._params['shift']

            err = 0.0
            derr = np.zeros(2)
            for idx_tr, idx_val in splits:

                a, b, _c = solve_twostep(
                    gamma * x[idx_tr], y[idx_tr],
                    mf, lmbda, 1.0,
                    self._params['smoothingParam'],
                    shift,
                )
                al = a / lmbda

                idx_sv = a != 0.0
                al_sv = al[idx_sv]
                xs_val = gamma[:, None] * x[idx_val]
                xs_tr = gamma[:, None] * x[idx_tr]
                xs_sv = xs_tr[idx_sv]
                k_val_sv = kernel(xs_val, xs_sv)
                ka_val = k_val_sv.dot(al_sv)
                y_val_pred = ka_val + b
                l_val = mf[0](1.0 - y_val_pred * y[idx_val]).sum()
                err += l_val

                k_tr_sv = kernel(xs_tr, xs_sv)
                ka = k_tr_sv.dot(al_sv)

                dec = ka + b
                f = shift - y[idx_tr] * dec
                h = mf[2](f)

                dl_val = -y[idx_val] * mf[1](1.0 - y_val_pred * y[idx_val])

                idx_p = h > 0
                idx_p_in_sv = h[idx_sv] > 0
                h_p = h[idx_p]
                if h_p.size == 0:
                    return np.inf, np.zeros_like(derr)

                k_val_p = k_val_sv[:, idx_p_in_sv]
                dk_val_sv = -2.0 * k_val_sv * diff(xs_val, xs_sv)

                k_p_sv = k_tr_sv[idx_p, :]
                k_p_p = k_p_sv[:, idx_p_in_sv]

                lhs_p = dl_val.dot(k_val_p) / lmbda
                lhs_0 = dl_val.sum()
                gamma_p, _ = solve_rendered_transposed(
                    k_p_p, h_p,
                    lhs_p, lhs_0,
                    lmbda,
                )

                # dlmbda:
                rhs_p = h_p * ka[idx_p]
                derr[0] += (
                    gamma_p.dot(rhs_p)
                    +
                    -dl_val.dot(ka_val)
                )

                # dgamma:
                dk_tr_p = -2.0 * k_p_sv * diff(xs_tr[idx_p], xs_sv)
                rhs_p = -h_p * dk_tr_p.dot(al_sv)
                derr[1] += (
                    gamma_p.dot(rhs_p)
                    +
                    dl_val.dot(dk_val_sv.dot(al_sv))
                )

            print(lmbda, gamma[0], err, derr)
            return err, derr

        p0 = np.log([self._params['lmbda'], self._params['gamma']])
        # check_grad(objective_function, p0)
        # raise

        res = minimize(
            objective_function, p0,
            method='L-BFGS-B',
            options=dict(disp=True),
            jac=True,
        )
        p_opt = res.x
        self._params.update(dict(
            lmbda=np.exp(p_opt[0]),
            gamma=np.exp(p_opt[1:]),
        ))

    def fit(self, x, y):
        splits = list(KFold(5).split(x))
        # n = x.shape[0]
        # splits = [(slice(0, n//2), slice(n//2, n))]
        self.optimize_params(x, y, splits)
        self._svm = SimpleSVM(self._params)
        self._svm.fit(x, y)

    def predict(self, x):
        return self._svm.predict(x)


if __name__ == '__main__':
    np.random.seed(42)
    n, nft = 1000, 10
    x = np.random.rand(n, nft)
    yr = np.sin(np.pi * x).sum(axis=1)
    y = np.where(yr > np.median(yr), 1.0, -1.0)

    svm = OptimizedSVM(dict(
        lmbda=0.1,
        gamma=0.5,
        smoothingParam=0.2,
        shift=1.0
    ))
    svm.fit(x[:800], y[:800])

    yest = svm.predict(x)

    import matplotlib.pyplot as plt
    plt.plot(y[:800], yest[:800], 'kx')
    plt.plot(y[800:] * 0.9, yest[800:], 'r+')
    plt.axhline(1.0)
    plt.axhline(0.0)
    plt.axhline(-1.0)
    plt.show()

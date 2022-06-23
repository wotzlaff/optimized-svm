from newton import solve_twostep

import numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import KFold


import rendered
from smooth_max import get_maxfun
from kernel import rbf_kernel as kernel, diff


class SimpleSVM(BaseEstimator, RegressorMixin):

    def __init__(self, params):
        self._params = params

    def fit(self, x, y):
        res = solve_twostep(
            self._params['gamma'] * x, y, kernel,
            dict(self._params, gamma=1.0),
        )
        self._a, self._b, self._c = res['a'], res['b'], res['c']
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

    def __init__(self, params, verbose=0):
        self._params = params
        self._verbose = verbose

    def optimize_params(self, x, y, splits):
        m, dm, ddm = mf = get_maxfun(
            self._params['smoothingParam'],
            self._params['maxfunDegree'],
        )
        shift = self._params['shift']
        max_asum = self._params.get('maxAsum', float('inf'))
        has_asum = ~np.isinf(max_asum)
        s = y
        
        def get_scaling(scaling_params):
            assert scaling_params.size == 1
            return scaling_params

        def loss(t, y, mf):
            shift = 1.0
            m, dm = mf[:2]
            ell = m(shift - t * y).sum()
            dell = -y * dm(shift - t * y)
            return ell, dell

        def objective_function(param):
            log_lmbda = param[0]
            lmbda = np.exp(log_lmbda)
            log_gamma = get_scaling(param[1:])
            gamma = np.exp(log_gamma)

            if self._verbose > 0:
                print(f'lmbda = {lmbda:14.8f}, ', end='')
                print(f'gamma = {gamma[0]:14.8f}, ', end='')

            err = 0.0
            derr = np.zeros(2)
            for idx_tr, idx_val in splits:
                res = solve_twostep(
                    gamma * x[idx_tr], y[idx_tr], kernel,
                    dict(self._params, lmbda=lmbda, gamma=1.0),
                )
                a, b, c = res['a'], res['b'], res['c']
                # extract solution
                al = a / lmbda
                idx_sv = a != 0.0
                al_sv = al[idx_sv]

                # scale inputs
                xs_val = gamma[:, None] * x[idx_val]
                xs_tr = gamma[:, None] * x[idx_tr]
                xs_sv = xs_tr[idx_sv]

                # compute validation estimates and error
                k_val_sv = kernel(xs_val, xs_sv)
                ka_val = k_val_sv.dot(al_sv)
                y_val_pred = ka_val + b
                l_val, dl_val = loss(y_val_pred, y[idx_val], mf)
                err += l_val

                # compute decision function values
                k_tr_sv = kernel(xs_tr, xs_sv)
                ka = k_tr_sv.dot(al_sv)
                dec = ka + b + c * y[idx_tr]
                f = shift - y[idx_tr] * dec
                h = ddm(f)

                # extract relevant part
                idx_p = h > 0
                idx_p_in_sv = h[idx_sv] > 0
                h_p = h[idx_p]
                if h_p.size == 0:
                    return np.inf, np.zeros_like(derr)
                k_p_sv = k_tr_sv[idx_p, :]
                k_p_p = k_p_sv[:, idx_p_in_sv]
                k_val_p = k_val_sv[:, idx_p_in_sv]

                if (idx_sv & (~idx_p)).any():
                    k_p_p = kernel(xs_tr[idx_p], xs_tr[idx_p])
                    k_val_p = kernel(xs_val, xs_tr[idx_p])

                # solve transposed system
                lhs_p = dl_val.dot(k_val_p) / lmbda
                lhs_1 = dl_val.sum()
                if has_asum:
                    hu_p, _ = rendered.solve(
                        k_p_p / lmbda + np.diag(1.0 / h_p),
                        (np.ones_like(lhs_p), s[idx_tr][idx_p]),
                        lhs_p, (lhs_1, 0.0),
                    )
                else:
                    hu_p, _ = rendered.solve(
                        k_p_p / lmbda + np.diag(1.0 / h_p),
                        np.ones_like(lhs_p),
                        lhs_p, lhs_1,
                    )

                # dlmbda:
                rhs_p = ka[idx_p]
                derr[0] += hu_p.dot(rhs_p) - dl_val.dot(ka_val)

                # dgamma:
                dk_val_sv = -2.0 * k_val_sv * diff(xs_val, xs_sv)
                dk_tr_p = -2.0 * k_p_sv * diff(xs_tr[idx_p], xs_sv)
                rhs_p = -dk_tr_p.dot(al_sv)
                derr[1] += hu_p.dot(rhs_p) + dl_val.dot(dk_val_sv.dot(al_sv))

            if self._verbose > 0:
                print(f'err = {err:14.8f}, derr = {derr}')
            return err, derr

        p0 = np.log([self._params['lmbda'], self._params['gamma']])
        # check_grad(objective_function, p0)
        # raise

        res = minimize(
            objective_function, p0,
            method='L-BFGS-B',
            options=dict(disp=True),
            jac=True,
            bounds=[(-10, 5), (-10, 5)],
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
        smoothingParam=0.1,
        shift=1.0,
        maxfunDegree='2',
        tolSmo=1e-2,
        tolNewton=1e-10,
        # verbose=10000,
        maxAsum=10.0,
    ))
    svm.fit(x[:n//2], y[:n//2])

    yest = svm.predict(x)

    import matplotlib.pyplot as plt
    plt.plot(y[:n//2], yest[:n//2], 'kx')
    plt.plot(y[n//2:] * 0.9, yest[n//2:], 'r+')
    plt.axhline(1.0)
    plt.axhline(0.0)
    plt.axhline(-1.0)
    plt.show()

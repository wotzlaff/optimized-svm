from functools import partial

import numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import KFold

from newton import solve_twostep
from sample import data
import rendered
import problem
from smooth_max import get_maxfun


class SimpleSVM(BaseEstimator, ClassifierMixin):

    def __init__(self, params):
        self._params = params

    def fit(self, x, y):
        self._prob = problem.Classification(
            self._params['gamma'] * x, y,
            dict(self._params, gamma=1.0),
        )
        res = solve_twostep(self._prob)
        self._a, self._b, self._c = res['a'], res['b'], res['c']
        self._idx_sv = idx_sv = self._a != 0.0
        self._asv = self._a[idx_sv]
        self._xsv = x[idx_sv]

    def predict(self, x):
        lmbda = self._params['lmbda']
        gamma = np.array(self._params['gamma'])
        xs = gamma[:, None] * x
        ka = self._prob.kernel(x=xs, idx0=self._idx_sv).T.dot(self._asv)
        return ka / lmbda + self._b


class OptimizedSVM(BaseEstimator, ClassifierMixin):

    def __init__(self, params, verbose=0):
        self._params = params
        self._verbose = verbose
        self._validation_loss = self._validation_loss_regression

    def _validation_loss_regression(self, t, y):
        m, dm = get_maxfun(
            self._params['smoothingParam'],
            self._params['maxfunDegree'],
        )[:2]
        ell = m(t - y).sum() + m(y - t).sum()
        dell = dm(t - y) - dm(y - t)
        return ell, dell

    def _validation_loss_classification(self, t, y):
        shift = 1.0
        m, dm = get_maxfun(
            self._params['smoothingParam'],
            self._params['maxfunDegree'],
        )[:2]
        ell = m(shift - t * y).sum()
        dell = -y * dm(shift - t * y)
        return ell, dell

    def _get_scaling(self, scaling_params):
        assert scaling_params.size == 1
        return scaling_params

    def _objective_function(self, param, x, y, splits):
        log_lmbda = param[0]
        lmbda = np.exp(log_lmbda)
        log_gamma = self._get_scaling(param[1:])
        gamma = np.exp(log_gamma)

        if self._verbose > 0:
            print(f'lmbda = {lmbda:14.8f}, ', end='')
            print(f'gamma = {gamma[0]:14.8f}, ', end='')

        err = 0.0
        derr = np.zeros(2)
        for idx_tr, idx_val in splits:
            prob = problem.Classification(
                gamma * x[idx_tr], y[idx_tr],
                dict(self._params, lmbda=lmbda, gamma=1.0),
            )
            res = solve_twostep(prob)
            z = a, b, c = res['a'], res['b'], res['c']
            # extract solution
            al = a / lmbda
            idx_sv = a != 0.0
            al_sv = al[idx_sv]

            # scale inputs
            xs_val = gamma[:, None] * x[idx_val]

            # compute validation estimates and error
            k_val_sv, dk_val_sv = prob.kernel(
                x=xs_val, idx0=idx_sv, deriv=True
            )
            ka_val = k_val_sv.T.dot(al_sv)
            y_val_pred = ka_val + b
            l_val, dl_val = self._validation_loss(y_val_pred, y[idx_val])
            err += l_val

            # compute decision function values
            k_tr_sv, dk_tr_sv = prob.kernel(idx=idx_sv, deriv=True)
            ka = k_tr_sv.dot(al_sv)
            dec = prob.decision(z, ka)
            h = prob.ddloss(dec)

            # extract relevant part
            idx_p = h > 0
            idx_p_in_sv = h[idx_sv] > 0
            h_p = h[idx_p]
            if h_p.size == 0:
                return np.inf, np.zeros_like(derr)
            k_p_sv = k_tr_sv[idx_p, :]
            k_p_p = k_p_sv[:, idx_p_in_sv]
            k_val_p = k_val_sv.T[:, idx_p_in_sv]

            if (idx_sv & (~idx_p)).any():
                k_p_p = prob.kernel(idx=idx_p, idx0=idx_p)
                k_val_p = prob.kernel(x=xs_val, idx0=idx_p).T

            # solve transposed system
            lhs_p = dl_val.dot(k_val_p) / lmbda
            lhs_1 = dl_val.sum()
            if prob.has_asum:
                hu_p, _ = rendered.solve(
                    k_p_p / lmbda + np.diag(1.0 / h_p),
                    (np.ones_like(lhs_p), prob.sign[idx_p]),
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
            rhs_p = -dk_tr_sv[idx_p].dot(al_sv)
            derr[1] += hu_p.dot(rhs_p) + dl_val.dot(dk_val_sv.T.dot(al_sv))

        if self._verbose > 0:
            print(f'err = {err:14.8f}, derr = {derr}')
        return err, derr

    def optimize_params(self, x, y, splits):
        p0 = np.log([self._params['lmbda'], self._params['gamma']])

        res = minimize(
            partial(self._objective_function, x=x, y=y, splits=splits),
            p0,
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
        self.optimize_params(x, y, splits)
        self._svm = SimpleSVM(self._params)
        self._svm.fit(x, y)

    def predict(self, x):
        return self._svm.predict(x)


def main():
    n = 1000
    x, y = data(n, nft=10)
    svm = OptimizedSVM(dict(
        lmbda=0.1,
        gamma=0.5,
        smoothingParam=0.1,
        shift=1.0,
        maxfunDegree='6',
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


if __name__ == '__main__':
    main()

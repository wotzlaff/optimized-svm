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


class TrainingMixin:
    def _create_problem(self, x, y, params):
        if params['kind'] == 'classification':
            return problem.Classification(x, y, params)
        elif params['kind'] == 'regression':
            return problem.Regression(x, y, params)


class KernelMixin:
    def __init__(self, kernel_type='rbf', blocks=None):
        self._kernel_type = kernel_type
        self._blocks = blocks

    def scale(self, x, gamma):
        return gamma * x


class SimpleSVM(BaseEstimator, ClassifierMixin, TrainingMixin, KernelMixin):
    def __init__(self, params):
        self._params = params

    def fit(self, x, y):
        xs = self.scale(x, self._params['gamma'])
        self._prob = self._create_problem(
            xs, y,
            dict(self._params, gamma=1.0),
        )
        res = solve_twostep(self._prob)
        a, b = res['a'], res['b']
        self._b = b
        idx_sv = a != 0.0
        self._idx_sv = idx_sv
        self._asv = a[idx_sv]

    def predict(self, x):
        lmbda = self._params['lmbda']
        xs = self.scale(x, self._params['gamma'])
        ka = self._prob.kernel(x=xs, idx0=self._idx_sv).T.dot(self._asv)
        return ka / lmbda + self._b


class ValidationMixin:
    def __init__(self, params=dict()):
        smoothing = params.get('smoothingParam', 0.5)
        degree = params.get('smoothingDegree', '6')
        self._m, self._dm = get_maxfun(smoothing, degree)[:2]
        self._shift = params.get('shift', 1.0)
        kind = params.get('kind', 'classification')
        if kind == 'classification':
            self._validation_loss = self._validation_loss_classification
        elif kind == 'regression':
            self._validation_loss = self._validation_loss_regression
        else:
            raise ValueError(f'unknown validation kind "{kind}"')

    def _validation_loss_classification(self, t, y):
        shift = self._shift
        ell = self._m(shift - t * y).sum()
        dell = -y * self._dm(shift - t * y)
        return ell, dell

    def _validation_loss_regression(self, t, y):
        ell = self._m(t - y).sum() + self._m(y - t).sum()
        dell = self._dm(t - y) - self._dm(y - t)
        return ell, dell


class ParameterOptimizationBase(ValidationMixin, TrainingMixin, KernelMixin):

    def __init__(self, params, validation_params=dict(), verbose=0):
        ValidationMixin.__init__(self, validation_params)
        self._params = params
        self._verbose = verbose

    def _objective_function(self, param, x, y, splits):
        log_lmbda = param[0]
        lmbda = np.exp(log_lmbda)
        log_gamma = param[1:]
        gamma = np.exp(log_gamma)

        if self._verbose > 0:
            print(f'lmbda = {lmbda:14.8f}, ', end='')
            print(f'gamma = {gamma[0]:14.8f}, ', end='')

        err = 0.0
        derr = np.zeros_like(param)
        for idx_tr, idx_val in splits:
            prob = self._create_problem(
                self.scale(x[idx_tr], gamma), y[idx_tr],
                dict(self._params, lmbda=lmbda, gamma=1.0),
            )
            res = solve_twostep(prob)
            z = a, b, c = res['a'], res['b'], res['c']
            # extract solution
            al = a / lmbda
            idx_sv = a != 0.0
            al_sv = al[idx_sv]

            # scale inputs
            xs_val = self.scale(x[idx_val], gamma)

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

            if k_p_p.shape[0] != k_p_p.shape[1]:
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
            derr[1:] += hu_p.dot(rhs_p) + dl_val.dot(dk_val_sv.T.dot(al_sv))

        if self._verbose > 0:
            print(f'err = {err:14.8f}, derr = {derr}')
        return err, derr

    def _optimize_params(self, x, y, splits):
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
        self._optimize_params(x, y, splits)
        self._svm = SimpleSVM(self._params)
        self._svm.fit(x, y)

    def predict(self, x):
        return self._svm.predict(x)


class OptimizedSVM(BaseEstimator, ClassifierMixin, ParameterOptimizationBase):
    def __init__(self, params, validation_params=dict(), verbose=0):
        ParameterOptimizationBase.__init__(
            self,
            params=params,
            validation_params=validation_params,
            verbose=verbose,
        )


def main():
    classification = True
    n = 10000
    x, y = data(n, nft=10, classification=classification)
    params = dict(
        lmbda=1.0,
        gamma=1.0,
        smoothingParam=0.02,
        epsilon=1e-6,
        shift=1.0,
        maxfunDegree='6',
        tolSmo=1e-1,
        tolNewton=1e-10,
        # verbose=10000,
        # maxAsum=0.01 * n * 0.8,
        kind='classification' if classification else 'regression',
    )
    svm0 = SimpleSVM(params)
    svm0.fit(x[:n//2], y[:n//2])
    yest0 = svm0.predict(x)

    svm = OptimizedSVM(
        params=params,
        validation_params=dict(
            kind='classification' if classification else 'regression',
            # kind='regression',
            shift=0.0,
        ),
    )
    svm.fit(x[:n//2], y[:n//2])
    yest = svm.predict(x)

    import matplotlib.pyplot as plt
    if svm._params['kind'] == 'classification':
        plt.plot(y[:n//2], yest[:n//2], 'kx')
        plt.plot(y[n//2:] * 0.8, yest0[n//2:], 'r+')
        mc0 = (y[n//2:] * yest0[n//2:] <= 0.0).mean()
        plt.plot(y[n//2:] * 0.9, yest[n//2:], 'g+')
        mc = (y[n//2:] * yest[n//2:] <= 0.0).mean()
        print(f'{mc0} -> {mc}')
        plt.axhline(1.0)
        plt.axhline(0.0)
        plt.axhline(-1.0)
    elif svm._params['kind'] == 'regression':
        plt.plot(y[:n//2], yest[:n//2], 'kx')
        plt.plot(y[n//2:], yest0[n//2:], 'r+')
        plt.plot(y[n//2:], yest[n//2:], 'g+')
        vmin = min(y.min(),  yest.min())
        vmax = max(y.max(),  yest.max())
        plt.plot([vmin, vmax], [vmin, vmax], '--')
    plt.show()


if __name__ == '__main__':
    main()

from functools import partial

import numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import KFold

from .newton import solve_twostep
from .sample import data
from .rendered import solve as solve_rendered
from .problem import Classification, Regression
from .smooth_max import get_maxfun
from .kernel import rbf_kernel
from .deriv_check import check_grad


class TrainingMixin:
    def _create_problem(self, x, y, params, kernel_fun):
        if params['kind'] == 'classification':
            return Classification(x, y, params, kernel_fun)
        elif params['kind'] == 'regression':
            return Regression(x, y, params, kernel_fun)


class FeatureScalingMixin:
    def __init__(self, blocks=None):
        self._blocks = blocks

    def scale(self, x, gamma):
        blocks = self._blocks
        if blocks is None:
            return gamma * x
        else:
            xs = x.copy()
            assert not isinstance(gamma, float) or len(blocks) == 1
            assert isinstance(gamma, float) or len(gamma) == len(blocks)
            for gamma_k, block_k in zip(gamma, blocks):
                xs[:, block_k] *= gamma_k
            return xs


class SimpleSVM(
    BaseEstimator, ClassifierMixin, TrainingMixin, FeatureScalingMixin,
):
    def __init__(self, params, blocks=None, kernel_fun=None, verbose=0):
        FeatureScalingMixin.__init__(self, blocks)
        self._params = params
        self._kernel_fun = kernel_fun
        self._verbose = verbose

    def fit(self, x, y):
        xs = self.scale(x, self._params['scaling'])
        self._prob = self._create_problem(
            xs, y,
            self._params,
            self._kernel_fun,
        )
        res = solve_twostep(self._prob)
        a, b = res['a'], res['b']
        self._b = b
        idx_sv = a != 0.0
        self._idx_sv = idx_sv
        self._asv = a[idx_sv]
        if self._verbose > 0:
            print(f'Got {idx_sv.sum()} of {len(idx_sv)} SVs', end='')
            print(f' = {100 * idx_sv.sum() / len(idx_sv):.1f}%')

    def predict(self, x):
        lmbda = self._params['lmbda']
        xs = self.scale(x, self._params['scaling'])
        ka = self._prob.k(x=xs, idx0=self._idx_sv).T.dot(self._asv)
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


class OptimizedSVM(
    BaseEstimator, ClassifierMixin,
    ValidationMixin, TrainingMixin, FeatureScalingMixin,
):
    def __init__(
        self,
        params,
        validation_params=dict(),
        blocks=None,
        splits=None,
        kernel_fun=None,
        verbose=0,
        debug=False,
    ):
        FeatureScalingMixin.__init__(self, blocks)
        ValidationMixin.__init__(self, validation_params)
        self._splits = splits
        self._params = params
        self._verbose = verbose
        self._kernel_fun = kernel_fun
        self._debug = debug

    def _objective_function(self, param, x, y, splits):
        log_lmbda = param[0]
        lmbda = np.exp(log_lmbda)
        log_scaling = param[1:]
        scaling = np.exp(log_scaling)

        if self._verbose > 0:
            print(f'lmbda = {lmbda:14.8f}, ', end='')
            print(f'scaling = {scaling[0]:14.8f}, ', end='')

        err = 0.0
        derr = np.zeros_like(param)
        for idx_tr, idx_val in splits:
            prob = self._create_problem(
                self.scale(x[idx_tr], scaling), y[idx_tr],
                dict(self._params, lmbda=lmbda, kernel_param=dict(
                    self._params['kernel_param'],
                    blocks=self._blocks,
                )),
                self._kernel_fun,
            )
            res = solve_twostep(prob)
            z = a, b, c = res['a'], res['b'], res['c']
            # extract solution
            al = a / lmbda
            idx_sv = a != 0.0
            al_sv = al[idx_sv]

            # scale inputs
            xs_val = self.scale(x[idx_val], scaling)

            # compute validation estimates and error
            k_val_sv, dk_val_sv = prob.k(
                x=xs_val, idx0=idx_sv, deriv=True
            )
            ka_val = k_val_sv.T.dot(al_sv)
            y_val_pred = ka_val + b
            l_val, dl_val = self._validation_loss(y_val_pred, y[idx_val])
            err += l_val

            # compute decision function values
            k_tr_sv, dk_tr_sv = prob.k(idx=idx_sv, deriv=True)
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
                k_p_p = prob.k(idx=idx_p, idx0=idx_p)
                k_val_p = prob.k(x=xs_val, idx0=idx_p).T

            # solve transposed system
            lhs_p = dl_val.dot(k_val_p) / lmbda
            lhs_1 = dl_val.sum()
            if prob.has_asum:
                hu_p, _ = solve_rendered(
                    k_p_p / lmbda + np.diag(1.0 / h_p),
                    (np.ones_like(lhs_p), prob.sign[idx_p]),
                    lhs_p, (lhs_1, 0.0),
                )
            else:
                hu_p, _ = solve_rendered(
                    k_p_p / lmbda + np.diag(1.0 / h_p),
                    np.ones_like(lhs_p),
                    lhs_p, lhs_1,
                )

            # dlmbda:
            rhs_p = ka[idx_p]
            derr[0] += hu_p.dot(rhs_p) - dl_val.dot(ka_val)

            # dscaling:
            rhs_p = -np.tensordot(dk_tr_sv[idx_p], al_sv, axes=(1, 0))
            derr[1:] += hu_p.dot(rhs_p) + dk_val_sv.T.dot(al_sv).dot(dl_val)

        if self._verbose > 0:
            print(f'err = {err:14.8f}, derr = {derr}')
        return err, derr

    def _optimize_params(self, x, y, splits):
        scaling = self._params['scaling']
        n_blocks = 1 if self._blocks is None else len(self._blocks)
        if isinstance(scaling, float):
            print(f'extending scaling parameter to {n_blocks} dims')
            scaling = [scaling] * n_blocks
        p0 = np.log(np.concatenate((
            [self._params['lmbda']],
            scaling,
        )))

        if self._debug:
            check_grad(
                partial(self._objective_function, x=x, y=y, splits=splits),
                p0,
            )

        res = minimize(
            partial(self._objective_function, x=x, y=y, splits=splits),
            p0,
            method='L-BFGS-B',
            options=dict(disp=True),
            jac=True,
            bounds=[(-10, 5)] + [(-10, 5)] * n_blocks,
        )
        p_opt = res.x
        self._params.update(dict(
            lmbda=np.exp(p_opt[0]),
            scaling=np.exp(p_opt[1:]),
        ))

    def fit(self, x, y):
        if self._splits is None:
            splits = list(KFold(5).split(x))
        else:
            splits = self._splits
        self._optimize_params(x, y, splits)
        self._svm = SimpleSVM(
            self._params,
            blocks=self._blocks,
            kernel_fun=self._kernel_fun,
            verbose=self._verbose,
        )
        print(self._svm._params)
        self._svm.fit(x, y)

    def predict(self, x):
        return self._svm.predict(x)


def main():
    verbose = 1
    classification = False
    n = 1000
    x, y = data(n, nft=2, classification=classification)
    # x[:, 0] *= 10
    params = dict(
        lmbda=1.0,
        scaling=1.0,
        kernel_param=dict(gamma=1.0),
        smoothingParam=0.02,
        epsilon=1e-6,
        shift=1.0,
        maxfunDegree='2',
        tolSmo=1e-1,
        tolNewton=1e-12,
        # verbose=10000,
        # maxAsum=0.01 * n * 0.8,
        kind='classification' if classification else 'regression',
    )
    svm0 = SimpleSVM(params, kernel_fun=rbf_kernel, verbose=verbose)
    svm0.fit(x[:n//2], y[:n//2])
    yest0 = svm0.predict(x)

    svm = OptimizedSVM(
        params=params,
        validation_params=dict(
            kind='classification' if classification else 'regression',
            # kind='regression',
            shift=1.0,
        ),
        kernel_fun=rbf_kernel,
        # blocks=[[0], [1]],
        # debug=True,
        verbose=verbose,
    )
    svm.fit(x[:n//2], y[:n//2])
    yest = svm.predict(x)

    import matplotlib.pyplot as plt
    if svm._params['kind'] == 'classification':
        plt.plot(
            y[:n//2] * 0.9, y[:n//2] * yest[:n//2],
            'kx', label='after(train)',
        )
        plt.plot(
            y[n//2:], y[n//2:] * yest[n//2:],
            'g+', label='after(val)',
        )
        plt.plot(
            y[:n//2] * 0.7, y[:n//2] * yest0[:n//2],
            'bx', label='intial(train)',
        )
        plt.plot(
            y[n//2:] * 0.8, y[n//2:] * yest0[n//2:],
            'r+', label='initial(val)',
        )
        plt.legend()

        mc0 = (y[n//2:] * yest0[n//2:] <= 0.0).mean()
        mc = (y[n//2:] * yest[n//2:] <= 0.0).mean()
        print(f'MC: {mc0} -> {mc}')

        plt.axhline(1.0)
        plt.axhline(0.0)
    elif svm._params['kind'] == 'regression':
        plt.plot(y[:n//2], yest0[:n//2], 'bo', label='intial(train)')
        plt.plot(y[n//2:], yest0[n//2:], 'ro', label='initial(val)')
        plt.plot(y[:n//2], yest[:n//2], 'kx', label='after(train)')
        plt.plot(y[n//2:], yest[n//2:], 'g+', label='after(val)')
        vmin = min(y.min(),  yest.min())
        vmax = max(y.max(),  yest.max())
        plt.plot([vmin, vmax], [vmin, vmax], '--')
        plt.legend()

        mae0 = np.abs(y[n//2:] - yest0[n//2:]).mean()
        mae = np.abs(y[n//2:] - yest[n//2:]).mean()
        print(f'MAE: {mae0} -> {mae}')
    plt.show()


if __name__ == '__main__':
    main()

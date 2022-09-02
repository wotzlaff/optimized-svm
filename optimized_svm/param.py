from functools import partial
import time
from tqdm import tqdm

import numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import KFold

from .newton import solve_twostep
from .rendered import solve as solve_rendered
from .problem import Classification, Regression
from .smooth_max import get_maxfun
from .deriv_check import check_grad


class TrainingMixin:
    def _create_problem(
        self,
        x, y,
        kind, params,
        kernel_fun, kernel_param
    ):
        Problem = dict(
            classification=Classification,
            regression=Regression,
        )[kind]
        if 'ratioAsum' in params:
            assert 'maxAsum' not in params
            params['maxAsum'] = len(y) * params['ratioAsum']
            del params['ratioAsum']
        return Problem(
            x, y,
            kernel_fun, kernel_param,
            **params,
        )


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
    BaseEstimator, ClassifierMixin,
    TrainingMixin, FeatureScalingMixin,
):
    def __init__(
        self,
        kind,
        params,
        scaling,
        kernel_fun,
        kernel_param,
        blocks=None,
        params_smo=dict(),
        params_newton=dict(),
        verbose=0,
    ):
        FeatureScalingMixin.__init__(self, blocks)
        self.__kind = kind
        self.__params = params
        self.__scaling = scaling
        self.__kernel_fun = kernel_fun
        self.__kernel_param = kernel_param
        self.__params_smo = params_smo
        self.__params_newton = params_newton
        self.__verbose = verbose

    def fit(self, x, y):
        xs = self.scale(x, self.__scaling)
        self._prob = self._create_problem(
            xs, y,
            self.__kind,
            self.__params,
            self.__kernel_fun,
            self.__kernel_param,
        )
        res = solve_twostep(
            self._prob,
            self.__params_smo,
            self.__params_newton,
        )
        a, b = res['a'], res['b']
        self._b = b
        idx_sv = a != 0.0
        self._idx_sv = idx_sv
        self._asv = a[idx_sv]
        self._xsv = self._prob._x[idx_sv]
        if self.__verbose > 0:
            print(f'Got {idx_sv.sum()} of {len(idx_sv)} SVs', end='')
            print(f' = {100 * idx_sv.sum() / len(idx_sv):.1f}%')

    def set_solution(self, x, a, b):
        xs = self.scale(x, self.__scaling)
        self._b = b
        idx_sv = a != 0.0
        self._idx_sv = idx_sv
        self._asv = a[idx_sv]
        self._xsv = xs[idx_sv]

    def predict(self, x):
        lmbda = self.__params['lmbda']
        xs = self.scale(x, self.__scaling)
        ka = self.__kernel_fun(
            xs, self._xsv,
            self.__kernel_param,
        ).dot(self._asv)
        return ka / lmbda + self._b


class ValidationMixin:
    def __init__(self, params=dict()):
        self.validation_params = params

    @property
    def validation_params(self):
        return self._validation_params

    @validation_params.setter
    def validation_params(self, params):
        self._validation_params = params
        smoothing = params.get('smoothingParam', 0.5)
        degree = params.get('smoothingDegree', '6')
        self._m, self._dm = get_maxfun(smoothing, degree)[:2]
        self._shift = params.get('shift', 1.0)
        kind = params.get('kind', 'classification')
        self._validation_kind = kind
        self._validation_loss = dict(
            classification=self._validation_loss_classification,
            regression=self._validation_loss_regression,
        )[kind]

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
        kind,
        params,
        kernel_fun,
        kernel_param,
        initial_params,
        validation_params=dict(),
        params_smo=dict(),
        params_newton=dict(),
        splits=None,
        verbose=0,
        debug=False,
    ):
        FeatureScalingMixin.__init__(self, kernel_param.get('blocks', None))
        ValidationMixin.__init__(self, validation_params)
        self.splits = splits
        self.__kind = kind
        self.__params = params
        self.__initial_params = initial_params
        self.verbose = verbose
        self.__kernel_fun = kernel_fun
        self.__kernel_param = kernel_param
        self.__params_smo = params_smo
        self.__params_newton = params_newton
        self.debug = debug

        self.__last_fit = None
        self.__last_value = None
        self.__time_start = None
        self.__funevals = 0

        self.__svm = None
        self.__no_nan = None

    def _objective_function(self, param, x, y, splits):
        self.__funevals += 1
        log_lmbda = param[0]
        lmbda = np.exp(log_lmbda)
        log_scaling = param[1:]
        scaling = np.exp(log_scaling)

        xs = self.scale(x, scaling)

        if self.verbose > 1:
            print('trial: ', end='')
            print(f'lmbda = {lmbda:14.8f}, ', end='')
            print(f'scaling = {np.mean(scaling):14.8f}, ', end='')

        a_avg = np.zeros(x.shape[0])
        b_avg = 0.0

        nerr = 0.0
        err = 0.0
        derr = np.zeros_like(param)
        it = enumerate(tqdm(splits) if self.verbose == 1 else splits)
        for _, (idx_tr, idx_val) in it:
            if self.verbose > 2:
                print(f'\n## {_+1} of {len(splits)}')
            if self.verbose > 3:
                print(f'  |T| = {len(y[idx_tr])}, |V| = {len(y[idx_val])}')
            prob = self._create_problem(
                xs[idx_tr], y[idx_tr],
                kind=self.__kind,
                params=dict(self.__params, lmbda=lmbda),
                kernel_fun=self.__kernel_fun,
                kernel_param=self.__kernel_param,
            )
            res = solve_twostep(prob, self.__params_smo, self.__params_newton)
            z = a, b, c = res['a'], res['b'], res['c']
            # extract solution
            al = a / lmbda
            idx_sv = a != 0.0
            al_sv = al[idx_sv]

            # capture avg
            a_avg[idx_tr] += prob.extract_original_a(a) / len(splits)
            b_avg += b / len(splits)

            # scale inputs
            xs_val = xs[idx_val]

            # compute validation estimates and error
            k_val_sv, dk_val_sv = prob.k(
                x=xs_val, idx0=idx_sv, deriv=True
            )
            ka_val = k_val_sv.T.dot(al_sv)
            y_val_pred = ka_val + b
            l_val, dl_val = self._validation_loss(y_val_pred, y[idx_val])
            err += l_val
            if self._validation_loss == self._validation_loss_classification:
                nerr += ((y_val_pred * y[idx_val]) < 0.0).mean()
            else:
                err_std = np.abs(y[idx_val] - np.median(y[idx_val])).sum()
                nerr += np.abs(y[idx_val] - y_val_pred).sum() / err_std

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

        if self._validation_loss == self._validation_loss_classification:
            n_val = sum(len(y[idx_val]) for _, idx_val in splits)
            err /= n_val
            derr /= n_val
        else:
            err_val = sum(
                np.sum(np.abs(y[idx_val] - np.median(y[idx_val])))
                for _, idx_val in splits
            )
            err /= err_val
            derr /= err_val

        self.__last_fit = SimpleSVM(
            kind=self.__kind,
            params=dict(self.__params, lmbda=lmbda),
            scaling=scaling,
            kernel_fun=self.__kernel_fun,
            kernel_param=self.__kernel_param,
            blocks=self._blocks,
            params_smo=self.__params_smo,
            params_newton=self.__params_newton,
        )
        self.__last_fit.set_solution(x, a_avg, b_avg)

        if self.verbose > 1:
            print(', '.join([
                f'err = {err:14.8f}',
                f'nerr = {nerr / len(splits):10.6f}',
                f'derr = {derr}',
            ]))
        self.__last_value = err, derr
        return err, derr

    def _optimize_params(self, x, y, splits):
        self.__time_start = time.time()
        scaling = self.__initial_params['scaling']
        nblocks = 1 if self._blocks is None else len(self._blocks)
        if isinstance(scaling, float):
            print(f'extending scaling parameter to {nblocks} dims')
            scaling = [scaling] * nblocks
        p0 = np.log(np.concatenate((
            [self.__initial_params['lmbda']],
            scaling,
        )))

        if self.debug:
            check_grad(
                partial(self._objective_function, x=x, y=y, splits=splits),
                p0,
            )
            return

        def callback(param):
            ypred = self.__last_fit.predict(x)
            if self._validation_kind == 'regression':
                ymed = np.median(y)
                err_tr = np.abs(y - ypred).sum() / np.abs(y - ymed).sum()
                err_val = self.__last_value[0]
            if self._validation_kind == 'classification':
                err_tr = (y * ypred <= 0.0).mean()
                err_val = self.__last_value[0]

            print(' / '.join([
                f'{self.__funevals} fevals',
                f'time = {time.time() - self.__time_start:.1f}s',
                f'lmbda = {np.exp(param[0])}',
                f'scaling = {np.exp(param[1:]).mean()}',
                f'nerr_tr = {err_tr:.8f}',
                f'nerr_val = {err_val:.8f}',
                f'ratio = {err_tr / err_val:.3f}'
            ]))

        res = minimize(
            partial(self._objective_function, x=x, y=y, splits=splits),
            p0,
            method='L-BFGS-B',
            options=dict(disp=True),
            jac=True,
            bounds=[(-10, 5)] + [(-10, 5)] * nblocks,
            callback=callback if self.verbose > 0 else None,
        )
        p_opt = res.x
        return dict(
            lmbda=np.exp(p_opt[0]),
            scaling=np.exp(p_opt[1:]),
        )

    def fit(self, x, y):
        x = np.array(x).astype(np.double)
        self.__no_nan = ~np.isnan(x).any(axis=0)
        x = x[:, self.__no_nan]
        y = np.array(y).astype(np.double)
        if self.splits is None:
            splits = list(KFold(5).split(x))
        else:
            splits = self.splits
        optimized_params = self._optimize_params(x, y, splits)
        self.__svm = SimpleSVM(
            kind=self.__kind,
            params=dict(self.__params, lmbda=optimized_params['lmbda']),
            scaling=optimized_params['scaling'],
            kernel_fun=self.__kernel_fun,
            kernel_param=self.__kernel_param,
            blocks=self._blocks,
            params_smo=self.__params_smo,
            params_newton=self.__params_newton,
        )
        print(optimized_params)
        self.__svm.fit(x, y)

    def predict(self, x):
        x = np.array(x).astype(np.double)
        if self.__no_nan is not None:
            x = x[:, self.__no_nan]
        if self.__svm is None:
            return np.zeros(x.shape[0])
        return self.__svm.predict(x)

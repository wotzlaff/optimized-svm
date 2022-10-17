from functools import partial
import time
from tqdm import tqdm

import numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import KFold

from optimized_svm.kernel import rbf_kernel

from .newton import solve_twostep
from .rendered import solve as solve_rendered
from .problem import TrainingMixin
from .deriv_check import check_grad
from .simple import SimpleSVM
from .scaling import FeatureScalingMixin
from .validation import ValidationMixin


class OptimizedSVM(
    BaseEstimator, ValidationMixin, TrainingMixin, FeatureScalingMixin,
):
    default_scaling0 = 'auto'
    default_lmbda0 = 1.0

    def __init__(
        self,
        kind,
        params,
        kernel_fun=rbf_kernel,
        kernel_param=dict(gamma=1.0),
        params_hpo=dict(),
        params_smo=dict(),
        params_newton=dict(),
        splits=None,
        verbose=0,
        debug=False,
    ):
        FeatureScalingMixin.__init__(self, kernel_param.get('blocks', None))
        ValidationMixin.__init__(self, params_hpo)
        self.splits = splits
        self.kind = kind
        self.params = params
        self.verbose = verbose
        self.kernel_fun = kernel_fun
        self.kernel_param = kernel_param
        self.params_hpo = params_hpo
        self.params_smo = params_smo
        self.params_newton = params_newton
        self.debug = debug

        self.__last_fit = None
        self.__last_value = None
        self.__time_start = None
        self.__funevals = 0

        self.__svm = None
        self.__no_nan = None

    def _objective_function_part(self, x_tr, y_tr, x_val, y_val, lmbda):
        status = dict()
        prob = self._create_problem(
            x_tr, y_tr,
            kind=self.kind,
            params=dict(self.params, lmbda=lmbda),
            kernel_fun=self.kernel_fun,
            kernel_param=self.kernel_param,
        )
        res = solve_twostep(prob, self.params_smo, self.params_newton)

        # extract status
        t = status['t'] = res['t_smo'] + res['t_refine']
        r_smo = status['r_smo'] = res['t_smo'] / t
        if self.verbose > 2:
            print(f'{t:.1f}s (SMO: {100*r_smo:.1f}%)')
        z = a, b, c = res['a'], res['b'], res['c']
        status['n_sv'] = (a != 0.0).sum()
        status['r_sv'] = (a != 0.0).mean()

        # extract solution
        al = a / lmbda
        idx_sv = a != 0.0
        al_sv = al[idx_sv]
        solution = dict(a=prob.extract_original_a(a), b=b, c=c)

        # compute validation estimates and error
        k_val_sv, dk_val_sv = prob.k(
            x=x_val, idx0=idx_sv, deriv=True,
        )
        n_param = 1 + dk_val_sv.shape[-1]
        ka_val = k_val_sv.T.dot(al_sv)
        y_val_pred = ka_val + b
        l_val, dl_val = self._validation_loss(y_val_pred, y_val)
        err = l_val
        if self.params_hpo['kind'] == 'classification':
            nerr = ((y_val_pred * y_val) < 0.0).mean()
        elif self.params_hpo['kind'] == 'balanced_classification':
            w = self._balanced_weights(y_val)
            nerr = (w * ((y_val_pred * y_val) < 0.0)).mean()
        elif self.params_hpo['kind'] == 'regression':
            err_std = np.abs(y_val - np.median(y_val)).sum()
            nerr = np.abs(y_val - y_val_pred).sum() / err_std
        else:
            raise ValueError('unknown validation loss')

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
            return np.inf, np.zeros(n_param)
        k_p_sv = k_tr_sv[idx_p, :]
        k_p_p = k_p_sv[:, idx_p_in_sv]
        k_val_p = k_val_sv.T[:, idx_p_in_sv]

        if k_p_p.shape[0] != k_p_p.shape[1]:
            k_p_p = prob.k(idx=idx_p, idx0=idx_p)
            k_val_p = prob.k(x=x_val, idx0=idx_p).T

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

        # compute error derivative
        derr = np.zeros(n_param)
        # dlmbda:
        rhs_p = ka[idx_p]
        derr[0] = hu_p.dot(rhs_p) - dl_val.dot(ka_val)
        # dscaling:
        rhs_p = -np.tensordot(dk_tr_sv[idx_p], al_sv, axes=(1, 0))
        derr[1:] = hu_p.dot(rhs_p) + dk_val_sv.T.dot(al_sv).dot(dl_val)

        return status, solution, err, derr, nerr

    def _objective_function(self, param, x, y, splits):
        self.__funevals += 1
        log_lmbda = param[0]
        lmbda = np.exp(log_lmbda)
        log_scaling = param[1:]
        scaling = np.exp(log_scaling)

        xs = self.scale(x, scaling)

        if self.verbose > 1:
            print(f'lmbda = {lmbda:g}, ', end='')
            print(f'scaling = {np.mean(scaling):g}, ', end='')

        a_avg = np.zeros(x.shape[0])
        b_avg = 0.0

        nerr = 0.0
        err = 0.0
        derr = np.zeros_like(param)
        it = enumerate(tqdm(splits) if self.verbose == 1 else splits)
        status = dict(r_smo=[], t=[], n_sv=[], r_sv=[])
        for _, (idx_tr, idx_val) in it:
            if self.verbose > 2:
                print(f'## {_+1} of {len(splits)}')
            if self.verbose > 3:
                print(f'  |T| = {len(y[idx_tr])}, |V| = {len(y[idx_val])}')
            status_t, sol_t, err_t, derr_t, nerr_t = self._objective_function_part(
                xs[idx_tr], y[idx_tr],
                xs[idx_val], y[idx_val],
                lmbda,
            )
            for key, val in status_t.items():
                status[key].append(val)
            a_avg[idx_tr] += sol_t['a']
            b_avg += sol_t['b']
            err += err_t
            derr += derr_t
            nerr += nerr_t
        
        a_avg /= len(splits)
        b_avg /= len(splits)

        t = sum(status['t'])
        r_smo = np.exp(np.log(status['r_smo']).mean())
        n_sv = np.mean(status['n_sv'])
        r_sv = np.exp(np.log(status['r_sv']).mean())

        if self.params_hpo['kind'] in [
            'classification',
            'balanced_classification',
        ]:
            n_val = sum(len(y[idx_val]) for _, idx_val in splits)
            err /= n_val
            derr /= n_val
        elif self.params_hpo['kind'] == 'regression':
            err_val = sum(
                np.sum(np.abs(y[idx_val] - np.median(y[idx_val])))
                for _, idx_val in splits
            )
            err /= err_val
            derr /= err_val
        else:
            raise ValueError('unknown validation loss')
        nerr /= len(splits)

        self.__last_fit = SimpleSVM(
            kind=self.kind,
            params=dict(self.params, lmbda=lmbda),
            scaling=scaling,
            kernel_fun=self.kernel_fun,
            kernel_param=self.kernel_param,
            blocks=self.blocks,
            params_smo=self.params_smo,
            params_newton=self.params_newton,
        )
        if self.kind == 'classes_':
            self.__last_fit.classes_ = self.classes_
        self.__last_fit.set_solution(x, a_avg, b_avg)

        if self.verbose > 1:
            print(', '.join([
                f'err = {err:g}',
                f'nerr = {nerr:g}',
                f'derr = {derr}',
                f'time: {t:.1f}s',
                f'SMO: {100*r_smo:.1f}%',
                f'avg #SVs: {n_sv:.0f} = {100*r_sv:.1f}%',
            ]))
        self.__last_value = err, derr
        return err, derr

    def _optimize_params(self, x, y, splits):
        self.__time_start = time.time()
        scaling = self.params_hpo.get('scaling0', OptimizedSVM.default_scaling0)
        if scaling == 'auto':
            scaling = np.sqrt(np.log(2) / ((x - x[0]) ** 2).sum(axis=1).mean())
        nblocks = 1 if self.blocks is None else len(self.blocks)
        if isinstance(scaling, float):
            scaling = [scaling] * nblocks
        p0 = np.log(np.concatenate((
            [self.params_hpo.get('lmbda0', OptimizedSVM.default_lmbda0)],
            scaling,
        )))

        if self.debug:
            check_grad(
                partial(self._objective_function, x=x, y=y, splits=splits),
                p0,
            )
            return

        def callback(param):
            ypred = self.__last_fit.decision_function(x)
            if self.params_hpo['kind'] == 'classification':
                err_tr = (y * ypred <= 0.0).mean()
                err_val = self.__last_value[0]
            elif self.params_hpo['kind'] == 'balanced_classification':
                w = self._balanced_weights(y)
                err_tr = (w * (y * ypred <= 0.0)).mean()
                err_val = self.__last_value[0]
            elif self.params_hpo['kind'] == 'regression':
                ymed = np.median(y)
                err_tr = np.abs(y - ypred).sum() / np.abs(y - ymed).sum()
                err_val = self.__last_value[0]
            else:
                raise ValueError('unknown validation loss')

            print(' / '.join([
                f'{self.__funevals} fevals',
                f'time = {time.time() - self.__time_start:.1f}s',
                f'lmbda = {np.exp(param[0]):g}',
                f'scaling = {np.exp(param[1:]).mean():g}',
                f'nerr_tr = {err_tr:.6f}',
                f'nerr_val = {err_val:.6f}',
                f'ratio = {err_tr / err_val:.3f}'
            ]))

        res = minimize(
            partial(self._objective_function, x=x, y=y, splits=splits),
            p0,
            method='L-BFGS-B',
            options=dict(
                ftol=self.params_hpo.get('ftol', 1e-6),
                gtol=self.params_hpo.get('gtol', 1e-6),
                maxfun=self.params_hpo.get('maxfun', 15000),
                maxiter=self.params_hpo.get('maxiter', 15000),
                maxls=self.params_hpo.get('maxls', 20),
                iprint=self.params_hpo.get('iprint', -1),
            ),
            jac=True,
            bounds=[
                np.log(self.params_hpo.get('lmbda_bounds', [1e-6, 1.0]))
            ] + nblocks * [
                np.log(self.params_hpo.get('scaling_bounds', [1e-6, 1.0]))
            ],
            callback=callback if self.verbose > 2 else None,
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
        if self.kind in ['classification', 'balanced_classification']:
            self.classes_ = np.unique(y)
            assert len(self.classes_) == 2
            y = np.where(y == self.classes_[0], -1.0, +1.0)
        y = np.array(y).astype(np.double)
        if self.splits is None:
            splits = list(KFold(5).split(x))
        else:
            splits = self.splits

        if self.blocks == 'all':
            self.kernel_param['blocks'] = self.blocks = None
            optimized_params = self._optimize_params(x, y, splits)
            if self.verbose > 1:
                print('optimized parameters:', optimized_params)
            self.params_hpo['lmbda0'] = optimized_params['lmbda']
            self.params_hpo['scaling0'] = optimized_params['scaling'][0]
            self.blocks = [[ft] for ft in range(x.shape[1])]
            self.kernel_param['blocks'] = self.blocks
        optimized_params = self._optimize_params(x, y, splits)
        self.__svm = SimpleSVM(
            kind=self.kind,
            params=dict(self.params, lmbda=optimized_params['lmbda']),
            scaling=optimized_params['scaling'],
            kernel_fun=self.kernel_fun,
            kernel_param=self.kernel_param,
            blocks=self.blocks,
            params_smo=self.params_smo,
            params_newton=self.params_newton,
        )
        if self.verbose > 1:
            print('optimized parameters:', optimized_params)
        self.__svm.fit(x, y)

    def decision_function(self, x):
        x = np.array(x).astype(np.double)
        if self.__no_nan is not None:
            x = x[:, self.__no_nan]
        if self.__svm is None:
            return np.zeros(x.shape[0])
        return self.__svm.decision_function(x)

    def predict(self, x):
        d = self.decision_function(x)
        if self.kind in ['classification', 'balanced_classification']:
            if not hasattr(self, 'classes_'):
                raise NotFittedError()
            return np.where(d < 0.0, self.classes_[0], self.classes_[1])
        return d

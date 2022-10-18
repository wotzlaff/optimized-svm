from sklearn.base import BaseEstimator

from optimized_svm.normalization import NormalizationMixin
from .problem import TrainingMixin
from .scaling import FeatureScalingMixin
from .newton import solve_twostep


class SimpleSVM(
    BaseEstimator,
    NormalizationMixin,
    TrainingMixin,
    FeatureScalingMixin,
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
        self.kind = kind
        self.params = params
        self.scaling = scaling
        self.kernel_fun = kernel_fun
        self.kernel_param = kernel_param
        self.params_smo = params_smo
        self.params_newton = params_newton
        self.verbose = verbose

    def fit(self, x, y):
        y = self._normalize_labels(y)
        xs = self.scale(x, self.scaling)
        self._prob = self._create_problem(
            xs, y,
            self.kind,
            self.params,
            self.kernel_fun,
            self.kernel_param,
        )
        res = solve_twostep(
            self._prob,
            self.params_smo,
            self.params_newton,
        )
        a, b = res['a'], res['b']
        self._b = b
        idx_sv = a != 0.0
        self._idx_sv = idx_sv
        self._asv = a[idx_sv]
        self._xsv = self._prob._x[idx_sv]
        if self.verbose > 0:
            print(f'Got {idx_sv.sum()} of {len(idx_sv)} SVs', end='')
            print(f' = {100 * idx_sv.sum() / len(idx_sv):.1f}%')

    def set_solution(self, x, a, b):
        xs = self.scale(x, self.scaling)
        self._b = b
        idx_sv = a != 0.0
        self._idx_sv = idx_sv
        self._asv = a[idx_sv]
        self._xsv = xs[idx_sv]

    def decision_function(self, x):
        lmbda = self.params['lmbda']
        xs = self.scale(x, self.scaling)
        ka = self.kernel_fun(
            xs, self._xsv,
            self.kernel_param,
        ).dot(self._asv)
        return ka / lmbda + self._b

    def predict(self, x):
        d = self.decision_function(x)
        return self._denormalize_decisions(d)

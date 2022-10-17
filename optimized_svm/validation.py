from functools import partial
import numpy as np
from .smooth_max import get_maxfun


class ValidationMixin:
    def __init__(self, params=dict()):
        self._m, self._dm = get_maxfun(
            params.get('smoothingParam', 0.5),
            params.get('smoothingDegree', '6'),
        )[:2]
        self._validation_loss = partial(dict(
            classification=self._validation_loss_classification,
            balanced_classification=self._validation_loss_balanced_classification,
            regression=self._validation_loss_regression,
        )[params['kind']], params=params)

    def _validation_loss_classification(self, t, y, params):
        shift = params.get('shift', 1.0)
        ell = self._m(shift - t * y).sum()
        dell = -y * self._dm(shift - t * y)
        return ell, dell

    def _balanced_weights(self, y):
        ratio = (y == 1.0).mean()
        return 0.5 / np.where(y == 1.0, ratio, 1.0 - ratio)

    def _validation_loss_balanced_classification(self, t, y, params):
        w = self._balanced_weights(y)
        shift = params.get('shift', 1.0)
        ell = (w * self._m(shift - t * y)).sum()
        dell = -w * y * self._dm(shift - t * y)
        return ell, dell

    def _validation_loss_regression(self, t, y, params):
        ell = self._m(t - y).sum() + self._m(y - t).sum()
        dell = self._dm(t - y) - self._dm(y - t)
        return ell, dell


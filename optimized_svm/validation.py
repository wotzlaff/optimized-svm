import numpy as np
from .smooth_max import get_maxfun


def balanced_weights(y):
    ratio = (y == 1.0).mean()
    return 0.5 / np.where(y == 1.0, ratio, 1.0 - ratio)


class ValidationMixin:
    def __init__(self, params=dict()):
        m, dm = get_maxfun(
            params.get('smoothingParam', 0.5),
            params.get('smoothingDegree', '6'),
        )[:2]

        def _validation_loss_classification(t, y):
            shift = params.get('shift', 1.0)
            ell = self._m(shift - t * y).sum()
            dell = -y * self._dm(shift - t * y)
            return ell, dell

        def _validation_loss_balanced_classification(t, y):
            w = balanced_weights(y)
            shift = params.get('shift', 1.0)
            ell = (w * self._m(shift - t * y)).sum()
            dell = -w * y * self._dm(shift - t * y)
            return ell, dell

        def _validation_loss_regression(t, y):
            ell = self._m(t - y).sum() + self._m(y - t).sum()
            dell = self._dm(t - y) - self._dm(y - t)
            return ell, dell

        def _validation_loss_ls(t, y):
            e = t - y
            return 0.5 * (e ** 2), e

        self._validation_loss = dict(
            classification=_validation_loss_classification,
            balanced_classification=_validation_loss_balanced_classification,
            regression=_validation_loss_regression,
            ls=_validation_loss_ls,
        )[params['kind']]


import numpy as np
from scipy.special import factorial
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
            ell = m(shift - t * y).sum()
            dell = -y * dm(shift - t * y)
            return ell, dell

        def _validation_loss_balanced_classification(t, y):
            w = balanced_weights(y)
            shift = params.get('shift', 1.0)
            ell = (w * m(shift - t * y)).sum()
            dell = -w * y * dm(shift - t * y)
            return ell, dell

        def _validation_loss_regression(t, y):
            ell = m(t - y).sum() + m(y - t).sum()
            dell = dm(t - y) - dm(y - t)
            return ell, dell

        def _validation_loss_poisson(t, y):
            et = np.exp(t)
            ell = et.sum() - t.dot(y) + np.log(factorial(y)).sum()
            dell = et - y
            return ell, dell

        def _validation_loss_ls(t, y):
            e = t - y
            return 0.5 * (e ** 2).sum(), e

        self._validation_loss = dict(
            classification=_validation_loss_classification,
            balanced_classification=_validation_loss_balanced_classification,
            regression=_validation_loss_regression,
            ls=_validation_loss_ls,
            poisson=_validation_loss_poisson,
        )[params['kind']]


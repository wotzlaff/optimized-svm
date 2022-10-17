from .smooth_max import get_maxfun


class ValidationMixin:
    def __init__(self, params=dict()):
        self.params_validation = params

    @property
    def params_validation(self):
        return self._params_validation

    @params_validation.setter
    def params_validation(self, params):
        self._params_validation = params
        self._m, self._dm = get_maxfun(
            params.get('smoothingParam', 0.5),
            params.get('smoothingDegree', '6'),
        )[:2]
        self._validation_loss = dict(
            classification=self._validation_loss_classification,
            regression=self._validation_loss_regression,
        )[params.get('kind', 'classification')]

    def _validation_loss_classification(self, t, y):
        shift = self.params_validation.get('shift', 1.0)
        ell = self._m(shift - t * y).sum()
        dell = -y * self._dm(shift - t * y)
        return ell, dell

    def _validation_loss_regression(self, t, y):
        ell = self._m(t - y).sum() + self._m(y - t).sum()
        dell = self._dm(t - y) - self._dm(y - t)
        return ell, dell


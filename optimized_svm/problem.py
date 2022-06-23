import numpy as np
from smooth_max import get_maxfun
from kernel import rbf_kernel as kernel


class ParamBase:
    def __init__(self, params):
        self._params = params

    @property
    def params(self):
        return self._params

    def m(self, t):
        return get_maxfun(
            self._params['smoothingParam'],
            self._params['maxfunDegree'],
        )[0](t)

    def dm(self, t):
        return get_maxfun(
            self._params['smoothingParam'],
            self._params['maxfunDegree'],
        )[1](t)

    def ddm(self, t):
        return get_maxfun(
            self._params['smoothingParam'],
            self._params['maxfunDegree'],
        )[2](t)


class Classification(ParamBase):
    def __init__(self, x, y, params):
        super().__init__(params)
        self._x = x
        self._y = y

    @property
    def size(self):
        return self._x.shape[0]

    @property
    def sign(self):
        return self._y

    def kernel(self, x=None, idx=None, idx0=None):
        x0 = self._x if idx0 is None else self._x[idx0]
        if x is not None:
            return kernel(x0, x, self._params['gamma'])
        if idx is not None:
            return kernel(x0, self._x[idx], self._params['gamma'])

    def loss(self, t):
        return self.m(self._params['shift'] - self._y * t)

    def dloss(self, t):
        return -self._y * self.dm(self._params['shift'] - self._y * t)

    def ddloss(self, t):
        return self.ddm(self._params['shift'] - self._y * t)

    def decision_objective(self, z, ka):
        a, b, c = z
        obj = 0.5 * ka.dot(a)
        dec = ka + b + self.sign * c
        obj += self.loss(dec).sum()
        if ~np.isinf(self._params.get('maxAsum', float('inf'))):
            obj += self._params['maxAsum'] * c
        return dec, obj

    def objective(self, z, ka):
        return self.decision_objective(z, ka)[1]

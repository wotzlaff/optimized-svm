import numpy as np
from smooth_max import get_maxfun
from kernel import rbf_kernel as kernel, diff


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

    def decision(self, z, ka=None):
        a, b, c = z
        if ka is None:
            idx_sv = a != 0.0
            ka = self.kernel(idx=idx_sv).dot(a[idx_sv]) / self._params['lmbda']
        dec = ka + b + self.sign * c
        return dec

    def decision_objective(self, z, ka=None):
        a, b, c = z
        if ka is None:
            idx_sv = a != 0.0
            ka = self.kernel(idx=idx_sv).dot(a[idx_sv]) / self._params['lmbda']
        obj = 0.5 * ka.dot(a)
        dec = ka + b + self.sign * c
        obj += self.loss(dec).sum()
        if ~np.isinf(self._params.get('maxAsum', float('inf'))):
            obj += self._params['maxAsum'] * c
        return dec, obj

    def objective(self, z, ka):
        return self.decision_objective(z, ka)[1]


class KernelBase:
    def kernel(self, x=None, idx=None, idx0=None, deriv=False):
        if idx0 is None:
            x0 = self._x
        else:
            x0 = self._x[idx0]
        if x is not None:
            x1 = x
        if idx is not None:
            x1 = self._x[idx]
        return kernel(x0, x1, self._params['gamma'], deriv=deriv)


class Classification(ParamBase, KernelBase):
    def __init__(self, x, y, params):
        ParamBase.__init__(self, params)
        self._x = x
        self._y = y

    @property
    def size(self):
        return self._x.shape[0]

    @property
    def sign(self):
        return self._y

    def loss(self, t):
        return self.m(self._params['shift'] - self._y * t)

    def dloss(self, t):
        return -self._y * self.dm(self._params['shift'] - self._y * t)

    def ddloss(self, t):
        return self.ddm(self._params['shift'] - self._y * t)

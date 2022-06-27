import numpy as np
from .smooth_max import get_maxfun


class ParamBase:
    def __init__(self, params):
        self._params = params

    @property
    def params(self):
        return self._params

    @property
    def has_asum(self):
        return not np.isinf(self._params.get('maxAsum', float('inf')))

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
            ka = self.k(idx=idx_sv).dot(a[idx_sv]) / self._params['lmbda']
        dec = ka + b + self.sign * c
        return dec

    def decision_objective(self, z, ka=None):
        a, b, c = z
        if ka is None:
            idx_sv = a != 0.0
            ka = self.k(idx=idx_sv).dot(a[idx_sv]) / self._params['lmbda']
        obj = 0.5 * ka.dot(a)
        dec = ka + b + self.sign * c
        obj += self.loss(dec).sum()
        if ~np.isinf(self._params.get('maxAsum', float('inf'))):
            obj += self._params['maxAsum'] * c
        return dec, obj

    def objective(self, z, ka):
        return self.decision_objective(z, ka)[1]


class KernelBase:
    def __init__(self, kernel_fun):
        self._kernel_fun = kernel_fun

    def k(self, x=None, idx=None, idx0=None, deriv=False):
        if idx0 is None:
            x0 = self._x
        else:
            x0 = self._x[idx0]
        if x is not None:
            x1 = x
        if idx is not None:
            x1 = self._x[idx]
        return self._kernel_fun(
            x0, x1,
            self._params['kernel_param'],
            deriv=deriv,
        )


class Classification(ParamBase, KernelBase):
    def __init__(self, x, y, params, kernel_fun):
        ParamBase.__init__(self, params)
        KernelBase.__init__(self, kernel_fun)
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


class Regression(ParamBase, KernelBase):
    def __init__(self, x, y, params, kernel_fun):
        ParamBase.__init__(self, params)
        KernelBase.__init__(self, kernel_fun)
        self._x = np.concatenate((x, x))
        self._y = y

    @property
    def size(self):
        return self._x.shape[0]

    @property
    def n(self):
        return self._y.size

    @property
    def sign(self):
        return np.concatenate((
            +np.ones_like(self._y),
            -np.ones_like(self._y),
        ))

    def loss(self, t):
        r = t.copy()
        r[:self.n] -= self._y
        r[self.n:] -= self._y
        f = -self.sign * r - self._params['epsilon']
        return self.m(f)

    def dloss(self, t):
        r = t.copy()
        r[:self.n] -= self._y
        r[self.n:] -= self._y
        f = -self.sign * r - self._params['epsilon']
        return -self.sign * self.dm(f)

    def ddloss(self, t):
        r = t.copy()
        r[:self.n] -= self._y
        r[self.n:] -= self._y
        f = -self.sign * r - self._params['epsilon']
        return self.ddm(f)

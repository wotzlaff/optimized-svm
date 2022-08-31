import numpy as np
from .smooth_max import get_maxfun


class ParamBase:
    def __init__(
        self,
        lmbda,
        smoothingParam,
        maxfunDegree,
        maxAsum=float('inf')
    ):
        self.__lmbda = lmbda
        self.__smoothingParam = smoothingParam
        self.__maxfunDegree = maxfunDegree
        self.__maxAsum = maxAsum

    @property
    def lmbda(self):
        return self.__lmbda

    @property
    def smoothingParam(self):
        return self.__smoothingParam

    @property
    def maxfunDegree(self):
        return self.__maxfunDegree

    @property
    def maxAsum(self):
        return self.__maxAsum

    @property
    def has_asum(self):
        return not np.isinf(self.__maxAsum)

    def m(self, t):
        return get_maxfun(self.__smoothingParam, self.__maxfunDegree)[0](t)

    def dm(self, t):
        return get_maxfun(self.__smoothingParam, self.__maxfunDegree)[1](t)

    def ddm(self, t):
        return get_maxfun(self.__smoothingParam, self.__maxfunDegree)[2](t)

    def decision(self, z, ka=None):
        a, b, c = z
        if ka is None:
            idx_sv = a != 0.0
            ka = self.k(idx=idx_sv).dot(a[idx_sv]) / self.__lmbda
        dec = ka + b + self.sign * c
        return dec

    def decision_objective(self, z, ka=None):
        a, b, c = z
        if ka is None:
            idx_sv = a != 0.0
            ka = self.k(idx=idx_sv).dot(a[idx_sv]) / self.__lmbda
        obj = 0.5 * ka.dot(a)
        dec = ka + b + self.sign * c
        obj += self.loss(dec).sum()
        if not np.isinf(self.__maxAsum):
            obj += self.__maxAsum * c
        return dec, obj

    def objective(self, z, ka):
        return self.decision_objective(z, ka)[1]


class KernelBase:
    def __init__(self, kernel_fun, kernel_param):
        self.__kernel_fun = kernel_fun
        self.__kernel_param = kernel_param

    def k(self, x=None, idx=None, idx0=None, deriv=False):
        if idx0 is None:
            x0 = self._x
        else:
            x0 = self._x[idx0]
        if x is not None:
            x1 = x
        if idx is not None:
            x1 = self._x[idx]
        return self.__kernel_fun(
            x0, x1,
            self.__kernel_param,
            deriv=deriv,
        )


class Classification(ParamBase, KernelBase):
    def __init__(
        self,
        x, y,
        kernel_fun, kernel_param,
        shift,
        lmbda, smoothingParam,
        maxfunDegree='2',
        maxAsum=float('inf'),
    ):
        ParamBase.__init__(self, lmbda, smoothingParam, maxfunDegree, maxAsum)
        KernelBase.__init__(self, kernel_fun, kernel_param)
        self._x = x
        self._y = y
        self.__shift = shift

    @property
    def shift(self):
        return self.__shift

    @property
    def size(self):
        return self._x.shape[0]

    @property
    def sign(self):
        return self._y

    def extract_original_a(self, a):
        return a

    def loss(self, t):
        return self.m(self.__shift - self._y * t)

    def dloss(self, t):
        return -self._y * self.dm(self.__shift - self._y * t)

    def ddloss(self, t):
        return self.ddm(self.__shift - self._y * t)


class Regression(ParamBase, KernelBase):
    def __init__(
        self,
        x, y,
        kernel_fun, kernel_param,
        epsilon,
        lmbda, smoothingParam,
        maxfunDegree='2',
        maxAsum=float('inf'),
    ):
        ParamBase.__init__(self, lmbda, smoothingParam, maxfunDegree, maxAsum)
        KernelBase.__init__(self, kernel_fun, kernel_param)
        self._x = np.concatenate((x, x))
        self._y = y
        self.__epsilon = epsilon

    @property
    def epsilon(self):
        return self.__epsilon

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

    def extract_original_a(self, a):
        return a[:self.n] + a[self.n:]

    def loss(self, t):
        r = t.copy()
        r[:self.n] -= self._y
        r[self.n:] -= self._y
        f = -self.sign * r - self.__epsilon
        return self.m(f)

    def dloss(self, t):
        r = t.copy()
        r[:self.n] -= self._y
        r[self.n:] -= self._y
        f = -self.sign * r - self.__epsilon
        return -self.sign * self.dm(f)

    def ddloss(self, t):
        r = t.copy()
        r[:self.n] -= self._y
        r[self.n:] -= self._y
        f = -self.sign * r - self.__epsilon
        return self.ddm(f)

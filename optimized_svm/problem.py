import numpy as np
import numpy.typing as npt
from .smooth_max import get_maxfun

Arr = npt.NDArray[np.float64]
Vars = tuple[Arr, float, float]


class ParamBase:
    def __init__(
        self,
        lmbda: float,
        smoothingParam: float,
        maxfunDegree: str,
        maxAsum: float = float('inf')
    ):
        self.lmbda = lmbda
        self.smoothingParam = smoothingParam
        self.maxfunDegree = maxfunDegree
        self.maxAsum = maxAsum


    @property
    def has_asum(self):
        return not np.isinf(self.maxAsum)

    def m(self, t: Arr):
        return get_maxfun(self.smoothingParam, self.maxfunDegree)[0](t)

    def dm(self, t: Arr):
        return get_maxfun(self.smoothingParam, self.maxfunDegree)[1](t)

    def ddm(self, t: Arr):
        return get_maxfun(self.smoothingParam, self.maxfunDegree)[2](t)

    def decision(self, z: Vars, ka: Arr | None = None):
        a, b, c = z
        if ka is None:
            idx_sv = a != 0.0
            ka: Arr = self.k(idx=idx_sv).dot(a[idx_sv]) / self.lmbda
        dec: Arr = ka + b + self.sign * c
        return dec

    def decision_objective(
        self, z: Vars, ka: Arr | None = None
    ):
        a, b, c = z
        if ka is None:
            idx_sv = a != 0.0
            ka: Arr = self.k(idx=idx_sv).dot(a[idx_sv]) / self.lmbda
        obj: float = 0.5 * ka.dot(a)
        dec: Arr = ka + b + self.sign * c
        obj += self.loss(dec).sum()
        if not np.isinf(self.maxAsum):
            obj += self.maxAsum * c
        return dec, obj

    def objective(self, z: Vars, ka: Arr | None):
        return self.decision_objective(z, ka)[1]


class KernelBase:
    _x: Arr

    def __init__(self, kernel_fun, kernel_param):
        self.kernel_fun = kernel_fun
        self.kernel_param = kernel_param

    def k(
        self,
        x: Arr | None = None,
        idx=None,
        idx0=None,
        deriv: bool = False,
    ):
        if idx0 is None:
            x0 = self._x
        else:
            x0 = self._x[idx0]
        if x is not None:
            x1 = x
        if idx is not None:
            x1 = self._x[idx]
        return self.kernel_fun(
            x0, x1,
            self.kernel_param,
            deriv=deriv,
        )


class Classification(ParamBase, KernelBase):
    def __init__(
        self,
        x: Arr, y: Arr,
        kernel_fun, kernel_param,
        w: Arr | None = None,
        shift: float = 1.0,
        lmbda: float = 1.0,
        smoothingParam: float = 0.5,
        maxfunDegree: str = '2',
        maxAsum: float = float('inf'),
    ):
        ParamBase.__init__(self, lmbda, smoothingParam, maxfunDegree, maxAsum)
        KernelBase.__init__(self, kernel_fun, kernel_param)
        self._x = x
        self._y = y
        self._w = np.ones_like(y) if w is None else w
        self.shift = shift

    @property
    def size(self):
        return self._x.shape[0]

    @property
    def sign(self):
        return self._y

    def extract_original_a(self, a: Arr):
        return a

    def loss(self, t: Arr):
        return self._w * self.m(self.shift - self._y * t)

    def dloss(self, t: Arr):
        return -self._w * self._y * self.dm(self.shift - self._y * t)

    def ddloss(self, t: Arr):
        return self._w * self.ddm(self.shift - self._y * t)


class Regression(ParamBase, KernelBase):
    def __init__(
        self,
        x: Arr, y: Arr,
        kernel_fun, kernel_param,
        w: Arr | None = None,
        epsilon: float = 1e-6,
        lmbda: float = 1.0,
        smoothingParam: float = 0.5,
        maxfunDegree: str = '2',
        maxAsum: float = float('inf'),
    ):
        ParamBase.__init__(self, lmbda, smoothingParam, maxfunDegree, maxAsum)
        KernelBase.__init__(self, kernel_fun, kernel_param)
        self._x = np.concatenate((x, x))
        self._y = y
        self._w = np.ones_like(y) if w is None else w
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

    def extract_original_a(self, a: Arr):
        return a[:self.n] + a[self.n:]

    def loss(self, t: Arr):
        r = t.copy()
        r[:self.n] -= self._y
        r[self.n:] -= self._y
        f = -self.sign * r - self.__epsilon
        res = self.m(f)
        res[:self.n] *= self._w
        res[self.n:] *= self._w
        return res

    def dloss(self, t: Arr):
        r = t.copy()
        r[:self.n] -= self._y
        r[self.n:] -= self._y
        f = -self.sign * r - self.__epsilon
        res = -self.sign * self.dm(f)
        res[:self.n] *= self._w
        res[self.n:] *= self._w
        return res

    def ddloss(self, t: Arr):
        r = t.copy()
        r[:self.n] -= self._y
        r[self.n:] -= self._y
        f = -self.sign * r - self.__epsilon
        res = self.ddm(f)
        res[:self.n] *= self._w
        res[self.n:] *= self._w
        return res

from functools import partial
import numpy as np
import numpy.typing as npt
from scipy.special import gammaln
from .kernel import KernelMixin
from .decision import DecisionMixin
from .smooth_max import SmoothMaxMixin

Arr = npt.NDArray[np.float64]
Vars = tuple[Arr, float, float]


class Classification(SmoothMaxMixin, DecisionMixin, KernelMixin):
    def __init__(
        self,
        x: Arr, y: Arr,
        kernel_fun, kernel_param,
        w: Arr | None | str = None,
        shift: float = 1.0,
        lmbda: float = 1.0,
        smoothingParam: float = 0.5,
        maxfunDegree: str = '2',
        maxAsum: float = float('inf'),
    ):
        SmoothMaxMixin.__init__(self, smoothingParam, maxfunDegree)
        DecisionMixin.__init__(self, lmbda, maxAsum)
        self._x = x
        self._y = y
        KernelMixin.__init__(self, kernel_fun, kernel_param)
        if w is None:
            self._w = np.ones_like(y)
        elif w == 'balanced':
            ratio = (y == 1.0).mean()
            self._w = 0.5 / np.where(y == 1.0, ratio, 1.0 - ratio)
        else:
            self._w = w
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


class Regression(SmoothMaxMixin, DecisionMixin, KernelMixin):
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
        SmoothMaxMixin.__init__(self, smoothingParam, maxfunDegree)
        DecisionMixin.__init__(self, lmbda, maxAsum)
        self._x = np.concatenate((x, x))
        self._y = y
        KernelMixin.__init__(self, kernel_fun, kernel_param)
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


class LSSVR(DecisionMixin, KernelMixin):
    def __init__(
        self,
        x: Arr, y: Arr,
        kernel_fun, kernel_param,
        w: Arr | None = None,
        lmbda: float = 1.0,
    ):
        DecisionMixin.__init__(self, lmbda, maxAsum=float('inf'))
        self._x = x
        self._y = y
        KernelMixin.__init__(self, kernel_fun, kernel_param)
        self._w = np.ones_like(y) if w is None else w

    @property
    def size(self):
        return self._x.shape[0]

    @property
    def sign(self):
        return np.zeros(self.size)

    def extract_original_a(self, a: Arr):
        return a

    def loss(self, t: Arr):
        e = t - self._y
        return 0.5 * self._w * (e ** 2)

    def dloss(self, t: Arr):
        e = t - self._y
        return self._w * e

    def ddloss(self, t: Arr):
        return self._w


class PoissonSVM(DecisionMixin, KernelMixin):
    def __init__(
        self,
        x: Arr, y: Arr,
        kernel_fun, kernel_param,
        w: Arr | None = None,
        lmbda: float = 1.0,
    ):
        DecisionMixin.__init__(self, lmbda, maxAsum=float('inf'))
        self._x = x
        self._y = y
        KernelMixin.__init__(self, kernel_fun, kernel_param)
        self._w = np.ones_like(y) if w is None else w

    @property
    def size(self):
        return self._x.shape[0]

    @property
    def sign(self):
        return np.zeros(self.size)

    def extract_original_a(self, a: Arr):
        return a

    def loss(self, t: Arr):
        et = np.exp(t)
        return self._w * (et - self._y * t + gammaln(self._y+1))

    def dloss(self, t: Arr):
        et = np.exp(t)
        return self._w * (et - self._y)

    def ddloss(self, t: Arr):
        et = np.exp(t)
        return self._w * et


class PoissonSVM(DecisionMixin, KernelMixin):
    def __init__(
        self,
        x: Arr, y: Arr,
        kernel_fun, kernel_param,
        w: Arr | None = None,
        lmbda: float = 1.0,
    ):
        DecisionMixin.__init__(self, lmbda, maxAsum=float('inf'))
        self._x = x
        self._y = y
        KernelMixin.__init__(self, kernel_fun, kernel_param)
        self._w = np.ones_like(y) if w is None else w

    @property
    def size(self):
        return self._x.shape[0]

    @property
    def sign(self):
        return np.zeros(self.size)

    def extract_original_a(self, a: Arr):
        return a

    def loss(self, t: Arr):
        et = np.exp(t)
        return self._w * (et - self._y * t + gammaln(self._y+1))

    def dloss(self, t: Arr):
        et = np.exp(t)
        return self._w * (et - self._y)

    def ddloss(self, t: Arr):
        et = np.exp(t)
        return self._w * et


class TrainingMixin:
    def _create_problem(
        self,
        x, y,
        kind, params,
        kernel_fun, kernel_param
    ):
        Problem = dict(
            classification=Classification,
            regression=Regression,
            balanced_classification=partial(Classification, w='balanced'),
            ls=LSSVR,
            poisson=PoissonSVM,
        )[kind]
        if 'ratioAsum' in params:
            assert 'maxAsum' not in params
            params['maxAsum'] = len(y) * params['ratioAsum']
            del params['ratioAsum']
        return Problem(
            x, y,
            kernel_fun, kernel_param,
            **params,
        )

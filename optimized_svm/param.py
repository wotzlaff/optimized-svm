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

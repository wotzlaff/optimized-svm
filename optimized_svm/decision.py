import numpy as np
import numpy.typing as npt

Arr = npt.NDArray[np.float64]
Vars = tuple[Arr, float, float]


class DecisionMixin:
    def __init__(
        self,
        lmbda: float,
        maxAsum: float = float('inf')
    ):
        self.lmbda = lmbda
        self.maxAsum = maxAsum

    @property
    def has_asum(self):
        return not np.isinf(self.maxAsum)

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

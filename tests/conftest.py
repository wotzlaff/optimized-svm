import numpy as np
import pytest


@pytest.fixture
def data():
    rng = np.random.default_rng(1)
    n = 90
    x = rng.normal(size=(n, 3))
    y = np.where(x[:, 0] + 0.5 * x[:, 1] ** 2 - 0.5 + 0.3 * rng.normal(size=n) > 0, 1.0, -1.0)
    t = np.sin(x[:, 0]) + 0.3 * x[:, 1] + 0.1 * rng.normal(size=n)
    counts = rng.poisson(np.exp(0.5 * t)).astype(float)
    return x, y, t, counts

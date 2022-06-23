import numpy as np


def rbf_kernel(x1, x2, gamma):
    x1sqr = (x1 * x1).sum(axis=1)
    x2sqr = (x2 * x2).sum(axis=1)
    d = x1sqr[:, None] + x2sqr[None, :] - 2.0 * x1.dot(x2.T)
    return np.exp(-gamma * d)

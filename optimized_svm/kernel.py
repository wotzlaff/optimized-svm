import numpy as np


def diff(x1, x2, idx=slice(None)):
    x1sqr = (x1[:, idx] * x1[:, idx]).sum(axis=1)
    x2sqr = (x2[:, idx] * x2[:, idx]).sum(axis=1)
    d = x1sqr[:, None] + x2sqr[None, :] - 2.0 * x1[:, idx].dot(x2[:, idx].T)
    return d


def rbf_kernel(x1, x2, gamma=1.0):
    return np.exp(-gamma * diff(x1, x2, slice(None)))

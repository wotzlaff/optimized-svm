import numpy as np


def data(n, nft, scale=1.0, classification=True):
    np.random.seed(42)
    x = np.random.rand(n, nft)
    yr = np.sin(scale * np.pi * x).sum(axis=1)
    if classification:
        y = np.where(yr > np.median(yr), 1.0, -1.0)
    else:
        y = yr
    return x, y

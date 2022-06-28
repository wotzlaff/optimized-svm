import numpy as np


def data(n, nft, scale=1.0, relevant_ratio=1.0, classification=True):
    np.random.seed(42)
    x = np.random.rand(n, nft)
    n_rel = int(relevant_ratio * nft + 0.5)
    yr = np.sin(scale * np.pi * x[:, :n_rel]).sum(axis=1)
    if classification:
        y = np.where(yr > np.median(yr), 1.0, -1.0)
    else:
        y = yr
    return x, y

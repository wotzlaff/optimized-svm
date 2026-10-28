"""Reproduces the protocol of Table 10.4.1 (Strasdat, Sec. 10.4.2) on LIBSVM regression data.

For every dataset and split (80 % for hyperparameter optimisation + training, 20 % for testing):
features and targets are standardised on the training part, lambda and the RBF scaling are
optimised by minimising the 5-fold CV error (smoothed absolute error, delta = 0.1) and the
test MSE/MAE are reported in original units. Models: smoothed epsilon-SVR (cubic smoothing,
delta = epsilon / 4) for epsilon in {0.1, 0.25, 0.5} and the LS-SVM.

Usage::

    python benchmarks/thesis_regression.py --datasets pyrim triazines --splits 30
"""

from __future__ import annotations

import argparse
import time
import urllib.request
from pathlib import Path

import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from optimized_svm import OptimizedSVR

URL = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/{}"
FILES = {
    "pyrim": "pyrim",
    "triazines": "triazines",
    "mpg": "mpg",
    "housing": "housing",
    "spacega": "space_ga",
    "abalone": "abalone",
    "cpusmall": "cpusmall",
}
#: values of Table 10.4.1 (validation loss 10.4.1): MSE / MAE for eps 0.1, 0.25, 0.5, LS-SVM
THESIS = {
    "pyrim": ([0.00656, 0.00625, 0.00645, 0.00597], [0.0532, 0.0512, 0.0526, 0.0491]),
    "triazines": ([0.0244, 0.0239, 0.0237, 0.0232], [0.107, 0.109, 0.111, 0.110]),
    "mpg": ([6.50, 6.58, 7.37, 6.63], [1.81, 1.84, 1.98, 1.82]),
    "housing": ([10.8, 10.9, 11.9, 9.54], [2.10, 2.17, 2.43, 2.07]),
    "spacega": ([0.0133, 0.0127, 0.0128, 0.0127], [0.0849, 0.0845, 0.0854, 0.0845]),
    "abalone": ([4.72, 4.61, 4.56, 4.53], [1.50, 1.50, 1.52, 1.53]),
    "cpusmall": ([19.1, 20.0, 32.6, 18.7], [2.70, 2.96, 3.66, 2.70]),
}
MODELS = [("eps=0.1", 0.1), ("eps=0.25", 0.25), ("eps=0.5", 0.5), ("LS-SVM", None)]


def load(name: str, cache: Path) -> tuple[np.ndarray, np.ndarray]:
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / FILES[name]
    if not path.exists():
        urllib.request.urlretrieve(URL.format(FILES[name]), path)
    x, y = load_svmlight_file(str(path))
    return np.asarray(x.todense()), np.asarray(y)


def model(eps: float | None, n_features: int, time_limit: float) -> OptimizedSVR:
    common = dict(
        lam_init=1e-2,
        scaling_init=float(np.sqrt(np.log(2.0) / n_features)),
        lam_bounds=(1e-6, 1.0),
        scaling_bounds=(1e-6, 1e2),
        validation="absolute",
        validation_smoothing=0.1,
        normalize_y=False,  # targets are standardised by the protocol
        time_limit=time_limit,
    )
    if eps is None:
        return OptimizedSVR(loss="squared", **common)
    return OptimizedSVR(epsilon=eps, smoothing=eps / 4, degree="3", **common)


def run(name: str, splits: int, time_limit: float, cache: Path, max_size: int) -> None:
    x, y = load(name, cache)
    rng = np.random.default_rng(0)
    mse = np.zeros((splits, len(MODELS)))
    mae = np.zeros_like(mse)
    t0 = time.perf_counter()
    for s in range(splits):
        if len(y) > max_size:
            idx = rng.choice(len(y), max_size, replace=False)
            xs, ys = x[idx], y[idx]
        else:
            xs, ys = x, y
        xa, xb, ya, yb = train_test_split(xs, ys, test_size=0.2, random_state=s)
        sx, sy = StandardScaler().fit(xa), StandardScaler().fit(ya[:, None])
        xa_n, xb_n = sx.transform(xa), sx.transform(xb)
        ya_n = sy.transform(ya[:, None]).ravel()
        for m, (_, eps) in enumerate(MODELS):
            est = model(eps, x.shape[1], time_limit).fit(xa_n, ya_n)
            pred = sy.inverse_transform(est.predict(xb_n)[:, None]).ravel()
            mse[s, m] = np.mean((pred - yb) ** 2)
            mae[s, m] = np.mean(np.abs(pred - yb))
    elapsed = time.perf_counter() - t0
    ref_mse, ref_mae = THESIS[name]
    print(f"\n{name} (n = {min(len(y), max_size)}, {splits} splits, {elapsed:.0f} s)")
    print(f"{'model':<10}{'MSE':>11}{'thesis':>11}{'MAE':>11}{'thesis':>11}")
    for m, (label, _) in enumerate(MODELS):
        print(
            f"{label:<10}{mse[:, m].mean():>11.4g}{ref_mse[m]:>11.4g}"
            f"{mae[:, m].mean():>11.4g}{ref_mae[m]:>11.4g}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--datasets", nargs="+", default=list(FILES), choices=list(FILES))
    parser.add_argument("--splits", type=int, default=30)
    parser.add_argument("--time-limit", type=float, default=600.0)
    parser.add_argument("--max-size", type=int, default=1000)
    parser.add_argument("--cache", type=Path, default=Path.home() / ".cache" / "optimized-svm")
    args = parser.parse_args()
    for name in args.datasets:
        run(name, args.splits, args.time_limit, args.cache, args.max_size)


if __name__ == "__main__":
    main()

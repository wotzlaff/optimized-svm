"""Tuning the loss parameters epsilon and delta by the hypergradient (exploratory study).

The CV error is differentiated w.r.t. ``log(epsilon)`` and ``log(delta)`` of the smoothed
epsilon-insensitive loss together with ``log(lambda)`` and the kernel scaling. The script compares
fixed parameters with tuned ones from several starting points on three data sets and reports the
CV error, the test error, the number of CV evaluations and Newton failures.

Usage::

    python benchmarks/hpo_loss_params.py
"""

import time
import warnings

import numpy as np
from scipy.optimize import minimize
from sklearn.datasets import load_diabetes, make_friedman1
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler

from optimized_svm import EpsilonInsensitiveLoss, TrainingProblem, cv_error_and_gradient, train
from optimized_svm.estimators import auto_scaling
from optimized_svm.kernels import rbf
from optimized_svm.validation import SmoothAbsoluteValidation


def datasets():
    x, y = make_friedman1(500, noise=1.0, random_state=0)
    yield "friedman", x, y
    rng = np.random.default_rng(0)
    x2, y2 = make_friedman1(500, noise=0.5, random_state=1)
    out = rng.random(500) < 0.08
    y2 = y2 + out * rng.normal(0, 8, 500)  # 8 % gross outliers
    yield "friedman+outliers", x2, y2
    d = load_diabetes()
    yield "diabetes", d.data, d.target


def prep(x, y, seed=0):
    xtr, xte, ytr, yte = train_test_split(x, y, test_size=0.3, random_state=seed)
    sx = StandardScaler().fit(xtr)
    med = np.median(ytr)
    mad = np.median(np.abs(ytr - med)) * 1.4826
    return sx.transform(xtr), sx.transform(xte), (ytr - med) / mad, (yte - med) / mad


def run(mode, xtr, ytr, eps0=0.1, d0=0.5, deg="2", max_iter=60):
    folds = list(KFold(5, shuffle=True, random_state=0).split(xtr))
    w = np.ones(len(ytr))
    val = SmoothAbsoluteValidation(0.1)
    base = EpsilonInsensitiveLoss(eps0, d0, deg)
    params = {
        "fixed": (),
        "eps": ("epsilon",),
        "both": ("epsilon", "smoothing"),
        "delta": ("smoothing",),
    }[mode]
    th0 = np.r_[np.log(1e-2), np.log(auto_scaling(xtr)), [np.log(getattr(base, p)) for p in params]]
    bounds = [(np.log(1e-6), 0.0), (np.log(1e-4), np.log(1e2))] + [
        (np.log(1e-3), np.log(2.0 if p == "epsilon" else 1.0)) for p in params
    ]
    warm = [None] * 5
    hist = []
    fails = [0]

    def fun(th):
        r = cv_error_and_gradient(
            th, xtr, ytr, w, folds, base, val, None, None, None, None, warm, params
        )
        fails[0] += sum(f.solution.info.get("newton") == "failed" for f in r.folds)
        hist.append(r.error)
        return r.error, r.gradient

    t = time.perf_counter()
    res = minimize(
        fun,
        th0,
        jac=True,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": max_iter, "ftol": 1e-9, "gtol": 1e-7},
    )
    return res, time.perf_counter() - t, len(hist), fails[0], params, base


def test_error(res, params, base, xtr, ytr, xte, yte):
    import dataclasses

    th = res.x
    lam = np.exp(th[0])
    s = np.exp(th[1])
    vals = {p: float(np.exp(th[2 + j])) for j, p in enumerate(params)}
    loss = dataclasses.replace(base, **vals)
    prob = TrainingProblem(xtr * s, ytr, loss, lam)
    sol = train(prob)
    f = rbf(xte * s, prob.x) @ (prob.sample_coefficients(sol.a) / lam) + sol.b
    return float(np.mean((f - yte) ** 2)), float(np.mean(np.abs(f - yte))), vals, lam, s


if __name__ == "__main__":
    for name, x, y in datasets():
        xtr, xte, ytr, yte = prep(x, y)
        print(f"== {name}: n_train={len(ytr)}")
        for mode, e0, d0 in [
            ("fixed", 0.1, 0.5),
            ("fixed", 0.1, 0.025),
            ("delta", 0.1, 0.5),
            ("eps", 0.1, 0.5),
            ("both", 0.1, 0.5),
            ("both", 0.5, 0.05),
            ("both", 1.0, 0.5),
            ("both", 0.02, 0.3),
        ]:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res, dt, nev, nf, params, base = run(mode, xtr, ytr, e0, d0)
                mse, mae, vals, lam, s = test_error(res, params, base, xtr, ytr, xte, yte)
            tag = f"{mode:5} start eps={e0:<4} delta={d0:<5}"
            fin = " ".join(f"{k}={v:.3g}" for k, v in vals.items())
            print(
                f"  {tag} -> cv={res.fun:.5f} test mse={mse:.4f} mae={mae:.4f} "
                f"lam={lam:.2e} s={s:.2f} {fin} | {nev} evals {dt:.1f}s "
                f"newton_fail={nf} {res.message[:30]}",
                flush=True,
            )

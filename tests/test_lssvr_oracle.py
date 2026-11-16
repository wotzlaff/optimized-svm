"""Agreement with the cross-validation objective of the older package ``optimized-lssvr``.

``OptimizedLSSVR(kernel=..., n_splits=5)`` minimises the relative MSE
``sum_v e_v^2 / n / var(y)`` over unshuffled 5-fold CV, where the fold systems use
``K_tr + lmbda * n_tr * I`` and ``K = exp(-sum_k gamma_k d_k^2)``. With equally sized folds the
objective of ``OptimizedSVR(loss="squared")`` coincides after the mapping

    lam = lmbda * n_tr,   s_k = sqrt(gamma_k),   labels scaled by std(y) (``normalize_y=True``),
    relative_mse = 2 * cv_error (the mean over the validation samples),
    d/d log s_k = 2 * d/d log gamma_k.

The reference numbers below were produced by ``benchmarks/lssvr_oracle_reference.py`` (old package,
scikit-learn 1.9.1, numpy 2.x, data and evaluation points defined there).
"""

import numpy as np
import pytest
from sklearn.model_selection import KFold

from optimized_svm import OptimizedSVR
from optimized_svm.hypergradient import cv_error_and_gradient
from optimized_svm.losses import SquaredLoss
from optimized_svm.validation import SquaredValidation

N, D = 120, 3
N_SPLITS = 5
N_TRAIN = N - N // N_SPLITS
GROUPS = [[0, 1], [2]]

# name -> (blocks, log lmbda, log gamma, old value, old gradient w.r.t. (log lmbda, log gamma))
CASES = {
    "rbf": (
        None,
        np.log(2e-3),
        np.log([0.7]),
        0.16514806178979435,
        [0.01904511707565077, 0.09667731175957815],
    ),
    "ard": (
        "features",
        np.log(5e-4),
        np.log([0.4, 1.3, 0.05]),
        0.1192682594315496,
        [0.0020012429933791097, 0.01518045952553465, 0.032044744453271, 0.008384078459107018],
    ),
    "group": (
        GROUPS,
        np.log(1e-2),
        np.log([0.9, 0.2]),
        0.20385444969062785,
        [0.053354273999503446, 0.06944010763655724, 0.02397898257402175],
    ),
}


def _data():
    # same as `make_data` in benchmarks/lssvr_oracle_reference.py
    rng = np.random.default_rng(0)
    x = rng.normal(size=(N, D))
    y = np.sin(x[:, 0]) + 0.5 * x[:, 1] ** 2 + 0.1 * rng.normal(size=N) + 3.0
    return x, y


def _new_objective(blocks, loglam, loggamma):
    """(relative_mse, gradient w.r.t. (log lmbda, log gamma_k)) from ``cv_error_and_gradient``."""
    x, y = _data()
    yn = (y - y.mean()) / y.std()  # what OptimizedSVR(normalize_y=True) does for loss="squared"
    folds = list(KFold(N_SPLITS).split(x))
    if blocks == "features":
        blocks = [[j] for j in range(D)]
    theta = np.concatenate(([loglam + np.log(N_TRAIN)], 0.5 * loggamma))
    res = cv_error_and_gradient(
        theta, x, yn, np.ones(N), folds, SquaredLoss(), SquaredValidation(), blocks
    )
    grad = res.gradient.copy()
    grad[1:] *= 0.5  # d/d log gamma = (1/2) d/d log s
    return 2.0 * res.error, 2.0 * grad


@pytest.mark.parametrize("name", list(CASES))
def test_cv_objective_matches_optimized_lssvr(name):
    blocks, loglam, loggamma, value, grad = CASES[name]
    new_value, new_grad = _new_objective(blocks, loglam, loggamma)
    assert new_value == pytest.approx(value, rel=1e-10)
    np.testing.assert_allclose(new_grad, grad, rtol=1e-9, atol=1e-12)


def test_estimator_objective_matches_optimized_lssvr():
    # the public estimator reports the same error at its initial point (max_iter=0 evaluations only)
    blocks, loglam, loggamma, value, _ = CASES["ard"]
    x, y = _data()
    reg = OptimizedSVR(
        loss="squared",
        cv=list(KFold(N_SPLITS).split(x)),
        blocks=blocks,
        two_stage=False,
        lam_init=float(np.exp(loglam) * N_TRAIN),
        scaling_init=float(np.sqrt(np.exp(loggamma[0]))),
        lam_bounds=(1e-12, 1e3),
        max_iter=1,
    ).fit(x, y)
    first = reg.history_[0]
    assert 2.0 * first["error"] == pytest.approx(
        _new_objective(blocks, loglam, np.log(np.exp(loggamma[0]) * np.ones(D)))[0], rel=1e-10
    )

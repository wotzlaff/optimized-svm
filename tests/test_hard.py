"""Indicator losses: independent convex oracle, feasibility and local sensitivities."""

import cvxpy as cp
import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn.base import clone
from sklearn.model_selection import KFold

from optimized_svm import (
    HardEpsilonInsensitiveLoss,
    MaximalMarginLoss,
    OptimizedSVC,
    OptimizedSVR,
    SmoothSVC,
    SmoothSVR,
    TrainingProblem,
    cv_error_and_gradient,
    train,
)
from optimized_svm.kernels import rbf
from optimized_svm.validation import SquaredValidation


@pytest.mark.parametrize("kind", ["classification", "regression"])
def test_cvxpy_oracle(kind):
    rng = np.random.default_rng(91)
    x = rng.normal(size=(14, 2))
    y = np.where(x[:, 0] > 0, 1.0, -1.0) if kind == "classification" else np.sin(x[:, 0])
    loss = MaximalMarginLoss() if kind == "classification" else HardEpsilonInsensitiveLoss(0.12)
    prob = TrainingProblem(x, y, loss, 0.3)
    sol = train(prob)
    k = rbf(x, x)
    coef, bias = cp.Variable(14), cp.Variable()
    dec = k @ coef + bias
    constraints = (
        [cp.multiply(y, dec) >= 1]
        if kind == "classification"
        else [dec >= y - 0.12, dec <= y + 0.12]
    )
    oracle = cp.Problem(cp.Minimize(0.5 * cp.quad_form(coef, cp.psd_wrap(k))), constraints)
    oracle.solve(solver="CLARABEL", tol_gap_abs=1e-10, tol_feas=1e-10, tol_gap_rel=1e-10)
    beta = sol.a / prob.lam
    assert_allclose(0.5 * beta @ k @ beta, oracle.value, rtol=2e-7, atol=2e-8)
    z = rng.normal(size=(8, 2))
    assert_allclose(rbf(z, x) @ beta + sol.b, rbf(z, x) @ coef.value + bias.value, atol=3e-5)


@pytest.mark.parametrize("kind", ["classification", "regression"])
def test_infeasible_duplicates(kind):
    x = np.array([[0.0], [0.0], [1.0]])
    y = np.array([-1.0, 1.0, 1.0])
    loss = MaximalMarginLoss() if kind == "classification" else HardEpsilonInsensitiveLoss(0.2)
    with pytest.raises(ValueError, match="infeasible"):
        train(TrainingProblem(x, y, loss, 1))


@pytest.mark.parametrize("kind", ["classification", "regression"])
def test_lambda_and_positive_weight_invariance(kind):
    x = np.linspace(-2, 2, 17)[:, None]
    y = np.where(x[:, 0] > 0, 1.0, -1.0) if kind == "classification" else np.sin(x[:, 0])
    loss = MaximalMarginLoss() if kind == "classification" else HardEpsilonInsensitiveLoss(0.1)
    solutions = [
        train(TrainingProblem(x, y, loss, lam, np.linspace(0.1, 5, 17))) for lam in [1e-4, 1.0, 1e4]
    ]
    for lam, sol in zip([1e-4, 1.0, 1e4], solutions, strict=True):
        assert_allclose(sol.ka + sol.b, solutions[1].ka + solutions[1].b, atol=1e-8)
        assert_allclose(sol.a / lam, solutions[1].a, atol=1e-8)
    plain = train(TrainingProblem(x, y, loss, 1))
    assert_allclose(plain.a, solutions[1].a, atol=1e-8)


def test_zero_weights_remove_constraints():
    x = np.array([[0.0], [0.0], [1.0]])
    y = np.array([-1.0, 1.0, 1.0])
    prob = TrainingProblem(x, y, MaximalMarginLoss(), 1, weights=np.array([0.0, 1.0, 1.0]))
    sol = train(prob)
    assert np.min(sol.ka[1:] + sol.b) >= 1 - 1e-8
    assert sol.a[0] == 0
    all_zero = train(TrainingProblem(x, y, MaximalMarginLoss(), 1, weights=np.zeros(3)))
    assert_allclose(all_zero.a, 0)
    assert all_zero.b == 0


def test_per_sample_tolerances_and_intercept_only():
    x = np.array([[0.0], [0.0], [1.0]])
    y = np.array([0.0, 1.0, 0.4])
    loss = HardEpsilonInsensitiveLoss(np.array([0.7, 0.4, 0.4]))
    sol = train(TrainingProblem(x, y, loss, 1))
    assert_allclose(sol.a, 0)
    assert_allclose(sol.b, 0.65)
    assert np.all(np.abs(sol.ka + sol.b - y) <= loss.epsilon)


@pytest.mark.parametrize("loss", [MaximalMarginLoss(), HardEpsilonInsensitiveLoss(0.1)])
def test_cv_gradient(loss):
    rng = np.random.default_rng(5)
    x = rng.normal(size=(18, 2))
    y = np.where(x[:, 0] > 0, 1.0, -1.0) if isinstance(loss, MaximalMarginLoss) else np.sin(x[:, 0])
    theta = np.log([0.2, 0.8, 1.1])
    folds = list(KFold(3, shuffle=True, random_state=2).split(x))

    def fun(t):
        return cv_error_and_gradient(
            t, x, y, np.ones(18), folds, loss, SquaredValidation(), blocks=[[0], [1]]
        )

    res = fun(theta)
    numeric = np.array(
        [(fun(theta + d).error - fun(theta - d).error) / 2e-5 for d in np.eye(3) * 1e-5]
    )
    assert_allclose(res.gradient, numeric, rtol=2e-5, atol=2e-6)
    assert res.gradient[0] == 0


def test_epsilon_gradient_and_per_sample_cv():
    x = np.linspace(-2, 2, 19)[:, None]
    y = np.sin(x[:, 0])
    w = np.ones(19)
    folds = list(KFold(3, shuffle=True, random_state=7).split(x))
    theta = np.log([0.2, 0.8, 0.15])

    def fun(t):
        return cv_error_and_gradient(
            t,
            x,
            y,
            w,
            folds,
            HardEpsilonInsensitiveLoss(),
            SquaredValidation(),
            loss_params=["epsilon"],
        )

    res = fun(theta)
    numeric = np.array(
        [(fun(theta + d).error - fun(theta - d).error) / 2e-5 for d in np.eye(3) * 1e-5]
    )
    assert_allclose(res.gradient, numeric, atol=2e-6, rtol=2e-5)
    vector = cv_error_and_gradient(
        theta[:2],
        x,
        y,
        w,
        folds,
        HardEpsilonInsensitiveLoss(np.full(19, 0.15)),
        SquaredValidation(),
    )
    assert_allclose(vector.error, res.error, atol=1e-12)
    assert_allclose(vector.gradient, res.gradient[:2], atol=1e-12)


def test_estimators_and_support_vectors():
    x = np.linspace(-2, 2, 15)[:, None]
    y = np.where(x[:, 0] > 0, "right", "left")
    svc = SmoothSVC(loss="hard_margin", scaling=1).fit(x, y)
    assert np.all(svc.predict(x) == y)
    assert 0 < svc.n_support_ < 15
    assert_allclose(abs(svc.decision_function(x)[svc.support_]), 1.0, atol=1e-7)
    svr = SmoothSVR(loss="hard_epsilon_insensitive", scaling=1, normalize_y=False).fit(
        x, np.sin(x[:, 0])
    )
    assert np.max(abs(svr.predict(x) - np.sin(x[:, 0]))) <= 0.1 + 1e-7
    assert_allclose(abs(svr.predict(x)[svr.support_] - np.sin(x[svr.support_, 0])), 0.1, atol=1e-7)
    assert_allclose(clone(svr).fit(x, np.sin(x[:, 0])).predict(x), svr.predict(x))


@pytest.mark.parametrize(
    "estimator",
    [
        OptimizedSVC(loss="hard_margin", max_iter=2, cv=2, scaling_bounds=(0.5, 2)),
        OptimizedSVR(
            loss="hard_epsilon_insensitive",
            max_iter=2,
            cv=2,
            scaling_bounds=(0.5, 2),
            normalize_y=False,
        ),
    ],
)
def test_optimized_estimators(estimator):
    x = np.linspace(-2, 2, 20)[:, None]
    y = np.where(x[:, 0] > 0, 1.0, -1.0) if isinstance(estimator, OptimizedSVC) else np.sin(x[:, 0])
    fitted = clone(estimator).fit(x, y)
    assert np.all(np.isfinite(fitted.predict(x)))
    assert fitted.n_support_ > 0


@pytest.mark.parametrize(
    "loss",
    [
        HardEpsilonInsensitiveLoss(-0.1),
        HardEpsilonInsensitiveLoss(np.array([0.1])),
        MaximalMarginLoss(0),
        MaximalMarginLoss(np.nan),
    ],
)
def test_invalid_loss_parameters(loss):
    with pytest.raises(ValueError):
        TrainingProblem(np.array([[0.0], [1.0]]), np.array([-1.0, 1.0]), loss, 1.0)


def test_indicator_values_and_unsupported_options():
    x = np.array([[0.0], [1.0]])
    y = np.array([-1.0, 1.0])
    loss = MaximalMarginLoss()
    terms = loss.terms(y, np.ones(2))
    assert_allclose(loss.value(y, terms), 0.0)
    assert np.all(np.isinf(loss.value(np.zeros(2), terms)))
    with pytest.raises(ValueError, match="max_asum"):
        TrainingProblem(x, y, loss, 1.0, max_asum=2.0)
    with pytest.raises(ValueError, match="offset"):
        TrainingProblem(x, y, loss, 1.0, offset=np.ones(2))
    with pytest.raises(ValueError, match="SMO/Newton"):
        train(TrainingProblem(x, y, loss, 1.0), newton_options={"tol": 1e-5})
    with pytest.raises(ValueError, match="n_basis"):
        SmoothSVC(loss="hard_margin", n_basis=2).fit(x, y)


def test_parallel_folds_match_serial():
    x = np.linspace(-2, 2, 20)[:, None]
    y = np.sin(x[:, 0])
    folds = list(KFold(3).split(x))
    args = (
        np.log([0.2, 0.8]),
        x,
        y,
        np.ones(20),
        folds,
        HardEpsilonInsensitiveLoss(0.15),
        SquaredValidation(),
    )
    serial = cv_error_and_gradient(*args)
    parallel = cv_error_and_gradient(*args, n_jobs=2)
    assert_allclose(parallel.error, serial.error, atol=1e-12)
    assert_allclose(parallel.gradient, serial.gradient, atol=1e-12)


def test_zero_epsilon_interpolates():
    x = np.linspace(-2, 2, 9)[:, None]
    y = np.sin(x[:, 0])
    fit = SmoothSVR(loss="hard_epsilon_insensitive", epsilon=0, normalize_y=False, scaling=1)
    assert_allclose(fit.fit(x, y).predict(x), y, atol=2e-7)


@pytest.mark.parametrize("lam", [np.nan, np.inf, 0, -1])
def test_invalid_lambda(lam):
    with pytest.raises(ValueError, match="lam"):
        SmoothSVC(loss="hard_margin", lam=lam).fit([[0.0], [1.0]], [-1, 1])


@pytest.mark.parametrize("weights", [np.array([np.nan, 1.0]), np.array([-1.0, 1.0])])
def test_invalid_weights(weights):
    with pytest.raises(ValueError, match="weights|non-negative"):
        TrainingProblem(
            np.array([[0.0], [1.0]]),
            np.array([-1.0, 1.0]),
            MaximalMarginLoss(),
            1.0,
            weights=weights,
        )


@pytest.mark.parametrize("lam", [1e-10, 1e10])
def test_cv_gradient_lambda_invariance(lam):
    x = np.linspace(-2, 2, 20)[:, None]
    y = np.sin(x[:, 0])
    folds = list(KFold(3).split(x))

    def fun(lam_value):
        return cv_error_and_gradient(
            np.log([lam_value, 0.8]),
            x,
            y,
            np.ones(20),
            folds,
            HardEpsilonInsensitiveLoss(0.15),
            SquaredValidation(),
        )

    reference, scaled = fun(1.0), fun(lam)
    assert_allclose(scaled.error, reference.error, atol=1e-12)
    assert_allclose(scaled.gradient, reference.gradient, atol=1e-12)

"""Kernel GLM estimators: scikit-learn checks, families, offset, hyperparameter optimisation."""

import numpy as np
import pytest
from scipy.special import expit
from sklearn.utils.estimator_checks import parametrize_with_checks

from optimized_svm import KernelGLMRegressor, OptimizedKernelGLMRegressor


def _expected_failures(est):
    # `scaling="auto"` depends on the data spread, which changes when samples are repeated
    return {
        "check_sample_weight_equivalence_on_dense_data": "scaling='auto' is data dependent",
        "check_regressors_no_decision_function": "decision_function is the link-scale predictor",
    }


@parametrize_with_checks(
    [
        KernelGLMRegressor(),
        KernelGLMRegressor(family="gamma"),
        KernelGLMRegressor(family="negative_binomial", dispersion=2.0),
        KernelGLMRegressor(family="tweedie"),
        OptimizedKernelGLMRegressor(cv=2, max_iter=2),
        OptimizedKernelGLMRegressor(
            family="negative_binomial", tune_dispersion=True, cv=2, max_iter=2
        ),
    ],
    expected_failed_checks=_expected_failures,
)
def test_sklearn_compatible(estimator, check):
    check(estimator)


def _data(family, n=250, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2, 2, (n, 2))
    eta = 0.5 + 0.8 * np.sin(2 * x[:, 0]) + 0.4 * x[:, 1]
    mu = np.exp(eta)
    w = None
    if family == "poisson":
        y = rng.poisson(mu)
    elif family == "negative_binomial":
        y = rng.negative_binomial(2.0, 2.0 / (2.0 + mu))
    elif family == "gamma":
        y = rng.gamma(4.0, mu / 4.0)
    elif family == "tweedie":
        # point mass at zero plus a gamma part
        y = np.where(rng.random(n) < np.exp(-mu / 2), 0.0, rng.gamma(2.0, mu / 2))
    else:
        w = rng.integers(5, 20, n).astype(float)
        y = rng.binomial(w.astype(int), expit(eta - 0.5)) / w
    return x, y.astype(float), w


@pytest.mark.parametrize("family", ["poisson", "binomial", "negative_binomial", "gamma", "tweedie"])
def test_family_beats_null_model(family):
    x, y, w = _data(family)
    est = KernelGLMRegressor(0.1, family=family, dispersion=2.0).fit(x, y, sample_weight=w)
    d2 = est.score(x, y, sample_weight=w)
    assert 0.05 < d2 <= 1.0
    p = est.predict(x)
    assert np.all(np.isfinite(p))
    if family == "binomial":
        assert np.all((p >= 0) & (p <= 1))
        np.testing.assert_allclose(p, expit(est.decision_function(x)))
    else:
        assert np.all(p >= 0)
        np.testing.assert_allclose(p, np.exp(est.decision_function(x)))


def test_score_of_null_model_is_zero():
    x, y, _ = _data("poisson")
    est = KernelGLMRegressor(1e8, family="poisson", scaling=1.0).fit(x, y)
    assert abs(est.score(x, y)) < 1e-3


@pytest.mark.parametrize(
    ("family", "y", "message"),
    [
        ("poisson", [1.0, -1.0, 2.0], "non-negative"),
        ("gamma", [1.0, 0.0, 2.0], "strictly positive"),
        ("binomial", [0.5, 1.5, 0.0], "proportions"),
        ("tweedie", [1.0, -2.0, 2.0], "non-negative"),
        ("nope", [1.0, 1.0, 2.0], "family must be one of"),
    ],
)
def test_invalid_targets_and_family(family, y, message):
    x = np.arange(6.0).reshape(3, 2)
    with pytest.raises(ValueError, match=message):
        KernelGLMRegressor(family=family).fit(x, y)


def test_invalid_parameters_and_offset():
    x, y, _ = _data("poisson", 20)
    with pytest.raises(ValueError, match="power"):
        KernelGLMRegressor(family="tweedie", power=2.5).fit(x, y)
    with pytest.raises(ValueError, match="dispersion"):
        KernelGLMRegressor(family="negative_binomial", dispersion=-1.0).fit(x, y)
    with pytest.raises(ValueError, match="offset must have shape"):
        KernelGLMRegressor().fit(x, y, offset=np.zeros(5))
    est = KernelGLMRegressor().fit(x, y)
    with pytest.raises(ValueError, match="offset must have shape"):
        est.predict(x, offset=np.zeros(3))


@pytest.mark.parametrize("family", ["poisson", "negative_binomial"])
def test_exposure_offset(family):
    rng = np.random.default_rng(3)
    n = 200
    x = rng.uniform(-1, 1, (n, 2))
    exposure = rng.uniform(0.5, 4.0, n)
    rate = np.exp(0.3 + x[:, 0])
    y = rng.poisson(rate * exposure).astype(float)
    kw = {"family": family, "dispersion": 5.0, "scaling": 1.0}
    est = KernelGLMRegressor(0.05, **kw).fit(x, y, offset=np.log(exposure))
    # the offset acts on the link scale and is part of the decision function and the prediction
    base = est.decision_function(x)
    off = np.log(exposure)
    np.testing.assert_allclose(est.decision_function(x, off), base + off)
    np.testing.assert_allclose(est.predict(x, off), est.predict(x) * exposure)
    # the offset model recovers the rate better than ignoring the exposure
    plain = KernelGLMRegressor(0.05, **kw).fit(x, y)
    assert est.score(x, y, offset=off) > plain.score(x, y)
    err = np.mean((est.predict(x) - rate) ** 2)
    assert err < np.mean((plain.predict(x) - rate) ** 2)


def test_binomial_trials_equal_expanded_bernoulli_data():
    rng = np.random.default_rng(4)
    n = 40
    x = rng.normal(size=(n, 2))
    trials = rng.integers(1, 6, n)
    succ = rng.binomial(trials, expit(x[:, 0]))
    kw = {"family": "binomial", "scaling": 0.7, "newton_options": {"tol": 1e-12}}
    grouped = KernelGLMRegressor(0.3, **kw).fit(
        x, succ / trials, sample_weight=trials.astype(float)
    )
    rows = np.repeat(np.arange(n), trials)
    labels = np.concatenate(
        [np.r_[np.ones(s), np.zeros(t - s)] for s, t in zip(succ, trials, strict=True)]
    )
    expanded = KernelGLMRegressor(0.3, **kw).fit(x[rows], labels)
    xt = rng.normal(size=(15, 2))
    np.testing.assert_allclose(
        grouped.decision_function(xt), expanded.decision_function(xt), atol=1e-6
    )


def test_optimized_improves_cv_deviance_with_offset():
    x, y, _ = _data("poisson", 200, seed=5)
    off = np.log(np.random.default_rng(5).uniform(0.5, 2.0, 200))
    est = OptimizedKernelGLMRegressor(cv=3, max_iter=15, lam_init=1.0, scaling_init=0.05)
    est.fit(x, y, offset=off)
    errors = [h["error"] for h in est.history_]
    assert est.cv_error_ < errors[0]
    assert not hasattr(est, "dispersion_")
    assert est.score(x, y, offset=off) > 0.05
    assert "dispersion" not in est.history_[0]


@pytest.mark.parametrize("family", ["binomial", "gamma", "tweedie"])
def test_optimized_other_families(family):
    x, y, w = _data(family, 150, seed=6)
    est = OptimizedKernelGLMRegressor(family=family, cv=3, max_iter=8)
    est.fit(x, y, sample_weight=w)
    assert est.cv_error_ <= est.history_[0]["error"]
    assert est.score(x, y, sample_weight=w) > 0.0


def test_optimized_with_blocks_and_fixed_dispersion():
    x, y, _ = _data("negative_binomial", 150, seed=7)
    est = OptimizedKernelGLMRegressor(
        family="negative_binomial", dispersion=2.0, blocks="features", cv=3, max_iter=6
    ).fit(x, y)
    assert est.dispersion_ == 2.0
    assert est.scaling_opt_.shape == (2,)
    assert all("dispersion" not in h for h in est.history_)


def test_tune_dispersion_moves_towards_overdispersion():
    rng = np.random.default_rng(8)
    n = 400
    x = rng.uniform(-2, 2, (n, 2))
    mu = np.exp(1.0 + 0.8 * np.sin(2 * x[:, 0]))
    y = rng.negative_binomial(1.0, 1.0 / (1.0 + mu)).astype(float)  # r = 1
    est = OptimizedKernelGLMRegressor(
        family="negative_binomial",
        dispersion=100.0,
        tune_dispersion=True,
        dispersion_bounds=(0.05, 200.0),
        cv=4,
        max_iter=30,
    ).fit(x, y)
    assert est.dispersion_ < 20.0  # started close to the Poisson limit, truth is 1
    assert est.history_[0]["dispersion"] == pytest.approx(100.0)
    assert est.history_[-1]["dispersion"] != est.history_[0]["dispersion"]
    assert est.dispersion_ == est.loss_params_opt_["dispersion"]
    assert 0.05 <= est.dispersion_ <= 200.0
    assert est.score(x, y) > 0.0


def test_tune_dispersion_through_both_stages():
    x, y, _ = _data("negative_binomial", 120, seed=9)
    est = OptimizedKernelGLMRegressor(
        family="negative_binomial",
        tune_dispersion=True,
        blocks="features",
        cv=3,
        max_iter=4,
    ).fit(x, y)
    stages = {h["stage"] for h in est.history_}
    assert stages == {"global", "features"}
    assert all("dispersion" in h for h in est.history_)
    assert est.scaling_opt_.shape == (2,)


def test_invalid_optimizer_combinations():
    x, y, _ = _data("poisson", 60)
    with pytest.raises(ValueError, match="negative_binomial"):
        OptimizedKernelGLMRegressor(tune_dispersion=True).fit(x, y)
    with pytest.raises(ValueError, match="n_basis"):
        OptimizedKernelGLMRegressor(
            family="negative_binomial", tune_dispersion=True, n_basis=10
        ).fit(x, y)
    with pytest.raises(ValueError, match="dispersion_bounds"):
        OptimizedKernelGLMRegressor(
            family="negative_binomial", tune_dispersion=True, dispersion_bounds=(2.0, 1.0)
        ).fit(x, y)

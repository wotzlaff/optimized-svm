"""GLM loss families (binomial, negative binomial, gamma, Tweedie)."""

import numpy as np
import pytest
from sklearn.model_selection import KFold

from optimized_svm import TrainingProblem, cv_error_and_gradient, train
from optimized_svm.losses import BinomialLoss, GammaLoss, NegativeBinomialLoss, TweedieLoss
from optimized_svm.validation import (
    BinomialDeviance,
    GammaDeviance,
    NegativeBinomialDeviance,
    TweedieDeviance,
)

TIGHT_SMO = {"tol": 1e-10}
TIGHT_NEWTON = {"tol": 1e-26, "max_steps": 200}

LOSSES = {
    "binomial": BinomialLoss(),
    "nb": NegativeBinomialLoss(2.5),
    "gamma": GammaLoss(),
    "tweedie": TweedieLoss(1.4),
}
VALIDATIONS = {
    "binomial": BinomialDeviance(),
    "nb": NegativeBinomialDeviance(2.5),
    "gamma": GammaDeviance(),
    "tweedie": TweedieDeviance(1.4),
}
# targets (including boundary values where the family allows them)
TARGETS = {
    "binomial": np.array([0.0, 0.2, 0.5, 0.9, 1.0, 0.35]),
    "nb": np.array([0.0, 1.0, 3.0, 7.0, 0.0, 2.0]),
    "gamma": np.array([0.2, 0.7, 1.0, 2.5, 6.0, 1.3]),
    "tweedie": np.array([0.0, 0.3, 1.0, 2.5, 0.0, 5.0]),
}
WEIGHTS = np.array([1.0, 0.5, 2.0, 3.0, 1.5, 0.7])
T = np.array([-1.5, -0.4, 0.0, 0.3, 1.1, 0.8])
NAMES = list(LOSSES)


def fd(fun, t, h=1e-6):
    return (fun(t + h) - fun(t - h)) / (2 * h)


@pytest.mark.parametrize("name", NAMES)
def test_derivatives_match_finite_differences(name):
    loss, y = LOSSES[name], TARGETS[name]
    terms = loss.terms(y, WEIGHTS)
    for t0 in (-2.0, -0.5, 0.0, 0.7, 1.5):
        t = np.full(y.size, t0)
        np.testing.assert_allclose(
            loss.d1(t, terms), fd(lambda s: loss.value(s, terms), t), rtol=1e-6, atol=1e-7
        )
        np.testing.assert_allclose(
            loss.d2(t, terms), fd(lambda s: loss.d1(s, terms), t), rtol=1e-6, atol=1e-7
        )
        assert np.all(loss.d2(t, terms) > 0)


@pytest.mark.parametrize("name", NAMES)
def test_value_is_nonnegative_with_zero_minimum(name):
    loss, y = LOSSES[name], TARGETS[name]
    terms = loss.terms(y, WEIGHTS)
    grid = np.linspace(-6, 6, 101)
    vals = np.array([loss.value(np.full(y.size, g), terms) for g in grid])
    assert vals.min() > -1e-12
    pos = y > 0 if name != "binomial" else (y > 0) & (y < 1)
    # minimiser: mu = y (logit for binomial, log otherwise); skip the boundary targets
    t_min = np.log(y[pos] / (1 - y[pos])) if name == "binomial" else np.log(y[pos])
    sub = loss.terms(y[pos], WEIGHTS[pos])
    np.testing.assert_allclose(loss.value(t_min, sub), 0.0, atol=1e-12)
    np.testing.assert_allclose(loss.d1(t_min, sub), 0.0, atol=1e-12)
    # y = 0 (or 1): value goes to 0 asymptotically
    zero = y == (1.0 if name == "binomial" else 0.0)
    if zero.any():
        sub = loss.terms(y[zero], WEIGHTS[zero])
        t_far = np.full(zero.sum(), 12.0 if name == "binomial" else -40.0)
        assert np.all(loss.value(t_far, sub) < 1e-5)


def test_negative_binomial_param_ratio_matches_finite_differences():
    y, r = TARGETS["nb"], 2.5
    terms = NegativeBinomialLoss(r).terms(y, WEIGHTS)
    for t0 in (-1.0, 0.2, 1.3):
        t = np.full(y.size, t0)
        h = 1e-6
        dphi_dr = (
            NegativeBinomialLoss(r + h).d1(t, terms) - NegativeBinomialLoss(r - h).d1(t, terms)
        ) / (2 * h)
        ratio = NegativeBinomialLoss(r).param_ratio("dispersion", t, terms)
        np.testing.assert_allclose(
            ratio, dphi_dr / NegativeBinomialLoss(r).d2(t, terms), rtol=1e-6, atol=1e-8
        )
    with pytest.raises(ValueError, match="tunable"):
        NegativeBinomialLoss(r).param_ratio("nope", t, terms)
    for other in (BinomialLoss(), GammaLoss(), TweedieLoss()):
        with pytest.raises(ValueError, match="tunable"):
            other.param_ratio("power", t, other.terms(np.ones(1) * 0.5, np.ones(1)))


@pytest.mark.parametrize("name", NAMES)
def test_overflow_behaviour(name):
    loss, y = LOSSES[name], TARGETS[name]
    terms = loss.terms(y, WEIGHTS)
    for t0 in (800.0, -800.0):
        t = np.full(y.size, t0)
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            v, g, h = loss.value(t, terms), loss.d1(t, terms), loss.d2(t, terms)
        assert not np.any(np.isnan(v)) and not np.any(np.isnan(g)) and not np.any(np.isnan(h))
        assert np.all(v >= 0)
    if name in ("binomial", "nb"):  # logistic-type: everything stays finite
        t = np.full(y.size, 800.0)
        assert np.all(np.isfinite(loss.value(t, terms)))
        assert np.all(np.isfinite(loss.d1(t, terms)))


def test_domain_validation():
    one = np.ones(2)
    with pytest.raises(ValueError):
        GammaLoss().terms(np.array([1.0, 0.0]), one)
    with pytest.raises(ValueError):
        BinomialLoss().terms(np.array([0.5, 1.2]), one)
    with pytest.raises(ValueError):
        NegativeBinomialLoss(1.0).terms(np.array([1.0, -1.0]), one)
    with pytest.raises(ValueError):
        NegativeBinomialLoss(0.0).terms(one, one)
    with pytest.raises(ValueError):
        TweedieLoss(1.0).terms(one, one)
    with pytest.raises(ValueError):
        TweedieLoss(2.0).terms(one, one)
    with pytest.raises(ValueError):
        TweedieLoss(1.5).terms(np.array([1.0, -0.1]), one)


@pytest.mark.parametrize("name", NAMES)
def test_initial_bias(name):
    loss, y = LOSSES[name], TARGETS[name]
    terms = loss.terms(y, WEIGHTS)
    b = loss.initial_bias(terms)
    # intercept-only optimum for the binomial/log-link families with unit curvature structure:
    # the start is the closed-form (moment) estimate, so the gradient is small but not exactly 0
    if name in ("binomial", "nb"):
        g = float(loss.d1(np.full(y.size, b), terms).sum())
        assert abs(g) < 0.05 * WEIGHTS.sum()
    # all-zero counts: finite start
    if name in ("nb", "tweedie"):
        z = loss.terms(np.zeros(4), np.ones(4))
        assert np.isfinite(loss.initial_bias(z))
    # offset shifts the start
    off = np.full(y.size, 0.7)
    shifted = type(terms)(terms.kernel_index, terms.y, terms.w, terms.sign, off)
    assert loss.initial_bias(shifted) == pytest.approx(b - 0.7, abs=1e-9)


@pytest.mark.parametrize("name", NAMES)
def test_validation_gradient_and_consistency(name):
    val, loss, y = VALIDATIONS[name], LOSSES[name], TARGETS[name]
    f = np.array([-0.8, 0.1, 0.4, -0.2, 0.9, 0.3])
    w = WEIGHTS if val.weighted else None
    v, g = val(f, y, w)
    ones = np.ones_like(y) if w is None else w
    assert v == pytest.approx(float(loss.value(f, loss.terms(y, ones)).sum()))
    num = np.array(
        [(val(f + h, y, w)[0] - val(f - h, y, w)[0]) / 2e-6 for h in 1e-6 * np.eye(f.size)]
    )
    np.testing.assert_allclose(g, num, rtol=1e-6, atol=1e-7)
    assert val.weighted == (name == "binomial")
    if name != "binomial":  # weights are ignored
        assert val(f, y, WEIGHTS)[0] == pytest.approx(v)


def synthetic(name, n=80, seed=3):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3))
    eta = 0.6 * np.sin(x[:, 0]) + 0.3 * x[:, 1]
    if name == "binomial":
        trials = rng.integers(1, 12, size=n).astype(float)
        y = rng.binomial(trials.astype(int), 1 / (1 + np.exp(-eta))) / trials
        return x, y, trials
    mu = np.exp(0.5 + eta)
    if name == "nb":
        r = 2.5
        y = rng.poisson(rng.gamma(r, mu / r)).astype(float)
    elif name == "gamma":
        y = rng.gamma(4.0, mu / 4.0)
    else:  # compound Poisson-gamma with power 1.4
        p, phi = 1.4, 1.0
        lam = mu ** (2 - p) / (phi * (2 - p))
        alpha = (2 - p) / (p - 1)
        theta = phi * (p - 1) * mu ** (p - 1)
        k = rng.poisson(lam)
        y = np.array(
            [rng.gamma(a * alpha, th) if a > 0 else 0.0 for a, th in zip(k, theta, strict=True)]
        )
        assert (y == 0).any()
    return x, y, np.ones(n)


@pytest.mark.parametrize("name", NAMES)
def test_training_converges_from_zero_bias(name):
    x, y, w = synthetic(name)
    prob = TrainingProblem(0.6 * x, y, LOSSES[name], 0.05, w)
    sol = train(prob)
    assert sol.residual < 1e-12, sol.residual
    assert abs(sol.a.sum()) < 1e-9
    dec = prob.decision(sol.b, sol.c, sol.ka)
    assert np.all(np.isfinite(dec))
    # the fit must beat the intercept-only model
    terms = prob.terms
    const = np.full(y.size, LOSSES[name].initial_bias(terms))
    assert LOSSES[name].value(dec, terms).sum() < LOSSES[name].value(const, terms).sum()


@pytest.mark.parametrize("name", NAMES)
def test_hypergradient_matches_finite_differences(name):
    x, y, w = synthetic(name, n=60)
    folds = list(KFold(3, shuffle=True, random_state=0).split(x))
    loss = LOSSES[name]
    theta = np.r_[np.log(0.05), np.log(0.6)]

    def f(th):
        return cv_error_and_gradient(
            th, x, y, w, folds, loss, VALIDATIONS[name], None, None, TIGHT_SMO, TIGHT_NEWTON
        )

    res = f(theta)
    eps = 1e-6
    num = np.array(
        [(f(theta + eps * e).error - f(theta - eps * e).error) / (2 * eps) for e in np.eye(2)]
    )
    np.testing.assert_allclose(res.gradient, num, rtol=1e-4, atol=1e-7)


def test_hypergradient_wrt_dispersion():
    x, y, w = synthetic("nb", n=60)
    folds = list(KFold(3, shuffle=True, random_state=0).split(x))
    loss = NegativeBinomialLoss(2.5)
    theta = np.r_[np.log(0.05), np.log(0.6), np.log(2.5)]

    def f(th):
        return cv_error_and_gradient(
            th, x, y, w, folds, loss, VALIDATIONS["nb"], None, None, TIGHT_SMO, TIGHT_NEWTON,
            loss_params=("dispersion",),
        )  # fmt: skip

    res = f(theta)
    eps = 1e-6
    num = np.array(
        [(f(theta + eps * e).error - f(theta - eps * e).error) / (2 * eps) for e in np.eye(3)]
    )
    np.testing.assert_allclose(res.gradient, num, rtol=1e-4, atol=1e-7)
    assert abs(res.gradient[-1]) > 1e-6

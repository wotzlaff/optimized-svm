import numpy as np
import pytest

from optimized_svm.losses import (
    ApproxPoissonLoss,
    EpsilonInsensitiveLoss,
    HingeLoss,
    LogisticLoss,
    PoissonLoss,
    SquaredLoss,
)
from optimized_svm.smooth_max import smooth_max

T = np.linspace(-2, 2, 4001)


@pytest.mark.parametrize("degree", ["2", "3", "6"])
@pytest.mark.parametrize("delta", [0.1, 0.5, 1.0])
def test_smooth_max_properties(degree, delta):
    m, dm, ddm = smooth_max(delta, degree)
    # equals max(0, t) outside (-delta, delta), above it inside, and convex
    outside = np.abs(T) >= delta
    np.testing.assert_allclose(m(T)[outside], np.maximum(T, 0)[outside])
    assert np.all(m(T) >= np.maximum(T, 0) - 1e-15)
    assert np.all(m(T) - np.maximum(T, 0) <= delta / 2)
    assert np.all(ddm(T) >= 0)
    # derivatives agree with finite differences and are continuous
    h = 1e-6
    # central differences are only O(h / delta) accurate where ddm jumps (t = +-delta)
    np.testing.assert_allclose(dm(T), (m(T + h) - m(T - h)) / (2 * h), atol=1e-5 / delta)
    inner = np.abs(np.abs(T) - delta) > 1e-3  # ddm may jump at +-delta (degree 2)
    np.testing.assert_allclose(
        ddm(T)[inner], ((dm(T + h) - dm(T - h)) / (2 * h))[inner], atol=1e-4 / delta
    )
    assert np.max(np.abs(np.diff(dm(T)))) < 0.01 / delta


def test_smooth_max_invalid():
    with pytest.raises(ValueError):
        smooth_max(-1.0)
    with pytest.raises(ValueError):
        smooth_max(0.5, "4")
    m, dm, ddm = smooth_max(0.0)
    np.testing.assert_array_equal(m(np.array([-1.0, 2.0])), [0.0, 2.0])


LOSSES = [
    (HingeLoss(1.0, 0.3, "2"), "labels"),
    (HingeLoss(0.5, 0.4, "6"), "labels"),
    (EpsilonInsensitiveLoss(0.2, 0.3, "3"), "real"),
    (LogisticLoss(), "labels"),
    (SquaredLoss(), "real"),
    (PoissonLoss(), "counts"),
    (ApproxPoissonLoss(), "counts"),
]


@pytest.mark.parametrize("loss,target", LOSSES)
def test_loss_derivatives(loss, target):
    rng = np.random.default_rng(0)
    n = 50
    y = {
        "labels": np.where(rng.random(n) > 0.5, 1.0, -1.0),
        "real": rng.normal(size=n),
        "counts": rng.poisson(2.0, size=n).astype(float),
    }[target]
    terms = loss.terms(y, rng.uniform(0.5, 2.0, size=n))
    t = rng.normal(size=terms.size)
    h = 1e-6
    np.testing.assert_allclose(
        loss.d1(t, terms),
        (loss.value(t + h, terms) - loss.value(t - h, terms)) / (2 * h),
        rtol=1e-5,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        loss.d2(t, terms),
        (loss.d1(t + h, terms) - loss.d1(t - h, terms)) / (2 * h),
        rtol=1e-4,
        atol=1e-4,
    )
    assert np.all(loss.d2(t, terms) >= 0)


def test_epsilon_terms_structure():
    terms = EpsilonInsensitiveLoss().terms(np.array([1.0, 2.0]), np.ones(2))
    np.testing.assert_array_equal(terms.kernel_index, [0, 1, 0, 1])
    np.testing.assert_array_equal(terms.sign, [1, 1, -1, -1])


def test_logistic_loss_is_stable_for_large_margins():
    terms = LogisticLoss().terms(np.array([1.0, -1.0, 1.0]), np.ones(3))
    t = np.array([800.0, 800.0, -800.0])  # margins +800, -800, -800
    loss = LogisticLoss()
    np.testing.assert_allclose(loss.value(t, terms), [0.0, 800.0, 800.0], atol=1e-12)
    np.testing.assert_allclose(loss.d1(t, terms), [0.0, 1.0, -1.0], atol=1e-12)
    assert np.all(np.isfinite(loss.d2(t, terms)))

"""Stable soft-max solves against original equations and independent convex reference."""

import numpy as np
import pytest
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold

from optimized_svm.multiclass import (
    MulticlassProblem,
    MultinomialNLL,
    SoftmaxFactorization,
    cv_error_and_gradient_multiclass,
    newton_multiclass,
)


def data():
    rng = np.random.default_rng(88)
    x = rng.normal(size=(24, 3))
    y = np.arange(24) % 3
    w = np.ones(24)
    w[::8] = 0
    return x, y, w


def test_spd_reduction_solves_original_forward_and_adjoint_equations():
    x, y, w = data()
    x[2] = x[1]
    prob = MulticlassProblem(x[:9], y[:9], 3, 0.04, w[:9])
    rng = np.random.default_rng(21)
    h = prob.curvature(rng.normal(size=(9, 3)))
    factor = SoftmaxFactorization.build(prob, h)
    ra, rb = rng.normal(size=(9, 3)), rng.normal(size=3)
    da, db = factor.solve(ra, rb)
    jac = prob.jacobian(h)
    np.testing.assert_allclose(jac @ np.r_[da.ravel(), db], np.r_[ra.ravel(), rb], atol=2e-11)
    ref = np.linalg.solve(jac.T, np.r_[ra.ravel(), rb])
    expected = np.einsum("ikl,il->ik", h, ref[:-3].reshape(9, 3) + ref[-3:])
    np.testing.assert_allclose(factor.adjoint_curvature(ra, rb), expected, atol=2e-11)


@pytest.mark.parametrize("lam", [1e-6, 1e-10])
@pytest.mark.parametrize("scale", [1.0, 0.03])
def test_extreme_regularization_wide_kernel_and_zero_weights(lam, scale):
    x, y, w = data()
    x[2] = x[1]  # singular kernel, without altering the model with jitter
    prob = MulticlassProblem(scale * x, y, 3, lam, w)
    # Check the advertised tolerance: duplicates/wide kernels can have a residual floor
    # around 1e-17 even with extended accumulation, depending on NumPy/BLAS rounding.
    sol = newton_multiclass(prob, tol=1e-12, max_steps=200)
    assert np.isfinite(prob.objective(sol.a, sol.b, sol.ka))
    assert np.all(np.isfinite(sol.t))
    assert sol.residual < 1e-12
    fa, fb = prob.stationarity(sol.a, sol.b, sol.t)
    np.testing.assert_allclose(sol.residual, np.mean(fa**2) + np.sum(fb**2), rtol=1e-13)
    np.testing.assert_allclose(sol.a.sum(axis=1), 0, atol=1e-10)
    np.testing.assert_allclose(sol.a[w == 0], 0, atol=1e-10)
    assert abs(sol.b.sum()) < 1e-9


def test_small_lambda_hypergradient_matches_finite_differences():
    x, y, w = data()
    folds = list(StratifiedKFold(3, shuffle=True, random_state=4).split(x, y))
    theta = np.log([1e-6, 0.7])

    def fun(t):
        return cv_error_and_gradient_multiclass(
            t,
            x,
            y,
            w,
            folds,
            n_classes=3,
            newton_options={"tol": 1e-25, "max_steps": 200},
        )

    result = fun(theta)
    eps = 1e-4
    numeric = [(fun(theta + d).error - fun(theta - d).error) / (2 * eps) for d in np.eye(2) * eps]
    np.testing.assert_allclose(result.gradient, numeric, rtol=2e-4, atol=2e-5)


def test_training_objective_matches_independent_primal_lbfgs():
    x, y, w = data()
    x[2] = x[1]
    prob = MulticlassProblem(x, y, 3, 0.01, w)
    sol = newton_multiclass(prob, tol=1e-24)
    eigenvalues, vectors = np.linalg.eigh(prob.k / prob.lam)
    root = vectors * np.sqrt(np.maximum(eigenvalues, 0))
    validation = MultinomialNLL(weighted=True)

    def objective(z):
        u, b = z[:-3].reshape(24, 3), z[-3:]
        error, derivative = validation(root @ u + b, y, w)
        return 0.5 * np.sum(u**2) + error, np.r_[
            (u + root.T @ derivative).ravel(), derivative.sum(axis=0)
        ]

    reference = minimize(
        objective,
        np.zeros(75),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 2000, "gtol": 1e-10, "ftol": 1e-14},
    )
    assert reference.success
    np.testing.assert_allclose(prob.objective(sol.a, sol.b, sol.ka), reference.fun, atol=1e-9)


def test_saturated_gradient_and_curvature_retain_small_probabilities():
    prob = MulticlassProblem(np.arange(3)[:, None], np.arange(3), 3, 1.0)
    t = np.array([[50.0, 0.0, -10.0], [-10.0, 50.0, 0.0], [0.0, -10.0, 50.0]])
    gradient = prob.gradient(t)
    h = prob.curvature(t)
    assert np.all(gradient[np.arange(3), np.arange(3)] < 0)
    assert np.all(h[np.arange(3), np.arange(3), np.arange(3)] > 0)
    assert prob.loss(t) > 0
    np.testing.assert_allclose(gradient.sum(axis=1), 0, atol=1e-36)
    np.testing.assert_allclose(h.sum(axis=2), 0, atol=1e-36)


def test_class_with_zero_effective_weight_has_no_finite_bias():
    with pytest.raises(ValueError, match="positive total training weight"):
        MulticlassProblem(np.zeros((3, 1)), np.arange(3), 3, 1.0, np.array([1.0, 1.0, 0.0]))


def test_extremes_without_extended_accumulation(monkeypatch):
    import optimized_svm.multiclass as multiclass

    monkeypatch.setattr(multiclass, "_kernel_product", lambda k, a, lam: k @ a / lam)
    x, y, w = data()
    x[2] = x[1]
    for scale in (1.0, 0.03):
        for lam in (1e-6, 1e-10):
            prob = MulticlassProblem(scale * x, y, 3, lam, w)
            sol = newton_multiclass(prob, max_steps=200)
            assert sol.residual < 1e-10
            assert np.isfinite(prob.objective(sol.a, sol.b, sol.ka))


def test_completely_saturated_bias_schur_is_rejected():
    x, y, w = data()
    prob = MulticlassProblem(x, y, 3, 0.1, w)
    h = prob.curvature(np.tile([1000.0, -1000.0, -1000.0], (24, 1)))
    assert np.count_nonzero(h) == 0
    with pytest.raises(np.linalg.LinAlgError):
        SoftmaxFactorization.build(prob, h)

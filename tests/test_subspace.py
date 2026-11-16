"""Subspace SVM: equivalence with the dual problem with the projected kernel, hypergradient."""

import warnings
from pathlib import Path

import numpy as np
import pytest
import smo_solver
from sklearn.model_selection import KFold

from optimized_svm import EpsilonInsensitiveLoss, HingeLoss, LogisticLoss, SquaredLoss
from optimized_svm.kernels import rbf, scale_features
from optimized_svm.subspace import (
    SubspaceProblem,
    _newton_subspace,
    choose_basis,
    fold_error_and_gradient_subspace,
    newton_subspace,
    nystroem_residual,
    select_basis,
)
from optimized_svm.training import ConvergenceError
from optimized_svm.validation import SmoothAbsoluteValidation, SmoothHingeValidation


def make(n=120, d=3, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d))
    y = np.where(x[:, 0] + 0.5 * x[:, 1] ** 2 - 0.5 + 0.3 * rng.normal(size=n) > 0, 1.0, -1.0)
    t = np.sin(x[:, 0]) + 0.3 * x[:, 1] + 0.1 * rng.normal(size=n)
    return 0.6 * x, y, t


def test_projected_kernel_dual_gives_the_same_model():
    # Cor. 5.9: the dual of the subspace problem is the standard dual with K_s = K_XZ K_ZZ^-1 K_ZX
    x, y, _ = make()
    z = x[choose_basis(len(x), 30, seed=1)]
    lam, delta = 0.05, 0.4
    prob = SubspaceProblem(x, z, y, HingeLoss(smoothing=delta), lam)
    sol = newton_subspace(prob)
    assert sol.residual < 1e-12
    k_s = prob.psi @ prob.psi.T
    ref = smo_solver.solve_classification(
        k_s, y, lam, kernel="precomputed", smoothing=delta, tol=1e-10, shrinking=False
    )
    f_ref = np.asarray(ref.ka) + ref.b
    np.testing.assert_allclose(sol.f, f_ref, atol=1e-6)
    np.testing.assert_allclose(sol.b, ref.b, atol=1e-6)
    # prediction through the basis points reproduces the training decision values
    np.testing.assert_allclose(rbf(x, z) @ sol.beta + sol.b, sol.f, atol=1e-6)


def test_full_basis_recovers_the_full_model():
    x, y, _ = make(60)
    lam, delta = 0.05, 0.4
    sub = newton_subspace(SubspaceProblem(x, x, y, HingeLoss(smoothing=delta), lam))
    full = smo_solver.solve_classification(
        x, y, lam, gamma=1.0, smoothing=delta, tol=1e-10, shrinking=False
    )
    np.testing.assert_allclose(sub.f, np.asarray(full.ka) + full.b, atol=1e-5)


@pytest.mark.parametrize("name", ["hinge", "eps", "logistic", "squared"])
def test_subspace_newton_converges_for_all_losses(name):
    x, y, t = make()
    loss, target = {
        "hinge": (HingeLoss(smoothing=0.5, degree="3"), y),
        "eps": (EpsilonInsensitiveLoss(0.1, 0.3), t),
        "logistic": (LogisticLoss(), y),
        "squared": (SquaredLoss(), t),
    }[name]
    z = x[choose_basis(len(x), 25)]
    sol = newton_subspace(SubspaceProblem(x, z, target, loss, 0.02))
    assert sol.residual < 1e-12 and sol.newton_steps < 40


def test_nonsmooth_loss_is_rejected():
    x, y, _ = make(30)
    with pytest.raises(ValueError, match="smooth"):
        SubspaceProblem(x, x[:5], y, HingeLoss(smoothing=0.0), 0.1)


CASES = {
    "hinge": (HingeLoss(smoothing=0.5), "y", None),
    "hinge-ard": (HingeLoss(smoothing=0.4, degree="6"), "y", [[0], [1], [2]]),
    "eps": (EpsilonInsensitiveLoss(0.1, 0.3, "3"), "t", None),
    "eps-blocks": (EpsilonInsensitiveLoss(0.05, 0.3), "t", [[0, 1], [2]]),
    "logistic": (LogisticLoss(), "y", None),
    "squared": (SquaredLoss(), "t", [[0], [1, 2]]),
}


@pytest.mark.parametrize("name", list(CASES))
def test_subspace_hypergradient_matches_finite_differences(name):
    loss, which, blocks = CASES[name]
    x, y, t = make(90)
    target = y if which == "y" else t
    validation = SmoothHingeValidation() if which == "y" else SmoothAbsoluteValidation()
    folds = list(KFold(3, shuffle=True, random_state=0).split(x))
    bases = [tr[choose_basis(len(tr), 20, seed=k)] for k, (tr, _) in enumerate(folds)]
    nb = 1 if blocks is None else len(blocks)
    theta = np.r_[np.log(0.05), np.full(nb, np.log(0.6))]
    tight = {"tol": 1e-26, "max_steps": 200}

    def f(th):
        xs = scale_features(x, np.exp(th[1:]), blocks)
        total, grad = 0.0, np.zeros_like(th)
        for (tr, va), basis in zip(folds, bases, strict=True):
            r = fold_error_and_gradient_subspace(
                xs[tr], target[tr], np.ones(len(tr)), xs[basis], xs[va], target[va],
                loss, validation, float(np.exp(th[0])), blocks, tight,
            )  # fmt: skip
            total += r.error
            grad += r.gradient
        return total, grad

    err, grad = f(theta)
    eps = 1e-6
    fd = np.array(
        [(f(theta + eps * e)[0] - f(theta - eps * e)[0]) / (2 * eps) for e in np.eye(theta.size)]
    )
    np.testing.assert_allclose(grad, fd, rtol=1e-5, atol=1e-7)
    assert np.all(np.abs(grad) > 1e-6)


def test_select_basis_reaches_the_tolerance():
    x = make(200)[0] * 0.3
    idx, ratio = select_basis(x, lam=0.05, tol=1.0, max_size=200)
    assert ratio <= 1.0 and len(set(idx)) == len(idx) and np.all(np.diff(idx) > 0)
    # the reported ratio is the Nystroem residual trace divided by lam
    k = rbf(x, x)
    resid = k - k[:, idx] @ np.linalg.solve(k[np.ix_(idx, idx)] + 1e-10 * np.eye(len(idx)), k[idx])
    assert np.trace(resid) / 0.05 == pytest.approx(ratio, rel=1e-4, abs=1e-6)
    assert len(select_basis(x, lam=0.05, tol=10.0, max_size=200)[0]) <= len(idx)


def test_select_basis_respects_the_size_limit():
    x = make(200)[0]
    idx, ratio = select_basis(x, lam=1e-6, tol=1.0)
    assert len(idx) == 20 and ratio > 1.0
    with pytest.raises(ValueError):
        select_basis(x, lam=-1.0)


# --- estimators -------------------------------------------------------------------------------

from optimized_svm import OptimizedSVC, OptimizedSVR, SmoothSVC, SmoothSVR  # noqa: E402


def _reg_data(n=300, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, (n, 3))
    return x, np.sin(2 * x[:, 0]) + 0.5 * x[:, 1] ** 2 + 0.05 * rng.normal(size=n)


def test_smooth_svr_subspace_matches_the_full_model():
    x, y = _reg_data()
    full = SmoothSVR(lam=1e-3, scaling=0.3).fit(x, y)
    sub = SmoothSVR(lam=1e-3, scaling=0.3, n_basis="auto", basis_tol=10.0).fit(x, y)
    assert len(sub.basis_) < len(x) and sub.n_support_ == len(sub.basis_)
    assert np.max(np.abs(sub.predict(x) - full.predict(x))) < 0.1
    fixed = SmoothSVR(lam=1e-3, scaling=0.3, n_basis=40).fit(x, y)
    assert len(fixed.basis_) == 40
    assert fixed.score(x, y) > 0.9


def test_smooth_svc_subspace_and_validation():
    x, y = _reg_data()
    labels = (y > np.median(y)).astype(int)
    clf = SmoothSVC(lam=1e-2, scaling=0.5, n_basis=30).fit(x, labels)
    assert clf.score(x, labels) > 0.9
    with pytest.raises(ValueError, match="smooth"):
        SmoothSVC(smoothing=0.0, n_basis=10).fit(x, labels)
    with pytest.raises(ValueError, match="max_asum"):
        SmoothSVR(max_asum=1.0, n_basis=10).fit(x, y)
    with pytest.raises(ValueError, match="n_basis"):
        SmoothSVR(n_basis=0).fit(x, y)
    with pytest.raises(ValueError, match="n_basis"):
        SmoothSVR(n_basis="many").fit(x, y)


def test_auto_basis_falls_back_to_the_full_model_with_a_warning():
    x, y = _reg_data(100)
    with pytest.warns(UserWarning, match="full model"):
        est = SmoothSVR(lam=1e-8, scaling=3.0, n_basis="auto").fit(x, y)
    assert not hasattr(est, "basis_")


def test_optimized_svr_with_a_basis():
    x, y = _reg_data(400)
    full = OptimizedSVR(max_iter=15, cv=3).fit(x, y)
    sub = OptimizedSVR(max_iter=15, cv=3, n_basis=60).fit(x, y)
    assert sub.history_[0]["n_basis"] == 60 and len(sub.basis_) == 60
    assert sub.cv_error_ < 1.5 * full.cv_error_ + 0.02
    assert sub.score(x, y) > 0.9
    # ARD blocks: two stages with a basis re-selected for each
    ard = OptimizedSVR(max_iter=8, cv=3, n_basis=60, blocks="features").fit(x, y)
    assert len(ard.scaling_opt_) == 3


def test_optimized_svc_with_a_basis_and_one_vs_rest():
    x, y = _reg_data(300)
    labels = np.digitize(y, np.quantile(y, [1 / 3, 2 / 3]))
    clf = OptimizedSVC(max_iter=8, cv=3, n_basis=50).fit(x, labels)
    assert clf.score(x, labels) > 0.8


def test_nystroem_residual_matches_the_dense_trace():
    x, _, _ = make(60)
    idx = choose_basis(60, 12, seed=1)
    k_xz = rbf(x, x[idx])
    dense = 60 - np.trace(k_xz @ np.linalg.solve(rbf(x[idx], x[idx]), k_xz.T))
    assert nystroem_residual(x, idx) == pytest.approx(dense, abs=1e-6)
    assert nystroem_residual(x, np.arange(60)) < 1e-6


def _adaptive_data(n=200):
    rng = np.random.default_rng(3)
    x = rng.uniform(-3, 3, size=(n, 1))
    return x, np.sin(2 * x[:, 0]) + 0.05 * rng.normal(size=n)


def test_auto_basis_is_reselected_when_lambda_drifts():
    x, y = _adaptive_data()
    est = OptimizedSVR(
        n_basis="auto", basis_tol=10.0, lam_init=1.0, scaling_init=0.5, max_iter=30, cv=3
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        est.fit(x, y)
    rounds = [h["basis_round"] for h in est.history_]
    assert max(rounds) >= 1 and rounds == sorted(rounds)
    assert est.n_iter_ >= max(rounds) + 1


def test_integer_basis_is_never_reselected():
    x, y = _adaptive_data()
    est = OptimizedSVR(n_basis=30, lam_init=1.0, scaling_init=0.5, max_iter=30, cv=3).fit(x, y)
    assert {h["basis_round"] for h in est.history_} == {0}


@pytest.mark.parametrize("n_basis", [None, 30, "auto"])
def test_cv_error_is_the_error_at_the_returned_hyperparameters(n_basis):
    x, y = _adaptive_data()
    est = OptimizedSVR(
        n_basis=n_basis, basis_tol=10.0, lam_init=1.0, scaling_init=0.5, max_iter=15, cv=3
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        est.fit(x, y)
    match = [
        h
        for h in est.history_
        if h["lam"] == pytest.approx(est.lam_, rel=1e-12)
        and np.allclose(h["scaling"], est.scaling_opt_, rtol=1e-12)
    ]
    assert match and est.cv_error_ == pytest.approx(match[-1]["error"])


def test_newton_falls_back_to_a_cold_start_from_a_hopeless_warm_start():
    # Captured during HPO (Friedman #1, n_basis="auto", lam = 1e-6): the warm start has coefficients
    # of order 1e6, the loss is linear in all samples (no curvature but lam) and the damped Newton
    # method finds no descent step; the cold start u = 0 converges in a few steps.
    data = np.load(Path(__file__).parent / "data_subspace_warm_start.npz")
    loss = EpsilonInsensitiveLoss(0.1, 0.5)
    prob = SubspaceProblem(data["x"], data["z"], data["y"], loss, float(data["lam"]))
    with pytest.raises(ConvergenceError):
        _newton_subspace(prob, data["beta0"], 0.0)
    cold = newton_subspace(prob)
    sol = newton_subspace(prob, data["beta0"], 0.0)
    assert sol.residual < 1e-12
    np.testing.assert_allclose(sol.f, cold.f, atol=1e-6)

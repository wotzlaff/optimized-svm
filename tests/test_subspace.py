"""Subspace SVM: equivalence with the dual problem with the projected kernel, hypergradient."""

import numpy as np
import pytest
import smo_solver
from sklearn.model_selection import KFold

from optimized_svm import EpsilonInsensitiveLoss, HingeLoss, LogisticLoss, SquaredLoss
from optimized_svm.kernels import rbf, scale_features
from optimized_svm.subspace import (
    SubspaceProblem,
    choose_basis,
    fold_error_and_gradient_subspace,
    newton_subspace,
    select_basis,
)
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

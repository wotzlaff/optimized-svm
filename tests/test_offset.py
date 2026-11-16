"""Offsets in the training problem, loss-specific initial bias, weighted validation losses."""

from dataclasses import dataclass

import numpy as np
import pytest
from sklearn.model_selection import KFold

from optimized_svm import (
    HingeLoss,
    PoissonLoss,
    SquaredLoss,
    cv_error_and_gradient,
)
from optimized_svm.kernels import scale_features
from optimized_svm.problem import TrainingProblem
from optimized_svm.subspace import (
    SubspaceProblem,
    choose_basis,
    fold_error_and_gradient_subspace,
    newton_subspace,
)
from optimized_svm.training import newton, train
from optimized_svm.validation import (
    PoissonValidation,
    SmoothHingeValidation,
    SquaredValidation,
    ValidationLoss,
)

TIGHT_NEWTON = {"tol": 1e-26, "max_steps": 200}


@dataclass(frozen=True)
class WeightedSquaredValidation(ValidationLoss):
    """``1/2 sum_v w_v (f_v - y_v)^2`` with per-sample weights."""

    weighted = True

    def __call__(self, f, y, w=None):
        w = np.ones_like(y) if w is None else w
        e = f - y
        return 0.5 * float(w @ e**2), w * e


def _offset(data):
    x = data[0]
    return 0.4 * np.cos(2.0 * x[:, 1]) + 0.2


def _cases(data):
    _, _, t, counts = data
    return {
        "poisson": (PoissonLoss(), PoissonValidation(), counts),
        "squared": (SquaredLoss(), SquaredValidation(), t),
        "squared-weighted": (SquaredLoss(), WeightedSquaredValidation(), t),
    }


@pytest.mark.parametrize("name", ["poisson", "squared", "squared-weighted"])
def test_full_hypergradient_with_offset_matches_finite_differences(name, data):
    loss, validation, target = _cases(data)[name]
    x = data[0]
    offset = _offset(data)
    rng = np.random.default_rng(3)
    w = rng.uniform(0.5, 2.0, size=len(target))
    folds = list(KFold(3, shuffle=True, random_state=0).split(x))
    theta = np.r_[np.log(0.05), np.log(0.6)]

    def f(th):
        return cv_error_and_gradient(
            th, x, target, w, folds, loss, validation,
            newton_options=TIGHT_NEWTON, offset=offset,
        )  # fmt: skip

    res = f(theta)
    eps = 1e-6
    fd = np.array(
        [(f(theta + eps * e).error - f(theta - eps * e).error) / (2 * eps) for e in np.eye(2)]
    )
    np.testing.assert_allclose(res.gradient, fd, rtol=1e-5, atol=1e-7)
    # the offset must matter
    assert abs(res.error - f_no_offset(theta, x, target, w, folds, loss, validation)) > 1e-3


def f_no_offset(theta, x, target, w, folds, loss, validation):
    return cv_error_and_gradient(
        theta, x, target, w, folds, loss, validation, newton_options=TIGHT_NEWTON
    ).error


@pytest.mark.parametrize("name", ["poisson", "squared"])
def test_subspace_hypergradient_with_offset_matches_finite_differences(name, data):
    loss, validation, target = _cases(data)[name]
    x = data[0]
    offset = _offset(data)
    folds = list(KFold(3, shuffle=True, random_state=0).split(x))
    bases = [tr[choose_basis(len(tr), 20, seed=k)] for k, (tr, _) in enumerate(folds)]
    theta = np.r_[np.log(0.05), np.log(0.6)]

    def f(th):
        xs = scale_features(x, np.exp(th[1:]), None)
        total, grad = 0.0, np.zeros_like(th)
        for (tr, va), basis in zip(folds, bases, strict=True):
            r = fold_error_and_gradient_subspace(
                xs[tr], target[tr], np.ones(len(tr)), xs[basis], xs[va], target[va],
                loss, validation, float(np.exp(th[0])), None, TIGHT_NEWTON,
                None, offset[tr], None, offset[va],
            )  # fmt: skip
            total += r.error
            grad += r.gradient
        return total, grad

    _, grad = f(theta)
    eps = 1e-6
    fd = np.array([(f(theta + eps * e)[0] - f(theta - eps * e)[0]) / (2 * eps) for e in np.eye(2)])
    np.testing.assert_allclose(grad, fd, rtol=1e-5, atol=1e-7)


def test_subspace_cv_dispatch_passes_offset_and_weights(data):
    x, _, t, _ = data
    offset = _offset(data)
    w = np.random.default_rng(0).uniform(0.5, 2.0, size=len(t))
    folds = list(KFold(3, shuffle=True, random_state=0).split(x))
    bases = [tr[choose_basis(len(tr), 20, seed=k)] for k, (tr, _) in enumerate(folds)]
    theta = np.r_[np.log(0.05), np.log(0.6)]

    def f(th):
        return cv_error_and_gradient(
            th, x, t, w, folds, SquaredLoss(), WeightedSquaredValidation(),
            newton_options=TIGHT_NEWTON, basis=bases, offset=offset,
        )  # fmt: skip

    res = f(theta)
    eps = 1e-6
    fd = np.array(
        [(f(theta + eps * e).error - f(theta - eps * e).error) / (2 * eps) for e in np.eye(2)]
    )
    np.testing.assert_allclose(res.gradient, fd, rtol=1e-5, atol=1e-7)


def test_constant_offset_with_squared_loss_equals_shifting_y(data):
    x, _, t, _ = data
    c = 0.7
    folds = list(KFold(3, shuffle=True, random_state=0).split(x))
    w = np.ones(len(t))
    theta = np.r_[np.log(0.05), np.log(0.6)]
    with_offset = cv_error_and_gradient(
        theta, x, t, w, folds, SquaredLoss(), SquaredValidation(),
        newton_options=TIGHT_NEWTON, offset=np.full(len(t), c),
    )  # fmt: skip
    # offset c: f = g + c with g fitted to y - c; the validation compares g + c with y
    shifted = cv_error_and_gradient(
        theta, x, t - c, w, folds, SquaredLoss(), SquaredValidation(), newton_options=TIGHT_NEWTON
    )
    assert with_offset.error == pytest.approx(shifted.error, rel=1e-9)
    np.testing.assert_allclose(with_offset.gradient, shifted.gradient, rtol=1e-8, atol=1e-10)
    # the fitted models agree too
    a = [r.solution.a for r in with_offset.folds]
    b = [r.solution.a for r in shifted.folds]
    for ai, bi in zip(a, b, strict=True):
        np.testing.assert_allclose(ai, bi, atol=1e-8)


def test_weights_are_only_passed_to_weighted_validations(data):
    x, y, _, _ = data
    seen = []

    class Spy(SmoothHingeValidation):
        def __call__(self, f, y, w=None):
            seen.append(w)
            return super().__call__(f, y)

    folds = list(KFold(3, shuffle=True, random_state=0).split(x))
    w = np.linspace(1.0, 2.0, len(y))
    theta = np.r_[np.log(0.05), np.log(0.6)]
    cv_error_and_gradient(theta, x, y, w, folds, HingeLoss(smoothing=0.5), Spy())
    assert all(s is None for s in seen)
    Spy.weighted = True  # type: ignore[misc]
    seen.clear()
    cv_error_and_gradient(theta, x, y, w, folds, HingeLoss(smoothing=0.5), Spy())
    for s, (_, va) in zip(seen, folds, strict=True):
        np.testing.assert_array_equal(s, w[va])


def test_offset_is_rejected_for_smo_losses(data):
    x, y, _, _ = data
    with pytest.raises(ValueError, match="offset"):
        TrainingProblem(x, y, HingeLoss(), 0.1, offset=np.zeros(len(y)))
    with pytest.raises(ValueError, match="offset"):
        SubspaceProblem(x, x[:5], y, HingeLoss(smoothing=0.5), 0.1, offset=np.zeros(len(y)))


def test_offset_enters_the_decision_values(data):
    x, _, _, counts = data
    off = _offset(data)
    prob = TrainingProblem(x, counts, PoissonLoss(), 0.1, offset=off)
    ka = np.linspace(-1, 1, len(counts))
    np.testing.assert_allclose(prob.decision(0.3, 0.0, ka), ka + 0.3 + off)
    sol = newton(prob, np.zeros(prob.size), 0.0, 0.0, tol=1e-24)
    assert sol.residual < 1e-20


def test_initial_bias_is_the_minimiser_for_zero_coefficients(data):
    _, _, t, counts = data
    off = _offset(data)
    w = np.random.default_rng(0).uniform(0.5, 2.0, size=len(t))
    for loss, y in ((PoissonLoss(), counts), (SquaredLoss(), t)):
        for o in (None, off):
            terms = loss.terms(y, w)
            if o is not None:
                import dataclasses

                terms = dataclasses.replace(terms, offset=o)
            b = loss.initial_bias(terms)
            grad = float(loss.d1(b + (0.0 if o is None else o), terms).sum())
            assert abs(grad) < 1e-9 * w.sum()
    # all-zero counts: finite start
    zero = PoissonLoss().terms(np.zeros(5), np.ones(5))
    assert np.isfinite(PoissonLoss().initial_bias(zero))
    assert HingeLoss().initial_bias(HingeLoss().terms(np.ones(3), np.ones(3))) == 0.0


@dataclass(frozen=True)
class DomainPoisson(PoissonLoss):
    """Poisson loss that is ``inf`` for ``t > bound`` (exercises the line search)."""

    bound: float = 3.0
    start: float = -6.0

    def initial_bias(self, terms):
        return self.start

    def value(self, t, terms):
        v = super().value(t, terms)
        return np.where(t > self.bound, np.inf, v)


def test_newton_line_searches_reject_infinite_objective_values(data):
    x, _, _, counts = data
    loss = DomainPoisson()
    # from b = -6 the first full Newton step jumps far beyond the bound
    prob = TrainingProblem(x, counts, loss, 0.5)
    b0 = loss.initial_bias(prob.terms)
    sol = newton(prob, np.zeros(prob.size), b0, 0.0, tol=1e-24, max_steps=200)
    assert sol.residual < 1e-20 and np.isfinite(sol.b)
    assert sol.newton_steps > 1
    assert np.all(np.isfinite(loss.value(prob.decision(sol.b, 0.0, sol.ka), prob.terms)))
    sol2 = train(prob, newton_options={"tol": 1e-24, "max_steps": 200})
    assert sol2.b == pytest.approx(sol.b, abs=1e-8)

    sub = SubspaceProblem(x, x[::6], counts, loss, 0.5)
    ssol = newton_subspace(sub, tol=1e-24, max_steps=200)
    assert ssol.residual < 1e-20 and np.isfinite(ssol.b)
    assert np.all(np.isfinite(ssol.f))

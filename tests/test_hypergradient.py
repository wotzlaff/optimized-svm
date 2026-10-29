"""The hypergradient must agree with central finite differences of the CV error."""

import numpy as np
import pytest
from sklearn.model_selection import KFold

from optimized_svm import (
    EpsilonInsensitiveLoss,
    HingeLoss,
    LogisticLoss,
    PoissonLoss,
    SquaredLoss,
    cv_error_and_gradient,
)
from optimized_svm.training import Factorization
from optimized_svm.validation import (
    LogisticValidation,
    PoissonValidation,
    SmoothAbsoluteValidation,
    SmoothHingeValidation,
    SquaredValidation,
)

TIGHT_SMO = {"tol": 1e-10}
TIGHT_NEWTON = {"tol": 1e-26, "max_steps": 200}


def case(name, data):
    x, y, t, counts = data
    return {
        "hinge": (HingeLoss(smoothing=0.5), SmoothHingeValidation(), y, None, None),
        "hinge-balanced": (
            HingeLoss(smoothing=0.5),
            SmoothHingeValidation(balanced=True),
            y,
            None,
            None,
        ),
        "hinge6-ard": (
            HingeLoss(smoothing=0.5, degree="6"),
            SmoothHingeValidation(),
            y,
            None,
            [[0], [1], [2]],
        ),
        "hinge-nu": (HingeLoss(smoothing=0.5), SmoothHingeValidation(), y, 8.0, None),
        "eps3-blocks": (
            EpsilonInsensitiveLoss(0.1, 0.3, "3"),
            SmoothAbsoluteValidation(),
            t,
            None,
            [[0, 1], [2]],
        ),
        "eps-nu": (EpsilonInsensitiveLoss(0.05, 0.3), SmoothAbsoluteValidation(), t, 5.0, None),
        "logistic": (LogisticLoss(), SmoothHingeValidation(), y, None, None),
        "logistic-logloss": (LogisticLoss(), LogisticValidation(), y, None, None),
        "logistic-ard": (
            LogisticLoss(),
            SmoothHingeValidation(),
            y,
            None,
            [[0], [1], [2]],
        ),
        "squared": (SquaredLoss(), SquaredValidation(), t, None, None),
        "poisson": (PoissonLoss(), PoissonValidation(), counts, None, None),
    }[name]


@pytest.mark.parametrize(
    "name",
    [
        "hinge",
        "hinge-balanced",
        "hinge6-ard",
        "hinge-nu",
        "eps3-blocks",
        "eps-nu",
        "logistic",
        "logistic-logloss",
        "logistic-ard",
        "squared",
        "poisson",
    ],
)
def test_gradient_matches_finite_differences(name, data):
    loss, validation, target, max_asum, blocks = case(name, data)
    x = data[0]
    folds = list(KFold(3, shuffle=True, random_state=0).split(x))
    w = np.ones(len(target))
    n_blocks = 1 if blocks is None else len(blocks)
    theta = np.r_[np.log(0.05), np.full(n_blocks, np.log(0.6))]

    def f(th):
        return cv_error_and_gradient(
            th, x, target, w, folds, loss, validation, blocks, max_asum, TIGHT_SMO, TIGHT_NEWTON
        )

    res = f(theta)
    if max_asum is not None:
        assert all(fold.solution.c > 0 for fold in res.folds), "constraint should be active"
    eps = 1e-6
    fd = np.array(
        [
            (f(theta + eps * e).error - f(theta - eps * e).error) / (2 * eps)
            for e in np.eye(theta.size)
        ]
    )
    np.testing.assert_allclose(res.gradient, fd, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("name", ["hinge", "eps3-blocks"])
def test_reused_factorization_gives_same_gradient(name, data, monkeypatch):
    loss, validation, target, max_asum, blocks = case(name, data)
    x = data[0]
    folds = list(KFold(3, shuffle=True, random_state=0).split(x))
    w = np.ones(len(target))
    n_blocks = 1 if blocks is None else len(blocks)
    theta = np.r_[np.log(0.05), np.full(n_blocks, np.log(0.6))]

    def run():
        return cv_error_and_gradient(
            theta, x, target, w, folds, loss, validation, blocks, max_asum, TIGHT_SMO, TIGHT_NEWTON
        )

    reused = run()
    monkeypatch.setattr(Factorization, "matches", lambda self, p, h: False)
    fresh = run()
    assert any(f.solution.factor is not None for f in reused.folds)
    np.testing.assert_allclose(reused.gradient, fresh.gradient, rtol=1e-10, atol=1e-12)
    assert reused.error == pytest.approx(fresh.error, rel=1e-12)


LOSS_PARAM_CASES = {
    "eps-deg2": (EpsilonInsensitiveLoss(0.3, 0.1, "2"), ("epsilon", "smoothing"), None, None),
    "eps-deg3-blocks": (
        EpsilonInsensitiveLoss(0.3, 0.15, "3"),
        ("epsilon", "smoothing"),
        None,
        [[0, 1], [2]],
    ),
    "eps-deg6-nu": (EpsilonInsensitiveLoss(0.25, 0.1, "6"), ("epsilon", "smoothing"), 5.0, None),
    "eps-only-epsilon": (EpsilonInsensitiveLoss(0.4, 0.1, "2"), ("epsilon",), None, None),
    "hinge-deg2": (HingeLoss(1.0, 0.5, "2"), ("smoothing",), None, None),
    "hinge-deg6-nu": (HingeLoss(1.0, 0.5, "6"), ("smoothing",), 8.0, None),
}


@pytest.mark.parametrize("name", list(LOSS_PARAM_CASES))
def test_gradient_wrt_loss_parameters_matches_finite_differences(name, data):
    loss, params, max_asum, blocks = LOSS_PARAM_CASES[name]
    x, y, t, _ = data
    target = t if isinstance(loss, EpsilonInsensitiveLoss) else y
    validation = SmoothAbsoluteValidation() if target is t else SmoothHingeValidation()
    folds = list(KFold(3, shuffle=True, random_state=0).split(x))
    w = np.ones(len(target))
    n_blocks = 1 if blocks is None else len(blocks)
    values = [getattr(loss, p) for p in params]
    theta = np.r_[np.log(0.05), np.full(n_blocks, np.log(0.6)), np.log(values)]

    def f(th):
        return cv_error_and_gradient(
            th,
            x,
            target,
            w,
            folds,
            loss,
            validation,
            blocks,
            max_asum,
            TIGHT_SMO,
            TIGHT_NEWTON,
            loss_params=params,
        )

    res = f(theta)
    assert res.gradient.size == theta.size
    eps = 1e-6
    fd = np.array(
        [
            (f(theta + eps * e).error - f(theta - eps * e).error) / (2 * eps)
            for e in np.eye(theta.size)
        ]
    )
    np.testing.assert_allclose(res.gradient, fd, rtol=1e-4, atol=1e-7)
    # the loss parameters must have an effect, otherwise the test would be vacuous
    assert np.all(np.abs(res.gradient[-len(params) :]) > 1e-6)


def test_unknown_loss_parameter_is_rejected(data):
    x, y, _, _ = data
    folds = list(KFold(3, shuffle=True, random_state=0).split(x))
    with pytest.raises(ValueError, match="tunable"):
        cv_error_and_gradient(
            np.r_[np.log(0.05), np.log(0.6), 0.0],
            x,
            y,
            np.ones(len(y)),
            folds,
            LogisticLoss(),
            SmoothHingeValidation(),
            loss_params=("smoothing",),
        )


def test_epsilon_is_not_identifiable_below_the_smoothing_width(data):
    # For quadratic smoothing and eps < delta the two smoothed sides overlap and their sum is
    # (t - y)^2 / (2 delta) + const(eps): eps only shifts the loss by a constant, so the solution
    # and the CV error do not depend on it and the hypergradient vanishes exactly.
    x, _, t, _ = data
    folds = list(KFold(3, shuffle=True, random_state=0).split(x))

    def run(eps):
        loss = EpsilonInsensitiveLoss(eps, 0.3, "2")
        theta = np.r_[np.log(0.05), np.log(0.6), np.log(eps)]
        return cv_error_and_gradient(
            theta, x, t, np.ones(len(t)), folds, loss, SmoothAbsoluteValidation(),
            smo_options=TIGHT_SMO, newton_options=TIGHT_NEWTON, loss_params=("epsilon",),
        )  # fmt: skip

    a, b = run(0.02), run(0.05)
    assert a.error == pytest.approx(b.error, abs=1e-12)
    assert abs(a.gradient[-1]) < 1e-12
    assert abs(run(0.2).gradient[-1]) > 1e-5

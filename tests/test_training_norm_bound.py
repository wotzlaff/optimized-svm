"""The rough SMO stage must not decide the final norm-constraint activity."""

import numpy as np
import pytest
import smo_solver
from numpy.testing import assert_allclose

from optimized_svm import (
    EpsilonInsensitiveLoss,
    HingeLoss,
    TrainingProblem,
    cv_error_and_gradient,
    train,
)
from optimized_svm.validation import SquaredValidation


@pytest.mark.parametrize("regression", [False, True])
@pytest.mark.parametrize("warm", [False, True])
def test_newton_activates_bound_missed_by_rough_smo(regression, warm):
    rng = np.random.default_rng(41)
    x = 0.6 * rng.normal(size=(28, 2))
    target = np.sin(x[:, 0]) if regression else np.where(x[:, 0] > 0, 1.0, -1.0)
    loss = EpsilonInsensitiveLoss(0.05, 0.2) if regression else HingeLoss(smoothing=0.2)
    prob = TrainingProblem(x, target, loss, 0.02, max_asum=1.5)
    solve = smo_solver.solve_regression if regression else smo_solver.solve_classification
    kwargs = {"epsilon": 0.05} if regression else {}
    reference = solve(x, target, 0.02, smoothing=0.2, max_asum=1.5, tol=1e-9, **kwargs)
    # A projected warm start is just inside the face; the solver reports c=0 there.
    start = reference.alpha * (1.0 - 1e-12) if warm else None
    result = train(prob, {"max_steps": 0}, alpha0=start)
    assert result.residual < 1e-12
    assert result.c > 0
    assert abs(prob.sign @ result.a - 1.5) < 1e-8
    assert_allclose(result.ka + result.b, reference.ka + reference.b, atol=2e-6)
    assert_allclose(result.a, reference.alpha, atol=2e-6)


def test_newton_can_release_a_wrongly_active_warm_start(monkeypatch):
    import optimized_svm.training as training

    x = np.linspace(-1, 1, 12)[:, None]
    y = np.sin(x[:, 0])
    loss = EpsilonInsensitiveLoss(0.05, 0.2)
    unconstrained = train(TrainingProblem(x, y, loss, 0.1))
    prob = TrainingProblem(x, y, loss, 0.1, max_asum=100.0)
    monkeypatch.setattr(
        training,
        "smo_warm_start",
        lambda *a, **kw: (unconstrained.a.copy(), unconstrained.b, 0.01, 0),
    )
    result = train(prob)
    assert result.c == 0
    assert prob.sign @ result.a < 100
    assert result.residual < 1e-12
    assert_allclose(result.ka + result.b, unconstrained.ka + unconstrained.b, atol=2e-6)


def test_cv_error_and_gradient_do_not_depend_on_projected_warm_start():
    from sklearn.model_selection import KFold

    rng = np.random.default_rng(7)
    x = rng.normal(size=(36, 3))
    y = np.sin(x[:, 0]) + 0.2 * x[:, 1]
    args = (
        np.log([0.01, 0.6]),
        x,
        y,
        np.ones(36),
        list(KFold(3, shuffle=True, random_state=0).split(x)),
        EpsilonInsensitiveLoss(0.05, 0.2),
        SquaredValidation(),
    )
    reference = cv_error_and_gradient(*args, max_asum=3.0, smo_options={"tol": 1e-9})
    warm = [fold.solution.a * (1 - 1e-12) for fold in reference.folds]
    result = cv_error_and_gradient(
        *args, max_asum=3.0, smo_options={"max_steps": 0}, warm_start=warm
    )
    assert_allclose(result.error, reference.error, atol=1e-9)
    assert_allclose(result.gradient, reference.gradient, atol=1e-8)
    for fold in result.folds:
        assert abs(fold.solution.a).sum() <= 3.0 + 1e-8

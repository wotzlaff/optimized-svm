import numpy as np
import pytest
import svm_smo

from optimized_svm import (
    ApproxPoissonLoss,
    EpsilonInsensitiveLoss,
    HingeLoss,
    PoissonLoss,
    SquaredLoss,
    TrainingProblem,
    newton,
    train,
)


def cases(data):
    x, y, t, counts = data
    return [
        (HingeLoss(smoothing=0.5), y, None),
        (HingeLoss(smoothing=0.5, degree="6"), y, None),
        (HingeLoss(smoothing=0.3), y, 8.0),
        (EpsilonInsensitiveLoss(0.1, 0.2), t, None),
        (EpsilonInsensitiveLoss(0.1, 0.2, "3"), t, 6.0),
        (SquaredLoss(), t, None),
        (PoissonLoss(), counts, None),
        (ApproxPoissonLoss(), counts, None),
    ]


def test_all_losses_are_solved_to_optimality(data):
    x = 0.6 * data[0]
    for loss, target, max_asum in cases(data):
        prob = TrainingProblem(x, target, loss, 0.05, max_asum=max_asum)
        sol = train(prob)
        assert sol.residual < 1e-12, (loss, sol.residual)
        assert abs(sol.a.sum()) < 1e-9
        if max_asum is not None:
            assert (prob.sign * sol.a).sum() <= max_asum * (1 + 1e-9)
            assert sol.c >= 0


def test_newton_matches_exact_smo_for_quadratic_smoothing(data):
    x, y, _, _ = data
    loss = HingeLoss(smoothing=0.4)
    prob = TrainingProblem(0.6 * x, y, loss, 0.02)
    ref = svm_smo.solve_classification(prob.x, y, 0.02, gamma=1.0, smoothing=0.4, tol=1e-10)
    # Newton from zero (no SMO warm start)
    sol = newton(prob, np.zeros(prob.size), 0.0, 0.0)
    np.testing.assert_allclose(sol.a, ref.alpha, atol=1e-7)
    np.testing.assert_allclose(sol.b, ref.b, atol=1e-7)


def test_non_smooth_loss_uses_smo_only(data):
    x, y, _, _ = data
    prob = TrainingProblem(x, y, HingeLoss(smoothing=0.0), 0.1)
    sol = train(prob)
    assert sol.newton_steps == 0 and sol.smo_steps > 0


def test_warm_start_reduces_smo_steps(data):
    x, _, t, _ = data
    loss = EpsilonInsensitiveLoss(0.05, 0.2)
    opts = {"tol": 1e-6}
    first = train(TrainingProblem(x, t, loss, 0.01), opts)
    cold = train(TrainingProblem(x, t, loss, 0.011), opts)
    warm = train(TrainingProblem(x, t, loss, 0.011), opts, alpha0=first.a)
    assert warm.smo_steps < cold.smo_steps
    # both are refined by Newton up to its residual tolerance
    np.testing.assert_allclose(warm.a, cold.a, atol=1e-5)


def test_invalid_problem():
    x = np.zeros((3, 1))
    with pytest.raises(ValueError):
        TrainingProblem(x, np.zeros(3), SquaredLoss(), 0.0)
    with pytest.raises(ValueError):
        TrainingProblem(x, np.zeros(3), SquaredLoss(), 1.0, max_asum=1.0)

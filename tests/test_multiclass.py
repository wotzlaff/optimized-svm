"""Multi-class soft-max SVM: training, r = 2 equivalence with the logistic model, hypergradient."""

import numpy as np
import pytest
from sklearn.model_selection import StratifiedKFold

from optimized_svm import LogisticLoss
from optimized_svm.hypergradient import fold_error_and_gradient
from optimized_svm.multiclass import (
    MulticlassProblem,
    MultinomialNLL,
    cv_error_and_gradient_multiclass,
    decision_function,
    newton_multiclass,
    predict_proba,
)
from optimized_svm.validation import LogisticValidation

TIGHT = {"tol": 1e-28, "max_steps": 200}


def make(n=60, r=3, d=3, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(r, d)) * 1.5
    y = np.arange(n) % r
    x = centers[y] + rng.normal(size=(n, d))
    w = rng.uniform(0.5, 2.0, size=n)
    return 0.5 * x, y, w


def folds_of(x, y, k=3):
    return list(StratifiedKFold(k, shuffle=True, random_state=0).split(x, y))


@pytest.mark.parametrize("r", [3, 4])
def test_objective_gradient_and_newton(r):
    x, y, w = make(r=r)
    prob = MulticlassProblem(x, y, r, 0.3, w)
    rng = np.random.default_rng(1)
    a, b = 0.1 * rng.normal(size=(prob.n, r)), rng.normal(size=r)
    ka, t = prob.decision(a, b)
    f_a, f_b = prob.stationarity(a, b, t)
    # gradient of the objective is (K F_a / lam, F_b)
    eps = 1e-6
    for idx in [(0, 0), (5, r - 1)]:
        da = np.zeros_like(a)
        da[idx] = eps
        fd = (
            prob.objective(a + da, b, prob.decision(a + da, b)[0])
            - prob.objective(a - da, b, prob.decision(a - da, b)[0])
        ) / (2 * eps)
        assert fd == pytest.approx((prob.k @ f_a / prob.lam)[idx], rel=1e-6, abs=1e-8)
    db = np.zeros(r)
    db[1] = eps
    fd = (
        prob.objective(a, b + db, prob.decision(a, b + db)[0])
        - prob.objective(a, b - db, prob.decision(a, b - db)[0])
    ) / (2 * eps)
    assert fd == pytest.approx(f_b[1], rel=1e-6, abs=1e-8)
    sol = newton_multiclass(prob)
    assert sol.residual < 1e-12
    assert np.abs(sol.a.sum(axis=1)).max() < 1e-10  # sum_k a_k = 0
    assert abs(sol.b.sum()) < 1e-10  # gauge
    # warm start from the solution needs no further step
    assert newton_multiclass(prob, sol.a, sol.b).newton_steps == 0


def test_two_classes_match_the_binary_logistic_model():
    # softmax with lam_s equals the logistic model with lam_b = lam_s / 2 and t = T_1 - T_0
    x, y, w = make(r=2, n=50)
    lam = 0.2
    prob = MulticlassProblem(x, y, 2, lam, w)
    sol = newton_multiclass(prob, **TIGHT)
    from optimized_svm.problem import TrainingProblem
    from optimized_svm.training import train

    ysgn = 2.0 * y - 1.0
    bin_prob = TrainingProblem(x, ysgn, LogisticLoss(), lam / 2, w)
    bsol = train(bin_prob, newton_options=TIGHT)
    t_bin = bin_prob.decision(bsol.b, bsol.c, bsol.ka)
    np.testing.assert_allclose(sol.t[:, 1] - sol.t[:, 0], t_bin, atol=1e-8)
    # alpha_bin = (a_1 - a_0) lam_b / lam_s = a_1 (a_0 = -a_1)
    np.testing.assert_allclose(sol.a[:, 1], bsol.a, atol=1e-8)
    # fold error and hypergradient coincide (log lam shifts by log 2, same derivative)
    tr, va = np.arange(35), np.arange(35, 50)
    ms = fold_error_and_gradient_multiclass_wrapper(x, y, w, tr, va, 2, lam)
    ref = fold_error_and_gradient(
        x[tr], ysgn[tr], w[tr], x[va], ysgn[va], LogisticLoss(), LogisticValidation(), lam / 2,
        None, newton_options=TIGHT,
    )  # fmt: skip
    assert ms.error == pytest.approx(ref.error, rel=1e-8)
    np.testing.assert_allclose(ms.gradient, ref.gradient, rtol=1e-6, atol=1e-9)


def fold_error_and_gradient_multiclass_wrapper(x, y, w, tr, va, r, lam, blocks=None, wv=None):
    from optimized_svm.multiclass import fold_error_and_gradient_multiclass

    val = MultinomialNLL(weighted=wv is not None)
    return fold_error_and_gradient_multiclass(
        x[tr], y[tr], w[tr], x[va], y[va], r, val, lam, blocks, TIGHT, None,
        None if wv is None else wv[va],
    )  # fmt: skip


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("ard", [False, True])
def test_hypergradient_matches_finite_differences(ard, weighted, capsys):
    x, y, w = make(n=60, r=3)
    w = w if weighted else np.ones_like(w)
    blocks = [[0], [1], [2]] if ard else None
    theta = np.array([np.log(0.2)] + [np.log(0.9), np.log(1.1), np.log(0.7)][: 3 if ard else 1])
    folds = folds_of(x, y)
    val = MultinomialNLL(weighted=weighted)
    kw = {"n_classes": 3, "validation": val, "blocks": blocks, "newton_options": TIGHT}
    res = cv_error_and_gradient_multiclass(theta, x, y, w, folds, **kw)
    fd = np.zeros_like(theta)
    for j in range(theta.size):
        e = np.zeros_like(theta)
        e[j] = 1e-5
        fd[j] = (
            cv_error_and_gradient_multiclass(theta + e, x, y, w, folds, **kw).error
            - cv_error_and_gradient_multiclass(theta - e, x, y, w, folds, **kw).error
        ) / 2e-5
    rel = np.abs(res.gradient - fd) / np.abs(fd)
    print(f"ard={ard} weighted={weighted} rel. FD error: {np.array2string(rel, precision=2)}")
    np.testing.assert_allclose(res.gradient, fd, rtol=1e-5)


def test_cv_warm_start_and_invariances():
    x, y, w = make(n=60, r=3)
    theta = np.array([np.log(0.2), np.log(0.9)])
    folds = folds_of(x, y)
    ws = [None] * len(folds)
    r1 = cv_error_and_gradient_multiclass(theta, x, y, w, folds, warm_start=ws)
    r2 = cv_error_and_gradient_multiclass(theta, x, y, w, folds, warm_start=ws)
    assert all(f.solution.newton_steps == 0 for f in r2.folds)
    assert r2.error == pytest.approx(r1.error, rel=1e-10)
    # relabelling the classes permutes the outputs; probabilities sum to one
    perm = np.array([2, 0, 1])
    prob = MulticlassProblem(x, y, 3, 0.2, w)
    prob_p = MulticlassProblem(x, perm[y], 3, 0.2, w)
    s, sp = newton_multiclass(prob), newton_multiclass(prob_p)
    np.testing.assert_allclose(sp.t[:, perm], s.t, atol=1e-8)
    np.testing.assert_allclose(sp.a[:, perm], s.a, atol=1e-8)
    p = predict_proba(x[:7], prob.x, s.a, s.b, 0.2)
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-14)
    np.testing.assert_allclose(decision_function(x, prob.x, s.a, s.b, 0.2), s.t, atol=1e-10)
    # CV error and gradient are invariant under the relabelling
    r3 = cv_error_and_gradient_multiclass(theta, x, perm[y], w, folds)
    assert r3.error == pytest.approx(r1.error, rel=1e-9)
    np.testing.assert_allclose(r3.gradient, r1.gradient, rtol=1e-6, atol=1e-9)

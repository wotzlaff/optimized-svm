"""``multi_class="softmax"`` in SmoothSVC and OptimizedSVC."""

import warnings

import numpy as np
import pytest
from scipy.optimize import minimize
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold

from optimized_svm import OptimizedSVC, SmoothSVC
from optimized_svm.kernels import scale_features
from optimized_svm.multiclass import cv_error_and_gradient_multiclass


def data(n=90, r=3, seed=0):
    x, y = make_classification(
        n, 5, n_informative=4, n_redundant=0, n_classes=r, n_clusters_per_class=1, random_state=seed
    )
    return x / 2, y


@pytest.mark.parametrize("r", [3, 4])
def test_smooth_fit_predict_proba(r):
    x, y = data(r=r)
    est = SmoothSVC(0.1, loss="logistic", multi_class="softmax").fit(x, y)
    p = est.predict_proba(x)
    assert p.shape == (len(y), r)
    np.testing.assert_allclose(p.sum(axis=1), 1.0)
    dec = est.decision_function(x)
    assert dec.shape == (len(y), r)
    np.testing.assert_array_equal(est.predict(x), est.classes_[dec.argmax(axis=1)])
    np.testing.assert_allclose(p, np.exp(dec) / np.exp(dec).sum(axis=1, keepdims=True))
    assert est.score(x, y) > 0.8


def test_string_labels_and_default_is_ovr():
    x, y = data()
    labels = np.array(["a", "b", "c"])[y]
    est = SmoothSVC(0.1, loss="logistic", multi_class="softmax").fit(x, labels)
    assert set(est.predict(x)) <= {"a", "b", "c"}
    ovr = SmoothSVC(0.1, loss="logistic").fit(x, y)
    assert ovr.multi_class == "ovr" and ovr.estimators_ is not None


def test_two_classes_equal_binary_model_with_half_lam():
    x, y = data(r=2)
    soft = SmoothSVC(0.2, loss="logistic", multi_class="softmax").fit(x, y)
    binary = SmoothSVC(0.1, loss="logistic").fit(x, y)
    np.testing.assert_allclose(soft.decision_function(x), binary.decision_function(x))
    np.testing.assert_allclose(soft.predict_proba(x), binary.predict_proba(x))
    # and it is the soft-max model: compare with the 3-class code on a 2-class problem
    from optimized_svm.multiclass import MulticlassProblem, newton_multiclass

    xs = scale_features(x, soft.scaling_, soft.blocks_)
    sol = newton_multiclass(
        MulticlassProblem(xs, y.astype(float), 2, 0.2), tol=1e-26, max_steps=200
    )
    np.testing.assert_allclose(sol.t[:, 1] - sol.t[:, 0], soft.decision_function(x), atol=1e-6)


def test_class_and_sample_weights():
    x, y = data(r=3)
    base = SmoothSVC(0.1, loss="logistic", multi_class="softmax")
    p0 = clone(base).fit(x, y).predict_proba(x)
    pc = clone(base).set_params(class_weight={0: 5.0}).fit(x, y).predict_proba(x)
    assert pc[:, 0].mean() > p0[:, 0].mean()
    w = np.where(y == 0, 5.0, 1.0)
    ps = clone(base).fit(x, y, sample_weight=w).predict_proba(x)
    np.testing.assert_allclose(pc, ps)
    bal = clone(base).set_params(class_weight="balanced").fit(x, y)
    counts = np.bincount(y)
    w_bal = (len(y) / (3 * counts))[y]
    np.testing.assert_allclose(
        bal.predict_proba(x), clone(base).fit(x, y, sample_weight=w_bal).predict_proba(x)
    )
    with pytest.raises(ValueError, match="class_weight"):
        clone(base).set_params(class_weight="nope").fit(x, y)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"loss": "hinge"}, "logistic"),
        ({"n_basis": 10}, "n_basis"),
        ({"max_asum": 2.0}, "max_asum"),
        ({"multi_class": "foo"}, "multi_class"),
    ],
)
def test_invalid_parameters(kwargs, match):
    x, y = data()
    params = {"loss": "logistic", "multi_class": "softmax", **kwargs}
    for cls, extra in ((SmoothSVC, {}), (OptimizedSVC, {"max_iter": 1})):
        with pytest.raises(ValueError, match=match):
            cls(**params, **extra).fit(x, y)


def test_missing_class_in_training_fold():
    x, y = data(n=60)
    # fold 0 trains without class 2
    folds = [(np.flatnonzero(y != 2), np.flatnonzero(y == 2)), (np.arange(60), np.arange(10))]
    with pytest.raises(ValueError, match="fold 0.*all 3 classes"):
        OptimizedSVC(loss="logistic", multi_class="softmax", cv=folds, max_iter=1).fit(x, y)


def test_cost_warning():
    x, y = data(n=60)
    with pytest.warns(UserWarning, match="r \\* n"):
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            from optimized_svm import estimators

            old = estimators.SOFTMAX_MAX_SIZE
            estimators.SOFTMAX_MAX_SIZE = 100
            try:
                SmoothSVC(0.1, loss="logistic", multi_class="softmax").fit(x, y)
            finally:
                estimators.SOFTMAX_MAX_SIZE = old


def test_optimized_matches_direct_lbfgs_and_cv_error():
    x, y = data(n=60)
    kw = {"loss": "logistic", "multi_class": "softmax", "cv": 3, "max_iter": 50, "tol": 1e-9}
    est = OptimizedSVC(**kw).fit(x, y)
    folds = [(tr, va) for tr, va in StratifiedKFold(3, shuffle=True, random_state=0).split(x, y)]
    w = np.ones(len(y))
    theta = np.log([est.lam_, *est.scaling_opt_])
    res = cv_error_and_gradient_multiclass(theta, x, y, w, folds, 3)
    assert res.error == pytest.approx(est.cv_error_, rel=1e-6)
    assert len(est.history_) > 1 and est.history_[-1]["n_support"] > 0
    # a direct run from the same start reaches the same optimum
    theta0 = np.log([est.lam_init, est.history_[0]["scaling"][0]])
    direct = minimize(
        lambda t: (lambda r: (r.error, r.gradient))(
            cv_error_and_gradient_multiclass(t, x, y, w, folds, 3)
        ),
        theta0,
        jac=True,
        method="L-BFGS-B",
        bounds=[tuple(np.log(est.lam_bounds)), tuple(np.log(est.scaling_bounds))],
        options={"ftol": 1e-9, "gtol": 1e-9, "maxiter": 50},
    )
    assert est.cv_error_ == pytest.approx(direct.fun, rel=1e-6)
    assert est.predict_proba(x).shape == (60, 3)


def test_optimized_n_jobs_equals_serial():
    x, y = data(n=60)
    fits = [
        OptimizedSVC(loss="logistic", multi_class="softmax", max_iter=8, cv=3, n_jobs=j).fit(x, y)
        for j in (None, 2)
    ]
    np.testing.assert_allclose(fits[0].lam_, fits[1].lam_, rtol=1e-6)
    np.testing.assert_allclose(fits[0].scaling_opt_, fits[1].scaling_opt_, rtol=1e-6)
    np.testing.assert_allclose(fits[0].cv_error_, fits[1].cv_error_, rtol=1e-5)
    np.testing.assert_allclose(
        fits[0].predict_proba(x), fits[1].predict_proba(x), rtol=1e-5, atol=1e-8
    )


def test_optimized_two_classes_reports_softmax_lam():
    x, y = data(n=60, r=2)
    soft = OptimizedSVC(loss="logistic", multi_class="softmax", max_iter=30, tol=1e-9).fit(x, y)
    binary = OptimizedSVC(
        loss="logistic", lam_init=5e-3, lam_bounds=(5e-7, 0.5), max_iter=30, tol=1e-9
    ).fit(x, y)
    assert soft.lam_ == pytest.approx(2 * binary.lam_, rel=1e-4)
    np.testing.assert_allclose(soft.cv_error_, binary.cv_error_, rtol=1e-6)


def test_warm_start_does_not_change_the_result():
    x, y = data(n=60)
    fits = [
        OptimizedSVC(loss="logistic", multi_class="softmax", max_iter=6, cv=3, warm_start=w).fit(
            x, y
        )
        for w in (True, False)
    ]
    np.testing.assert_allclose(fits[0].cv_error_, fits[1].cv_error_, rtol=1e-5)

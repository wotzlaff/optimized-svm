import pickle

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.datasets import make_classification, make_friedman1
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.estimator_checks import parametrize_with_checks

from optimized_svm import OptimizedSVC, OptimizedSVR, SmoothSVC, SmoothSVR

BINARY_ONLY = "binary classifier; use OneVsRestClassifier for more classes"
_CLASSIFIER_XFAIL: dict[str, str] = {}


def _expected_failures(est):
    # `scaling="auto"` depends on the data spread, which changes when samples are repeated
    failures = {"check_sample_weight_equivalence_on_dense_data": "scaling='auto' is data dependent"}
    if isinstance(est, (SmoothSVC, OptimizedSVC)):
        failures.update(_CLASSIFIER_XFAIL)
    return failures


@parametrize_with_checks(
    [
        SmoothSVR(),
        SmoothSVR(loss="squared"),
        SmoothSVC(),
        OptimizedSVR(cv=2, max_iter=2),
        OptimizedSVC(cv=2, max_iter=2),
    ],
    expected_failed_checks=_expected_failures,
)
def test_sklearn_compatible(estimator, check):
    check(estimator)


@pytest.fixture
def classification():
    x, y = make_classification(300, 6, n_informative=3, random_state=0)
    return train_test_split(StandardScaler().fit_transform(x), y, random_state=0)


@pytest.fixture
def regression():
    x, y = make_friedman1(300, noise=0.5, random_state=0)
    return train_test_split(x, y, random_state=0)


def test_optimized_svc_improves_cv_error(classification):
    xa, xb, ya, yb = classification
    est = OptimizedSVC(lam_init=1.0, scaling_init=0.05).fit(xa, ya)
    errors = [h["error"] for h in est.history_]
    assert est.cv_error_ < errors[0]
    assert est.score(xb, yb) > 0.85
    assert set(est.predict(xb)) <= set(ya)


def test_ard_identifies_irrelevant_features(regression):
    xa, xb, ya, yb = regression
    est = OptimizedSVR(blocks="features", max_iter=30).fit(xa, ya)
    # make_friedman1 uses features 0..4, features 5..9 are noise
    assert est.scaling_opt_[:5].min() > 2 * est.scaling_opt_[5:].max()
    assert est.score(xb, yb) > 0.85
    assert {h["stage"] for h in est.history_} == {"global", "features"}


@pytest.mark.parametrize("loss", ["epsilon_insensitive", "squared"])
def test_optimized_svr_beats_default(regression, loss):
    xa, xb, ya, yb = regression
    opt = OptimizedSVR(loss=loss).fit(xa, ya)
    default = SmoothSVR(loss=loss).fit(xa, ya)
    assert opt.score(xb, yb) > default.score(xb, yb)


def test_poisson_regression():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 2))
    y = rng.poisson(np.exp(0.8 * np.sin(x[:, 0]) + 0.3 * x[:, 1]))
    est = OptimizedSVR(loss="poisson", max_iter=10).fit(x, y)
    pred = est.predict(x)
    assert np.all(pred > 0)
    assert np.corrcoef(pred, np.exp(0.8 * np.sin(x[:, 0]) + 0.3 * x[:, 1]))[0, 1] > 0.8


def test_time_limit_keeps_best_parameters(regression):
    xa, _, ya, _ = regression
    est = OptimizedSVR(time_limit=0.0).fit(xa, ya)
    assert est.history_ == [] and np.isclose(est.lam_, est.lam_init)


def test_nu_constraint_and_weights(classification):
    xa, xb, ya, yb = classification
    w = np.where(ya == 1, 2.0, 1.0)
    est = SmoothSVC(lam=0.01, max_asum=20.0).fit(xa, ya, sample_weight=w)
    assert est.training_info_["c"] > 0
    assert est.score(xb, yb) > 0.8


def test_decision_gradient_matches_finite_differences(regression):
    xa, xb, ya, _ = regression
    est = SmoothSVR(lam=0.01, scaling=0.3).fit(xa, ya)
    g = est.decision_gradient(xb[:5])
    h = 1e-6
    fd = np.stack(
        [
            (est._decision(xb[:5] + h * e) - est._decision(xb[:5] - h * e)) / (2 * h)
            for e in np.eye(xb.shape[1])
        ],
        axis=1,
    )
    np.testing.assert_allclose(g, fd, rtol=1e-5, atol=1e-6)


def test_sklearn_tools(classification):
    xa, xb, ya, yb = classification
    search = GridSearchCV(
        make_pipeline(StandardScaler(), SmoothSVC()), {"smoothsvc__lam": [0.01, 0.1]}, cv=3
    ).fit(xa, ya)
    assert search.best_score_ > 0.8
    x3, y3 = make_classification(150, 5, n_informative=3, n_classes=3, random_state=0)
    ovr = OneVsRestClassifier(SmoothSVC(lam=0.05)).fit(x3, y3)
    builtin = SmoothSVC(lam=0.05).fit(x3, y3)
    np.testing.assert_array_equal(ovr.predict(x3), builtin.predict(x3))
    assert builtin.decision_function(x3).shape == (150, 3)
    est = OptimizedSVC(max_iter=3).fit(xa, ya)
    again = pickle.loads(pickle.dumps(est))
    np.testing.assert_array_equal(again.decision_function(xb), est.decision_function(xb))
    assert clone(est).get_params()["max_iter"] == 3

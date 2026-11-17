import pickle

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.datasets import make_classification, make_friedman1
from sklearn.metrics import log_loss
from sklearn.model_selection import (
    GridSearchCV,
    GroupKFold,
    KFold,
    LeaveOneGroupOut,
    StratifiedKFold,
    train_test_split,
)
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
        SmoothSVC(loss="logistic"),
        OptimizedSVR(cv=2, max_iter=2),
        OptimizedSVC(cv=2, max_iter=2),
        OptimizedSVC(loss="logistic", cv=2, max_iter=2),
        SmoothSVC(loss="logistic", multi_class="softmax"),
        OptimizedSVC(loss="logistic", multi_class="softmax", cv=2, max_iter=2),
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


class TestLogisticClassifier:
    def test_probabilities_binary(self, classification):
        xtr, xte, ytr, yte = classification
        est = SmoothSVC(lam=0.05, loss="logistic").fit(xtr, ytr)
        proba = est.predict_proba(xte)
        assert proba.shape == (len(xte), 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0)
        np.testing.assert_array_equal(est.classes_[proba.argmax(axis=1)], est.predict(xte))
        np.testing.assert_allclose(proba[:, 1], 1 / (1 + np.exp(-est.decision_function(xte))))
        assert log_loss(yte, proba) < 0.45
        assert est.score(xte, yte) > 0.8

    def test_probabilities_multiclass(self):
        x, y = make_classification(
            240, 6, n_informative=4, n_classes=3, n_clusters_per_class=1, random_state=0
        )
        est = SmoothSVC(lam=0.05, loss="logistic").fit(x, y)
        proba = est.predict_proba(x)
        assert proba.shape == (240, 3)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0)
        np.testing.assert_array_equal(est.classes_[proba.argmax(axis=1)], est.predict(x))

    def test_predict_proba_only_for_logistic(self, classification):
        assert not hasattr(SmoothSVC(), "predict_proba")
        assert not hasattr(OptimizedSVC(), "predict_proba")
        assert hasattr(OptimizedSVC(loss="logistic"), "predict_proba")

    def test_parameter_validation(self, classification):
        xtr, _, ytr, _ = classification
        with pytest.raises(ValueError, match="loss"):
            SmoothSVC(loss="nope").fit(xtr, ytr)
        with pytest.raises(ValueError, match="max_asum"):
            SmoothSVC(loss="logistic", max_asum=5.0).fit(xtr, ytr)

    def test_hyperparameter_optimization(self, classification):
        xtr, xte, ytr, yte = classification
        est = OptimizedSVC(loss="logistic", cv=3, max_iter=8).fit(xtr, ytr)
        errors = [h["error"] for h in est.history_]
        assert min(errors) < errors[0]
        proba = est.predict_proba(xte)
        assert log_loss(yte, proba) < 0.45
        # more confident than a badly regularised model, i.e. the optimisation did something
        bad = SmoothSVC(lam=10.0, loss="logistic").fit(xtr, ytr)
        assert log_loss(yte, proba) < log_loss(yte, bad.predict_proba(xte))


# --- cross-validation splitters and groups --------------------------------------------------


def _group_data():
    x, y = make_friedman1(60, noise=0.5, random_state=0)
    return x, y, np.repeat(np.arange(6), 10)


def _fold_signature(folds):
    return [(tuple(tr), tuple(va)) for tr, va in folds]


def test_cv_leave_one_group_out_uses_groups():
    x, y, groups = _group_data()
    est = OptimizedSVR(loss="squared", cv=LeaveOneGroupOut(), max_iter=2).fit(x, y, groups=groups)
    assert len(est.history_) > 0
    folds = est._folds(x, y, groups)
    assert len(folds) == 6
    for (_, va), g in zip(folds, np.unique(groups), strict=True):
        assert set(groups[va]) == {g}


def test_cv_group_kfold_matches_explicit_folds():
    x, y, groups = _group_data()
    splitter = GroupKFold(n_splits=3)
    explicit = [(tr, va) for tr, va in splitter.split(x, y, groups)]
    kw = dict(loss="squared", max_iter=3)
    a = OptimizedSVR(cv=splitter, **kw).fit(x, y, groups=groups)
    b = OptimizedSVR(cv=explicit, **kw).fit(x, y)
    assert a.cv_error_ == pytest.approx(b.cv_error_)
    assert a.lam_ == pytest.approx(b.lam_)
    assert _fold_signature(a._folds(x, y, groups)) == _fold_signature(explicit)


def test_cv_splitter_without_groups_and_int_unchanged():
    x, y, _ = _group_data()
    kf = KFold(3, shuffle=True, random_state=1)
    a = OptimizedSVR(cv=kf, loss="squared", max_iter=2).fit(x, y)
    b = OptimizedSVR(cv=[(tr, va) for tr, va in kf.split(x)], loss="squared", max_iter=2).fit(x, y)
    assert a.cv_error_ == pytest.approx(b.cv_error_)
    # int: shuffled KFold seeded by random_state, as before
    est = OptimizedSVR(cv=4, random_state=3)
    expected = list(KFold(4, shuffle=True, random_state=3).split(x, y))
    assert _fold_signature(est._folds(x, y)) == _fold_signature(expected)
    clf = OptimizedSVC(cv=4, random_state=3)
    yc = (y > np.median(y)).astype(int)
    expected = list(StratifiedKFold(4, shuffle=True, random_state=3).split(x, yc))
    assert _fold_signature(clf._folds(x, yc)) == _fold_signature(expected)


def test_cv_groups_required_error():
    x, y, _ = _group_data()
    with pytest.raises(ValueError, match="groups"):
        OptimizedSVR(cv=LeaveOneGroupOut(), loss="squared").fit(x, y)


def test_cv_groups_classifier_one_vs_rest():
    x, y = make_classification(90, 5, n_informative=3, n_classes=3, random_state=0)
    groups = np.repeat(np.arange(6), 15)
    clf = OptimizedSVC(cv=GroupKFold(3), max_iter=2).fit(x, y, groups=groups)
    assert len(clf.estimators_) == 3
    assert clf.predict(x).shape == y.shape
    with pytest.raises(ValueError, match="groups"):
        OptimizedSVC(cv=GroupKFold(3), max_iter=2).fit(x, y)
    # binary case
    yb = (y > 0).astype(int)
    OptimizedSVC(cv=LeaveOneGroupOut(), max_iter=2).fit(x, yb, groups=groups)


def test_optimized_estimators_give_the_same_result_with_n_jobs():
    from sklearn.datasets import make_classification, make_friedman1

    from optimized_svm import OptimizedSVC, OptimizedSVR

    x, t = make_friedman1(120, noise=1.0, random_state=0)
    reg = [OptimizedSVR(max_iter=8, n_jobs=j).fit(x, t) for j in (None, 3)]
    # equal up to rounding: the BLAS thread count differs
    np.testing.assert_allclose(reg[0].lam_, reg[1].lam_, rtol=1e-6)
    np.testing.assert_allclose(reg[0].scaling_opt_, reg[1].scaling_opt_, rtol=1e-6)
    assert len(reg[0].history_) == len(reg[1].history_)
    xc, yc = make_classification(120, n_classes=3, n_informative=4, random_state=0)
    clf = [OptimizedSVC(max_iter=6, n_jobs=j).fit(xc, yc) for j in (None, 2)]
    for a, b in zip(clf[0].estimators_, clf[1].estimators_, strict=True):
        np.testing.assert_allclose(a.lam_, b.lam_, rtol=1e-6)

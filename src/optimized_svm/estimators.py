"""scikit-learn compatible estimators.

* :class:`KernelGLMRegressor`, :class:`OptimizedKernelGLMRegressor`: kernel generalized linear
  models (Poisson, binomial, negative binomial, Gamma, Tweedie) with offset, fixed or tuned
  hyperparameters.

* :class:`SmoothSVC`, :class:`SmoothSVR`: kernel SVMs with smoothed losses and *fixed*
  hyperparameters, trained by SMO + Newton.
* :class:`OptimizedSVC`, :class:`OptimizedSVR`: the same models whose regularization parameter
  and kernel scalings are tuned by minimising a cross-validation error with exact gradients
  (bilevel approach of Strasdat, Ch. 9).
"""

from __future__ import annotations

import dataclasses
import time
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize
from scipy.special import expit, gammaln, softmax, xlogy
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.utils.metaestimators import available_if
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import _check_sample_weight, check_is_fitted, validate_data

from .hypergradient import Fold, cv_error_and_gradient
from .kernels import Blocks, rbf, scale_features
from .losses import (
    ApproxPoissonLoss,
    BinomialLoss,
    EpsilonInsensitiveLoss,
    GammaLoss,
    HingeLoss,
    LogisticLoss,
    Loss,
    NegativeBinomialLoss,
    PoissonLoss,
    SquaredLoss,
    TweedieLoss,
)
from .multiclass import (
    MulticlassProblem,
    MultinomialNLL,
    cv_error_and_gradient_multiclass,
    newton_multiclass,
)
from .multiclass import decision_function as multiclass_decision
from .problem import TrainingProblem
from .smooth_max import Degree
from .subspace import SubspaceProblem, newton_subspace, nystroem_residual, select_basis
from .training import train
from .validation import (
    BinomialDeviance,
    GammaDeviance,
    LogisticValidation,
    NegativeBinomialDeviance,
    NegativeBinomialNLL,
    PoissonValidation,
    SmoothAbsoluteValidation,
    SmoothHingeValidation,
    SquaredValidation,
    TweedieDeviance,
    ValidationLoss,
)

Array = NDArray[np.float64]


# --------------------------------------------------------------------------------------------
# shared helpers


def resolve_blocks(blocks: Any, n_features: int) -> Blocks:
    """``None`` (one scaling), ``"features"`` (one per feature) or explicit index lists."""
    if blocks is None:
        return None
    if isinstance(blocks, str):
        if blocks != "features":
            raise ValueError(f"blocks must be None, 'features' or a list, got {blocks!r}")
        return [[j] for j in range(n_features)]
    blocks = [list(map(int, b)) for b in blocks]
    used = sorted(j for b in blocks for j in b)
    if used != list(range(n_features)):
        raise ValueError("blocks must partition the feature indices 0..n_features-1")
    return blocks


def auto_scaling(x: Array) -> float:
    """Initial scaling with ``E|x - z|^2 * s^2 = log 2`` (kernel value 1/2 on average)."""
    spread = float(((x - x.mean(axis=0)) ** 2).sum(axis=1).mean())
    return float(np.sqrt(np.log(2.0) / (2.0 * spread))) if spread > 0 else 1.0


class _LabelScaler:
    """Affine normalisation of regression targets."""

    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.offset, self.scale = 0.0, 1.0

    def fit(self, y: Array) -> _LabelScaler:
        if self.mode == "median":
            self.offset = float(np.median(y))
            self.scale = float(np.abs(y - self.offset).mean())
        elif self.mode == "mean":
            self.offset = float(y.mean())
            self.scale = float(y.std())
        if not self.scale > 0:
            self.scale = 1.0
        return self

    def transform(self, y: Array) -> Array:
        return (y - self.offset) / self.scale

    def inverse(self, f: Array) -> Array:
        return f * self.scale + self.offset


@dataclass
class _Model:
    """Kernel expansion ``f(x) = sum_k coef_k exp(-|S x - S sv_k|^2) + b``."""

    support_vectors: Array  # scaled
    coef: Array
    intercept: float

    def decision(self, xs: Array) -> Array:
        return rbf(xs, self.support_vectors) @ self.coef + self.intercept


class _SmoothBase(BaseEstimator):
    """Fixed-hyperparameter training shared by :class:`SmoothSVC` and :class:`SmoothSVR`."""

    def _loss(self) -> Loss:
        raise NotImplementedError

    def _check_params(self) -> None:
        if not isinstance(self.lam, Real) or self.lam <= 0:
            raise ValueError(f"lam must be positive, got {self.lam!r}")
        if not isinstance(self.smoothing, Real) or self.smoothing < 0:
            raise ValueError(f"smoothing must be non-negative, got {self.smoothing!r}")

    def _scaling(self, x: Array) -> Array:
        n = 1 if self.blocks_ is None else len(self.blocks_)
        if isinstance(self.scaling, str):
            if self.scaling != "auto":
                raise ValueError(f"scaling must be 'auto' or positive, got {self.scaling!r}")
            return np.full(n, auto_scaling(x))
        s = np.atleast_1d(np.asarray(self.scaling, dtype=np.float64))
        if s.size == 1:
            s = np.full(n, s[0])
        if s.size != n or np.any(s <= 0):
            raise ValueError(f"scaling must be positive with {n} entries")
        return s

    def _basis(self, xs: Array, lam: float) -> NDArray[np.intp] | None:
        """Basis indices of the Subspace SVM for scaled features ``xs`` (``None``: full model)."""
        mode = self.n_basis
        if mode is None:
            return None
        if self.max_asum is not None:
            raise ValueError("n_basis is not supported together with max_asum")
        if getattr(self._loss(), "smoothing", 1.0) <= 0.0:
            raise ValueError("n_basis needs a smooth loss (smoothing > 0)")
        n = xs.shape[0]
        if isinstance(mode, str):
            if mode != "auto":
                raise ValueError(
                    f"n_basis must be None, a positive integer or 'auto', got {mode!r}"
                )
            if not isinstance(self.basis_tol, Real) or self.basis_tol <= 0:
                raise ValueError(f"basis_tol must be positive, got {self.basis_tol!r}")
            idx, ratio = select_basis(xs, lam, float(self.basis_tol))
            if ratio > self.basis_tol:
                warnings.warn(
                    f"n_basis='auto': {idx.size} basis points (n // 10) do not reach "
                    f"basis_tol={self.basis_tol:g} (residual / lam = {ratio:.3g}); "
                    "using the full model",
                    stacklevel=3,
                )
                return None
            return idx
        if not isinstance(mode, Integral) or isinstance(mode, bool) or not 1 <= int(mode) <= n:
            raise ValueError(
                f"n_basis must be None, 'auto' or an integer in [1, {n}], got {mode!r}"
            )
        return select_basis(xs, lam, 1e-12, max_size=int(mode))[0]

    def _fit_subspace(
        self,
        xs: Array,
        y: Array,
        w: Array,
        idx: NDArray[np.intp],
        offset: Array | None = None,
    ) -> None:
        prob = SubspaceProblem(xs, xs[idx], y, self._loss(), self.lam, w, offset)
        sol = newton_subspace(prob, **(self.newton_options or {}))
        self.basis_ = idx
        self.support_ = idx
        self.dual_coef_ = sol.beta
        self.intercept_ = sol.b
        self.n_support_ = idx.size
        self.training_info_ = {
            "residual": sol.residual,
            "smo_steps": 0,
            "newton_steps": sol.newton_steps,
            "time_smo": 0.0,
            "time_newton": sol.time_newton,
            "c": 0.0,
        }
        self._model = _Model(xs[idx], sol.beta, sol.b)

    def _fit_model(
        self, x: Array, y: Array, w: Array, offset: Array | None = None, lam: float | None = None
    ) -> None:
        lam = float(self.lam if lam is None else lam)
        self.blocks_ = resolve_blocks(self.blocks, x.shape[1])
        self.scaling_ = self._scaling(x)
        xs = scale_features(x, self.scaling_, self.blocks_)
        self.__dict__.pop("basis_", None)
        idx = self._basis(xs, float(self.lam))
        if idx is not None:
            self._fit_subspace(xs, y, w, idx, offset)
            return
        prob = TrainingProblem(xs, y, self._loss(), lam, w, self.max_asum, offset)
        sol = train(prob, self.smo_options, self.newton_options)
        beta = prob.sample_coefficients(sol.a)
        support = np.flatnonzero(beta)
        self.support_ = support
        self.dual_coef_ = beta[support] / lam
        self.intercept_ = sol.b
        self.n_support_ = support.size
        self.training_info_ = {
            "residual": sol.residual,
            "smo_steps": sol.smo_steps,
            "newton_steps": sol.newton_steps,
            "time_smo": sol.time_smo,
            "time_newton": sol.time_newton,
            "c": sol.c,
        }
        self._model = _Model(xs[support], self.dual_coef_, sol.b)

    def _fit_softmax_model(self, x: Array, yi: NDArray[np.intp], w: Array, r: int) -> None:
        """Soft-max model for ``r >= 3`` classes (labels ``yi`` in ``0..r-1``) at ``self.lam``."""
        self.blocks_ = resolve_blocks(self.blocks, x.shape[1])
        self.scaling_ = self._scaling(x)
        xs = scale_features(x, self.scaling_, self.blocks_)
        self.__dict__.pop("basis_", None)
        _warn_softmax_cost(r, x.shape[0])
        lam = float(self.lam)
        prob = MulticlassProblem(xs, yi.astype(np.float64), r, lam, w)
        sol = newton_multiclass(prob, **(self.newton_options or {}))
        self.support_ = np.arange(x.shape[0])
        self.dual_coef_ = sol.a / lam
        self.intercept_ = sol.b  # type: ignore[assignment]  # one per class
        self.n_support_ = x.shape[0]
        self.training_info_ = {
            "residual": sol.residual,
            "smo_steps": 0,
            "newton_steps": sol.newton_steps,
            "time_smo": 0.0,
            "time_newton": sol.time_newton,
            "c": 0.0,
        }
        self._mc = _SoftmaxModel(xs, sol.a, sol.b, lam)

    def _decision(self, X: ArrayLike) -> Array:
        check_is_fitted(self)
        x = validate_data(self, X, reset=False, dtype=np.float64)
        return self._model.decision(scale_features(x, self.scaling_, self.blocks_))

    def decision_gradient(self, X: ArrayLike) -> Array:
        """Gradient of the decision function w.r.t. the input features, shape ``(n, d)``.

        Useful as a local sensitivity / feature relevance measure.
        """
        check_is_fitted(self)
        if getattr(self, "_mc", None) is not None:
            raise NotImplementedError(
                "decision_gradient is not available for multi_class='softmax'"
            )
        x = validate_data(self, X, reset=False, dtype=np.float64)
        xs = scale_features(x, self.scaling_, self.blocks_)
        sv = self._model.support_vectors
        k = rbf(xs, sv) * self._model.coef  # (n, n_sv)
        grad_scaled = -2.0 * (k.sum(axis=1)[:, None] * xs - k @ sv)
        factor = scale_features(np.ones((1, x.shape[1])), self.scaling_, self.blocks_)[0]
        return grad_scaled * factor


@dataclass
class _SoftmaxModel:
    """Soft-max model ``T(x) = K(x, X) A / lam + 1 b^T`` on scaled features."""

    x: Array
    a: Array
    b: Array
    lam: float

    def decision(self, xs: Array) -> Array:
        return multiclass_decision(xs, self.x, self.a, self.b, self.lam)


#: warn if the dense Newton system of the soft-max model (size ``r * n``) exceeds this
SOFTMAX_MAX_SIZE = 4000


def _warn_softmax_cost(r: int, n: int) -> None:
    if r * n > SOFTMAX_MAX_SIZE:
        warnings.warn(
            f"multi_class='softmax' solves dense Newton systems of size r * n = {r * n} "
            f"(> {SOFTMAX_MAX_SIZE}, cost O((r n)^3) per step); this can be very slow and "
            "memory hungry, consider multi_class='ovr' or fewer samples",
            UserWarning,
            stacklevel=4,
        )


_SMOOTH_DOC = """
    lam : float
        Regularization parameter ``lambda > 0`` (``1 / C`` in LIBSVM terms).
    scaling : float, array or "auto"
        Feature scaling(s) ``s_k``; the kernel is ``exp(-sum_k s_k^2 |x_B_k - z_B_k|^2)``.
        "auto" chooses ``s`` such that the mean kernel value is about 1/2.
    blocks : None, "features" or list of lists
        Feature blocks sharing one scaling (None: a single scaling, i.e. an RBF kernel).
    smoothing : float
        Smoothing parameter ``delta`` of the loss (``0``: non-smooth, SMO only).
    degree : {"2", "3", "6"}
        Degree of the smoothed maximum function (``"2"`` is solved exactly by SMO).
    max_asum : float, optional
        Bound on ``sum |alpha|`` (generalised nu-SVM).
    n_basis : None, int or "auto"
        Subspace SVM (Strasdat, Sec. 5.3): restrict the model to the span of this many basis
        points, chosen by greedy pivoted Cholesky, and train by Newton's method with systems of
        that size. Needs a smooth loss (``smoothing > 0``) and no ``max_asum``. ``"auto"`` adds
        points until the Nystroem residual ``tr(K - K_s)`` is at most ``basis_tol * lam`` (at most
        ``n // 10`` points; otherwise a warning is issued and the full model is used). Pays off
        for fits with fixed hyperparameters on large regression problems with wide kernels;
        experimental in the optimisers, where it has not yet saved time without losing accuracy
        (see the theory page on the subspace SVM). The optimisers choose the basis at the
        starting values of each stage and keep it fixed during L-BFGS-B. With ``"auto"`` they
        then check the residual at the result and, if it exceeds ``basis_tol * lam`` in some
        fold, choose a new basis there and optimise again (at most 5 rounds per stage;
        ``history_`` entries carry ``basis_round``). The result of a stage is the best
        evaluation of its last round. An integer keeps its basis for the stage.
    basis_tol : float
        Tolerance of ``n_basis="auto"`` (1: CV error within about 1 % of the full model, 10: about
        10 %).
    smo_options, newton_options : dict, optional
        Options of :func:`smo_solver.solve_classification` / :func:`optimized_svm.training.newton`.
"""


class SmoothSVC(ClassifierMixin, _SmoothBase):
    __doc__ = (
        """SVM classifier with a smoothed hinge or a logistic loss and fixed hyperparameters.

    Solves ``min lam/2 |w|^2 + sum_i c_i m_delta(shift - y_i f(x_i))`` (``loss="hinge"``) or
    ``min lam/2 |w|^2 + sum_i c_i log(1 + exp(-y_i f(x_i)))`` (``loss="logistic"``).

    Parameters
    ----------"""
        + _SMOOTH_DOC
        + """    loss : {"hinge", "logistic"}
        ``hinge``: smoothed hinge loss, trained by SMO + Newton, sparse solution. ``logistic``:
        logistic loss (Strasdat, Sec. 4.1.3), trained by Newton's method only; the solution is
        dense and every Newton system has the full size ``n``. It enables
        :meth:`predict_proba` (``sigmoid`` of the decision values, normalised for more than two
        classes) and does not support ``max_asum``; ``smoothing``, ``degree`` and ``shift`` are
        ignored.
    shift : float
        Margin of the hinge loss.
    class_weight : "balanced", dict or None
        Per-class weights multiplied with ``sample_weight``.
    multi_class : {"ovr", "softmax"}
        Treatment of more than two classes. ``"ovr"``: one binary model per class (one-vs-rest,
        each with its own hyperparameters in the optimising estimators). ``"softmax"``: one joint
        multinomial model (Strasdat, Sec. 5.1) with a decision function per class, a common
        ``lam`` and common scalings, trained by Newton's method on the multinomial logistic loss.
        It needs ``loss="logistic"`` and no ``n_basis`` or ``max_asum``; ``predict_proba`` returns
        the soft-max probabilities (no normalisation of one-vs-rest sigmoids) and
        ``decision_function`` has shape ``(n, n_classes)``. The Newton systems are dense of size
        ``r (n + 1)`` for ``r`` classes (cost ``O((r n)^3)`` per step; a ``UserWarning`` is issued
        if ``r n > 4000``). With two classes the binary logistic model is used, which is the same
        model: soft-max with ``lam`` equals the binary model with ``lam / 2``, and the binary
        model with ``lam / 2`` is what is fitted (``lam`` keeps the soft-max meaning for any
        number of classes). ``newton_options`` then takes the options of
        :func:`optimized_svm.multiclass.newton_multiclass` (``tol``, ``max_steps``, ...).
    """
    )

    def __init__(
        self,
        lam: float = 1.0,
        *,
        loss: str = "hinge",
        scaling: float | ArrayLike | str = "auto",
        blocks: Any = None,
        smoothing: float = 0.5,
        degree: Degree = "2",
        shift: float = 1.0,
        max_asum: float | None = None,
        n_basis: int | str | None = None,
        basis_tol: float = 1.0,
        class_weight: Any = None,
        multi_class: str = "ovr",
        smo_options: dict[str, Any] | None = None,
        newton_options: dict[str, Any] | None = None,
    ) -> None:
        self.lam = lam
        self.loss = loss
        self.scaling = scaling
        self.blocks = blocks
        self.smoothing = smoothing
        self.degree = degree
        self.shift = shift
        self.max_asum = max_asum
        self.n_basis = n_basis
        self.basis_tol = basis_tol
        self.class_weight = class_weight
        self.multi_class = multi_class
        self.smo_options = smo_options
        self.newton_options = newton_options

    def _loss(self) -> Loss:
        if self.loss == "hinge":
            return HingeLoss(self.shift, self.smoothing, self.degree)
        if self.loss == "logistic":
            if self.max_asum is not None:
                raise ValueError("max_asum is not supported for the logistic loss")
            return LogisticLoss()
        raise ValueError(f"loss must be one of {_CLASSIFICATION_LOSSES}, got {self.loss!r}")

    def fit(self, X: ArrayLike, y: ArrayLike, sample_weight: ArrayLike | None = None) -> SmoothSVC:
        self._check_params()
        _check_multi_class(self)
        self._loss()  # validates `loss`
        x, y = validate_data(self, X, y, dtype=np.float64)
        y = np.asarray(y)
        if self.multi_class == "softmax":
            yi, w, r = _softmax_targets(self, y, x, sample_weight)
            if r >= 3:
                self._fit_softmax_model(x, yi, w, r)
            else:  # the same model as the binary one with lam / 2
                self._fit_model(x, np.where(yi == 1, 1.0, -1.0), w, lam=float(self.lam) / 2)
            return self
        if _fit_one_vs_rest(self, x, y, sample_weight):
            return self
        labels, w = _binary_labels(self, y, x, sample_weight)
        self._fit_model(x, labels, w)
        return self

    def decision_function(self, X: ArrayLike) -> Array:
        """Signed decision values: shape ``(n,)`` for two classes (positive values predict
        ``classes_[1]``), otherwise ``(n, n_classes)`` (one-vs-rest)."""
        return _ovr_decision(self, X)

    def predict(self, X: ArrayLike) -> NDArray[Any]:
        return _ovr_predict(self, X)

    @available_if(lambda est: est.loss == "logistic")  # type: ignore[untyped-decorator]
    def predict_proba(self, X: ArrayLike) -> Array:
        """Class probabilities ``sigmoid(f(x))`` (only for ``loss="logistic"``); for more than
        two classes the one-vs-rest probabilities are normalised to sum to one."""
        return _ovr_proba(self, X)


_CLASSIFICATION_LOSSES = ("hinge", "logistic")


def _fit_one_vs_rest(
    est: Any,
    x: Array,
    y: Array,
    sample_weight: ArrayLike | None,
    groups: ArrayLike | None = None,
) -> bool:
    """Fits one binary clone per class if there are more than two classes.

    ``groups`` (only for the optimizing estimators) is passed on to every binary fit."""
    check_classification_targets(y)
    classes = np.unique(y)
    est.estimators_ = None
    est._mc = None
    if classes.size <= 2:
        return False
    est.classes_ = classes
    w = _weights(sample_weight, x)
    if isinstance(est.class_weight, dict):
        w = w * np.array([est.class_weight.get(c, 1.0) for c in y])
    params = {"class_weight": None} if isinstance(est.class_weight, dict) else {}
    extra = {} if groups is None else {"groups": groups}
    est.estimators_ = [
        clone(est).set_params(**params).fit(x, (y == c).astype(int), sample_weight=w, **extra)
        for c in classes
    ]
    if hasattr(est.estimators_[0], "n_iter_"):
        est.n_iter_ = max(e.n_iter_ for e in est.estimators_)
    return True


def _check_multi_class(est: Any) -> None:
    if est.multi_class not in ("ovr", "softmax"):
        raise ValueError(f"multi_class must be 'ovr' or 'softmax', got {est.multi_class!r}")
    if est.multi_class == "softmax":
        if est.loss != "logistic":
            raise ValueError("multi_class='softmax' needs loss='logistic'")
        if est.n_basis is not None:
            raise ValueError("multi_class='softmax' does not support n_basis")
        if est.max_asum is not None:
            raise ValueError("multi_class='softmax' does not support max_asum")


def _softmax_targets(
    est: Any, y: Array, x: Array, sample_weight: ArrayLike | None
) -> tuple[NDArray[np.intp], Array, int]:
    """Integer labels, weights (``sample_weight`` times class weights) and number of classes."""
    check_classification_targets(y)
    est.estimators_ = None
    est._mc = None
    est.classes_, yi = np.unique(y, return_inverse=True)
    r = int(est.classes_.size)
    if r < 2:
        raise ValueError(f"The number of classes has to be greater than one; got {r} class")
    w = _weights(sample_weight, x)
    if est.class_weight == "balanced":
        counts = np.bincount(yi, minlength=r)
        w = w * (yi.size / (r * counts[yi]))
    elif isinstance(est.class_weight, dict):
        w = w * np.array([est.class_weight.get(c, 1.0) for c in est.classes_])[yi]
    elif est.class_weight is not None:
        raise ValueError(f"invalid class_weight {est.class_weight!r}")
    return yi.astype(np.intp), w, r


def _ovr_decision(est: Any, X: ArrayLike) -> Array:
    check_is_fitted(est)
    if getattr(est, "_mc", None) is not None:
        x = validate_data(est, X, reset=False, dtype=np.float64)
        return est._mc.decision(scale_features(x, est.scaling_, est.blocks_))
    if est.estimators_ is None:
        return est._decision(X)
    x = validate_data(est, X, reset=False, dtype=np.float64)
    return np.stack([e.decision_function(x) for e in est.estimators_], axis=1)


def _ovr_proba(est: Any, X: ArrayLike) -> Array:
    if getattr(est, "_mc", None) is not None:
        return np.asarray(softmax(_ovr_decision(est, X), axis=1), dtype=np.float64)
    p = expit(_ovr_decision(est, X))
    if p.ndim == 1:
        return np.stack([1.0 - p, p], axis=1)
    return np.asarray(p / p.sum(axis=1, keepdims=True), dtype=np.float64)


def _ovr_predict(est: Any, X: ArrayLike) -> NDArray[Any]:
    dec = _ovr_decision(est, X)
    idx = (dec > 0).astype(int) if dec.ndim == 1 else dec.argmax(axis=1)
    return est.classes_[idx]


def _binary_labels(
    est: Any, y: ArrayLike, x: Array, sample_weight: ArrayLike | None
) -> tuple[Array, Array]:
    check_classification_targets(y)
    est.classes_, yi = np.unique(y, return_inverse=True)
    if est.classes_.size != 2:
        raise ValueError(
            f"The number of classes has to be greater than one; got {est.classes_.size} class"
        )
    w = _weights(sample_weight, x)
    if est.class_weight == "balanced":
        counts = np.bincount(yi, minlength=2)
        w = w * (yi.size / (2.0 * counts[yi]))
    elif isinstance(est.class_weight, dict):
        w = w * np.array([est.class_weight.get(c, 1.0) for c in est.classes_])[yi]
    elif est.class_weight is not None:
        raise ValueError(f"invalid class_weight {est.class_weight!r}")
    return np.where(yi == 1, 1.0, -1.0), w


def _weights(sample_weight: ArrayLike | None, x: Array) -> Array:
    w = _check_sample_weight(sample_weight, x, dtype=np.float64, ensure_non_negative=True)
    if not np.any(w > 0):
        raise ValueError("All sample weights are zero.")
    return w


_REGRESSION_LOSSES = ("epsilon_insensitive", "squared", "poisson", "approx_poisson")


class SmoothSVR(RegressorMixin, _SmoothBase):
    __doc__ = (
        """SVM regression with a smoothed loss and fixed hyperparameters.

    Parameters
    ----------"""
        + _SMOOTH_DOC
        + """    loss : {"epsilon_insensitive", "squared", "poisson", "approx_poisson"}
        ``epsilon_insensitive``: smoothed epsilon-SVR; ``squared``: LS-SVM; ``poisson``:
        Poisson regression with log link (predictions are ``exp(f)``).
    epsilon : float
        Width of the insensitive tube (in normalised units if ``normalize_y``).
    normalize_y : bool
        Normalise targets (median/mean absolute deviation for ``epsilon_insensitive``,
        mean/std for ``squared``; ignored for Poisson losses).
    """
    )

    def __init__(
        self,
        lam: float = 1.0,
        *,
        loss: str = "epsilon_insensitive",
        epsilon: float = 0.1,
        scaling: float | ArrayLike | str = "auto",
        blocks: Any = None,
        smoothing: float = 0.5,
        degree: Degree = "2",
        max_asum: float | None = None,
        n_basis: int | str | None = None,
        basis_tol: float = 1.0,
        normalize_y: bool = True,
        smo_options: dict[str, Any] | None = None,
        newton_options: dict[str, Any] | None = None,
    ) -> None:
        self.lam = lam
        self.loss = loss
        self.epsilon = epsilon
        self.scaling = scaling
        self.blocks = blocks
        self.smoothing = smoothing
        self.degree = degree
        self.max_asum = max_asum
        self.n_basis = n_basis
        self.basis_tol = basis_tol
        self.normalize_y = normalize_y
        self.smo_options = smo_options
        self.newton_options = newton_options

    def _loss(self) -> Loss:
        if self.loss == "epsilon_insensitive":
            return EpsilonInsensitiveLoss(self.epsilon, self.smoothing, self.degree)
        if self.loss == "squared":
            return SquaredLoss()
        if self.loss == "poisson":
            return PoissonLoss()
        if self.loss == "approx_poisson":
            return ApproxPoissonLoss()
        raise ValueError(f"loss must be one of {_REGRESSION_LOSSES}, got {self.loss!r}")

    def _label_mode(self) -> str:
        if not self.normalize_y or self.loss in ("poisson", "approx_poisson"):
            return "none"
        return "median" if self.loss == "epsilon_insensitive" else "mean"

    def fit(self, X: ArrayLike, y: ArrayLike, sample_weight: ArrayLike | None = None) -> SmoothSVR:
        self._check_params()
        x, y = validate_data(self, X, y, dtype=np.float64, y_numeric=True)
        y = np.asarray(y, dtype=np.float64)
        w = _weights(sample_weight, x)
        self._loss()  # validates `loss`
        if self.loss in ("poisson", "approx_poisson") and np.any(y < 0):
            raise ValueError("Poisson losses require non-negative targets")
        self.label_scaler_ = _LabelScaler(self._label_mode()).fit(y)
        self._fit_model(x, self.label_scaler_.transform(y), w)
        return self

    def predict(self, X: ArrayLike) -> Array:
        check_is_fitted(self)
        f = self.label_scaler_.inverse(self._decision(X))
        return np.exp(f) if self.loss in ("poisson", "approx_poisson") else f


# --------------------------------------------------------------------------------------------
# hyperparameter optimization


class _Timeout(Exception):
    pass


class _OptimizedMixin:
    """L-BFGS-B on the cross-validation error w.r.t. ``(log lam, log scalings)``."""

    # hyperparameters, set by the concrete estimators
    lam_init: float
    scaling_init: float | str
    blocks: Any
    two_stage: bool
    cv: Any
    lam_bounds: tuple[float, float]
    scaling_bounds: tuple[float, float]
    time_limit: float | None
    max_iter: int
    tol: float
    random_state: int | None
    verbose: int
    warm_start: bool
    n_jobs: int | None
    max_asum: float | None
    n_basis: int | str | None
    basis_tol: float
    smo_options: dict[str, Any] | None
    newton_options: dict[str, Any] | None
    _loss: Callable[[], Loss]
    _basis: Callable[[Array, float], NDArray[np.intp] | None]

    def _validation(self) -> ValidationLoss:
        raise NotImplementedError

    def _folds(self, x: Array, y: Array, groups: ArrayLike | None = None) -> list[Fold]:
        if hasattr(self.cv, "split"):  # scikit-learn splitter object
            return [(np.asarray(tr), np.asarray(va)) for tr, va in self.cv.split(x, y, groups)]
        if not isinstance(self.cv, Integral):
            return [(np.asarray(tr), np.asarray(va)) for tr, va in self.cv]
        splitter_cls = StratifiedKFold if isinstance(self, ClassifierMixin) else KFold
        splitter = splitter_cls(int(self.cv), shuffle=True, random_state=self.random_state)
        return [(tr, va) for tr, va in splitter.split(x, y)]

    def _optimize(
        self,
        x: Array,
        y: Array,
        w: Array,
        groups: ArrayLike | None = None,
        offset: Array | None = None,
        loss_params: Mapping[str, tuple[float, tuple[float, float]]] | None = None,
        n_classes: int | None = None,
        lam_factor: float = 1.0,
    ) -> None:
        """Minimises the cross-validation error.

        ``offset`` is a per-sample offset of the decision values. ``loss_params`` maps names of
        tunable loss parameters (e.g. ``"dispersion"``) to ``(initial value, bounds)``; their
        logarithms are appended to ``theta`` (after the scalings, through both stages) and the
        optimised values are stored in ``loss_params_opt_``.

        With ``n_classes`` the labels ``y`` are integers ``0..n_classes-1`` of the soft-max model
        (validated by the multinomial negative log-likelihood, weighted if there are class
        weights). ``lam_factor`` rescales ``lam_init`` and ``lam_bounds`` internally (the binary
        model equivalent to the soft-max model has ``lam / 2``); ``lam_`` and ``history_`` are
        reported in the unscaled convention.
        """
        softmax_model = n_classes is not None
        loss = self._loss()
        validation: Any = (
            MultinomialNLL(weighted=getattr(self, "class_weight", None) is not None)
            if softmax_model
            else self._validation()
        )
        params = dict(loss_params or {})
        names = tuple(params)
        n_extra = len(names)
        if n_extra and self.n_basis is not None:
            raise ValueError("tuning loss parameters is not supported together with n_basis")

        def scalings(theta: Array) -> Array:
            return np.exp(theta[1 : theta.size - n_extra])

        folds = self._folds(x, y, groups)
        if n_classes is not None:
            for i, (tr, _) in enumerate(folds):
                if np.unique(y[tr]).size < n_classes:
                    raise ValueError(
                        f"multi_class='softmax': the training set of fold {i} does not contain "
                        f"all {n_classes} classes; use fewer folds or a stratified splitter"
                    )
            _warn_softmax_cost(n_classes, max(len(tr) for tr, _ in folds))
        start = time.perf_counter()
        history: list[dict[str, Any]] = []
        lam_bounds = np.log(self.lam_bounds) + np.log(lam_factor)
        s_bounds = np.log(self.scaling_bounds)
        smo_options = dict(self.smo_options or {})
        warm: list[Any] | None = [None] * len(folds) if self.warm_start else None

        def select_bases(
            blocks: Blocks, theta: Array
        ) -> tuple[list[NDArray[np.intp]], list[NDArray[np.intp]]] | None:
            """Subspace bases of all folds for the kernel and lambda at ``theta``: global indices
            and indices relative to the training set of the fold (``None``: full model). Fixed
            during a round of L-BFGS-B, so that the objective stays smooth."""
            if self.n_basis is None:
                return None
            xs = scale_features(x, scalings(theta), blocks)
            lam = float(np.exp(theta[0]))
            bases, local = [], []
            for tr, _ in folds:
                idx = self._basis(xs[tr], lam)
                if idx is None:
                    return None
                bases.append(np.asarray(tr)[idx])
                local.append(idx)
            return bases, local

        def basis_adequate(blocks: Blocks, theta: Array, local: list[NDArray[np.intp]]) -> bool:
            """Whether the bases still meet ``basis_tol`` at the hyperparameters ``theta``."""
            xs = scale_features(x, scalings(theta), blocks)
            lam = float(np.exp(theta[0]))
            return all(
                nystroem_residual(xs[tr], idx) / lam <= self.basis_tol
                for (tr, _), idx in zip(folds, local, strict=True)
            )

        def run_stage(blocks: Blocks, theta0: Array, error0: float = np.nan) -> tuple[Array, float]:
            theta, error = theta0, error0
            adaptive = self.n_basis == "auto"
            max_rounds = 5 if adaptive else 1
            for rnd in range(max_rounds):
                selected = select_bases(blocks, theta)
                bases = None if selected is None else selected[0]
                if (bases is not None or rnd > 0) and warm is not None:
                    warm[:] = [None] * len(folds)  # coefficients of an earlier basis do not fit
                first = len(history)

                def fun(
                    theta: Array,
                    bases: list[NDArray[np.intp]] | None = bases,
                    rnd: int = rnd,
                ) -> tuple[float, Array]:
                    if (
                        self.time_limit is not None
                        and time.perf_counter() - start > self.time_limit
                    ):
                        raise _Timeout

                    def record(
                        theta: Array,
                        cv_err: float,
                        cv_grad: Array,
                        n_sup: float,
                        bases: list[NDArray[np.intp]] | None,
                    ) -> tuple[float, Array]:
                        history.append(
                            {
                                "lam": float(np.exp(theta[0])),
                                "scaling": scalings(theta),
                                "error": cv_err,
                                "gradient_norm": float(np.linalg.norm(cv_grad)),
                                "n_support": n_sup,
                                "n_basis": None
                                if bases is None
                                else float(np.mean([len(b) for b in bases])),
                                "time": time.perf_counter() - start,
                                "stage": "features"
                                if blocks is not None and self.blocks is not None
                                else "global",
                                "basis_round": rnd,
                                **{
                                    name: float(np.exp(v))
                                    for name, v in zip(
                                        names, theta[theta.size - n_extra :], strict=True
                                    )
                                },
                            }
                        )
                        if self.verbose:
                            h = history[-1]
                            print(
                                f"[{len(history):3d}] lam = {h['lam']:.3e}, mean scaling = "
                                f"{np.mean(h['scaling']):.3e}, cv error = {h['error']:.6f}"
                            )
                        return cv_err, cv_grad

                    if n_classes is not None:
                        mc = cv_error_and_gradient_multiclass(
                            theta, x, y, w, folds, n_classes, validation, blocks,
                            self.newton_options, warm, n_jobs=self.n_jobs,
                        )  # fmt: skip
                        n_sup = float(np.mean([len(tr) for tr, _ in folds]))
                        return record(theta, mc.error, mc.gradient, n_sup, bases)
                    res = cv_error_and_gradient(
                        theta,
                        x,
                        y,
                        w,
                        folds,
                        loss,
                        validation,
                        blocks,
                        self.max_asum,
                        smo_options,
                        self.newton_options,
                        warm,
                        names,
                        basis=bases,
                        offset=offset,
                        n_jobs=self.n_jobs,
                    )
                    n_sup = float(np.mean([f.n_support for f in res.folds]))
                    return record(theta, res.error, res.gradient, n_sup, bases)

                bounds = (
                    [tuple(lam_bounds)]
                    + [tuple(s_bounds)] * (theta.size - 1 - n_extra)
                    + [tuple(np.log(params[name][1])) for name in names]
                )
                timeout = False
                try:
                    result = minimize(
                        fun,
                        theta,
                        jac=True,
                        method="L-BFGS-B",
                        bounds=bounds,
                        options={"maxiter": self.max_iter, "ftol": self.tol, "gtol": self.tol},
                    )
                    self.optimize_result_ = result
                    self.n_iter_ = getattr(self, "n_iter_", 0) + int(result.nit)
                except _Timeout:
                    self.optimize_result_ = None
                    timeout = True
                round_entries = history[first:]
                if round_entries:
                    best = min(round_entries, key=lambda h: h["error"])
                    theta = np.log(
                        np.concatenate(
                            ([best["lam"]], best["scaling"], [best[name] for name in names])
                        )
                    )
                    error = best["error"]
                if timeout or selected is None or not round_entries or not adaptive:
                    break
                if basis_adequate(blocks, theta, selected[1]):
                    break
                if rnd == max_rounds - 1:
                    warnings.warn(
                        f"n_basis='auto': the basis still misses basis_tol={self.basis_tol:g} "
                        f"after {max_rounds} rounds; cv_error_ is that of a too small subspace",
                        stacklevel=2,
                    )
                    break
            return theta, error

        self.n_iter_ = 0
        blocks = resolve_blocks(self.blocks, x.shape[1])
        s0 = auto_scaling(x) if self.scaling_init == "auto" else float(self.scaling_init)
        theta = np.log([self.lam_init * lam_factor, s0, *(params[name][0] for name in names)])
        if blocks is not None and self.two_stage:
            # first a common scaling for all blocks, then individual ones
            theta, error = run_stage(None, theta)
        else:
            error = np.nan
        if blocks is not None:
            theta = np.concatenate(([theta[0]], np.full(len(blocks), theta[1]), theta[2:]))
            theta, error = run_stage(blocks, theta, error)
        else:
            theta, error = run_stage(None, theta)
        self.lam_ = float(np.exp(theta[0])) / lam_factor
        for h in history:
            h["lam"] /= lam_factor
        self.scaling_opt_ = scalings(theta)
        self.loss_params_opt_ = {
            name: float(np.exp(v))
            for name, v in zip(names, theta[theta.size - n_extra :], strict=True)
        }
        self.blocks_opt_ = blocks
        self.history_ = history
        self.cv_error_ = float(error)
        self.time_ = time.perf_counter() - start


_OPT_DOC = """
    lam_init : float
        Initial value of ``lambda``.
    scaling_init : float or "auto"
        Initial scaling (shared by all blocks).
    blocks : None, "features" or list of lists
        Feature blocks with individual scalings (``"features"``: ARD kernel).
    two_stage : bool
        With blocks, first optimise one common scaling, then the individual ones.
    cv : int, cross-validation splitter or iterable of (train, validation) index arrays
        Folds of the cross-validation error. An int gives shuffled (stratified for classifiers)
        ``k``-fold splits seeded by ``random_state``. Any scikit-learn splitter (an object with a
        ``split`` method, e.g. ``GroupKFold`` or ``LeaveOneGroupOut``) is called as
        ``cv.split(X, y, groups)`` with the ``groups`` passed to ``fit``; for ``OptimizedSVR`` ``y``
        is the normalised target. An iterable of index pairs is used as is.
    lam_bounds, scaling_bounds : tuple of float
        Box constraints of the hyperparameters.
    time_limit : float, optional
        Wall-clock limit in seconds; the best parameters found so far are used.
    max_iter, tol : int, float
        Options of L-BFGS-B.
    random_state : int, optional
        Seed of the fold shuffling.
    verbose : int
        Print progress.
    warm_start : bool
        Start SMO at the solution of the previous evaluation (per fold).
    n_jobs : int, optional
        Number of folds trained in parallel (threads; the BLAS threads are divided among them).
        ``None`` or 1: serial, -1: all cores. The result equals the serial one up to floating-point
        rounding (the BLAS thread count differs). The gain is largest when SMO dominates
        (classification); for large Newton-dominated problems the serial run already uses
        multithreaded BLAS. The memory grows with the number of parallel folds.

    Attributes
    ----------
    lam_, scaling_opt_ : float, ndarray
        Optimised hyperparameters.
    history_ : list of dict
        All evaluations of the cross-validation error.
    best_estimator_ : estimator
        Model refitted on all data with the optimised hyperparameters.
"""


class OptimizedSVC(_OptimizedMixin, SmoothSVC):
    __doc__ = (
        """:class:`SmoothSVC` whose ``lam`` and kernel scalings are optimised by minimising a
    smoothed cross-validation hinge error (``loss="hinge"``) or the cross-validation logistic
    loss (``loss="logistic"``) with exact gradients. With ``multi_class="softmax"`` (and more than
    two classes) one joint multinomial model is fitted whose common ``lam`` and scalings are
    tuned by the cross-validation multinomial negative log-likelihood (weighted by the sample and
    class weights if ``class_weight`` is given); every training fold must contain all classes.

    Parameters
    ----------"""
        + _OPT_DOC
        + "\n    Remaining parameters: see :class:`SmoothSVC`.\n"
    )

    def __init__(
        self,
        *,
        lam_init: float = 1e-2,
        scaling_init: float | str = "auto",
        blocks: Any = None,
        two_stage: bool = True,
        cv: Any = 5,
        lam_bounds: tuple[float, float] = (1e-6, 1.0),
        scaling_bounds: tuple[float, float] = (1e-4, 1e2),
        time_limit: float | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        random_state: int | None = 0,
        verbose: int = 0,
        warm_start: bool = True,
        n_jobs: int | None = None,
        loss: str = "hinge",
        smoothing: float = 0.5,
        degree: Degree = "2",
        shift: float = 1.0,
        max_asum: float | None = None,
        n_basis: int | str | None = None,
        basis_tol: float = 1.0,
        class_weight: Any = None,
        multi_class: str = "ovr",
        validation_smoothing: float = 0.5,
        smo_options: dict[str, Any] | None = None,
        newton_options: dict[str, Any] | None = None,
    ) -> None:
        self.lam_init = lam_init
        self.scaling_init = scaling_init
        self.blocks = blocks
        self.two_stage = two_stage
        self.cv = cv
        self.lam_bounds = lam_bounds
        self.scaling_bounds = scaling_bounds
        self.time_limit = time_limit
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose
        self.warm_start = warm_start
        self.n_jobs = n_jobs
        self.loss = loss
        self.smoothing = smoothing
        self.degree = degree
        self.shift = shift
        self.max_asum = max_asum
        self.n_basis = n_basis
        self.basis_tol = basis_tol
        self.class_weight = class_weight
        self.multi_class = multi_class
        self.validation_smoothing = validation_smoothing
        self.smo_options = smo_options
        self.newton_options = newton_options

    @property
    def lam(self) -> float:  # type: ignore[override]  # used by the fixed-parameter base class
        return getattr(self, "lam_", self.lam_init)

    @property
    def scaling(self) -> Any:  # type: ignore[override]
        return getattr(self, "scaling_opt_", self.scaling_init)

    def _validation(self) -> ValidationLoss:
        balanced = self.class_weight == "balanced"
        if self.loss == "logistic":
            return LogisticValidation(balanced=balanced)
        return SmoothHingeValidation(self.shift, self.validation_smoothing, "6", balanced=balanced)

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: ArrayLike | None = None,
        groups: ArrayLike | None = None,
    ) -> OptimizedSVC:
        """Fits the model; ``groups`` is passed to ``cv.split`` (splitter objects only)."""
        _check_multi_class(self)
        for name in ("lam_", "scaling_opt_"):
            self.__dict__.pop(name, None)
        x, y = validate_data(self, X, y, dtype=np.float64)
        y = np.asarray(y)
        if self.multi_class == "softmax":
            yi, w, r = _softmax_targets(self, y, x, sample_weight)
            if r >= 3:
                self._optimize(x, yi.astype(np.float64), w, groups, n_classes=r)
                self._fit_softmax_model(x, yi, w, r)
            else:  # the same model as the binary one with lam / 2: tune that, report lam
                labels = np.where(yi == 1, 1.0, -1.0)
                self._optimize(x, labels, w, groups, lam_factor=0.5)
                self._fit_model(x, labels, w, lam=self.lam_ / 2)
            return self
        if _fit_one_vs_rest(self, x, y, sample_weight, groups):
            return self
        labels, w = _binary_labels(self, y, x, sample_weight)
        self._optimize(x, labels, w, groups)
        self._fit_model(x, labels, w)
        return self


class OptimizedSVR(_OptimizedMixin, SmoothSVR):
    __doc__ = (
        """:class:`SmoothSVR` whose ``lam`` and kernel scalings are optimised by minimising a
    cross-validation error with exact gradients.

    Parameters
    ----------"""
        + _OPT_DOC
        + """    validation : {"absolute", "squared", "poisson"} or None
        Validation error; default: smoothed absolute error (``epsilon_insensitive``),
        squared error (``squared``) or Poisson loss.
    validation_smoothing : float
        Smoothing of the absolute error (Huber-like, in normalised units).

    Remaining parameters: see :class:`SmoothSVR`.
    """
    )

    def __init__(
        self,
        *,
        loss: str = "epsilon_insensitive",
        epsilon: float = 0.1,
        lam_init: float = 1e-2,
        scaling_init: float | str = "auto",
        blocks: Any = None,
        two_stage: bool = True,
        cv: Any = 5,
        lam_bounds: tuple[float, float] = (1e-6, 1.0),
        scaling_bounds: tuple[float, float] = (1e-4, 1e2),
        time_limit: float | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        random_state: int | None = 0,
        verbose: int = 0,
        warm_start: bool = True,
        n_jobs: int | None = None,
        smoothing: float = 0.5,
        degree: Degree = "2",
        max_asum: float | None = None,
        n_basis: int | str | None = None,
        basis_tol: float = 1.0,
        normalize_y: bool = True,
        validation: str | None = None,
        validation_smoothing: float = 0.1,
        smo_options: dict[str, Any] | None = None,
        newton_options: dict[str, Any] | None = None,
    ) -> None:
        self.loss = loss
        self.epsilon = epsilon
        self.lam_init = lam_init
        self.scaling_init = scaling_init
        self.blocks = blocks
        self.two_stage = two_stage
        self.cv = cv
        self.lam_bounds = lam_bounds
        self.scaling_bounds = scaling_bounds
        self.time_limit = time_limit
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose
        self.warm_start = warm_start
        self.n_jobs = n_jobs
        self.smoothing = smoothing
        self.degree = degree
        self.max_asum = max_asum
        self.n_basis = n_basis
        self.basis_tol = basis_tol
        self.normalize_y = normalize_y
        self.validation = validation
        self.validation_smoothing = validation_smoothing
        self.smo_options = smo_options
        self.newton_options = newton_options

    @property
    def lam(self) -> float:  # type: ignore[override]
        return getattr(self, "lam_", self.lam_init)

    @property
    def scaling(self) -> Any:  # type: ignore[override]
        return getattr(self, "scaling_opt_", self.scaling_init)

    def _validation(self) -> ValidationLoss:
        kind = self.validation
        if kind is None:
            kind = {"epsilon_insensitive": "absolute", "squared": "squared"}.get(
                self.loss, "poisson"
            )
        if kind == "absolute":
            return SmoothAbsoluteValidation(self.validation_smoothing)
        if kind == "squared":
            return SquaredValidation()
        if kind == "poisson":
            return PoissonValidation()
        raise ValueError(f"unknown validation {kind!r}")

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: ArrayLike | None = None,
        groups: ArrayLike | None = None,
    ) -> OptimizedSVR:
        """Fits the model; ``groups`` is passed to ``cv.split`` (splitter objects only)."""
        for name in ("lam_", "scaling_opt_"):
            self.__dict__.pop(name, None)
        x, y = validate_data(self, X, y, dtype=np.float64, y_numeric=True)
        y = np.asarray(y, dtype=np.float64)
        w = _weights(sample_weight, x)
        self._loss()
        self.label_scaler_ = _LabelScaler(self._label_mode()).fit(y)
        yn = self.label_scaler_.transform(y)
        self._optimize(x, yn, w, groups)
        self._fit_model(x, yn, w)
        return self


# --------------------------------------------------------------------------------------------
# generalized linear models

_GLM_FAMILIES = ("poisson", "binomial", "negative_binomial", "gamma", "tweedie")


def _check_offset(offset: ArrayLike | None, n: int) -> Array | None:
    if offset is None:
        return None
    off = np.asarray(offset, dtype=np.float64)
    if off.shape != (n,):
        raise ValueError(f"offset must have shape ({n},), got {off.shape}")
    if not np.all(np.isfinite(off)):
        raise ValueError("offset must be finite")
    return off


def _check_glm_targets(family: str, y: Array) -> None:
    if family == "binomial":
        if np.any((y < 0) | (y > 1)):
            raise ValueError(
                "family='binomial' needs targets that are proportions in [0, 1] "
                "(pass the numbers of trials as sample_weight)"
            )
    elif family == "gamma":
        if np.any(y <= 0):
            raise ValueError("family='gamma' needs strictly positive targets")
    elif np.any(y < 0):
        raise ValueError(f"family={family!r} needs non-negative targets")


def _glm_loss(family: str, dispersion: float, power: float) -> Loss:
    if family == "poisson":
        return PoissonLoss()
    if family == "binomial":
        return BinomialLoss()
    if family == "negative_binomial":
        if not isinstance(dispersion, Real) or not dispersion > 0:
            raise ValueError(f"dispersion must be positive, got {dispersion!r}")
        return NegativeBinomialLoss(float(dispersion))
    if family == "gamma":
        return GammaLoss()
    if family == "tweedie":
        if not isinstance(power, Real) or not 1.0 < power < 2.0:
            raise ValueError(f"power must lie in (1, 2), got {power!r}")
        return TweedieLoss(float(power))
    raise ValueError(f"family must be one of {_GLM_FAMILIES}, got {family!r}")


def _glm_deviance(family: str, loss: Loss, eta: Array, y: Array, w: Array) -> float:
    """Summed weighted half deviance of ``eta`` (zero for a perfect fit)."""
    value = float(loss.value(eta, loss.terms(y, w)).sum())
    if (
        family == "poisson"
    ):  # PoissonLoss is the negative log-likelihood; subtract the saturated one
        value -= float(w @ (y - xlogy(y, y) + gammaln(y + 1)))
    return value


_GLM_DOC = """
    lam : float
        Regularization parameter ``lambda > 0``.
    family : {"poisson", "binomial", "negative_binomial", "gamma", "tweedie"}
        Distribution family. The link is ``log`` except for ``binomial`` (``logit``). Targets:
        ``y >= 0`` (Poisson, negative binomial, Tweedie), ``y > 0`` (Gamma), proportions in
        ``[0, 1]`` (binomial; ``sample_weight`` are the numbers of trials).
    dispersion : float
        Dispersion ``r`` of the negative binomial family (variance ``mu + mu^2 / r``).
    power : float
        Variance power ``p`` in ``(1, 2)`` of the Tweedie family (variance ``mu^p``).
    scaling : float, array or "auto"
        Feature scaling(s), see :class:`SmoothSVR`.
    blocks : None, "features" or list of lists
        Feature blocks sharing one scaling.
    n_basis : None, int or "auto"
        Subspace SVM, see :class:`SmoothSVR`.
    basis_tol : float
        Tolerance of ``n_basis="auto"``.
    newton_options : dict, optional
        Options of :func:`optimized_svm.training.newton`.
"""


class KernelGLMRegressor(RegressorMixin, _SmoothBase):
    __doc__ = (
        """Kernel generalized linear model with fixed hyperparameters.

    Minimises ``lam/2 |f|^2 + sum_i w_i d(y_i, g^-1(f(x_i) + offset_i))`` over the RKHS of the
    Gaussian kernel, where ``d`` is the half unit deviance of the family and ``g`` its link.
    Trained by Newton's method (no SMO); the solution is dense.

    Parameters
    ----------"""
        + _GLM_DOC
        + """
    Notes
    -----
    ``fit(X, y, sample_weight=None, offset=None)``: ``offset`` is an additive term on the link
    scale, e.g. ``log(exposure)``. ``decision_function(X, offset=None)`` is the linear predictor
    (including the offset), ``predict(X, offset=None)`` the mean (``exp`` of it, ``expit`` for
    the binomial family, i.e. the predicted proportion). ``score`` is the deviance explained
    ``D^2 = 1 - deviance(model) / deviance(null model)``, the null model being the
    intercept-only fit (on the link scale it is ``b + offset``).
    """
    )

    #: no smoothing parameter, no 1-norm constraint, no SMO: read by the shared training code
    smoothing = 1.0
    max_asum: float | None = None
    smo_options: dict[str, Any] | None = None

    def __init__(
        self,
        lam: float = 1.0,
        *,
        family: str = "poisson",
        dispersion: float = 1.0,
        power: float = 1.5,
        scaling: float | ArrayLike | str = "auto",
        blocks: Any = None,
        n_basis: int | str | None = None,
        basis_tol: float = 1.0,
        newton_options: dict[str, Any] | None = None,
    ) -> None:
        self.lam = lam
        self.family = family
        self.dispersion = dispersion
        self.power = power
        self.scaling = scaling
        self.blocks = blocks
        self.n_basis = n_basis
        self.basis_tol = basis_tol
        self.newton_options = newton_options

    def __sklearn_tags__(self) -> Any:
        tags = super().__sklearn_tags__()
        tags.target_tags.positive_only = self.family != "binomial"
        tags.regressor_tags.poor_score = True  # D^2 on arbitrary targets, as in sklearn's GLMs
        return tags

    def _dispersion(self) -> float:
        return self.dispersion

    def _loss(self) -> Loss:
        return _glm_loss(self.family, self._dispersion(), self.power)

    def _prepare(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: ArrayLike | None,
        offset: ArrayLike | None,
    ) -> tuple[Array, Array, Array, Array | None]:
        self._check_params()
        self._loss()  # validates family and its parameters
        x, y = validate_data(self, X, y, dtype=np.float64, y_numeric=True)
        y = np.asarray(y, dtype=np.float64)
        _check_glm_targets(self.family, y)
        w = _weights(sample_weight, x)
        return x, y, w, _check_offset(offset, x.shape[0])

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: ArrayLike | None = None,
        offset: ArrayLike | None = None,
    ) -> KernelGLMRegressor:
        x, y, w, off = self._prepare(X, y, sample_weight, offset)
        if self.family == "negative_binomial":
            self.dispersion_ = float(self.dispersion)
        self._fit_model(x, y, w, off)
        return self

    def decision_function(self, X: ArrayLike, offset: ArrayLike | None = None) -> Array:
        """Linear predictor on the link scale, including ``offset``."""
        eta = self._decision(X)
        off = _check_offset(offset, eta.shape[0])
        return eta if off is None else eta + off

    def predict(self, X: ArrayLike, offset: ArrayLike | None = None) -> Array:
        """Predicted mean: ``exp(eta)``, for the binomial family the proportion ``expit(eta)``."""
        eta = self.decision_function(X, offset)
        return np.asarray(expit(eta) if self.family == "binomial" else np.exp(eta))

    def score(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: ArrayLike | None = None,
        offset: ArrayLike | None = None,
    ) -> float:
        """Deviance explained ``D^2 = 1 - deviance(model) / deviance(null model)``.

        The null model is the intercept-only fit to ``(y, sample_weight, offset)``: the weighted
        mean (with ``offset``: the maximum likelihood ``b`` with ``mu_i = g^-1(b + offset_i)``).
        """
        x, y, w, off = self._prepare_score(X, y, sample_weight, offset)
        loss = self._loss()
        terms = loss.terms(y, w)
        if off is not None:
            terms = dataclasses.replace(terms, offset=off)
        null = np.full(y.shape, loss.initial_bias(terms))
        if off is not None:
            null = null + off
        d0 = _glm_deviance(self.family, loss, null, y, w)
        d1 = _glm_deviance(self.family, loss, self.decision_function(x, off), y, w)
        return 1.0 - d1 / d0 if d0 > 0 else 0.0

    def _prepare_score(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: ArrayLike | None,
        offset: ArrayLike | None,
    ) -> tuple[Array, Array, Array, Array | None]:
        check_is_fitted(self)
        x, y = validate_data(self, X, y, reset=False, dtype=np.float64, y_numeric=True)
        y = np.asarray(y, dtype=np.float64)
        _check_glm_targets(self.family, y)
        return x, y, _weights(sample_weight, x), _check_offset(offset, x.shape[0])


class OptimizedKernelGLMRegressor(_OptimizedMixin, KernelGLMRegressor):
    __doc__ = (
        """:class:`KernelGLMRegressor` whose ``lam`` and kernel scalings (and optionally the
    negative binomial dispersion) are optimised by minimising the cross-validation deviance with
    exact gradients.

    The validation loss is the half deviance of the family (binomial: weighted by the numbers of
    trials, i.e. ``sample_weight``); the offset enters the validation predictions as well.

    Parameters
    ----------"""
        + _OPT_DOC
        + _GLM_DOC
        + """    tune_dispersion : bool
        Optimise the dispersion of the negative binomial family together with the other
        hyperparameters (not with ``n_basis``). The validation loss is then the full negative
        binomial negative log-likelihood (:class:`NegativeBinomialNLL`), which depends on the
        dispersion; the hypergradient contains its direct derivative.
    dispersion_bounds : tuple of float
        Box constraints of the dispersion.

    Attributes
    ----------
    dispersion_ : float
        Optimised (``tune_dispersion``) or given dispersion (negative binomial only).
    """
    )

    def __init__(
        self,
        *,
        family: str = "poisson",
        dispersion: float = 1.0,
        power: float = 1.5,
        tune_dispersion: bool = False,
        dispersion_bounds: tuple[float, float] = (1e-2, 1e3),
        lam_init: float = 1e-2,
        scaling_init: float | str = "auto",
        blocks: Any = None,
        two_stage: bool = True,
        cv: Any = 5,
        lam_bounds: tuple[float, float] = (1e-6, 1.0),
        scaling_bounds: tuple[float, float] = (1e-4, 1e2),
        time_limit: float | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        random_state: int | None = 0,
        verbose: int = 0,
        warm_start: bool = True,
        n_jobs: int | None = None,
        n_basis: int | str | None = None,
        basis_tol: float = 1.0,
        newton_options: dict[str, Any] | None = None,
    ) -> None:
        self.family = family
        self.dispersion = dispersion
        self.power = power
        self.tune_dispersion = tune_dispersion
        self.dispersion_bounds = dispersion_bounds
        self.lam_init = lam_init
        self.scaling_init = scaling_init
        self.blocks = blocks
        self.two_stage = two_stage
        self.cv = cv
        self.lam_bounds = lam_bounds
        self.scaling_bounds = scaling_bounds
        self.time_limit = time_limit
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose
        self.warm_start = warm_start
        self.n_jobs = n_jobs
        self.n_basis = n_basis
        self.basis_tol = basis_tol
        self.newton_options = newton_options

    @property
    def lam(self) -> float:  # type: ignore[override]
        return getattr(self, "lam_", self.lam_init)

    @property
    def scaling(self) -> Any:  # type: ignore[override]
        return getattr(self, "scaling_opt_", self.scaling_init)

    def _dispersion(self) -> float:
        return getattr(self, "dispersion_", self.dispersion)

    def _validation(self) -> ValidationLoss:
        if self.family == "negative_binomial":
            if self.tune_dispersion:
                # full likelihood; the hypergradient rebuilds it with the current dispersion
                return NegativeBinomialNLL(float(self.dispersion))
            return NegativeBinomialDeviance(float(self.dispersion))
        if self.family == "poisson":
            return PoissonValidation()
        if self.family == "binomial":
            return BinomialDeviance()
        if self.family == "gamma":
            return GammaDeviance()
        return TweedieDeviance(float(self.power))

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: ArrayLike | None = None,
        offset: ArrayLike | None = None,
        groups: ArrayLike | None = None,
    ) -> OptimizedKernelGLMRegressor:
        """Fits the model; ``groups`` is passed to ``cv.split`` (splitter objects only)."""
        for name in ("lam_", "scaling_opt_", "dispersion_"):
            self.__dict__.pop(name, None)
        x, y, w, off = self._prepare(X, y, sample_weight, offset)
        tuned: dict[str, tuple[float, tuple[float, float]]] = {}
        if self.tune_dispersion:
            if self.family != "negative_binomial":
                raise ValueError("tune_dispersion needs family='negative_binomial'")
            if self.n_basis is not None:
                raise ValueError("tune_dispersion is not supported together with n_basis")
            lo, hi = self.dispersion_bounds
            if not 0 < lo < hi:
                raise ValueError(f"dispersion_bounds must satisfy 0 < low < high, got {(lo, hi)}")
            tuned["dispersion"] = (float(self.dispersion), (float(lo), float(hi)))
        self._optimize(x, y, w, groups, off, tuned)
        if self.family == "negative_binomial":
            self.dispersion_ = self.loss_params_opt_.get("dispersion", float(self.dispersion))
        self._fit_model(x, y, w, off)
        return self

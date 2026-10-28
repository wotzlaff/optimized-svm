"""scikit-learn compatible estimators.

* :class:`SmoothSVC`, :class:`SmoothSVR`: kernel SVMs with smoothed losses and *fixed*
  hyperparameters, trained by SMO + Newton.
* :class:`OptimizedSVC`, :class:`OptimizedSVR`: the same models whose regularization parameter
  and kernel scalings are tuned by minimising a cross-validation error with exact gradients
  (bilevel approach of Strasdat, Ch. 9).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import _check_sample_weight, check_is_fitted, validate_data

from .hypergradient import Fold, cv_error_and_gradient
from .kernels import Blocks, rbf, scale_features
from .losses import (
    ApproxPoissonLoss,
    EpsilonInsensitiveLoss,
    HingeLoss,
    Loss,
    PoissonLoss,
    SquaredLoss,
)
from .problem import TrainingProblem
from .training import train
from .validation import (
    PoissonValidation,
    SmoothAbsoluteValidation,
    SmoothHingeValidation,
    SquaredValidation,
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

    def _fit_model(self, x: Array, y: Array, w: Array) -> None:
        self.blocks_ = resolve_blocks(self.blocks, x.shape[1])
        self.scaling_ = self._scaling(x)
        xs = scale_features(x, self.scaling_, self.blocks_)
        prob = TrainingProblem(xs, y, self._loss(), self.lam, w, self.max_asum)
        sol = train(prob, self.smo_options, self.newton_options)
        beta = prob.sample_coefficients(sol.a)
        support = np.flatnonzero(beta)
        self.support_ = support
        self.dual_coef_ = beta[support] / self.lam
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

    def _decision(self, X: ArrayLike) -> Array:
        check_is_fitted(self)
        x = validate_data(self, X, reset=False, dtype=np.float64)
        return self._model.decision(scale_features(x, self.scaling_, self.blocks_))

    def decision_gradient(self, X: ArrayLike) -> Array:
        """Gradient of the decision function w.r.t. the input features, shape ``(n, d)``.

        Useful as a local sensitivity / feature relevance measure.
        """
        check_is_fitted(self)
        x = validate_data(self, X, reset=False, dtype=np.float64)
        xs = scale_features(x, self.scaling_, self.blocks_)
        sv = self._model.support_vectors
        k = rbf(xs, sv) * self._model.coef  # (n, n_sv)
        grad_scaled = -2.0 * (k.sum(axis=1)[:, None] * xs - k @ sv)
        factor = scale_features(np.ones((1, x.shape[1])), self.scaling_, self.blocks_)[0]
        return grad_scaled * factor


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
    smo_options, newton_options : dict, optional
        Options of :func:`smo_solver.solve_classification` / :func:`optimized_svm.training.newton`.
"""


class SmoothSVC(ClassifierMixin, _SmoothBase):
    __doc__ = (
        """Binary SVM classifier with a smoothed hinge loss and fixed hyperparameters.

    Solves ``min lam/2 |w|^2 + sum_i c_i m_delta(shift - y_i f(x_i))``.

    Parameters
    ----------"""
        + _SMOOTH_DOC
        + """    shift : float
        Margin of the hinge loss.
    class_weight : "balanced", dict or None
        Per-class weights multiplied with ``sample_weight``.
    """
    )

    def __init__(
        self,
        lam: float = 1.0,
        *,
        scaling: float | ArrayLike | str = "auto",
        blocks: Any = None,
        smoothing: float = 0.5,
        degree: str = "2",
        shift: float = 1.0,
        max_asum: float | None = None,
        class_weight: Any = None,
        smo_options: dict[str, Any] | None = None,
        newton_options: dict[str, Any] | None = None,
    ) -> None:
        self.lam = lam
        self.scaling = scaling
        self.blocks = blocks
        self.smoothing = smoothing
        self.degree = degree
        self.shift = shift
        self.max_asum = max_asum
        self.class_weight = class_weight
        self.smo_options = smo_options
        self.newton_options = newton_options

    def _loss(self) -> Loss:
        return HingeLoss(self.shift, self.smoothing, self.degree)

    def fit(self, X: ArrayLike, y: ArrayLike, sample_weight: ArrayLike | None = None) -> SmoothSVC:
        self._check_params()
        x, y = validate_data(self, X, y, dtype=np.float64)
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


def _fit_one_vs_rest(est: Any, x: Array, y: Array, sample_weight: ArrayLike | None) -> bool:
    """Fits one binary clone per class if there are more than two classes."""
    check_classification_targets(y)
    classes = np.unique(y)
    est.estimators_ = None
    if classes.size <= 2:
        return False
    est.classes_ = classes
    w = _weights(sample_weight, x)
    if isinstance(est.class_weight, dict):
        w = w * np.array([est.class_weight.get(c, 1.0) for c in y])
    params = {"class_weight": None} if isinstance(est.class_weight, dict) else {}
    est.estimators_ = [
        clone(est).set_params(**params).fit(x, (y == c).astype(int), sample_weight=w)
        for c in classes
    ]
    if hasattr(est.estimators_[0], "n_iter_"):
        est.n_iter_ = max(e.n_iter_ for e in est.estimators_)
    return True


def _ovr_decision(est: Any, X: ArrayLike) -> Array:
    check_is_fitted(est)
    if est.estimators_ is None:
        return est._decision(X)
    x = validate_data(est, X, reset=False, dtype=np.float64)
    return np.stack([e.decision_function(x) for e in est.estimators_], axis=1)


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
        degree: str = "2",
        max_asum: float | None = None,
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

    def _validation(self) -> ValidationLoss:
        raise NotImplementedError

    def _folds(self, x: Array, y: Array) -> list[Fold]:
        if not isinstance(self.cv, Integral):
            return [(np.asarray(tr), np.asarray(va)) for tr, va in self.cv]
        splitter_cls = StratifiedKFold if isinstance(self, ClassifierMixin) else KFold
        splitter = splitter_cls(int(self.cv), shuffle=True, random_state=self.random_state)
        return [(tr, va) for tr, va in splitter.split(x, y)]

    def _optimize(self, x: Array, y: Array, w: Array) -> None:
        loss, validation = self._loss(), self._validation()
        folds = self._folds(x, y)
        start = time.perf_counter()
        history: list[dict[str, Any]] = []
        lam_bounds = np.log(self.lam_bounds)
        s_bounds = np.log(self.scaling_bounds)
        smo_options = dict(self.smo_options or {})
        warm: list[Array | None] = [None] * len(folds) if self.warm_start else None

        def run_stage(blocks: Blocks, theta0: Array) -> Array:
            def fun(theta: Array) -> tuple[float, Array]:
                if self.time_limit is not None and time.perf_counter() - start > self.time_limit:
                    raise _Timeout
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
                )
                history.append(
                    {
                        "lam": float(np.exp(theta[0])),
                        "scaling": np.exp(theta[1:]),
                        "error": res.error,
                        "gradient_norm": float(np.linalg.norm(res.gradient)),
                        "n_support": float(np.mean([f.n_support for f in res.folds])),
                        "time": time.perf_counter() - start,
                        "stage": "features"
                        if blocks is not None and self.blocks is not None
                        else "global",
                    }
                )
                if self.verbose:
                    h = history[-1]
                    print(
                        f"[{len(history):3d}] lam = {h['lam']:.3e}, mean scaling = "
                        f"{np.mean(h['scaling']):.3e}, cv error = {h['error']:.6f}"
                    )
                return res.error, res.gradient

            bounds = [tuple(lam_bounds)] + [tuple(s_bounds)] * (theta0.size - 1)
            best_before = len(history)
            try:
                result = minimize(
                    fun,
                    theta0,
                    jac=True,
                    method="L-BFGS-B",
                    bounds=bounds,
                    options={"maxiter": self.max_iter, "ftol": self.tol, "gtol": self.tol},
                )
                self.optimize_result_ = result
                self.n_iter_ = getattr(self, "n_iter_", 0) + int(result.nit)
            except _Timeout:
                self.optimize_result_ = None
            stage = history[best_before:]
            if not stage:
                return theta0
            best = min(stage, key=lambda h: h["error"])
            return np.log(np.concatenate(([best["lam"]], best["scaling"])))

        self.n_iter_ = 0
        blocks = resolve_blocks(self.blocks, x.shape[1])
        s0 = auto_scaling(x) if self.scaling_init == "auto" else float(self.scaling_init)
        theta = np.log([self.lam_init, s0])
        if blocks is not None and self.two_stage:
            # first a common scaling for all blocks, then individual ones
            theta = run_stage(None, theta)
        if blocks is not None:
            theta = np.concatenate(([theta[0]], np.full(len(blocks), theta[1])))
            theta = run_stage(blocks, theta)
        else:
            theta = run_stage(None, theta)
        self.lam_ = float(np.exp(theta[0]))
        self.scaling_opt_ = np.exp(theta[1:])
        self.blocks_opt_ = blocks
        self.history_ = history
        self.cv_error_ = min(h["error"] for h in history) if history else np.nan
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
    cv : int or iterable of (train, validation) index arrays
        Folds of the cross-validation error.
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
    smoothed cross-validation hinge error with exact gradients.

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
        smoothing: float = 0.5,
        degree: str = "2",
        shift: float = 1.0,
        max_asum: float | None = None,
        class_weight: Any = None,
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
        self.smoothing = smoothing
        self.degree = degree
        self.shift = shift
        self.max_asum = max_asum
        self.class_weight = class_weight
        self.validation_smoothing = validation_smoothing
        self.smo_options = smo_options
        self.newton_options = newton_options

    @property
    def lam(self) -> float:  # used by the fixed-parameter base class
        return getattr(self, "lam_", self.lam_init)

    @property
    def scaling(self) -> Any:
        return getattr(self, "scaling_opt_", self.scaling_init)

    def _validation(self) -> ValidationLoss:
        return SmoothHingeValidation(
            self.shift, self.validation_smoothing, "6", balanced=self.class_weight == "balanced"
        )

    def fit(
        self, X: ArrayLike, y: ArrayLike, sample_weight: ArrayLike | None = None
    ) -> OptimizedSVC:
        for name in ("lam_", "scaling_opt_"):
            self.__dict__.pop(name, None)
        x, y = validate_data(self, X, y, dtype=np.float64)
        if _fit_one_vs_rest(self, x, y, sample_weight):
            return self
        labels, w = _binary_labels(self, y, x, sample_weight)
        self._optimize(x, labels, w)
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
        smoothing: float = 0.5,
        degree: str = "2",
        max_asum: float | None = None,
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
        self.smoothing = smoothing
        self.degree = degree
        self.max_asum = max_asum
        self.normalize_y = normalize_y
        self.validation = validation
        self.validation_smoothing = validation_smoothing
        self.smo_options = smo_options
        self.newton_options = newton_options

    @property
    def lam(self) -> float:
        return getattr(self, "lam_", self.lam_init)

    @property
    def scaling(self) -> Any:
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
        self, X: ArrayLike, y: ArrayLike, sample_weight: ArrayLike | None = None
    ) -> OptimizedSVR:
        for name in ("lam_", "scaling_opt_"):
            self.__dict__.pop(name, None)
        x, y = validate_data(self, X, y, dtype=np.float64, y_numeric=True)
        w = _weights(sample_weight, x)
        self._loss()
        self.label_scaler_ = _LabelScaler(self._label_mode()).fit(y)
        yn = self.label_scaler_.transform(y)
        self._optimize(x, yn, w)
        self._fit_model(x, yn, w)
        return self

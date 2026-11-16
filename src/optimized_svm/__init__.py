"""Support vector machines with hyperparameters optimized by exact gradients.

The training problem is solved by SMO (``smo_solver``) followed by a semismooth Newton method;
regularization and kernel scalings are tuned by minimising a cross-validation error whose
gradient is obtained from the implicit function theorem (Strasdat, Ch. 8 and 9).
"""

from .estimators import (
    KernelGLMRegressor,
    OptimizedKernelGLMRegressor,
    OptimizedSVC,
    OptimizedSVR,
    SmoothSVC,
    SmoothSVR,
)
from .hypergradient import cv_error_and_gradient, fold_error_and_gradient
from .losses import (
    ApproxPoissonLoss,
    BinomialLoss,
    EpsilonInsensitiveLoss,
    GammaLoss,
    HingeLoss,
    LogisticLoss,
    NegativeBinomialLoss,
    PoissonLoss,
    SquaredLoss,
    TweedieLoss,
)
from .problem import TrainingProblem
from .training import Solution, newton, train
from .validation import (
    BinomialDeviance,
    GammaDeviance,
    NegativeBinomialDeviance,
    PoissonValidation,
    TweedieDeviance,
)

__version__ = "0.3.1"

__all__ = [
    "ApproxPoissonLoss",
    "BinomialDeviance",
    "BinomialLoss",
    "EpsilonInsensitiveLoss",
    "GammaDeviance",
    "GammaLoss",
    "HingeLoss",
    "KernelGLMRegressor",
    "LogisticLoss",
    "NegativeBinomialDeviance",
    "NegativeBinomialLoss",
    "OptimizedKernelGLMRegressor",
    "OptimizedSVC",
    "OptimizedSVR",
    "PoissonLoss",
    "PoissonValidation",
    "SmoothSVC",
    "SmoothSVR",
    "Solution",
    "SquaredLoss",
    "TrainingProblem",
    "TweedieDeviance",
    "TweedieLoss",
    "cv_error_and_gradient",
    "fold_error_and_gradient",
    "newton",
    "train",
]

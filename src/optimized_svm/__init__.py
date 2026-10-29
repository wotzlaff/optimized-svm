"""Support vector machines with hyperparameters optimized by exact gradients.

The training problem is solved by SMO (``smo_solver``) followed by a semismooth Newton method;
regularization and kernel scalings are tuned by minimising a cross-validation error whose
gradient is obtained from the implicit function theorem (Strasdat, Ch. 8 and 9).
"""

from .estimators import OptimizedSVC, OptimizedSVR, SmoothSVC, SmoothSVR
from .hypergradient import cv_error_and_gradient, fold_error_and_gradient
from .losses import (
    ApproxPoissonLoss,
    EpsilonInsensitiveLoss,
    HingeLoss,
    PoissonLoss,
    SquaredLoss,
)
from .problem import TrainingProblem
from .training import Solution, newton, train

__version__ = "0.1.1"

__all__ = [
    "ApproxPoissonLoss",
    "EpsilonInsensitiveLoss",
    "HingeLoss",
    "OptimizedSVC",
    "OptimizedSVR",
    "PoissonLoss",
    "SmoothSVC",
    "SmoothSVR",
    "Solution",
    "SquaredLoss",
    "TrainingProblem",
    "cv_error_and_gradient",
    "fold_error_and_gradient",
    "newton",
    "train",
]

"""The hypergradient must agree with central finite differences of the CV error."""

import numpy as np
import pytest
from sklearn.model_selection import KFold

from optimized_svm import (
    EpsilonInsensitiveLoss,
    HingeLoss,
    PoissonLoss,
    SquaredLoss,
    cv_error_and_gradient,
)
from optimized_svm.validation import (
    PoissonValidation,
    SmoothAbsoluteValidation,
    SmoothHingeValidation,
    SquaredValidation,
)

TIGHT_SMO = {"tol": 1e-10}
TIGHT_NEWTON = {"tol": 1e-26, "max_steps": 200}


def case(name, data):
    x, y, t, counts = data
    return {
        "hinge": (HingeLoss(smoothing=0.5), SmoothHingeValidation(), y, None, None),
        "hinge-balanced": (
            HingeLoss(smoothing=0.5),
            SmoothHingeValidation(balanced=True),
            y,
            None,
            None,
        ),
        "hinge6-ard": (
            HingeLoss(smoothing=0.5, degree="6"),
            SmoothHingeValidation(),
            y,
            None,
            [[0], [1], [2]],
        ),
        "hinge-nu": (HingeLoss(smoothing=0.5), SmoothHingeValidation(), y, 8.0, None),
        "eps3-blocks": (
            EpsilonInsensitiveLoss(0.1, 0.3, "3"),
            SmoothAbsoluteValidation(),
            t,
            None,
            [[0, 1], [2]],
        ),
        "eps-nu": (EpsilonInsensitiveLoss(0.05, 0.3), SmoothAbsoluteValidation(), t, 5.0, None),
        "squared": (SquaredLoss(), SquaredValidation(), t, None, None),
        "poisson": (PoissonLoss(), PoissonValidation(), counts, None, None),
    }[name]


@pytest.mark.parametrize(
    "name",
    [
        "hinge",
        "hinge-balanced",
        "hinge6-ard",
        "hinge-nu",
        "eps3-blocks",
        "eps-nu",
        "squared",
        "poisson",
    ],
)
def test_gradient_matches_finite_differences(name, data):
    loss, validation, target, max_asum, blocks = case(name, data)
    x = data[0]
    folds = list(KFold(3, shuffle=True, random_state=0).split(x))
    w = np.ones(len(target))
    n_blocks = 1 if blocks is None else len(blocks)
    theta = np.r_[np.log(0.05), np.full(n_blocks, np.log(0.6))]

    def f(th):
        return cv_error_and_gradient(
            th, x, target, w, folds, loss, validation, blocks, max_asum, TIGHT_SMO, TIGHT_NEWTON
        )

    res = f(theta)
    if max_asum is not None:
        assert all(fold.solution.c > 0 for fold in res.folds), "constraint should be active"
    eps = 1e-6
    fd = np.array(
        [
            (f(theta + eps * e).error - f(theta - eps * e).error) / (2 * eps)
            for e in np.eye(theta.size)
        ]
    )
    np.testing.assert_allclose(res.gradient, fd, rtol=1e-5, atol=1e-7)

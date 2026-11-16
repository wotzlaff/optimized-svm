"""Reference values of the cross-validation objective of ``optimized-lssvr``.

``tests/test_lssvr_oracle.py`` compares ``cv_error_and_gradient`` of this package (squared loss)
with the objective of the older package ``optimized-lssvr`` (``OptimizedLSSVR``). This script
recomputes the reference numbers stored in the test, using the *old* package (it must be
importable, e.g. ``PYTHONPATH=/path/to/sklearn-optimized-lssvr`` in an environment with numpy,
scipy and scikit-learn; it need not be installed in the project environment)::

    PYTHONPATH=/path/to/sklearn-optimized-lssvr python benchmarks/lssvr_oracle_reference.py

The old objective is captured by intercepting ``scipy.optimize.minimize`` inside
``OptimizedLSSVR.optimize_parameters``: the function it would minimise is evaluated at fixed
points ``p = (log lmbda, log gamma_1, ..., log gamma_K)`` and returns the relative MSE
``sum_v e_v^2 / n / var(y)`` over the 5 unshuffled folds and its gradient w.r.t. ``p``.
"""

from __future__ import annotations

import numpy as np

N, D = 120, 3
GROUPS = [[0, 1], [2]]
# (name, kernel, log lmbda, log gamma)
CASES = [
    ("rbf", "rbf", np.log(2e-3), [np.log(0.7)]),
    ("ard", "multi_rbf", np.log(5e-4), np.log([0.4, 1.3, 0.05])),
    ("group", "group_multi_rbf", np.log(1e-2), np.log([0.9, 0.2])),
]


def make_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(N, D))
    y = np.sin(x[:, 0]) + 0.5 * x[:, 1] ** 2 + 0.1 * rng.normal(size=N) + 3.0
    return x, y


def old_objective(kernel: str, x: np.ndarray, y: np.ndarray, p0: np.ndarray):
    """Returns the objective closure ``p -> (relative_mse, gradient)`` of the old package."""
    import scipy.optimize
    from optimized_lssvr import OptimizedLSSVR

    captured = {}

    def fake_minimize(fun, x0, **kwargs):
        captured["fun"] = fun
        return scipy.optimize.OptimizeResult(x=x0, fun=0.0, jac=np.zeros_like(x0))

    real = scipy.optimize.minimize
    scipy.optimize.minimize = fake_minimize
    try:
        est = OptimizedLSSVR(
            kernel=kernel,
            n_splits=5,
            lmbda0=float(np.exp(p0[0])),
            gamma0=np.exp(p0[1:]) if kernel != "rbf" else float(np.exp(p0[1])),
            feature_groups=GROUPS if kernel == "group_multi_rbf" else None,
        )
        est.optimize_parameters(x, y)
    finally:
        scipy.optimize.minimize = real
    return captured["fun"]


def main() -> None:
    x, y = make_data()
    np.set_printoptions(precision=17)
    for name, kernel, loglam, loggamma in CASES:
        p = np.concatenate(([loglam], loggamma))
        value, grad = old_objective(kernel, x, y, p)(p)
        print(f"{name!r}: dict(value={float(value)!r}, grad={[float(g) for g in grad]!r}),")


if __name__ == "__main__":
    main()

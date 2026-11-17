# Maximal margin classification and hard ε-SVR

Sections 4.1.2 and 4.2.3 of the thesis use **indicator losses**: the loss is zero when a
constraint is met and infinity otherwise. They are available as
`SmoothSVC(loss="hard_margin")` and `SmoothSVR(loss="hard_epsilon_insensitive")`.
The class names are retained for compatibility; these losses are not smoothed.

For classification the training problem is

$$
\min_{w,b}\frac{\lambda}{2}\|w\|^2
\quad\text{subject to}\quad y_i f(x_i)\geq\mathrm{shift}.
$$

For regression it is

$$
\min_{w,b}\frac{\lambda}{2}\|w\|^2
\quad\text{subject to}\quad |f(x_i)-y_i|\leq\varepsilon_i.
$$

The constraints are imposed directly, without a finite slack penalty. Every positive `lam`
therefore gives the same predictor; it only rescales the dual variables `Solution.a`. Positive
sample/class weights also leave the constraint unchanged; zero weights remove a constraint.

```python
from optimized_svm import SmoothSVC, SmoothSVR

classifier = SmoothSVC(loss="hard_margin", scaling=1.0).fit(X, labels)
regressor = SmoothSVR(
    loss="hard_epsilon_insensitive", epsilon=0.15,
    normalize_y=False, scaling=1.0,
).fit(X, targets)
```

With `normalize_y=True`, epsilon is in normalized target units, as for ordinary epsilon-SVR.
The estimators require scalar epsilon. The low-level `HardEpsilonInsensitiveLoss` also accepts
one epsilon per sample, including zero; `cv_error_and_gradient` slices this array per fold.

## Solver and infeasibility

The dedicated dense QP solver uses an eigenbasis of the RBF kernel, a linear feasibility
check, SLSQP, and active-set polishing. It verifies feasibility of the actual kernel expansion
and stationarity before returning a solution. It does not add diagonal jitter to the kernel.
Floating-point feasibility tolerance is `2e-7 * max(1, max(abs(constraint targets)))`.
`training_info_["residual"]` reports the final constraint/stationarity residual.

Opposite labels at identical inputs make hard classification infeasible. At identical
regression inputs all epsilon intervals must overlap. Such problems raise `ValueError`;
a failed numerical solve raises `ConvergenceError`. The feasibility check works in a numerical
eigenbasis: eigenvalues below `eps_machine * n * max(1, largest_eigenvalue)` are discarded.
Very wide kernels or nearly coincident inputs can consequently be numerically infeasible even
when the exact mathematical problem is feasible. Use moderate, well-conditioned data sets;
the dense eigenproblem and QP are not a large-scale replacement for SMO.

Neither `max_asum`, `n_basis`, offsets nor SMO/Newton options are supported for the hard route.
One-vs-rest classification works; the joint soft-max formulation requires the logistic loss.
Warm-start arrays are accepted by the shared training API but not used by the constrained solver.

## Hyperparameter gradients

`OptimizedSVC` and `OptimizedSVR` can tune the kernel scalings using the existing validation
losses. The log-lambda derivative is identically zero, so `lam_init` is retained (subject to
its bounds). The low-level CV API can also tune a scalar epsilon with `loss_params=("epsilon",)`.

For a fixed, nondegenerate active set, the active constraints replace the smooth-loss
stationarity equations. Differentiation uses the bordered kernel system on the support vectors,
with no loss-curvature diagonal. The resulting local gradient agrees with finite differences.
At active-set changes the predictor need not be differentiable. Singular or severely
ill-conditioned active systems raise `ValueError` rather than reporting a unique hypergradient.
Optimization also stops with an error if a trial kernel makes the constraints numerically
infeasible; choose scaling bounds that keep the kernel well-conditioned.

See the [executed example](../examples/hard_constraints.md) for hard/soft comparisons and an
infeasible-data example.

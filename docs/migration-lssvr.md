# Coming from optimized-lssvr

The package `optimized-lssvr` (`OptimizedLSSVR`, hyperparameter optimization for least squares SVR,
Fischer et al. 2015) is deprecated. `OptimizedSVR(loss="squared")` solves the same problem: the same
kernel, the same cross-validation objective and the same adjoint gradient. It adds scikit-learn
splitters, a time limit and the other losses and validation errors of this package.

```sh
pip install "optimized-svm>=0.3.1"   # splitters and groups need 0.3.1
```

## Parameter mapping

For `OptimizedLSSVR` with $n_\text{tr}$ training samples per fold and $n$ samples in total, the fold
systems are $(K + \texttt{lmbda}\, n_\text{tr} I)$ with $K = \exp(-\sum_k \texttt{gamma}_k\, d_k^2)$.
`OptimizedSVR(loss="squared")` solves $(K + \lambda I)$ with the same $K$ for $\gamma_k = s_k^2$, hence

$$
\lambda = \texttt{lmbda} \cdot n_\text{tr}, \qquad s_k = \sqrt{\texttt{gamma}_k}, \qquad
\texttt{relative\_mse} = 2 \cdot \texttt{cv\_error\_}.
$$

The cross-validation error of `OptimizedSVR` is the mean over all validation samples of
$\tfrac12 (f - y)^2$ in units of the standard deviation of $y$ (`normalize_y=True`), which is the old
relative MSE $\sum e^2 / n / \operatorname{var}(y)$ up to the factor 2. Gradients agree after the
chain rule: $\partial / \partial \log s_k = 2\, \partial / \partial \log \gamma_k$ and
$\partial / \partial \log \lambda = \partial / \partial \log \texttt{lmbda}$.
`tests/test_lssvr_oracle.py` checks objective and gradient (RBF, ARD, feature groups) against reference
values of the old package to 1e-9 relative.

| `OptimizedLSSVR` | `OptimizedSVR(loss="squared")` |
|---|---|
| `lmbda0` | `lam_init = lmbda0 * n_tr` |
| `gamma0` (scalar) | `scaling_init = sqrt(gamma0)` (one initial value for all blocks) |
| `kernel="rbf"` | `blocks=None` |
| `kernel="multi_rbf"` (ARD) | `blocks="features"` |
| `kernel="group_multi_rbf"`, `feature_groups` | `blocks=feature_groups` |
| `bounds={"lmbda": (l, u), "gamma": (l, u)}` | `lam_bounds=(l * n_tr, u * n_tr)`, `scaling_bounds=(sqrt(l), sqrt(u))`; the defaults (`1e-6..1`, `1e-4..1e2`) are narrower than the old unbounded search |
| `n_splits=5` (unshuffled `KFold`) | `cv=KFold(5)`; `cv=5` shuffles (`random_state`) |
| `fit(X, y, groups=g)` (leave one group out) | `cv=LeaveOneGroupOut()`, `fit(X, y, groups=g)`; any splitter with `split` works, e.g. `GroupKFold` |
| `method`, `tol` | L-BFGS-B always; `tol`, `max_iter` |
| `verbose` | `verbose` |
| `fit(..., optimize_parameters=False)` | `SmoothSVR(loss="squared", lam=..., scaling=..., blocks=...)` |
| `estimate_sensitivity(X)` | `decision_gradient(X)` |

| attribute | new |
|---|---|
| `params_["lmbda"]` | `lam_ / n_tr` |
| `params_["gamma"]` | `scaling_opt_ ** 2` |
| `relative_mse_` | `2 * cv_error_` |
| `dparams_` | `history_[-1]["gradient_norm"]` (norm of the last evaluated gradient only) |
| `alpha_`, `bias_` | `dual_coef_ * std(y)`, `intercept_ * std(y) + mean(y)` (the new model is trained on normalised labels; `dual_coef_` lists the nonzero coefficients at `support_`) |
| `yvar_` | `label_scaler_.scale ** 2` |

Two differences remain: `two_stage=True` (default) first optimizes one common scaling when `blocks`
are given, `OptimizedLSSVR` starts directly with individual `gamma`; set `two_stage=False` to follow the
old path. In the old package the initial `gamma0` may be an array per feature, here `scaling_init` is a
scalar.

## Before and after

```python
# before
from optimized_lssvr import OptimizedLSSVR

model = OptimizedLSSVR(kernel="multi_rbf", lmbda0=1e-3, gamma0=0.5, n_splits=5)
model.fit(X, y, groups=groups)          # groups: leave one group out
print(model.params_, model.relative_mse_)

# after
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from optimized_svm import OptimizedSVR

n_tr = len(X) - len(X) // 5             # only a unit conversion; folds of size n / 5
model = OptimizedSVR(
    loss="squared",
    blocks="features",
    two_stage=False,
    lam_init=1e-3 * n_tr,
    scaling_init=np.sqrt(0.5),
    lam_bounds=(1e-9, 1e3),
    scaling_bounds=(1e-4, 1e2),
    cv=LeaveOneGroupOut(),              # or KFold(5) without groups
)
model.fit(X, y, groups=groups)
print(model.lam_, model.scaling_opt_ ** 2, 2 * model.cv_error_)
```

## The regularization of the old final fit

`OptimizedLSSVR` optimizes `lmbda` with the diagonal term `lmbda * n_tr` in every fold, but its final
fit on all data uses `lmbda * n`. The refitted model is therefore regularized by a factor
$n / n_\text{tr}$ more strongly (25 % with 5 folds) than the model that was validated. `OptimizedSVR`
uses the selected `lam_` unchanged in the final fit; since it is the value validated in the folds, it
corresponds to `lmbda * n_tr`. To reproduce the predictions of an old model with given `params_`, use
`SmoothSVR(loss="squared", lam=params_["lmbda"] * n, scaling=np.sqrt(params_["gamma"]), ...)`
(verified to machine precision, with `blocks="features"` for the ARD kernel).

The mapping is exact for folds of equal size. Otherwise the old objective uses a different
`lmbda * n_tr` per fold, whereas `lam` is shared by all folds here.

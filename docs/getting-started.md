# Getting started

## Classification

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from optimized_svm import OptimizedSVC

x, y = load_breast_cancer(return_X_y=True)
x = StandardScaler().fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)

clf = OptimizedSVC(cv=5, verbose=1).fit(x_train, y_train)
print(clf.lam_, clf.scaling_opt_, clf.score(x_test, y_test))
```

`OptimizedSVC` minimizes the 5-fold cross-validation error of a smoothed hinge loss over
$\log\lambda$ and the log kernel scaling $\log s$ (the RBF parameter is $\gamma = s^2$). After the
search the model is refitted on all data with the best parameters.

!!! tip "Feature scaling"
    The kernel acts on the raw features. Standardize them (e.g. with `StandardScaler`) or use
    `blocks="features"` so that every feature gets its own scaling.

## Regression and feature relevance

```python
from sklearn.datasets import make_friedman1
from optimized_svm import OptimizedSVR

x, y = make_friedman1(600, noise=0.5, random_state=0)
reg = OptimizedSVR(epsilon=0.1, blocks="features").fit(x, y)
print(reg.scaling_opt_.round(3))   # small values = irrelevant features
```

With `blocks="features"` the kernel is $k(x, z) = \exp(-\sum_j s_j^2 (x_j - z_j)^2)$. With
`two_stage=True` (default) a single common scaling is optimized first and used as starting point.

## Choosing the model

| estimator | loss | parameters |
|---|---|---|
| `SmoothSVC` / `OptimizedSVC` (`loss="hinge"`) | smoothed hinge $w_i\, m_\delta(\text{shift} - y_i f(x_i))$ | `smoothing`, `degree`, `shift`, `class_weight`, `max_asum` |
| `loss="logistic"` | $w_i \log(1 + e^{-y_i f(x_i)})$, enables `predict_proba` | `class_weight` |
| `SmoothSVR` / `OptimizedSVR` (`loss="epsilon_insensitive"`) | $w_i (m_\delta(f_i - y_i - \varepsilon) + m_\delta(y_i - f_i - \varepsilon))$ | `epsilon`, `smoothing`, `degree` |
| `loss="squared"` | $\tfrac{w_i}{2} (f_i - y_i)^2$ (LS-SVM) | – |
| `loss="poisson"` | $w_i (e^{f_i} - y_i f_i)$, prediction $e^{f}$ | – |

The logistic loss needs no smoothing, but it yields a dense solution and is trained by Newton's method
only, so every training step solves a system of size $n$: prefer the hinge loss for large $n$ unless you need
probabilities. It does not support `max_asum`.

`smoothing` ($\delta$) controls how strongly the kinks of the loss are rounded; `degree` selects
the smoothing polynomial ("2": $C^1$, exactly solvable by SMO; "3" and "6": $C^2$).

## Useful attributes

- `lam_`, `scaling_opt_`: optimized hyperparameters
- `history_`: list of all CV evaluations (`lam`, `scaling`, `error`, `gradient_norm`, `time`, …)
- `cv_error_`, `time_`, `n_iter_`
- `support_`, `dual_coef_`, `intercept_`, `training_info_` of the final model
- `decision_gradient(X)`: derivative of the decision function w.r.t. the inputs

## Lower-level building blocks

```python
from optimized_svm import EpsilonInsensitiveLoss, TrainingProblem, train

prob = TrainingProblem(x * 0.5, y, EpsilonInsensitiveLoss(0.1, 0.05, "3"), lam=1e-3)
sol = train(prob)          # SMO warm start + Newton refinement
sol.a, sol.b, sol.residual
```

See `examples/` in the repository for complete scripts.

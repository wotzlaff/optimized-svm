# Subspace SVM

The model is restricted to the span of $m$ basis points $z_1,\dots,z_m$ (thesis, Sec. 5.3):

$$
w = \sum_j \beta_j \phi(z_j), \qquad f(x) = \sum_j \beta_j\, k(x, z_j) + b .
$$

## Training

By Corollary 5.9 the dual is the standard dual with the projected kernel
$K_s = K_{XZ} K_{ZZ}^{-1} K_{ZX}$ (a Nyström approximation). Equivalently the SVM has the explicit
feature map $\psi(x) = L^{-1} k_Z(x)$ with $K_{ZZ} = L L^\top$, and the primal problem has only
$m+1$ unknowns:

$$
\min_{u,b}\ \tfrac{\lambda}{2}\|u\|^2 + \sum_l \varphi_l(\psi_{k(l)}^\top u + b), \qquad \beta = L^{-\top} u .
$$

It is solved by a damped Newton method with systems of size $m+1$; one step costs
$O(nm^2 + m^3)$ instead of a factorization of size $n$. SMO is not used, so the loss must have
positive curvature somewhere (`smoothing > 0`, squared, logistic, Poisson); `smoothing = 0` is rejected.

## Hypergradient

The stationarity conditions in $(\beta, b)$ define the solution implicitly. One adjoint system of
size $m+1$ per fold gives the gradient of the cross-validation error with respect to $\log\lambda$ and
the log kernel scalings (including ARD blocks). It is checked against central differences for the
smoothed hinge, ε-insensitive (degrees 2/3/6), logistic and squared loss.

## Usage

All four estimators take `n_basis` and `basis_tol`:

```python
from optimized_svm import OptimizedSVR

est = OptimizedSVR(n_basis="auto", basis_tol=1.0).fit(X, y)   # basis size from the residual rule
est = OptimizedSVR(n_basis=200).fit(X, y)                      # fixed size
```

- `n_basis=None` (default): full model. An integer: that many basis points. `"auto"`: `select_basis`
  with tolerance `basis_tol` (see below); if the limit $n/10$ is hit, a warning is issued and the full model is used.
- Needs a smooth loss (`smoothing > 0`, or squared/logistic/Poisson) and no `max_asum`. Models are
  trained by Newton's method only (`newton_options`; `smo_options` is ignored).
- The optimizers (`OptimizedSVC/SVR`) choose the basis of every fold at the starting values of each stage
  (global scaling, then ARD scalings) and keep it fixed during L-BFGS-B, so the objective stays smooth. With
  `n_basis="auto"` the rule is then checked at the result (`nystroem_residual(x_scaled, idx) / lam <= basis_tol`
  in every fold); if it fails, e.g. because the optimizer lowered $\lambda$ by orders of magnitude, a new basis is
  chosen there and L-BFGS-B restarts (at most 5 rounds per stage). The stage keeps the best evaluation of its
  last round (earlier rounds used smaller subspaces and are not comparable); `history_` entries carry `basis_round`. An integer
  `n_basis` is never re-selected. The final
  model is trained on all data with a basis chosen at the optimized hyperparameters. After fitting,
  `basis_` holds its indices, `support_`/`dual_coef_` the basis points and expansion coefficients, and
  `history_` records `n_basis` per evaluation.
- Low-level: pass `basis` (one index array per fold, indices into `x`) to `cv_error_and_gradient`;
  `warm_start` then holds expansion coefficients. `loss_params` and `max_asum` are not supported with a basis.

!!! warning "`n_basis=\"auto\"` often falls back to the full model with the default starting values"
    The basis is chosen at the starting values of each stage. With the defaults (`lam_init=1e-2`,
    `scaling_init="auto"`) the residual rule did not hold within $n/10$ points for ε-SVR on Friedman #1 with
    $n = 5000$, so all three variants (`tol` 1, `tol` 10, full) ran the full model and gave identical
    results (17 evaluations, about 520 s each; `benchmarks/results/subspace_hpo_n5000_fallback.json`). A warning is issued, but
    no speed-up results. The gain shown below was measured at wide kernels (scale 0.05–0.1) and $\lambda \ge 10^{-2}$; start the
    optimization there (`scaling_init`, `lam_init`) if you want the subspace model to be used. An end-to-end comparison of
    the optimization with such starting values is still open.

## Choosing the basis

Study `benchmarks/subspace_params.py` (raw data `benchmarks/results/subspace_params.json`): ε-SVR and
classification on Friedman-type data, $n = 1000$ and $2500$, three kernel scales, $\lambda = 10^{-3}$ and
$10^{-1}$, 5 folds, basis sizes 25–800; the cross-validation error of one evaluation is compared with the
full model.

- **Selection rule: greedy pivoted Cholesky, not random.** At equal $m$ the two rules give about the same
  error (median excess over the full model at $m = 400$: +0.4 % pivoted, +0.7 % random on
  regression), so the choice does not matter for quality; pivoted Cholesky is used because it yields
  the residual trace needed for the size rule.
- **Size rule: stop when $\operatorname{tr}(K - K_s) \le 	au\lambda$.** The trace of the Nyström residual relative to $\lambda$
  predicts the loss of accuracy, the trace alone does not (with small $\lambda$ a residual of $10^{-3}\,n$
  already costs +55 % error). Excess CV error over the full model, all 24 settings × 6 sizes:

  | $\operatorname{tr}(K-K_s)/\lambda$ | runs | median | maximum |
  |---|---|---|---|
  | < 1 | 24 | +0.01 % | +0.9 % |
  | 1–10 | 10 | +0.6 % | +11 % |
  | 10–100 | 15 | +2 % | +14 % |
  | 100–1000 | 18 | +15 % | +82 % |
  | > 1000 | 65 | +17 % to +28 % | +135 % |

  `select_basis(x_scaled, lam, tol=1.0, max_size=None)` implements this with $\tau = 1$ (use `tol=10`
  to accept up to ~14 %). The default size limit is $\max(10, n/10)$; if it is hit, the returned ratio is
  above `tol` and the model is not a close approximation.
- **Check of the rule** (`select_basis` with the size limit $n/4$, $n = 2500$, one evaluation incl. 5 folds;
  excess = CV error over the full model):

  | task | kernel scale | $\lambda$ | $\tau$ | $m$ | excess | subspace (s) | full (s) |
  |---|---|---|---|---|---|---|---|
  | ε-SVR | 0.05 | $10^{-3}$ | 1 | 376 | +0.2 % | 6.1 | 7.0 |
  | ε-SVR | 0.05 | $10^{-3}$ | 10 | 230 | +2.9 % | 3.0 | 7.0 |
  | ε-SVR | 0.05 | $10^{-1}$ | 1 | 93 | +1.1 % | 1.2 | 4.1 |
  | ε-SVR | 0.05 | $10^{-1}$ | 10 | 47 | +12 % | 0.3 | 4.1 |
  | ε-SVR | 0.15 | $10^{-1}$ | 1 | 625 (limit) | +0.9 % | 9.3 | 5.7 |
  | classification | 0.05 | $10^{-3}$ | 1 | 238 | 0.0 % | 6.2 | 1.1 |
  | classification | 0.05 | $10^{-1}$ | 1 | 74 | +0.2 % | 0.7 | 0.9 |

  The rule keeps the error small whenever the ratio reaches $\tau$. Whether it also saves time depends on the
  problem: only for ε-SVR with a wide kernel (scale 0.05) and not too small $\lambda$ (3.4× faster at
  $\tau = 1$, 14× at $\tau = 10$ with +12 % error); for small $\lambda$ the basis grows and the gain
  disappears. With a narrower kernel (scale 0.15) the limit is hit and the full model is faster. For
  classification the full model (SMO) takes about 1 s here and the subspace model is not faster.
- **Larger $n$** (`benchmarks/subspace_large.py`, ε-SVR on Friedman #1, 5 folds, one evaluation; time of the subspace model
  includes `select_basis`; excess = CV error over the full model):

  | $n$ | scale | $\lambda$ | $\tau$ | $m$ | excess | subspace (s) | full (s) | speed-up |
  |---|---|---|---|---|---|---|---|---|
  | 5000 | 0.05 | 0.1 | 1 | 139 | +0.7 % | 2.4 | 14.0 | 5.8× |
  | 5000 | 0.05 | 0.01 | 1 | 282 | +0.3 % | 4.5 | 23.6 | 5.2× |
  | 5000 | 0.1 | 0.1 | 1 | 685 | +0.1 % | 11.5 | 20.7 | 1.8× |
  | 10000 | 0.05 | 0.1 | 1 | 189 | +0.4 % | 2.9 | 95.5 | 33× |
  | 10000 | 0.05 | 0.01 | 1 | 348 | +0.2 % | 6.9 | 154.8 | 22× |
  | 10000 | 0.1 | 0.1 | 1 | 939 | +0.1 % | 18.4 | 106.1 | 5.8× |
  | 10000 | 0.05 | 0.1 | 10 | 70 | +7.7 % | 1.2 | 95.5 | 80× |
  | 10000 | 0.05 | 0.01 | 10 | 189 | +2.3 % | 3.5 | 154.8 | 44× |
  | 10000 | 0.1 | 0.1 | 10 | 429 | +1.5 % | 12.0 | 106.1 | 8.8× |

  The gain grows with $n$: the basis size needed for a given $\tau$ grows slowly with $n$ while the full
  model gets much more expensive. At $n = 10000$ the subspace model is 6 to 33 times faster at $\tau = 1$ with at most 0.4 % excess error.
- The basis depends on the kernel scale and on $\lambda$; with `n_basis="auto"` the optimizers re-select
  it when the hyperparameters move too far (see below).

## Hyperparameter optimization (experimental)

The single-evaluation gains above do not carry over to a complete optimization so far
(`benchmarks/subspace_hpo.py`, `OptimizedSVR`, Friedman #1, $n = 5000$, 5 folds, `max_iter = 30`, starting
values $\lambda = 0.1$, scale 0.05; raw data `benchmarks/results/subspace_hpo_*.json`):

| noise | variant | time (s) | basis rounds | CV error | test MSE | final $\lambda$ |
|---|---|---|---|---|---|---|
| 1 | full | 932 | – | 0.2163 | 1.143 | $10^{-6}$ (bound) |
| 1 | `"auto"`, $\tau = 1$ | 1095 | 2, then full | 0.2163 | 1.143 | $10^{-6}$ |
| 1 | `"auto"`, $\tau = 10$ | 1256 | 3, then full | 0.2163 | 1.143 | $10^{-6}$ |
| 3 | full | 416 | – | 0.5352 | 9.68 | $6.3 \cdot 10^{-3}$ |
| 3 | `"auto"`, $\tau = 1$ | 242 | 2, then full | 0.5352 | 9.69 | $9.2 \cdot 10^{-3}$ |
| 3 | `"auto"`, $\tau = 10$ | 29 | 5, rule not met | 0.5611 | 11.11 | $1.2 \cdot 10^{-6}$ |

The optimizer drives $\lambda$ to values ($\le 10^{-2}$) where the residual rule needs more than the
$n/10$ basis points allowed per fold, so with $\tau = 1$ it ends up running the full model; the subspace
rounds act at best as a cheap warm-up (1.7× faster in one case, 17 % slower in the other). Before the
re-selection was added, a basis fixed at the starting values led to clearly wrong optima (CV error 0.42
instead of 0.216 at $\tau = 10$). With $\tau = 10$ the optimization can be much faster but ends in a
different, worse optimum (very wide kernel, $\lambda$ at the bound) and warns that the rule was not met.
For hyperparameter optimization the full model is therefore the recommended default; the subspace SVM is
useful for fits with fixed hyperparameters at large $n$ and wide kernels. Not run: $n = 10000$.

## Example

ε-SVR on Friedman #1, $n = 3000$, 5 folds, $\lambda = 10^{-3}$, one kernel scale, one evaluation of
error and gradient (cold start, one machine):

| basis size $m$ | CV error | time (s) |
|---|---|---|
| 50 | 0.351 | 0.7 |
| 100 | 0.253 | 0.9 |
| 200 | 0.234 | 3.9 |
| 400 | 0.1823 | 6.9 |
| 800 | 0.1819 | 9.3 |
| full model | 0.1818 | 44.3 |

A single evaluation at one hyperparameter value; it shows the trade-off, not a general speed-up.
The basis is random, so $m$ must be large enough for the kernel scale in use.

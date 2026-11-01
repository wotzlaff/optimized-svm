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

Low-level only: pass `basis` (one index array per fold, indices into `x`) to
`cv_error_and_gradient`; `warm_start` then holds expansion coefficients. `choose_basis` draws a random
subset. `loss_params` and `max_asum` are not supported with a basis.

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
- The basis depends on the kernel scale and on $\lambda$; choose it for the starting values
  of a hyperparameter optimization and re-select it if they change a lot.

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

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

# Hyperparameter gradients

## Bilevel formulation

Let $\theta = (\log\lambda, \log s_1, \dots, \log s_K)$ collect the hyperparameters. For folds
$(T_j, V_j)$ the upper level problem is

$$
\min_\theta\; E(\theta) = \frac{1}{|V|}\sum_j \sum_{v \in V_j} L\big(f_j(x_v; \theta), y_v\big),
$$

where $f_j$ is the decision function trained on $T_j$ (the lower level problem) and $L$ a smooth
validation loss: a smoothed hinge for classification, a Huber-type smoothed absolute error,
the squared error or the Poisson loss for regression (Sec. 9.1).

## Implicit function theorem

With smoothed losses, the training solution satisfies the (semismooth) optimality system

$$
a_l + \phi_l'(\mathrm{ka}_l + b + s_l c) = 0, \qquad \mathbf 1^\top a = 0, \qquad (s^\top a = A),
$$

with $\mathrm{ka} = K a/\lambda$. Only terms $P = \{l : \phi_l'' > 0\}$ react to small changes of
$\theta$; all others keep $a_l$ constant. Differentiating the system on $P$ gives
(Sec. 9.2)

$$
\begin{pmatrix} K_{PP}/\lambda + \operatorname{diag}(1/h_P) & \mathbf 1 & s_P\\ \mathbf 1^\top & 0 & 0\\ s_P^\top & 0 & 0\end{pmatrix}
\begin{pmatrix}\partial a_P\\ \partial b\\ \partial c\end{pmatrix}
= -\begin{pmatrix}\partial_\theta \mathrm{ka}_P\\ 0\\ 0\end{pmatrix},
\qquad h = \phi''.
$$

## Adjoint system

Instead of solving this system for every hyperparameter, the chain rule is applied in reverse:
one solves the transposed (symmetric) system with right-hand side
$(K_{VP}^\top \nabla L/\lambda,\; \mathbf 1^\top\nabla L,\; 0)$ for $u$ and obtains

$$
\frac{\partial E}{\partial\theta} = -u^\top \partial_\theta \mathrm{ka}_P + \nabla L^\top \partial_\theta f_V\big|_{a\ \text{fixed}},
$$

with $\partial_{\log\lambda}\mathrm{ka} = -\mathrm{ka}$ and
$\partial_{\log s_k} K = -2 K \odot D_k$ ($D_k$: squared distances in block $k$ of the scaled
features). The cost per fold is one factorization of a matrix of size $|P|$, independent of the
number of hyperparameters — this is what makes ARD kernels affordable. For the quadratic
smoothing $h$ is piecewise constant, so the Newton system of the last training step has the same
matrix as the adjoint system and its Cholesky factor is reused; for other degrees the matrix is
factorized again. Contractions with $\partial_{\log s_k} K$ do not form the matrices $D_k$:
$\sum_{ij} W_{ij}\lVert x_{i,B_k} - z_{j,B_k}\rVert^2$ is evaluated from row and column sums of
$W$ and one product $WZ$ for all blocks.

The implementation is tested against central finite differences for all losses, smoothing
degrees, ARD blocks and an active $\nu$-constraint (relative error about $10^{-8}$).

## Loss parameters ($\varepsilon$, $\delta$)

The parameters of the loss — the tube width $\varepsilon$ of the $\varepsilon$-insensitive loss and the
smoothing width $\delta$ — enter the optimality condition $a_l + \phi_l'(t_l; p) = 0$ directly, not
through the kernel. The same adjoint vector $u$ therefore gives their derivatives at no extra cost:

$$
\frac{\partial E}{\partial p} = -\sum_{l \in P} u_l\, \frac{\partial_p \phi_l'(t_l)}{\phi_l''(t_l)} .
$$

All smoothings are of the form $m_\delta(x) = \delta\, m_1(x/\delta)$, hence
$\partial_\delta m_\delta'(x) = -(x/\delta)\, m_\delta''(x)$ for every degree. The ratios are
$s_l$ for $\varepsilon$ and $s_l \cdot \mathrm{arg}_l/\delta$ for $\delta$, where $\mathrm{arg}_l$ is the
argument of the smoothed maximum and $s_l$ the term sign (for the hinge loss $y_l$ replaces $s_l$).
They are implemented as `Loss.param_ratio`. The function `cv_error_and_gradient` takes
`loss_params=("epsilon", "smoothing")`; `theta` is then extended by $\log\varepsilon$ and $\log\delta$.

```python
import numpy as np

from optimized_svm import EpsilonInsensitiveLoss, cv_error_and_gradient

theta = np.r_[np.log(lam), np.log(scaling), np.log(0.1), np.log(0.3)]  # lam, s, eps, delta
res = cv_error_and_gradient(
    theta, x, y, w, folds, EpsilonInsensitiveLoss(0.1, 0.3), validation,
    loss_params=("epsilon", "smoothing"),
)
```

The gradients are exact (checked against finite differences, also for $\delta$ down to 0.003), but the
estimators do not tune these parameters, because it rarely pays off:

- **$\varepsilon$ is not identifiable for $\varepsilon < \delta$** (quadratic smoothing): the two smoothed
  sides overlap and $\varepsilon$ only shifts the loss by a constant, so the solution, the CV error and the
  gradient do not depend on it.
- The CV error is **multimodal in $\delta$** (variations of about 1 %) and rough for very small $\delta$;
  starting with a large $\varepsilon$ (everything inside the tube, zero solution) can end in a poor basin.
- On data without outliers the tuned parameters do not improve the CV error. With 8 % gross outliers a
  smaller $\delta$ helps (test MSE 0.226 → 0.216), but a fixed small $\delta$ does as well, at a fraction of
  the cost (tuning needs two to four times as many CV evaluations).

The study is `benchmarks/hpo_loss_params.py`.

## Optimization

L-BFGS-B runs on $\theta$ within box constraints. Each evaluation warm-starts SMO with the fold
solutions of the previous evaluation (about ten times fewer SMO steps for regression). With `time_limit`, the best parameters found so far are
used.

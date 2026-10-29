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

## Optimization

L-BFGS-B runs on $\theta$ within box constraints. Each evaluation warm-starts SMO with the fold
solutions of the previous evaluation (about ten times fewer SMO steps for regression). With `time_limit`, the best parameters found so far are
used.

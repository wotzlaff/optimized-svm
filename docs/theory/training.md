# Training problem

## Standard training problem

For samples $(x_i, y_i)$, a feature map $\varphi$ with kernel $k(x, z) = \langle\varphi(x),\varphi(z)\rangle$
and convex losses $\ell_i$, the *standard training problem* (Strasdat, Def. 2.5) reads

$$
\min_{w, b}\; \frac{\lambda}{2}\|w\|^2 + \sum_{i=1}^n \ell_i(\langle w, \varphi(x_i)\rangle + b).
$$

Its Fenchel dual is (Cor. 3.11)

$$
\min_{\alpha}\; \frac{1}{2\lambda}\alpha^\top K \alpha + \sum_{i=1}^n \ell_i^\star(-\alpha_i)
\quad\text{s.t.}\quad \mathbf 1^\top\alpha = 0,
$$

and $w = \frac1\lambda \sum_i \alpha_i \varphi(x_i)$ (representer theorem). The decision function is
$f(x) = \frac1\lambda\sum_i \alpha_i k(x, x_i) + b$.

## Smoothed losses

Hinge and $\varepsilon$-insensitive losses are built from $\max(0, t)$. Replacing it by a smooth
approximation $m_\delta$ that agrees with $\max(0, t)$ for $|t| \ge \delta$ (Sec. 4.4) yields
differentiable losses. For the quadratic smoothing

$$
m_\delta(t) = \begin{cases} 0 & t \le -\delta\\ \frac{(t+\delta)^2}{4\delta} & |t| < \delta \\ t & t \ge \delta\end{cases}
$$

the conjugate is quadratic on an interval, so the dual is still a box-constrained QP that SMO
solves exactly. The higher degree variants ("3", "6") are $C^2$.

## Two-step training

1. **SMO** (`smo-solver`) solves the dual with quadratic smoothing to a rough tolerance
   (Ch. 7). Working set selection uses second-order information and shrinking removes
   variables fixed at their bounds.
2. **Semismooth Newton** (Ch. 8.4) solves the optimality system of the actual (possibly
   higher-degree) smoothed problem

   $$
   a + \ell'(\tfrac1\lambda K a + b\mathbf 1 + s c) = 0, \qquad \mathbf 1^\top a = 0
   $$

   Terms with vanishing curvature ($\ell'' = 0$) are fixed explicitly, so only a linear system on
   the terms with $\ell'' > 0$ has to be solved. Usually one to a few steps suffice.

The SMO stage can be tuned with `smo_options` (all options of `smo-solver`), for instance the
bound-aware stopping criterion of Sec. 7.6, `smo_options={"stopping": "bound_aware"}`. By default
SMO runs to the rough tolerance `tol = 0.3` or for at most five steps per loss term; see
[Switching from SMO to Newton](switching.md) for the reasons.

For non-smooth losses ($\delta = 0$) only SMO is used; for losses without an SMO formulation
(squared, logistic, Poisson) Newton starts at zero.

## Logistic loss

The logistic loss $\ell(t) = \log(1 + e^{-yt})$ (Sec. 4.1.3) is smooth without any smoothing and its
curvature $\ell''(t) = \sigma(-yt)\,\sigma(yt)$ is positive everywhere. Its conjugate involves
$x \log x$ (an entropy), so the dual is not a separable quadratic and SMO cannot be used. Newton's
method works as usual (about 5–9 steps from zero), but because no term has vanishing curvature the
solution is dense and every Newton system has the full size $n$ (cost $\mathcal O(n^3)$ per step).
In exchange the decision values have a probabilistic reading, $P(y = 1 \mid x) = \sigma(f(x))$,
used by `predict_proba`. On the checkerboard-like test sets of the benchmarks its accuracy matches the
smoothed hinge loss, the test log-loss is slightly better, and training is 3–6 times slower at $n = 2000$.
Starting Newton at the solution of neighbouring hyperparameters does not help: the dual
coefficients $a = -\ell'(f)$ change with the kernel, so the previous $a$ is no better than zero.

## Optional 1-norm constraint

`max_asum=A` adds $\sum_i |\alpha_i| \le A$ (a generalized $\nu$-SVM, Sec. 5.2). Its multiplier
$c \ge 0$ shifts the decision values of the training terms by $s_i c$.

# Multi-class soft-max SVM

For more than two classes the default is one-vs-rest: one binary model per class, each with its own
hyperparameters. `multi_class="softmax"` instead fits **one joint multinomial model**
(thesis, Sec. 5.1, Prop. 5.1) with a decision function per class, a common $\lambda$ and common
kernel scalings. It needs `loss="logistic"`; `predict_proba` returns the soft-max probabilities
directly (no Platt scaling, no renormalisation of one-vs-rest sigmoids).

## Formulation

With labels $y_i \in \{0,\dots,r-1\}$, coefficients $A \in \mathbb{R}^{n\times r}$ (columns $a_k$), biases
$b \in \mathbb{R}^r$ and $M = K/\lambda$, the decision values are

$$
T = M A + \mathbf{1} b^\top \in \mathbb{R}^{n\times r},
$$

and the model minimises

$$
\sum_{k=1}^r \frac{a_k^\top K a_k}{2\lambda} + \sum_i w_i\Big(\log\sum_k e^{T_{ik}} - T_{i,y_i}\Big).
$$

The weights $w_i$ are the sample weights times the class weights (`"balanced"`: $n/(r\,n_c)$).

## Identifiability gauge

The soft-max loss is invariant under $T_i \mapsto T_i + c_i\mathbf{1}$. Stationarity in $A$ gives
$A = -G$ with $G_{ik} = w_i(p_{ik} - [k = y_i])$, whose rows sum to zero, so $\sum_k a_k = 0$ holds at
every solution without a constraint. The unregularised biases are determined only up to a common
constant; it is fixed by $\sum_k b_k = 0$, enforced by the penalty $\tfrac{\rho}{2}(\mathbf{1}^\top b)^2$
with $\rho = \operatorname{mean}(w)$, which vanishes at the solution. It keeps the Hessian positive
definite, does not depend on the hyperparameters and therefore adds nothing to the hypergradient.

## Newton system

Newton's method is applied to the stationarity equations $F(A,b)=0$,
$F_A = A + G(T)$, $F_b = \sum_i G_i + \rho\,\mathbf{1}^\top b\,\mathbf{1}$. With the per-sample Hessians
$H_i = w_i(\operatorname{diag}p_i - p_ip_i^\top)$ the Jacobian is

$$
J = \begin{pmatrix} I + HM & HE \\ E^\top H M & E^\top H E + \rho \mathbf{1}\mathbf{1}^\top\end{pmatrix},
\qquad (HM)_{(ik),(jl)} = (H_i)_{kl} M_{ij},
$$

where $E$ stacks $r\times r$ identities. The nonsymmetric $J$ is kept as a mathematical
reference, but is no longer factorized by LU. That product formulation can become severely
ill-conditioned at small $\lambda$, especially when probabilities saturate.

Let $C\in\mathbb{R}^{r\times(r-1)}$ be orthonormal class contrasts, $C^\top\mathbf{1}=0$,
and factor each categorical covariance in this subspace as $H_i=R_iR_i^\top$.
With block-diagonal $R$, factor the symmetric positive definite matrix

$$
S=I+R^\top M R.
$$

Its eigenvalues are at least one for a positive semidefinite kernel. The bias is solved with
the small Schur complement $P^\top S^{-1}P$, where $P=R^\top E C$. The common bias direction
is handled analytically by the gauge penalty. No inverse or diagonal jitter of $K$ is needed;
duplicate samples and zero individual sample weights remain supported. Every class must have
positive total training weight; otherwise it has no finite bias optimum. A singular bias Schur
complement raises an explicit linear-algebra error rather than producing a non-finite gradient.

Backtracking on the same penalised objective globalises the iteration, with a preconditioned
gradient fallback if a Newton system cannot be solved. Solves check backward residuals in the
original $J$ equations. Covariance diagonals and target-class loss derivatives use sums of
non-target probabilities, preserving tiny probabilities when the target probability rounds to
one. Kernel products are recomputed at trial points; for $\lambda<10^{-5}$ they use NumPy
extended accumulation before division. This changes no kernel or objective. The extra precision
is platform dependent (`longdouble` equals ordinary double on some platforms, including Windows);
very ill-conditioned kernels still impose a floating-point accuracy floor. The reported final
residual is recomputed from the returned coefficients and predictions.

## Hypergradient

For a validation loss $E(T_{\mathrm{val}})$ with $e = \partial E/\partial T_{\mathrm{val}}$ (the
multinomial negative log-likelihood, `MultinomialNLL`), the explicit part uses
$T_{\mathrm{val}} = K_{vt}A/\lambda + \mathbf{1}b^\top$ at fixed $(A,b)$. The implicit part needs one
adjoint solve per fold,

$$
J^\top z = \begin{pmatrix} K_{vt}^\top e/\lambda \\ e^\top\mathbf{1}\end{pmatrix},
\qquad \frac{dE}{d\theta}\Big|_{\text{implicit}} = -\sum_i (z_{a,i} + z_b)^\top H_i \frac{\partial T_i}{\partial\theta},
$$

with $\partial T/\partial\log\lambda = -KA/\lambda$ and $\partial T/\partial\log s_k = (\partial_{s_k}K)\,A/\lambda$
at fixed $A$. The same covariance-root factors solve the adjoint directly for
$u=H(z_a+Ez_b)$, without forming $J^\top$. A factor from the last Newton direction is reused
only when its curvature equals the final curvature exactly; otherwise the final system is
factorized once. The bias gauge has no hyperparameter derivative.

## Cost

The SPD system has size $n(r-1)$, plus an $(r-1)$-dimensional bias solve. One Newton step
still costs $O((rn)^3)$ and needs $O((rn)^2)$ memory, per fold. The model is meant
for $rn$ up to a few thousand; a `UserWarning` is issued if $r\cdot n_{\text{train fold}} > 4000$. Use
`n_jobs` to train the folds in parallel (memory grows with the number of parallel folds) or one-vs-rest
for larger problems. Every training fold must contain all classes (a `ValueError` is raised otherwise;
the default folds are stratified). `n_basis` and `max_asum` are not supported.

## Two classes

For $r=2$ the soft-max model is the binary logistic model: with $d = T_{\cdot 1} - T_{\cdot 0}$ one has
$\log\sum_k e^{T_{ik}} - T_{i,y_i} = \log(1+e^{-\tilde y_i d_i})$, $a_0=-a_1$, and the penalty becomes
$a_1^\top K a_1/\lambda$. Setting $f = d = K\beta/\lambda'$ with $\beta = 2 a_1\lambda'/\lambda$ gives the
binary penalty $\beta^\top K\beta/(2\lambda')$ equal to it exactly for $\lambda' = \lambda/2$. Hence **soft-max
with $\lambda$ is the binary logistic model with $\lambda/2$**. With two classes the estimators use the
binary code with that halved value: `SmoothSVC(lam, multi_class="softmax")` fits the binary model with
$\lambda/2$, and `OptimizedSVC` tunes the binary model with `lam_init` and `lam_bounds` halved and reports
`lam_` and `history_` in the soft-max convention (twice the binary value), so that `lam_` means the same
for any number of classes.

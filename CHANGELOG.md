# Changelog

## Unreleased

### Added
- Exact maximal margin classification (`loss="hard_margin"`) and hard epsilon-SVR
  (`loss="hard_epsilon_insensitive"`), thesis Sec. 4.1.2/4.2.3: indicator losses, dense constrained
  training with feasibility/KKT checks and explicit errors for contradictory data; scalar or
  per-sample epsilon in the low-level API. Positive lambda and positive weights leave predictions
  unchanged; zero weights remove constraints. The estimators and CV API support local active-set
  hypergradients for kernel scalings (and scalar epsilon in the low-level API); log-lambda derivative
  is zero. No subspace, 1-norm constraint, offsets or SMO/Newton options for these losses. Added a
  theory page and an executed hard/soft comparison example. Intended for moderate, well-conditioned
  problems; active-set changes can be nondifferentiable.
- `benchmarks/primal_first_order.py` (thesis Sec. 10.3.1–10.3.2): gradient descent and preconditioned gradient
  descent (Ch. 8.2/8.3) for the primal problem, implemented for this study only, against SMO; empirical error
  bound `‖F(z)‖ ≈ 0.98 ‖z − z*‖`. Neither method is competitive (SMO reaches the same distance about 24 times
  faster in steps and orders of magnitude faster in time), so they are not part of the library.
- `benchmarks/newton_vs_smo.py` (Sec. 10.3.3): Newton's method from zero, SMO to `1e-8` and the default
  two-stage training over n = 100 … 40 000 and four λ. Newton needs a size-independent number of steps, its
  time grows about like n²; SMO and the two-stage training are faster at every size.

## 0.5.0 – 2026-11-17

### Added
- `multi_class="softmax"` for `SmoothSVC` and `OptimizedSVC` (default `"ovr"`, unchanged): one joint
  multinomial (soft-max) model for more than two classes (thesis Sec. 5.1) with a decision function per
  class, common `lam` and scalings, trained by a dense Newton method (`optimized_svm.multiclass`). Needs
  `loss="logistic"`; `predict_proba` returns the soft-max probabilities. The optimiser tunes `lam` and the
  scalings with exact hypergradients of the cross-validation multinomial negative log-likelihood
  (`cv_error_and_gradient_multiclass`, with `n_jobs` and warm starts). Cost is `O((r n)^3)` per step (a
  `UserWarning` above `r n = 4000`); every training fold must contain all classes. With two classes the
  binary logistic model with `lam / 2` is used (the same model). See the new theory page and example.
- `optimized_svm.hypergradient.run_folds`: shared serial/parallel fold runner behind `n_jobs`.
- `n_jobs` parameter of `OptimizedSVC`, `OptimizedSVR`, `OptimizedKernelGLMRegressor` and
  `cv_error_and_gradient`: trains the cross-validation folds in parallel threads (scikit-learn
  semantics, `None`/1 serial, -1 all cores); the BLAS threads are divided among the workers. Results equal
  the serial ones up to floating-point rounding (the BLAS thread count differs). The gain is largest when
  SMO dominates (classification); for large Newton-dominated problems the serial run already uses
  multithreaded BLAS, and memory grows with the number of parallel folds. Uses `joblib`
  (a dependency of scikit-learn).

### Fixed
- Subspace hypergradient: the derivative w.r.t. `log lam` ignored the jitter of the basis kernel
  (`lam/2 beta^T (K_ZZ + JITTER I) beta`), so it was only accurate to ~1e-5 (default jitter) up to
  ~1e-3 for ill-conditioned `K_ZZ` (wide kernel, large basis, small `lam`). It is now computed exactly
  in the Cholesky coordinates; the scaling components were not affected.

## 0.4.0 – 2026-11-16

### Added
- `KernelGLMRegressor` and `OptimizedKernelGLMRegressor`: kernel GLMs (`family` Poisson, binomial,
  negative binomial, Gamma, Tweedie) with per-sample `offset` (e.g. log exposure) in `fit`,
  `decision_function`, `predict` and `score` (deviance explained D^2), fixed or tuned `lam` and kernel
  scalings by the cross-validation deviance; optional tuning of the negative binomial dispersion
  (`tune_dispersion`, validated by the new full likelihood `NegativeBinomialNLL`; validation losses can declare tunable `params` with a `param_grad` that enters the hypergradient). `_optimize` accepts an
  offset and tunable loss parameters; the existing estimators are unchanged.
- GLM losses in the half unit deviance form (minimum 0 per term, log/logit links): `BinomialLoss`
  (proportions with trial counts as weights), `NegativeBinomialLoss` (tunable `dispersion`),
  `GammaLoss` and `TweedieLoss` (power in (1, 2)), plus the matching validation losses
  `BinomialDeviance`, `NegativeBinomialDeviance`, `GammaDeviance`, `TweedieDeviance`.
- Per-sample `offset` in the training problem (`TrainingProblem`, `SubspaceProblem`) and in
  `fold_error_and_gradient`, `fold_error_and_gradient_subspace` and `cv_error_and_gradient`
  (decision value `K a / lam + b + offset`; only for losses without SMO stage). The hypergradient stays exact.
- `Loss.initial_bias(terms)`: loss-specific starting bias for Newton without SMO warm start (squared and
  Poisson loss override it).
- `ValidationLoss.weighted` and an optional per-sample weight argument `w` of validation losses (passed
  from the cross-validation code only if `weighted` is set).
- Theory page `theory/glm.md` and example `examples/glm_counts.py` (count data with exposure, Poisson vs.
  negative binomial with tuned dispersion).

## 0.3.1 – 2026-11-16

### Added
- `cv` of `OptimizedSVC`/`OptimizedSVR` accepts scikit-learn splitter objects (anything with `split`,
  e.g. `GroupKFold`, `LeaveOneGroupOut`, unshuffled `KFold`); `fit` has a new argument `groups=None`
  that is passed to `cv.split(X, y, groups)` (also in the one-vs-rest classifier). Integers and iterables
  of index pairs behave as before.
- Migration page "Coming from optimized-lssvr" (`docs/migration-lssvr.md`) with the exact parameter
  mapping; `tests/test_lssvr_oracle.py` checks the cross-validation objective and its gradient
  (RBF, ARD, feature groups) against reference values of `optimized-lssvr`
  (`benchmarks/lssvr_oracle_reference.py` regenerates them).

### Changed
- Documentation built with Zensical instead of MkDocs Material (`zensical.toml`); the example scripts are
  executed by `scripts/build_examples.py` before the build (Zensical has no `mkdocs-jupyter`). API
  reference: new section for `optimized_svm.subspace`.

## 0.3.0 – 2026-11-16

### Added
- Subspace SVM (thesis Sec. 5.3), experimental: `optimized_svm.subspace` (Newton training in the span of
  $m$ basis points, systems of size $m+1$) and the `basis` argument of `cv_error_and_gradient` with an
  exact hypergradient (checked against finite differences). `select_basis` chooses the basis points by
  pivoted Cholesky until the Nyström residual trace is at most `tol * lam`; `nystroem_residual`. The
  estimators take `n_basis` (int or `"auto"`) and `basis_tol`; with `"auto"` the optimizers re-select the
  basis (up to 5 rounds per stage, `basis_round` in `history_`) when the residual exceeds
  `basis_tol * lam` at the result of L-BFGS-B. Studies: `benchmarks/subspace_params.py`,
  `subspace_large.py`, `subspace_hpo.py`. Pays off for fits with fixed hyperparameters at large $n$;
  in the optimizers no gain without loss of accuracy so far; see `theory/subspace.md`.
- Benchmark scripts `benchmarks/hpo_vs_search.py` (gradient method against grid, random and
  successive-halving search) and `benchmarks/hpo_timing.py` (wall-clock time of complete
  optimizations, for comparing releases). Results: https://wotzlaff.github.io/svm-site/benchmarks/
- Hypergradient with respect to the loss parameters `epsilon` and `smoothing`: `loss_params` of
  `cv_error_and_gradient`/`fold_error_and_gradient` and `Loss.param_ratio`. Exact (checked against finite
  differences), but not used by the estimators: tuning these parameters rarely pays off (`epsilon` is not
  identifiable below the smoothing width, the CV error is multimodal in `smoothing`). Study:
  `benchmarks/hpo_loss_params.py`.

### Changed
- `cv_error_` of the optimizers is the cross-validation error at the returned hyperparameters
  (before: the minimum over `history_`; the same value except with re-selected subspace bases).

### Fixed
- `newton_subspace` repeats the solve from a cold start when a warm start with huge coefficients
  (tiny `lam`) gives no descent step.

## 0.2.1 – 2026-10-29

### Changed
- Switch from SMO to Newton: the rough SMO stage now stops at tolerance 0.3 (was 0.1) or after five
  steps per loss term. Same results; ε-SVR hyperparameter optimization up to 2.5 times faster, and
  1–2 orders of magnitude faster with an active 1-norm bound, where SMO converges very slowly.
  Explicit `tol`/`max_steps` in `smo_options` replace the defaults. Study:
  `benchmarks/smo_newton_switch.py`.

### Added
- `smo_options` accepts the bound-aware stopping criterion of `smo-solver` (`stopping="bound_aware"`,
  `rho`); requires `smo-solver>=0.1.2`.

## 0.2.0 – 2026-10-29

### Added
- Logistic loss (thesis Sec. 4.1.3): `LogisticLoss`, `LogisticValidation` and `loss="logistic"` for
  `SmoothSVC`/`OptimizedSVC`, including `predict_proba` (sigmoid of the decision values, normalised
  one-vs-rest probabilities for more than two classes). Trained by Newton's method only, the solution is
  dense; `max_asum` is not supported. Hypergradients are covered by the finite-difference tests.

## 0.1.1 – 2026-10-29

### Changed
- Faster hypergradient: kernel matrices are handled per sample (ε-SVR uses `n` instead of `2n`
  columns), the validation kernel columns are reused for the adjoint system, and the kernel
  derivatives are contracted per block with one matrix product instead of forming a distance
  matrix per block.
- The Cholesky factor of the last Newton system is reused for the adjoint system when the
  curvature pattern is unchanged (always the case for smoothing of degree 2).
- Linear systems of size up to 1000 are factorized with a single BLAS thread; multi-threaded
  LAPACK was several times slower for them, notably on shared machines. New dependency:
  `threadpoolctl`.
- Typical effect (ε-SVR, `n = 450`, 5-fold, one BLAS thread): 3.2 s → 2.0 s; ARD 2.0 s → 0.9 s;
  classification (`n = 600`) 1.3 s → 0.7 s. Results are unchanged.

### Fixed
- Warm starts were counterproductive for classification: repairing the rounding error of
  `sum(alpha) = 0` moved a third of the variables off their bounds and made SMO take about four
  times as many steps as a cold start. Now about four times fewer than a cold start.

## 0.1.0 – 2026-10-28

Complete overhaul of the 2022 code (tag `legacy-2022`).

### Changed
- training backend: Rust solver `smo-solver` instead of the Nim package `nimsmo`
- new package layout (`src/`), `pyproject.toml`, snake_case API, scikit-learn compatible
  estimators `SmoothSVC`, `SmoothSVR`, `OptimizedSVC`, `OptimizedSVR`
- adjoint formulation of the hypergradient (one linear system per fold)

### Added
- one-vs-rest multi-class classification, class and sample weights
- warm starts across hyperparameter evaluations, rough SMO tolerance before Newton
- `decision_gradient()` (replaces the unfinished `estimate_sensitivity`)
- test suite (finite-difference checks of all gradients, scikit-learn estimator checks),
  documentation, examples and a reproduction script for thesis Table 10.4.1

### Fixed
- see `LEGACY_BUGS.md` in the project workspace (P1–P13), notably: regression HPO with
  `maxAsum` did not terminate, SMO and Newton used different kernel parameters, the 1-norm
  constraint was always treated as active

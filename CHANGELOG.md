# Changelog

## Unreleased

### Added
- Benchmark scripts `benchmarks/hpo_vs_search.py` (gradient method against grid, random and
  successive-halving search) and `benchmarks/hpo_timing.py` (wall-clock time of complete
  optimizations, for comparing releases). Results: https://wotzlaff.github.io/svm-site/benchmarks/

## Unreleased

### Added
- Hypergradient with respect to the loss parameters `epsilon` and `smoothing`: `loss_params` of
  `cv_error_and_gradient`/`fold_error_and_gradient` and `Loss.param_ratio`. Exact (checked against finite
  differences), but not used by the estimators: tuning these parameters rarely pays off (`epsilon` is not
  identifiable below the smoothing width, the CV error is multimodal in `smoothing`). Study:
  `benchmarks/hpo_loss_params.py`.

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

# Changelog

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

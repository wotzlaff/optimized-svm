# Changelog

## 0.2.0 – unreleased

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

# optimized-svm

[![PyPI](https://img.shields.io/pypi/v/optimized-svm)](https://pypi.org/project/optimized-svm/)
[![CI](https://github.com/wotzlaff/optimized-svm/actions/workflows/ci.yml/badge.svg)](https://github.com/wotzlaff/optimized-svm/actions/workflows/ci.yml)

**Support vector machines whose hyperparameters are tuned by exact gradients instead of grid search.**

`optimized-svm` treats hyperparameter selection as a bilevel optimization problem: the upper level
minimizes a (smoothed) cross-validation error, the lower level is the SVM training problem. Smoothing
the loss makes the training solution differentiable in the hyperparameters, so the gradient of the
CV error follows from the implicit function theorem and a quasi-Newton method (L-BFGS-B) can be used.
This works for the regularization parameter **and** for per-feature kernel scalings (automatic
relevance determination), where grid search is hopeless.

The methods are developed in

> N. Strasdat, *Duality, Derivative-Based Training Methods and Hyperparameter Optimization for
> Support Vector Machines*, Dissertation, Technische Universität Dresden, 2023.
> [urn:nbn:de:bsz:14-qucosa2-874731](https://nbn-resolving.org/urn:nbn:de:bsz:14-qucosa2-874731)

Training uses the Rust SMO solver [`smo-solver`](https://github.com/wotzlaff/smo-solver) as a warm start
followed by a semismooth Newton method on the smoothed primal problem.

## Features

- scikit-learn estimators: `OptimizedSVC`, `OptimizedSVR` (tuned) and `SmoothSVC`, `SmoothSVR`
  (fixed hyperparameters); binary and one-vs-rest classification
- losses: smoothed hinge, smoothed ε-insensitive, squared (LS-SVM), Poisson; smoothing of degree
  2, 3 or 6; sample/class weights; optional bound on `Σ|α|` (generalized ν-SVM)
- exact hypergradients via one adjoint linear system per fold (verified against finite differences)
- ARD kernels (`blocks="features"`) with two-stage optimization, warm starts, time limits,
  full optimization history
- `decision_gradient()` for local feature sensitivities

## Installation

```bash
pip install optimized-svm
```

For development (the solver [`smo-solver`](https://pypi.org/project/smo-solver/) is installed
from PyPI as a prebuilt wheel):

```bash
git clone https://github.com/wotzlaff/optimized-svm && cd optimized-svm
uv sync --group dev
uv run pytest
```

## Quick start

```python
from sklearn.datasets import make_friedman1
from optimized_svm import OptimizedSVR

x, y = make_friedman1(600, noise=0.5, random_state=0)
model = OptimizedSVR(blocks="features").fit(x, y)   # one kernel scaling per feature

model.lam_            # optimized regularization parameter
model.scaling_opt_    # optimized scalings: features 5..9 (pure noise) are switched off
model.history_        # every evaluation of the CV error
model.predict(x[:5])
```

On a 1000-sample classification task (`examples/quickstart.py`) the optimizer needs 16 CV
evaluations (≈10 s) versus 143 grid points (≈20 s) for `GridSearchCV` with `sklearn.svm.SVC`,
with comparable test accuracy (0.844 vs. 0.856).

## How it works

| step | method | thesis |
|---|---|---|
| training, warm start | SMO on the dual with quadratic smoothing (`smo-solver`) | Ch. 7 |
| training, refinement | semismooth Newton on `a + φ'(Ka/λ + b + sc) = 0` | Ch. 8.4 |
| CV error gradient | implicit function theorem, adjoint system per fold | Ch. 9.2 |
| hyperparameter search | L-BFGS-B over `(log λ, log s₁, …, log s_K)` | Ch. 9.1, 10.4 |

## Reproducing the thesis

`benchmarks/thesis_regression.py` repeats the protocol of Table 10.4.1 (LIBSVM regression data,
30 random 80/20 splits, ε-SVR with cubic smoothing and LS-SVM):

```bash
python benchmarks/thesis_regression.py --datasets pyrim triazines mpg housing --splits 30
```

## Documentation

Full documentation including the mathematical background: <https://wotzlaff.github.io/optimized-svm/>.
Build locally with `uv run --group docs mkdocs serve`.

## Citation

If you use this software, please cite the dissertation:

```bibtex
@phdthesis{strasdat2023,
  author = {Nico Strasdat},
  title  = {Duality, Derivative-Based Training Methods and
            Hyperparameter Optimization for Support Vector Machines},
  school = {Technische Universit{\"a}t Dresden},
  type   = {Dissertation},
  year   = {2023},
  url    = {https://nbn-resolving.org/urn:nbn:de:bsz:14-qucosa2-874731},
  note   = {urn:nbn:de:bsz:14-qucosa2-874731}
}
```

## License

BSD-3-Clause. The 2022 version (Nim backend) is preserved in the git tag `legacy-2022`.

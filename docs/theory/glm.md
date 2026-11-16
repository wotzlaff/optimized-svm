# GLM losses

For positive or count-valued targets the squared loss is a poor fit. The following losses are
generalized linear model (GLM) families with the *decision value* `t` acting on the link scale.
Each term is the **half unit deviance**, so that its minimum over `t` is exactly 0 (at `mu = y`),
which keeps the loss well scaled compared with the raw negative log-likelihood. All of them are
convex and smooth in `t` (no SMO warm start; they are solved by Newton's method).

| Loss | Link | Targets | Term `phi(t)` (times weight `w`) |
|---|---|---|---|
| `BinomialLoss` | `mu = expit(t)` | proportions `y` in [0, 1], `w` = trials | `y log(y/mu) + (1-y) log((1-y)/(1-mu))` |
| `NegativeBinomialLoss(dispersion=r)` | `mu = e^t` | counts `y >= 0` | `y log(y/mu) - (y+r) log((y+r)/(mu+r))` |
| `GammaLoss` | `mu = e^t` | `y > 0` | `y e^{-t} + t - log y - 1` |
| `TweedieLoss(power=p)` | `mu = e^t` | `y >= 0`, `1 < p < 2` | `y^{2-p}/((1-p)(2-p)) - y e^{(1-p)t}/(1-p) + e^{(2-p)t}/(2-p)` |

Derivatives (without the weight `w`):

| Loss | `phi'` | `phi''` |
|---|---|---|
| Binomial | `mu - y` | `mu (1 - mu)` |
| Negative binomial | `r (mu - y)/(mu + r)` | `(y + r) r mu/(mu + r)^2` |
| Gamma | `1 - y e^{-t}` | `y e^{-t}` |
| Tweedie | `e^{(1-p)t} (e^t - y)` | `(p-1) y e^{(1-p)t} + (2-p) e^{(2-p)t}` |

## When to use which

- **Binomial**: success proportions out of `n_i` trials (pass `n_i` as the sample weight); the
  case `n_i = 1` is logistic regression with labels in {0, 1}.
- **Negative binomial**: over-dispersed counts (variance `mu + mu^2/r`); `r -> infinity` recovers
  the Poisson loss. `r` can be tuned by the hypergradient with `loss_params=("dispersion",)`.
- **Gamma**: strictly positive, right-skewed targets with constant coefficient of variation
  (durations, costs).
- **Tweedie** with `1 < p < 2`: non-negative targets with a point mass at zero and a continuous
  positive part (insurance claims, rainfall); `p -> 1` is Poisson-like, `p -> 2` gamma-like.

## Dispersion hypergradient

With `phi'(t; r) = r (mu - y)/(mu + r)` one has
`d phi'/d r = mu (mu - y)/(mu + r)^2`, so the ratio used by the hypergradient is

```text
(d phi'/d r) / phi'' = (mu - y) / (r (y + r)).
```

The hypergradient code multiplies by `r`, which gives the derivative with respect to `log r`
(the parametrisation of `theta`).

## Validation

`BinomialDeviance`, `NegativeBinomialDeviance(dispersion)`, `GammaDeviance` and
`TweedieDeviance(power)` sum the same half unit deviances over the validation samples; the
prediction `f` is on the link scale. `BinomialDeviance.weighted` is true (it uses the number of
trials as validation weights), the others ignore weights.

## Starting value of the bias

`Loss.initial_bias` returns the intercept-only estimate (logit or log of the weighted mean of `y`,
corrected for the mean offset if present) that Newton can use when it starts without SMO.

## Usage

`KernelGLMRegressor` (fixed hyperparameters) and `OptimizedKernelGLMRegressor` (`lam`, the kernel
scalings and optionally the negative binomial dispersion tuned by the cross-validation deviance)
fit these models with `family` in `"poisson"`, `"binomial"`, `"negative_binomial"`, `"gamma"` and
`"tweedie"`. Counts with an exposure, e.g. claims per policy and time at risk, enter through the
`offset` (the log of the exposure), which is added to the decision value on the link scale in
training, validation and prediction:

```python
import numpy as np
from optimized_svm import OptimizedKernelGLMRegressor

rng = np.random.default_rng(0)
X = rng.uniform(-2, 2, (400, 2))
exposure = rng.uniform(0.5, 3.0, 400)
rate = np.exp(0.5 + np.sin(2 * X[:, 0]))
y = rng.negative_binomial(2.0, 2.0 / (2.0 + rate * exposure))

model = OptimizedKernelGLMRegressor(family="negative_binomial", dispersion=2.0, cv=5)
model.fit(X, y, offset=np.log(exposure))

model.predict(X, offset=np.log(exposure))  # expected counts
model.predict(X)                           # rates per unit exposure
model.decision_function(X)                 # linear predictor (link scale)
model.score(X, y, offset=np.log(exposure))  # D^2
```

For the binomial family `y` are proportions and `sample_weight` the numbers of trials; fitting
grouped data is equivalent to fitting the expanded Bernoulli data.

**Score.** `score` returns the deviance explained
`D^2 = 1 - deviance(model) / deviance(null model)`, where the null model is the intercept-only fit
to the same data (`mu_i = e^{b + offset_i}`, resp. the weighted mean without offset; computed by
`Loss.initial_bias`). It is 1 for a perfect fit and 0 for the intercept-only model.

**Dispersion tuning.** With `tune_dispersion=True` (negative binomial only, not with `n_basis`)
`log r` is appended to the hyperparameters and optimised within `dispersion_bounds` with the
hypergradient above; the result is `dispersion_` and each `history_` entry carries the current
`dispersion`. The half deviance of the validation samples depends on `r` itself and would be
minimised by degenerate values, so in this case the validation loss is the Poisson deviance of the
predicted mean (`PoissonValidation`), which is independent of `r`: the dispersion only changes how
the training problem weights large counts. Its signal is therefore weak; it moves `r` in the right
direction for clearly over-dispersed data, but the result should not be read as a maximum
likelihood estimate of `r`.

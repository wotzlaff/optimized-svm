# %% [markdown]
# # Count data with exposure (GLM)
#
# Claim counts, accidents or visits are observed over different periods of time or for different
# numbers of policies. The *exposure* `E_i` says how much "opportunity" observation `i` had, and the
# expected count is proportional to it: `E[y_i] = E_i * rate(x_i)`. In a GLM with log link this is
# an `offset`: we pass `log(E_i)`, which is added to the decision value on the link scale, so that
# the kernel model only has to learn the rate per unit exposure.
#
# Real counts are often *overdispersed*: their variance exceeds the mean, which the Poisson
# family cannot represent (variance = mean). The negative binomial family has
# `Var = mu + mu^2 / r` with a dispersion `r` (smaller `r` means more overdispersion), and
# `tune_dispersion=True` chooses `r` together with the other hyperparameters by cross-validation.
#
# The score of the GLM estimators is the deviance explained `D^2 = 1 - deviance(model) /
# deviance(null model)`; the null model is the intercept-only fit, which accounts for the offset.
# `D^2 = 1` is a perfect fit and `0` is no better than the intercept.
#
# Here the data are synthetic: two features, a smooth nonlinear rate, exposure `E ~ U(0.2, 2)` and
# negative binomial counts with true `r = 2`. Saves `glm_counts.png`.

# %%
import time

import matplotlib.pyplot as plt
import numpy as np

from optimized_svm import OptimizedKernelGLMRegressor

rng = np.random.default_rng(0)
true_r = 2.0


def true_rate(x):
    return np.exp(0.3 + np.sin(2.0 * x[:, 0]) + 0.5 * x[:, 1] ** 2 - 0.4)


def sample(n):
    x = rng.uniform(-1.5, 1.5, (n, 2))
    exposure = rng.uniform(0.2, 2.0, n)
    mu = true_rate(x) * exposure
    y = rng.negative_binomial(true_r, true_r / (true_r + mu))
    return x, exposure, y


x_train, e_train, y_train = sample(600)
x_test, e_test, y_test = sample(600)
print(f"mean count {y_train.mean():.2f}, variance {y_train.var():.2f}")

# %% [markdown]
# Both models get `offset=np.log(exposure)` in `fit` and `score`; the negative binomial model also
# tunes `r`. Note that each model's `D^2` is measured with its own family's deviance, so the two
# values are not directly comparable: the Poisson deviance penalises the large counts of an
# overdispersed sample differently than the negative binomial deviance. The tuned `r` and the rate
# curves below are the more informative comparison.

# %%
models = {
    "poisson": OptimizedKernelGLMRegressor(family="poisson", cv=5, max_iter=40),
    "negative_binomial": OptimizedKernelGLMRegressor(
        family="negative_binomial", tune_dispersion=True, cv=5, max_iter=40
    ),
}
for name, model in models.items():
    t0 = time.perf_counter()
    model.fit(x_train, y_train, offset=np.log(e_train))
    seconds = time.perf_counter() - t0
    d2 = model.score(x_test, y_test, offset=np.log(e_test))
    print(f"{name:18s} {seconds:5.1f} s  test D^2 {d2:.3f}  (lam = {model.lam_:.3g})")
nb = models["negative_binomial"]
print(f"tuned dispersion {nb.dispersion_:.2f} (true {true_r:.1f})")

# %% [markdown]
# `predict` with an offset returns expected counts, without it the rate per unit exposure. The
# expected count therefore scales linearly with the exposure:

# %%
x0 = x_test[:1]
counts = [float(nb.predict(x0, offset=np.log([e]))[0]) for e in (0.5, 1.0, 2.0)]
print(
    f"expected counts at exposure 0.5 / 1 / 2: {counts[0]:.3f} / {counts[1]:.3f} / {counts[2]:.3f}"
)

# %% [markdown]
# The plot shows the rate per unit exposure along the first feature (second feature fixed at 0),
# i.e. `predict` without offset, against the true rate:

# %%
grid = np.column_stack([np.linspace(-1.5, 1.5, 200), np.zeros(200)])
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.plot(grid[:, 0], true_rate(grid), "k:", label="true rate")
for name, model in models.items():
    ax.plot(grid[:, 0], model.predict(grid), label=name)
ax.set_xlabel("$x_0$ (with $x_1 = 0$)")
ax.set_ylabel("rate per unit exposure")
ax.legend()
fig.tight_layout()
fig.savefig("glm_counts.png", dpi=150)

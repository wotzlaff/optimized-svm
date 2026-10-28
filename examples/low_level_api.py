# %% [markdown]
# # Low-level API
#
# Build a training problem, train it, and evaluate the cross-validation error gradient.

# %%
import numpy as np
from sklearn.model_selection import KFold

from optimized_svm import EpsilonInsensitiveLoss, TrainingProblem, cv_error_and_gradient, train
from optimized_svm.validation import SmoothAbsoluteValidation

rng = np.random.default_rng(0)
x = rng.uniform(-2, 2, size=(300, 2))
y = np.sin(2 * x[:, 0]) * np.cos(x[:, 1]) + 0.1 * rng.normal(size=300)

# %% [markdown]
# A smoothed epsilon-SVR with a piecewise cubic smoothing ($C^2$ loss).

# %%
loss = EpsilonInsensitiveLoss(epsilon=0.1, smoothing=0.05, degree="3")
prob = TrainingProblem(x * 0.8, y, loss, lam=1e-3)
sol = train(prob)
print(f"SMO steps {sol.smo_steps}, Newton steps {sol.newton_steps}, residual {sol.residual:.1e}")

# %% [markdown]
# The CV error and its exact gradient with respect to $(\log\lambda, \log s)$.

# %%
folds = list(KFold(5, shuffle=True, random_state=0).split(x))
theta = np.log([1e-3, 0.8])
res = cv_error_and_gradient(
    theta, x, y, np.ones(len(y)), folds, loss, SmoothAbsoluteValidation(0.1)
)
print(f"CV error {res.error:.4f}, gradient {res.gradient}")

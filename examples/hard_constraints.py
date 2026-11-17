# %% [markdown]
# # Hard constraints: maximal margin classification and hard ε-SVR
#
# A hard constraint must be met. Ordinary hinge/epsilon-insensitive losses permit violations
# when reducing the RKHS norm is worth their finite penalty. Sections 4.1.2 and 4.2.3 instead
# minimize the norm subject to a fixed margin or a tube containing every training point.

# %%
import matplotlib.pyplot as plt
import numpy as np

from optimized_svm import SmoothSVC, SmoothSVR

x = np.linspace(-2, 2, 17)[:, None]
labels = np.where(x[:, 0] < 0, -1, 1)
xx = np.linspace(-2.3, 2.3, 300)[:, None]

hard = SmoothSVC(loss="hard_margin", scaling=1).fit(x, labels)
soft = SmoothSVC(lam=4, smoothing=0, scaling=1).fit(x, labels)
print(f"Smallest hard margin: {np.min(labels * hard.decision_function(x)):.8f}")
print(f"Smallest soft margin: {np.min(labels * soft.decision_function(x)):.8f}")

fig, ax = plt.subplots(figsize=(7, 3))
ax.plot(xx[:, 0], hard.decision_function(xx), label="Hard margin")
ax.plot(xx[:, 0], soft.decision_function(xx), label="Soft margin, λ=4")
ax.scatter(x[:, 0], labels, c=labels, cmap="coolwarm", edgecolors="black")
for level in [-1, 1]:
    ax.axhline(level, color="gray", ls=":")
ax.set(xlabel="x", ylabel="Decision value")
ax.legend()
plt.show()

# %% [markdown]
# Regression: all residuals stay inside the chosen tube. Here epsilon is in original target
# units because target normalization is disabled. Changing positive lambda has no effect.

# %%
targets = np.sin(x[:, 0]) + 0.04 * np.cos(9 * x[:, 0])
epsilon = 0.15
reg = SmoothSVR(loss="hard_epsilon_insensitive", epsilon=epsilon, normalize_y=False, scaling=1).fit(
    x, targets
)
reg_scaled = SmoothSVR(
    loss="hard_epsilon_insensitive", epsilon=epsilon, normalize_y=False, scaling=1, lam=100
).fit(x, targets)
assert np.max(abs(reg.predict(x) - targets)) <= epsilon + 1e-7
assert np.allclose(reg.predict(xx), reg_scaled.predict(xx), atol=1e-8)
print(f"Largest residual: {np.max(abs(reg.predict(x) - targets)):.8f}")
print(f"Support vectors: {reg.n_support_}/{len(x)}")
curve = reg.predict(xx)
fig, ax = plt.subplots(figsize=(7, 3))
ax.fill_between(xx[:, 0], curve - epsilon, curve + epsilon, alpha=0.2, label="Hard ε-tube")
ax.plot(xx[:, 0], curve, label="Prediction")
ax.scatter(x[:, 0], targets, color="black", s=20, label="Training targets")
ax.scatter(
    x[reg.support_, 0],
    targets[reg.support_],
    facecolors="none",
    edgecolors="red",
    s=80,
    label="Support vectors",
)
ax.set(xlabel="x", ylabel="y")
ax.legend()
plt.show()

# %% [markdown]
# Contradictory constraints produce an explicit error. At x=0 one prediction cannot be both
# below −0.8 and above +0.8. Widening epsilon to 1 permits the constant zero prediction.

# %%
conflicting_x = np.array([[0.0], [0.0]])
conflicting_y = np.array([-1.0, 1.0])
try:
    SmoothSVR(loss="hard_epsilon_insensitive", epsilon=0.2, normalize_y=False).fit(
        conflicting_x, conflicting_y
    )
except ValueError as exc:
    print(exc)
feasible = SmoothSVR(loss="hard_epsilon_insensitive", epsilon=1.0, normalize_y=False).fit(
    conflicting_x, conflicting_y
)
print("With ε=1:", feasible.predict(conflicting_x))

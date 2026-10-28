"""Automatic relevance determination: one kernel scaling per feature (Friedman #1 data).

Only the first five features influence the target; the optimised scalings of the remaining
five features shrink towards zero. Saves ``ard_feature_relevance.png``.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_friedman1
from sklearn.model_selection import train_test_split

from optimized_svm import OptimizedSVR

x, y = make_friedman1(600, n_features=10, noise=0.5, random_state=0)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)

common = OptimizedSVR(verbose=1).fit(x_train, y_train)
ard = OptimizedSVR(blocks="features", verbose=1).fit(x_train, y_train)
print(f"R^2 single scaling: {common.score(x_test, y_test):.3f}")
print(f"R^2 ARD:            {ard.score(x_test, y_test):.3f}")

fig, ax = plt.subplots(figsize=(6, 3))
ax.bar(np.arange(10), ard.scaling_opt_, color=["#3b6ea8"] * 5 + ["#b0b7c3"] * 5)
ax.set_xticks(np.arange(10), [f"x{j}" for j in range(10)])
ax.set_ylabel("optimised scaling $s_j$")
ax.set_title("ARD on Friedman #1 (x0-x4 relevant)")
fig.tight_layout()
fig.savefig("ard_feature_relevance.png", dpi=150)

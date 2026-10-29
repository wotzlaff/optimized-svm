# %% [markdown]
# # Class probabilities with the logistic loss
#
# `loss="logistic"` trains the SVM with the logistic loss of the thesis (Sec. 4.1.3) and enables
# `predict_proba`. We tune it with `OptimizedSVC` and compare it with the smoothed hinge loss, whose
# decision values we pass through a sigmoid for the comparison.
#
# The logistic loss needs no SMO warm start and gives a dense solution, so every Newton system has
# the full size `n`: it is the slower of the two and meant for moderate `n`.

# %%
import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.datasets import make_classification
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from optimized_svm import OptimizedSVC

x, y = make_classification(800, 8, n_informative=4, flip_y=0.1, random_state=0)
x = StandardScaler().fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)

# %%
results = {}
for loss in ("hinge", "logistic"):
    t0 = time.perf_counter()
    est = OptimizedSVC(loss=loss, cv=5).fit(x_train, y_train)
    seconds = time.perf_counter() - t0
    f = est.decision_function(x_test)
    p = 1.0 / (1.0 + np.exp(-f))  # sigmoid of the decision values (= predict_proba for "logistic")
    results[loss] = p
    print(
        f"{loss:9s} {seconds:5.1f} s  accuracy {est.score(x_test, y_test):.3f}  "
        f"log-loss {log_loss(y_test, p):.3f}  (lam = {est.lam_:.3g})"
    )

# %% [markdown]
# `predict_proba` returns the probabilities of both classes. A reliability diagram shows how well
# they are calibrated:

# %%
fig, ax = plt.subplots(figsize=(4.5, 4.5))
for loss, p in results.items():
    frac, mean = calibration_curve(y_test, p, n_bins=8)
    ax.plot(mean, frac, "o-", label=loss)
ax.plot([0, 1], [0, 1], "k:", label="perfect")
ax.set_xlabel("predicted probability")
ax.set_ylabel("observed frequency")
ax.legend()
fig.tight_layout()

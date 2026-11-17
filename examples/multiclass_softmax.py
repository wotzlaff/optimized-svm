# %% [markdown]
# # Multi-class: one-vs-rest or soft-max
#
# For more than two classes `OptimizedSVC(loss="logistic")` fits by default one binary model per
# class
# (`multi_class="ovr"`), each with its own `lam` and scalings, and normalises the sigmoids to
# probabilities. With `multi_class="softmax"` it fits **one joint multinomial model** (thesis,
# Sec. 5.1)
# with a common `lam` and common scalings, tuned by the cross-validation multinomial
# log-likelihood, and
# `predict_proba` returns the soft-max probabilities directly.
#
# The soft-max model solves dense Newton systems of size `r * n` (cost `O((r n)^3)` per step), so
# it is
# meant for a few thousand `r * n` at most. We compare both on the wine data (3 classes).

# %%
import time

import numpy as np
from sklearn.datasets import load_wine
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from optimized_svm import OptimizedSVC

x, y = load_wine(return_X_y=True)
x = StandardScaler().fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=1, stratify=y)

# %%
print(f"{'multi_class':12s} {'time [s]':>8s} {'accuracy':>9s} {'log loss':>9s}")
for multi_class in ("ovr", "softmax"):
    t0 = time.perf_counter()
    est = OptimizedSVC(loss="logistic", multi_class=multi_class, cv=5, n_jobs=2).fit(
        x_train, y_train
    )
    seconds = time.perf_counter() - t0
    proba = est.predict_proba(x_test)
    print(
        f"{multi_class:12s} {seconds:8.1f} {est.score(x_test, y_test):9.3f} "
        f"{log_loss(y_test, proba):9.3f}"
    )
    if multi_class == "softmax":
        print(f"\nsoft-max: lam = {est.lam_:.3g}, scaling = {est.scaling_opt_[0]:.3g}")
        print("probabilities sum to one:", np.allclose(proba.sum(axis=1), 1.0))

# %% [markdown]
# The one-vs-rest model has one set of hyperparameters per class (`est.estimators_[k].lam_`), the
# soft-max model a single `lam_` and `scaling_opt_`. Its cross-validation error `est.cv_error_`
# is the
# mean multinomial negative log-likelihood of the validation folds.

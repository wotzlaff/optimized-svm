"""Gradient-based hyperparameter optimisation vs. grid search on a classification task."""

import time

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from optimized_svm import OptimizedSVC

x, y = make_classification(1000, 8, n_informative=4, flip_y=0.05, random_state=0)
x = StandardScaler().fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)

t0 = time.perf_counter()
opt = OptimizedSVC(cv=5).fit(x_train, y_train)
t_opt = time.perf_counter() - t0
acc = opt.score(x_test, y_test)
print(
    f"OptimizedSVC: lam = {opt.lam_:.3g}, gamma = {opt.scaling_opt_[0] ** 2:.3g}, "
    f"{len(opt.history_)} CV evaluations, {t_opt:.1f} s, test accuracy {acc:.3f}"
)

grid = {"C": np.logspace(-2, 4, 13), "gamma": np.logspace(-4, 1, 11)}
t0 = time.perf_counter()
search = GridSearchCV(SVC(), grid, cv=5, n_jobs=1).fit(x_train, y_train)
t_grid = time.perf_counter() - t0
best = search.best_params_
print(
    f"GridSearchCV: C = {best['C']:.3g}, gamma = {best['gamma']:.3g}, "
    f"{len(search.cv_results_['params'])} grid points, {t_grid:.1f} s, "
    f"test accuracy {search.score(x_test, y_test):.3f}"
)

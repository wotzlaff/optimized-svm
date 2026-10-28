# Reproducing the thesis

`benchmarks/thesis_regression.py` follows the protocol of Sec. 10.4.2:

- LIBSVM regression datasets (downloaded and cached on first use),
- 30 random splits, 80 % for hyperparameter optimization and training, 20 % for testing,
- features and targets standardized on the training part,
- smoothed $\varepsilon$-SVR with cubic smoothing ($\delta = \varepsilon/4$) for
  $\varepsilon \in \{0.1, 0.25, 0.5\}$ and the LS-SVM,
- 5-fold CV with the smoothed absolute validation error ($\delta = 0.1$),
- start $\lambda = 10^{-2}$, $\gamma = \log 2 / d$; bounds $\lambda \in [10^{-6}, 1]$,
  $\gamma \in [10^{-12}, 10^4]$; time limit 600 s,
- datasets with more than 1000 samples are subsampled to 1000.

```bash
python benchmarks/thesis_regression.py --splits 30
```

RESULTS_PLACEHOLDER

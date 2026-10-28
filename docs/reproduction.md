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

## Results (2026-10-28, 30 splits)

Test MSE / MAE in original units; thesis values in parentheses.

| dataset | model | MSE | MAE |
|---|---|---:|---:|
| pyrim | ε = 0.1 | 0.00630 (0.00656) | 0.0505 (0.0532) |
| | ε = 0.25 | 0.00599 (0.00625) | 0.0498 (0.0512) |
| | ε = 0.5 | 0.00670 (0.00645) | 0.0529 (0.0526) |
| | LS-SVM | 0.00682 (0.00597) | 0.0518 (0.0491) |
| triazines | ε = 0.1 | 0.0209 (0.0244) | 0.102 (0.107) |
| | ε = 0.25 | 0.0213 (0.0239) | 0.103 (0.109) |
| | ε = 0.5 | 0.0206 (0.0237) | 0.104 (0.111) |
| | LS-SVM | 0.0213 (0.0232) | 0.107 (0.110) |
| mpg | ε = 0.1 | 7.75 (6.50) | 1.95 (1.81) |
| | ε = 0.25 | 7.61 (6.58) | 1.96 (1.84) |
| | ε = 0.5 | 8.55 (7.37) | 2.14 (1.98) |
| | LS-SVM | 7.51 (6.63) | 1.92 (1.82) |

pyrim and triazines match or improve on the thesis values (the random splits differ). For mpg
the errors are about 15 % (MSE) higher in all four models; the cause is still open (candidates:
different splits, a different local minimum of the CV error, feature preprocessing). The larger
datasets (housing, space_ga, abalone, cpusmall) are still to be run with 30 splits; a preliminary
run with 5 splits gave values close to the thesis for housing.

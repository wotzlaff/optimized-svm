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
| housing | ε = 0.1 | 12.24 (10.8) | 2.19 (2.10) |
| | ε = 0.25 | 12.32 (10.9) | 2.28 (2.17) |
| | ε = 0.5 | 13.82 (11.9) | 2.55 (2.43) |
| | LS-SVM | 10.85 (9.54) | 2.17 (2.07) |
| space_ga | ε = 0.1 | 0.01344 (0.0133) | 0.0835 (0.0849) |
| | ε = 0.25 | 0.01304 (0.0127) | 0.0829 (0.0845) |
| | ε = 0.5 | 0.01342 (0.0128) | 0.0844 (0.0854) |
| | LS-SVM | 0.01292 (0.0127) | 0.0829 (0.0845) |
| abalone | ε = 0.1 | 4.59 (4.72) | 1.49 (1.50) |
| | ε = 0.25 | 4.55 (4.61) | 1.49 (1.50) |
| | ε = 0.5 | 4.49 (4.56) | 1.52 (1.52) |
| | LS-SVM | 4.47 (4.53) | 1.52 (1.53) |
| cpusmall | ε = 0.1 | 18.35 (19.1) | 2.71 (2.70) |
| | ε = 0.25 | 17.9 (20.0) | 2.94 (2.96) |
| | ε = 0.5 | 22.63 (32.6) | 3.54 (3.66) |
| | LS-SVM | 17.12 (18.7) | 2.68 (2.70) |

Five of the seven datasets match the thesis or are slightly better (the random splits differ).
**mpg and housing are about 13–15 % worse** in the MSE, in all four models.

### Diagnosis of the mpg/housing deviation

- *Not the optimizer.* A dense `GridSearchCV` over (C, γ) (SVR) or `KernelRidge` (LS-SVM) with the
  same standardization and CV reaches the same test error on the same 30 splits: mpg 7.84 ± 0.36
  (SVR grid) and 7.51 (LS-SVM, identical to ours); housing 10.9 (LS-SVM grid, ours 10.85).
  The gradient method therefore does not get stuck in a poor local minimum, and for the LS-SVM the
  implementation agrees with an independent solver.
- *Split noise contributes but does not explain it alone.* The per-split MSE of mpg has a
  standard deviation of about 2; the mean over 30 splits varies by roughly ±0.4. For mpg the thesis
  value lies about 2 standard errors below ours, but housing deviates in the same direction.
- *Binarising nominal features does not help.* The OpenML versions of the data sets (autoMpg,
  id 196; boston, id 531) have nominal columns (mpg: `cylinders`, `model`, `origin`; housing:
  `CHAS`, `RAD`). One-hot encoding them (mpg: 7 → 25 columns, housing: 13 → 22) makes the results
  *worse* (mpg LS-SVM 7.96, housing LS-SVM 11.54).
- *Different protocols are not comparable.* The 2018/19 runs of `hpo4svm` on OpenML use the
  10-fold CV of the OpenML tasks (3/5/10 inner folds). A 10-fold CV of a `KernelRidge` grid on the
  LIBSVM data gives MAE 1.87 (mpg) and 2.09 (housing), close to our values.

- *The old code does not reproduce the splits.* The 2018/19 prototype `hpo4svm` draws its splits
  with `np.random.seed(42)` and `np.random.permutation`; replaying that sequence on the LIBSVM
  files (datasets in the order pyrim, triazines, mpg, housing) does not reproduce Table 10.4.1
  (pyrim LS-SVM MSE 0.0100 instead of 0.0060, mpg 8.37 instead of 6.63). That script is a later
  experiment (3-fold CV, different bounds); the script that produced the table and the input files
  (`~/daten/regression/<name>/data.mat`) are not available.

The cause of the deviation remains open. Remaining candidates are the exact data set versions
and preprocessing used in the thesis (not documented there; the protocol is adopted from
Fischer et al., *Optim. Methods Softw.* 30(6), 2015) and the specific random splits.

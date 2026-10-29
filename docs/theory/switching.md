# Switching from SMO to Newton

Training first runs SMO on the dual up to a rough tolerance and then refines the result with
Newton's method (see [Training problem](training.md)). The thesis leaves open *when* to switch
(Ch. 11). A late switch wastes SMO steps that Newton would do better; an early switch hands
Newton a poor starting point, so it needs more steps, each of which factorizes the matrix of the
terms with positive curvature. This page summarizes a measurement study; the script is
`benchmarks/smo_newton_switch.py`.

## Rule

By default SMO stops when the violation measure is at most `ROUGH_SMO_TOL = 0.3` **or** after
`ROUGH_SMO_STEPS = 5` steps per loss term, whichever comes first. If Newton's method fails, an
accurate SMO solution (tolerance $10^{-6}$, no budget) is computed instead. Passing `tol` or
`max_steps` in `smo_options` replaces these defaults.

## Findings

Total time of one training (SMO plus Newton) for tolerances from $10^{-3}$ to "no SMO", on
40 problems (ε-SVR and classification; $n$ from 300 to 3000; $\lambda$ from $10^{-4}$ to $10^{-1}$;
kernel width, smoothing $\delta$ and degree varied; with and without an active 1-norm bound), cold
start and as a warm-started step of a hyperparameter optimization. Newton's method never failed.

- **A tolerance of about 0.3 is best on average.** Relative to the best tolerance of each problem
  (geometric mean), 0.3 costs 1.12 (classification) and 1.20 (regression) in cold starts, 0.1 costs
  1.16 and 1.35, and $10^{-3}$ costs 1.3 and 2.2. Tolerances of 3 or more are 2.2 to 5.8 times slower:
  Newton then starts almost from zero and needs 8 to 35 steps.
- **Small $\lambda$ prefers an earlier hand-over.** SMO needs many steps when $\lambda$ is small
  (for classification with $n = 1000$ about 80 000 steps at tolerance 0.1 for $\lambda = 10^{-4}$), while
  Newton's cost hardly depends on $\lambda$. For classification the best tolerance rises from
  about 0.01 ($\lambda = 10^{-1}$) to 1 ($\lambda \le 10^{-3}$); for regression it is about 0.03 for
  $\lambda = 10^{-1}$ and between 0.1 and 1 for smaller $\lambda$.
- **A narrow kernel prefers a later hand-over (classification).** For large scalings the kernel
  matrix is close to the identity, SMO is cheap and Newton's method needs more steps from a rough
  start (best tolerance $10^{-3}$ for four times the default scaling).
- **Little influence:** $n$ (300 to 3000), the smoothing parameter $\delta$, the degree (2, 3, 6), the
  working set rule, shrinking, one- or two-sided search and the stopping criterion. The bound-aware
  criterion with $\rho = 1$ saves 10–25 % of the SMO steps at a tolerance of 0.3. The number of BLAS
  threads changes the absolute times but not the optimum (0.3 for ε-SVR with $n = 3000$ with one and
  with twelve threads).
- **A step budget removes the worst cases.** SMO can converge extremely slowly, in particular when
  the 1-norm bound (ν-type constraint) is active (zig-zagging between the two sign groups) and for very
  small $\lambda$. Newton's method solves these problems in one or two steps. With the budget the worst
  case relative to the best rule drops from 2.1 to 1.3 (classification, cold) and from 12 to 3.3
  (regression, warm-started step).

## Effect on a full optimization

Wall time of `fit` (shared machine, minimum of two runs); the optimized CV error is the same up to
$10^{-10}$, except for the ν-case (difference $7 \cdot 10^{-5}$ along a flat path).

| problem | tol 0.1 (before) | tol 0.3, budget 5n | speed-up |
|---|---:|---:|---:|
| classification, n = 600 | 0.9 s | 0.8 s | 1.1 |
| classification, n = 1500 | 6.0 s | 5.9 s | 1.0 |
| classification, degree 6, n = 600 | 1.2 s | 1.1 s | 1.1 |
| ε-SVR, n = 450 | 2.1 s | 1.6 s | 1.3 |
| ε-SVR, n = 1200 | 60 s | 24 s | 2.5 |
| ε-SVR, ARD, n = 450 | 3.5 s | 2.7 s | 1.3 |
| ε-SVR with ν-bound, n = 450 | 154 s | 3.5 s | 44 |

## Reproducing

```bash
python benchmarks/smo_newton_switch.py --out switch.csv         # tolerance sweep
python benchmarks/smo_newton_switch.py --summary switch.csv
python benchmarks/smo_newton_switch.py --rules --out rules.csv  # step budgets
python benchmarks/smo_newton_switch.py --summary-rules rules.csv
```

"""Newton's method for the primal problem over the problem size, and Newton vs. SMO runtime
(Strasdat, dissertation, Sec. 10.3.3, Figures 10.3.3 and 10.3.4; SMO reference: Sec. 10.2.9).

Setting (Sec. 10.1, 10.2.6): checkerboard data (n uniform points on [0, 1]^2, y = +1 iff
sin(4 pi x1) sin(4 pi x2) >= 0, fixed seed), Gaussian kernel with gamma = 30, smoothed hinge loss
with delta = 1/4 (quadratic smoothing, m(t) = (t + delta)^2 / (4 delta) on |t| <= delta), unit
weights. For every problem size n and regularization lambda three solvers are timed:

* ``nw_*``: Newton's method from zero (a = 0, b = 0), no SMO. As in the thesis the kernel matrix
  is precomputed once (``t_kernel``, built blockwise into one preallocated n x n array that is
  shared by all lambda of the same n); ``nw_t`` is the time of the Newton iterations alone,
  ``nw_total = t_kernel + nw_t``. ``nw_n_curv`` is the size of the curvature set (= number of
  variables strictly between the bounds, the size of the linear systems at the solution),
  ``nw_n_sv`` the number of nonzero dual variables, ``nw_failed`` flags a ``ConvergenceError``.
* ``smo_*``: SMO alone (shrinking on) until r(alpha) <= 1e-8 (Sec. 10.2.9), no kernel matrix.
  ``smo_timeout`` flags that the per-run limit was hit (the run then is not converged).
* ``two_*``: the default two-stage ``train`` (rough SMO stage, then Newton); ``two_t_smo`` and
  ``two_t_newton`` are the times of both stages (the kernel matrix is again precomputed:
  total cost is ``t_kernel + two_t_smo + two_t_newton``).

Every time is the minimum over ``--reps`` repetitions. Memory: one n x n float64 matrix
(12.8 GB for n = 40000) plus the matrices of the curvature set; ``--max-n`` limits the sweep and
once a run of a lambda exceeds ``--limit`` seconds, larger n are skipped for that lambda.

Usage (run on a quiet machine)::

    uv run python benchmarks/newton_vs_smo.py run > benchmarks/results/newton_vs_smo.csv
    uv run python benchmarks/newton_vs_smo.py run --smoke      # n = 100, 500, 1000, two lambda
    uv run python benchmarks/newton_vs_smo.py summary benchmarks/results/newton_vs_smo.csv
    uv run --with matplotlib python benchmarks/newton_vs_smo.py plot \\
        benchmarks/results/newton_vs_smo.csv OUTDIR
"""

from __future__ import annotations

import argparse
import csv
import gc
import sys
import time
from functools import partial
from typing import Any

import numpy as np

import optimized_svm.problem as problem_module
from optimized_svm import HingeLoss, TrainingProblem, train
from optimized_svm.training import ConvergenceError, newton, residual, smo_warm_start

SIZES = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 40000]
LAMBDAS = [1e-1, 1e-2, 1e-3, 1e-4]
GAMMA = 30.0
SMOOTHING = 0.25
SMO_TOL = 1e-8
SEED = 0
BLOCK = 2000
FIELDS = [
    "n", "lam", "reps", "t_kernel",
    "nw_t", "nw_total", "nw_steps", "nw_residual", "nw_n_curv", "nw_n_sv", "nw_failed",
    "smo_t", "smo_steps", "smo_residual", "smo_timeout",
    "two_t_smo", "two_t_newton", "two_smo_steps", "two_newton_steps", "two_failed",
]  # fmt: skip


def make_data(n: int, seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    """Checkerboard dataset of Sec. 10.1 (features scaled such that the kernel is gamma = 30)."""
    x = np.random.default_rng(seed).random((n, 2))
    y = np.where(np.sin(4 * np.pi * x[:, 0]) * np.sin(4 * np.pi * x[:, 1]) >= 0, 1.0, -1.0)
    return x * np.sqrt(GAMMA), y


def kernel_matrix(x: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
    """Gaussian kernel matrix exp(-|x_i - x_j|^2) (scaled features), blockwise into ``out``."""
    n = x.shape[0]
    k = np.empty((n, n)) if out is None else out
    sq = (x * x).sum(axis=1)
    for i in range(0, n, BLOCK):
        d = sq[i : i + BLOCK, None] + sq[None, :] - 2.0 * x[i : i + BLOCK] @ x.T
        np.maximum(d, 0.0, out=d)
        np.negative(d, out=d)
        np.exp(d, out=k[i : i + BLOCK])
    return k


def timed_min(fn: Any, reps: int, limit: float) -> tuple[float, Any]:
    """Minimum wall time of ``fn()`` over ``reps`` calls (no repetition after a run > limit/4)."""
    best, out = np.inf, None
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        dt = time.perf_counter() - t0
        best = min(best, dt)
        if dt > limit / 4:
            break
    return best, out


def measure(prob: TrainingProblem, reps: int, limit: float) -> dict[str, Any]:
    row: dict[str, Any] = {}
    # Newton from zero
    start_b = prob.loss.initial_bias(prob.terms)
    failed = [False]

    def run_newton() -> Any:
        try:
            failed[0] = False
            return newton(prob, np.zeros(prob.size), start_b, 0.0)
        except ConvergenceError:
            failed[0] = True
            return None

    t, sol = timed_min(run_newton, reps, limit)
    row.update(nw_t=t, nw_failed=int(failed[0]))
    if sol is not None:
        curv = int((prob.loss.d2(prob.decision(sol.b, sol.c, sol.ka), prob.terms) > 0).sum())
        row.update(
            nw_steps=sol.newton_steps,
            nw_residual=sol.residual,
            nw_n_curv=curv,
            nw_n_sv=int(np.count_nonzero(sol.a)),
        )
    del sol
    # SMO alone
    t, res = timed_min(lambda: smo_warm_start(prob, tol=SMO_TOL, time_limit=limit), reps, limit)
    a, b, c, steps = res
    row.update(smo_t=t, smo_steps=steps, smo_timeout=int(t >= limit))
    row["smo_residual"] = residual(prob, a, b, c, prob.ka(a), False)
    # default two-stage training
    best: Any = None
    for _ in range(reps):
        s = train(prob)
        if best is None or s.time_smo + s.time_newton < best[0] + best[1]:
            best = (s.time_smo, s.time_newton, s.smo_steps, s.newton_steps)
            failed2 = int(s.info.get("newton") == "failed")
        if s.time_smo + s.time_newton > limit / 4:
            break
    row.update(
        two_t_smo=best[0],
        two_t_newton=best[1],
        two_smo_steps=best[2],
        two_newton_steps=best[3],
        two_failed=failed2,
    )
    return row


def run(smoke: bool, reps: int, max_n: int, limit: float, lams: list[float]) -> None:
    sizes = [n for n in ([100, 500, 1000] if smoke else SIZES) if n <= max_n]
    if smoke:
        lams = [1e-2, 1e-3]
    problem_module.PRECOMPUTE_LIMIT = 0  # the matrix is built here, once per n
    writer = csv.DictWriter(sys.stdout, FIELDS)
    writer.writeheader()
    skip: set[float] = set()
    for n in sizes:
        active = [lam for lam in lams if lam not in skip]
        if not active:
            break
        x, y = make_data(n)
        buf = np.empty((n, n))
        t_kernel, _ = timed_min(partial(kernel_matrix, x, buf), reps, limit)
        print(f"n={n}: kernel {t_kernel:.2f} s", file=sys.stderr, flush=True)
        for lam in active:
            prob = TrainingProblem(x, y, HingeLoss(1.0, SMOOTHING, "2"), lam)
            prob._k = buf  # shared, never copied
            t0 = time.perf_counter()
            row = measure(prob, reps, limit)
            row["nw_total"] = t_kernel + row["nw_t"]
            writer.writerow({"n": n, "lam": lam, "reps": reps, "t_kernel": t_kernel, **row})
            sys.stdout.flush()
            wall = time.perf_counter() - t0
            print(f"  lam={lam:g}: {wall:.1f} s wall", file=sys.stderr, flush=True)
            if max(row["nw_t"], row["smo_t"], row["two_t_smo"] + row["two_t_newton"]) > limit:
                print(f"  lam={lam:g}: limit exceeded, skipping larger n", file=sys.stderr)
                skip.add(lam)
            del prob
        del buf
        gc.collect()


def load(path: str) -> list[dict[str, float]]:
    out = []
    for r in csv.DictReader(open(path)):
        out.append({k: float(v) if v != "" else np.nan for k, v in r.items()})
    return out


def exponent(n: np.ndarray, t: np.ndarray) -> float:
    ok = np.isfinite(t) & (t > 0)
    return float(np.polyfit(np.log(n[ok]), np.log(t[ok]), 1)[0]) if ok.sum() > 1 else np.nan


def summary(path: str) -> None:
    rows = load(path)
    for lam in sorted({r["lam"] for r in rows}, reverse=True):
        sel = [r for r in rows if r["lam"] == lam]
        print(f"\nlambda = {lam:g}")
        print(
            f"{'n':>6} {'t_kern':>8} {'nw_t':>8} {'nw_stp':>6} {'nw_res':>9} {'curv':>6} "
            f"{'sv':>6} {'fail':>4} {'smo_t':>8} {'smo_stp':>9} {'to':>2} {'two_smo':>8} "
            f"{'two_nw':>8} {'t_nw':>4}"
        )
        for r in sel:
            print(
                f"{int(r['n']):6d} {r['t_kernel']:8.3f} {r['nw_t']:8.3f} {r['nw_steps']:6.0f} "
                f"{r['nw_residual']:9.1e} {r['nw_n_curv']:6.0f} {r['nw_n_sv']:6.0f} "
                f"{r['nw_failed']:4.0f} {r['smo_t']:8.3f} {r['smo_steps']:9.0f} "
                f"{r['smo_timeout']:2.0f} {r['two_t_smo']:8.3f} {r['two_t_newton']:8.3f} "
                f"{r['two_newton_steps']:4.0f}"
            )
        n = np.array([r["n"] for r in sel])
        big = n >= 1000
        for name, key in (("Newton", "nw_t"), ("SMO", "smo_t")):
            t = np.array([r[key] for r in sel])
            if big.sum() > 1:
                order = exponent(n[big], t[big])
                print(f"  empirical order of {name} time for n >= 1000: {order:.2f}")
        steps = np.array([r["nw_steps"] for r in sel])
        print(f"  Newton steps: min {np.nanmin(steps):.0f}, max {np.nanmax(steps):.0f}")


def plot(path: str, outdir: str) -> None:
    import os

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(outdir, exist_ok=True)
    rows = load(path)
    lams = sorted({r["lam"] for r in rows}, reverse=True)

    def series(lam: float, key: str) -> tuple[np.ndarray, np.ndarray]:
        sel = [r for r in rows if r["lam"] == lam]
        return np.array([r["n"] for r in sel]), np.array([r[key] for r in sel])

    fig, ax = plt.subplots(figsize=(5, 3.5))
    for lam in lams:
        ax.plot(*series(lam, "nw_steps"), "o-", label=rf"$\lambda = 10^{{{np.log10(lam):.0f}}}$")
    ax.set(xscale="log", xlabel="n", ylabel="Newton steps")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{outdir}/newton_steps_size.png", dpi=150)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    for lam in lams:
        ax.plot(*series(lam, "nw_total"), "o-", label=rf"$\lambda = 10^{{{np.log10(lam):.0f}}}$")
    n_all = np.array(sorted({r["n"] for r in rows}))
    ref = np.nanmin([series(lam, "nw_total")[1][-1] / series(lam, "nw_total")[0][-1] ** 2
                     for lam in lams])  # fmt: skip
    ax.plot(n_all, ref * n_all**2, "k--", lw=0.8, label=r"$\propto n^2$")
    ax.set(xscale="log", yscale="log", xlabel="n", ylabel="runtime (s)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{outdir}/newton_runtime_size.png", dpi=150)

    fig, axes = plt.subplots(1, len(lams), figsize=(3.2 * len(lams), 3.2), sharey=True)
    for ax, lam in zip(np.atleast_1d(axes), lams, strict=True):
        ax.plot(*series(lam, "nw_total"), "o-", label="Newton (incl. kernel)")
        ax.plot(*series(lam, "nw_t"), "o:", label="Newton iterations")
        ax.plot(*series(lam, "smo_t"), "s-", label=r"SMO ($r \leq 10^{-8}$)")
        n, ts = series(lam, "two_t_smo")
        ax.plot(n, ts + series(lam, "two_t_newton")[1], "^-", label="SMO + Newton")
        ax.set(
            xscale="log", yscale="log", xlabel="n", title=rf"$\lambda = 10^{{{np.log10(lam):.0f}}}$"
        )
    np.atleast_1d(axes)[0].set_ylabel("runtime (s)")
    np.atleast_1d(axes)[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{outdir}/compare_newton_smo.png", dpi=150)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run", help="run the sweep, CSV to stdout")
    p.add_argument("--smoke", action="store_true", help="n = 100, 500, 1000 and two lambda")
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--max-n", type=int, default=max(SIZES))
    p.add_argument("--limit", type=float, default=1200.0, help="per-run time guard (seconds)")
    p.add_argument("--lams", type=float, nargs="+", default=LAMBDAS)
    p = sub.add_parser("summary", help="print tables for a CSV")
    p.add_argument("file")
    p = sub.add_parser("plot", help="write the figures for a CSV into a directory")
    p.add_argument("file")
    p.add_argument("outdir")
    a = ap.parse_args()
    if a.cmd == "run":
        run(a.smoke, a.reps, a.max_n, a.limit, a.lams)
    elif a.cmd == "summary":
        summary(a.file)
    else:
        plot(a.file, a.outdir)

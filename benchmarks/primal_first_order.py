"""First-order methods for the primal training problem and an empirical error bound.

Repeats Strasdat, Sec. 10.3.1 ("Performance of First-Order Methods") and Sec. 10.3.2
("Empirical Evaluation of an Error Bound Condition") on the checkerboard problem of Sec. 10.1
(n = 1000, Gaussian kernel with gamma = 30, lambda = 1e-3, c_i = 1) with the smoothed hinge loss
of Sec. 10.2.6 (delta = 1/4, ``m_delta(x) = (x + delta)^2 / (4 delta)`` on ``|x| <= delta``,
i.e. ``HingeLoss(smoothing=0.25)``). The methods are for this study only and are not part of the
library:

* ``gd``   simple gradient descent, Alg. 8.2.1, on ``psi(z)``, ``z = (alpha, b)``,
  direction ``-grad psi = -(K r / lam, 1^T g)`` with ``r = alpha + g(z)``.
* ``pgd``  preconditioned gradient descent, Alg. 8.3.1 / (8.3.4), direction
  ``(-r / lam, -1^T g / lam^2)`` (= gradient descent on the K^{1/2} formulation (6.3.1) with the
  scaling ``W = diag(I, lam^-2)``), implemented without ``K^{1/2}``.
* ``pgd_w1`` the same with ``W = I`` (direction ``(-r / lam, -1^T g)``), a check that the
  ``lam^-2`` scaling of ``b`` matters (Sec. 10.3.2).

Both use the Armijo rule of the thesis (``eta0 = lam``, ``eta = 1/10``, ``sigma = 1/10``) and
start at ``z0 = 0``. The reference solution ``z*`` comes from ``optimized_svm.train``
(SMO + Newton with a tiny tolerance). The optimality system is (6.5.1),
``F(alpha, b) = (alpha + g(alpha, b), 1^T alpha)``; the library's ``residual`` is the
mean-squared variant ``mean((alpha + g)^2) + (1^T g)^2``. For comparison with the dual method the
``smo_solver`` is run on the same problem with a sequence of tolerances.

Usage::

    uv run python benchmarks/primal_first_order.py run --aux --methods ""   # reference, SMO, timing
    uv run python benchmarks/primal_first_order.py run --methods gd          # or pgd, pgd_w1
    uv run python benchmarks/primal_first_order.py summary
    uv run --with matplotlib python benchmarks/primal_first_order.py plot OUTDIR
    uv run python benchmarks/primal_first_order.py timing   # re-measure cost per step

``run`` writes ``results/primal_first_order_{gd,pgd,pgd_w1}.csv.gz`` (every step up to 1000,
then every 10th/100th/1000th step per decade),
``..._smo.csv``, ``..._timing.csv``, ``..._run_<method>.csv``,
``..._ref.csv`` and ``..._zstar.csv.gz``. ``--methods`` allows splitting the (long) runs over
several processes; reference, SMO and timing are written by the process given ``--aux``
(which has to run first).
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse  # noqa: E402
import csv  # noqa: E402
import gzip  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import smo_solver  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

from optimized_svm import HingeLoss, TrainingProblem, train  # noqa: E402
from optimized_svm.training import residual  # noqa: E402

RESULTS = Path(__file__).parent / "results"
PREFIX = "primal_first_order"
N, GAMMA, LAM, DELTA, SEED = 1000, 30.0, 1e-3, 0.25, 0
ETA0, ETA, SIGMA = LAM, 0.1, 0.1  # thesis, Sec. 10.3.1
METHODS = ("gd", "pgd", "pgd_w1")
TRACE_FIELDS = [
    "step", "psi_gap", "grad", "grad_tilde", "F", "dist", "dist_alpha", "dist_b", "tau", "time",
    "matvecs",
]  # fmt: skip
SMO_TOLS = [3.0, 1.0, 0.3, 0.1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 1e-5, 1e-6, 1e-8]


def path(name: str) -> Path:
    return RESULTS / f"{PREFIX}_{name}"


class Setup:
    """Data, kernel matrix and reference solution."""

    def __init__(self) -> None:
        rng = np.random.default_rng(SEED)
        x = rng.uniform(size=(N, 2))
        self.y = np.where(np.sin(4 * np.pi * x[:, 0]) * np.sin(4 * np.pi * x[:, 1]) >= 0, 1.0, -1.0)
        self.x = x
        # the library works with scaled features and the RBF kernel with gamma = 1
        self.prob = TrainingProblem(
            x * np.sqrt(GAMMA), self.y, HingeLoss(smoothing=DELTA), LAM, weights=np.ones(N)
        )
        self.k = np.ascontiguousarray(self.prob.kernel())
        self.loss, self.terms = self.prob.loss, self.prob.terms

    def psi(self, a: np.ndarray, b: float, q: np.ndarray | None = None) -> float:
        """``psi = a^T K a / (2 lam) + sum_i m_delta(1 - y_i t_i)`` with ``t = K a / lam + b``."""
        q = self.k @ a if q is None else q
        return float(0.5 / LAM * (a @ q) + self.loss.value(q / LAM + b, self.terms).sum())

    def g(self, a: np.ndarray, b: float) -> np.ndarray:
        return self.loss.d1(self.k @ a / LAM + b, self.terms)

    def grad(self, a: np.ndarray, b: float) -> np.ndarray:
        """Gradient (6.1.4): ``(K (alpha + g) / lam, 1^T g)``."""
        g = self.g(a, b)
        return np.append(self.k @ (a + g) / LAM, g.sum())

    def f_norm(self, a: np.ndarray, b: float) -> float:
        """``|F(alpha, b)|`` for the system (6.5.1)."""
        r = a + self.g(a, b)
        return float(np.sqrt(r @ r + a.sum() ** 2))

    def reference(self) -> tuple[np.ndarray, float, dict[str, float]]:
        sol = train(
            self.prob,
            smo_options={"tol": 1e-6},
            newton_options={"tol": 1e-28, "max_steps": 50},
        )
        a, b = sol.a, float(sol.b)
        info = {
            "psi_star": self.psi(a, b),
            "F_star": self.f_norm(a, b),
            "grad_star": float(np.linalg.norm(self.grad(a, b))),
            "residual_library": float(residual(self.prob, a, b, 0.0, self.prob.ka(a), False)),
            "newton_steps": float(sol.newton_steps),
            "smo_steps": float(sol.smo_steps),
            "n_sv": float(np.sum(np.abs(a) > 1e-10)),
            "n_bound": float(np.sum(np.abs(np.abs(a) - 1.0) < 1e-10)),
            "b_star": b,
            "alpha_star_norm": float(np.linalg.norm(a)),
        }
        return a, b, info


def first_order(
    s: Setup,
    method: str,
    ref: tuple[np.ndarray, float],
    max_steps: int,
    patience: int = 50,
    refresh: int = 100,
    max_backtracks: int = 40,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Runs ``gd``, ``pgd`` or ``pgd_w1`` from ``z0 = 0`` and records every step."""
    k, lam = s.k, LAM
    a_star, b_star = ref
    psi_star = s.psi(a_star, b_star)
    omega = {"gd": 0.0, "pgd": lam**-2, "pgd_w1": 1.0}[method]
    a, b = np.zeros(N), 0.0
    q = np.zeros(N)
    psi = s.psi(a, b, q)
    tr = {f: np.full(max_steps + 1, np.nan) for f in TRACE_FIELDS}
    matvecs, stalls, stop, last = 0, 0, "", max_steps
    t0 = time.perf_counter()

    def record(i: int, grad2: float, gt2: float, rr: float, tau: float) -> None:
        da, db = a - a_star, b - b_star
        tr["step"][i] = i
        tr["psi_gap"][i] = psi - psi_star
        tr["grad"][i] = np.sqrt(grad2)
        tr["grad_tilde"][i] = np.sqrt(gt2)
        tr["F"][i] = np.sqrt(rr + a.sum() ** 2)
        tr["dist_alpha"][i] = np.sqrt(da @ da)
        tr["dist_b"][i] = abs(db)
        tr["dist"][i] = np.sqrt(da @ da + db * db)
        tr["tau"][i] = tau
        tr["time"][i] = time.perf_counter() - t0
        tr["matvecs"][i] = matvecs

    for i in range(max_steps + 1):
        if i % refresh == 0:  # remove accumulated rounding errors of the recursions
            q = k @ a
            psi = s.psi(a, b, q)
            matvecs += 1
        g = s.loss.d1(q / lam + b, s.terms)
        r = a + g
        sb = float(g.sum())
        kr = k @ r  # K r: the only matrix-vector product of a PGD step
        matvecs += 1
        rkr = float(r @ kr)
        grad2 = float(kr @ kr) / lam**2 + sb**2
        gt2 = rkr / lam**2 + sb**2
        if i == max_steps or grad2 == 0.0 or stop:
            record(i, grad2, gt2, float(r @ r), np.nan)
            stop = stop or ("max_steps" if i == max_steps else "zero_gradient")
            last = i
            break
        if method == "gd":
            kkr = k @ kr  # the extra matrix-vector product of simple gradient descent
            matvecs += 1
            slope, da_, db_, kda = grad2, -kr / lam, -sb, -kkr / lam
        else:
            slope = rkr / lam**2 + omega * sb**2
            da_, db_, kda = -r / lam, -omega * sb, -kr / lam
        tau, accepted = ETA0, False
        for _ in range(max_backtracks):
            a_new, b_new, q_new = a + tau * da_, b + tau * db_, q + tau * kda
            psi_new = s.psi(a_new, b_new, q_new)
            if psi_new <= psi - SIGMA * tau * slope:
                accepted = True
                break
            tau *= ETA
        record(i, grad2, gt2, float(r @ r), tau)
        if not accepted:
            stop, last = "line_search_failed", i
            break
        stalls = stalls + 1 if psi_new >= psi else 0
        a, b, q, psi = a_new, b_new, q_new, psi_new
        if stalls >= patience:
            stop = "no_numerical_descent"  # recorded at the next iteration
    tr = {f: v[: last + 1] for f, v in tr.items()}
    info = {
        "method": method,
        "steps": last,
        "stop_reason": stop,
        "time": float(tr["time"][-1]),
        "matvecs": int(tr["matvecs"][-1]),
        "psi_gap_final": float(tr["psi_gap"][-1]),
        "F_final": float(tr["F"][-1]),
        "dist_final": float(tr["dist"][-1]),
    }
    return tr, info


def thin_index(length: int) -> np.ndarray:
    """All steps up to 1000, then every 10th, 100th, 1000th step per decade, and the last one."""
    parts = [np.arange(min(1001, length))]
    for start, stride in ((1000, 10), (10_000, 100), (100_000, 1000)):
        parts.append(np.arange(start, min(length, 10 * start), stride))
    parts.append(np.arange(1_000_000, length, 1000))
    return np.union1d(np.unique(np.concatenate(parts)), [length - 1])


def write_trace(name: str, tr: dict[str, np.ndarray]) -> None:
    with gzip.open(path(f"{name}.csv.gz"), "wt", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(TRACE_FIELDS)
        for i in thin_index(len(tr["step"])):
            w.writerow([f"{tr[f][i]:.10g}" for f in TRACE_FIELDS])


def read_trace(name: str) -> dict[str, np.ndarray]:
    with gzip.open(path(f"{name}.csv.gz"), "rt") as fh:
        rows = list(csv.reader(fh))
    data = np.array(rows[1:], dtype=float)
    return {f: data[:, j] for j, f in enumerate(rows[0])}


def read_csv(name: str) -> list[dict[str, str]]:
    with open(path(name)) as fh:
        return list(csv.DictReader(fh))


def write_csv(name: str, rows: list[dict[str, object]]) -> None:
    with open(path(name), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def validate(s: Setup) -> dict[str, float]:
    """Gradient vs central finite differences; consistency of the PGD descent estimate (8.3.6)."""
    rng = np.random.default_rng(1)
    a, b = 0.01 * rng.normal(size=N), 0.1
    grad = s.grad(a, b)
    h, errs = 1e-5, []
    for _ in range(5):
        d = rng.normal(size=N + 1)
        d /= np.linalg.norm(d)
        fd = (s.psi(a + h * d[:-1], b + h * d[-1]) - s.psi(a - h * d[:-1], b - h * d[-1])) / (2 * h)
        errs.append(abs(fd - grad @ d) / abs(grad @ d))
    g = s.g(a, b)
    r = a + g
    slope = r @ s.k @ r / LAM**2 + g.sum() ** 2 / LAM**2  # descent estimate of (8.3.6)
    d_pgd = np.append(-r / LAM, -g.sum() / LAM**2)  # direction (8.3.5)
    u = d_pgd / np.linalg.norm(d_pgd)  # finite differences need a unit direction (kinks of m_delta)
    fd_dir = (s.psi(a + h * u[:-1], b + h * u[-1]) - s.psi(a - h * u[:-1], b - h * u[-1])) / (2 * h)
    return {
        "fd_rel_err_max": float(max(errs)),
        "pgd_grad_dot_dir": float(grad @ d_pgd),
        "pgd_minus_descent_estimate": float(-slope),
        "pgd_fd_rel_err": float(abs(fd_dir - grad @ u) / abs(grad @ u)),
        "kernel_check_abs": float(abs(s.k[3, 7] - np.exp(-GAMMA * np.sum((s.x[3] - s.x[7]) ** 2)))),
    }


def smo_study(s: Setup, a_star: np.ndarray, b_star: float) -> list[dict[str, object]]:
    psi_star = s.psi(a_star, b_star)
    rows = []
    for tol in SMO_TOLS:
        best = np.inf
        for _ in range(3):
            t0 = time.perf_counter()
            res = smo_solver.solve_classification(
                s.prob.x, s.y, LAM, gamma=1.0, weights=np.ones(N), smoothing=DELTA, tol=tol
            )
            best = min(best, time.perf_counter() - t0)
        a, b = np.asarray(res.alpha), float(res.b)
        da = a - a_star
        rows.append(
            {
                "tol": tol,
                "steps": int(res.steps),
                "time": best,
                "psi_gap": s.psi(a, b) - psi_star,
                "F": s.f_norm(a, b),
                "grad": float(np.linalg.norm(s.grad(a, b))),
                "dist": float(np.sqrt(da @ da + (b - b_star) ** 2)),
                "dist_alpha": float(np.linalg.norm(da)),
                "dist_b": abs(b - b_star),
            }
        )
        print("smo", rows[-1], flush=True)
    return rows


def timing_study(
    s: Setup, a_star: np.ndarray, b_star: float, steps: int = 3000, reps: int = 3
) -> list[dict[str, object]]:
    """Cost per step of the first ``steps`` steps (best of ``reps``), methods run one by one."""
    rows = []
    for m in METHODS:
        runs = [first_order(s, m, (a_star, b_star), steps)[1] for _ in range(reps)]
        best = min(runs, key=lambda i: i["time"])
        rows.append(
            {
                "method": m,
                "steps": best["steps"],
                "s_per_step": best["time"] / best["steps"],
                "matvecs_per_step": best["matvecs"] / best["steps"],
            }
        )
        print("timing", rows[-1], flush=True)
    return rows


def cmd_timing(_: argparse.Namespace) -> None:
    s = Setup()
    z = read_zstar()
    write_csv("timing.csv", timing_study(s, z[:-1], float(z[-1])))


def read_zstar() -> np.ndarray:
    with gzip.open(path("zstar.csv.gz"), "rt") as fh:
        return np.array(fh.read().split(), dtype=float)


def cmd_run(args: argparse.Namespace) -> None:
    RESULTS.mkdir(exist_ok=True)
    s = Setup()
    if args.aux:
        t0 = time.perf_counter()
        a_star, b_star, info = s.reference()
        info["time_reference"] = time.perf_counter() - t0
        info.update(validate(s))
        with gzip.open(path("zstar.csv.gz"), "wt") as fh:
            fh.write("\n".join(repr(float(v)) for v in np.append(a_star, b_star)))
        write_csv("ref.csv", [{"key": k, "value": f"{v:.12g}"} for k, v in info.items()])
        print("reference:", info, flush=True)
    z = read_zstar()
    a_star, b_star = z[:-1], float(z[-1])
    if args.aux:
        write_csv("smo.csv", smo_study(s, a_star, b_star))
        write_csv("timing.csv", timing_study(s, a_star, b_star))
    for m in [m for m in args.methods.split(",") if m]:
        tr, info = first_order(s, m, (a_star, b_star), args.steps)
        write_trace(m, tr)
        write_csv(f"run_{m}.csv", [info])
        print(m, info, flush=True)


def slope_fit(x: np.ndarray, y: np.ndarray, lo: float, hi: float) -> float:
    """Log-log slope of ``y`` against ``x`` over the points with ``lo <= x <= hi``."""
    m = (x >= lo) & (x <= hi) & (y > 0)
    return float(np.polyfit(np.log(x[m]), np.log(y[m]), 1)[0]) if m.sum() > 5 else float("nan")


def first_below(tr: dict[str, np.ndarray], key: str, thr: float) -> tuple[float, float]:
    idx = np.flatnonzero(tr[key] <= thr)
    return (tr["step"][idx[0]], tr["time"][idx[0]]) if len(idx) else (np.nan, np.nan)


def jitter(v: np.ndarray) -> float:
    """Median absolute change of ``log v`` between consecutive records."""
    return float(np.median(np.abs(np.diff(np.log(v)))))


def cmd_summary(_: argparse.Namespace) -> None:
    print("reference solution:")
    for r in read_csv("ref.csv"):
        print(f"  {r['key']:28s} {float(r['value']):.6g}")
    print("\nruns:")
    for m in METHODS:
        if path(f"run_{m}.csv").exists():
            print(" ", read_csv(f"run_{m}.csv")[0])
    print("\ntiming (first 3000 steps, run sequentially):")
    for r in read_csv("timing.csv"):
        print(" ", r)
    traces = {m: read_trace(m) for m in METHODS if path(f"{m}.csv.gz").exists()}
    print("\nmeasures at selected steps:")
    for m, tr in traces.items():
        print(f" {m}")
        for k in (1, 10, 100, 1000, 10_000, 100_000, 280_000, 1_000_000):
            j = np.flatnonzero(tr["step"] <= k)[-1]
            if tr["step"][j] < k - 10:
                continue
            print(
                f"  k={int(tr['step'][j]):8d} gap={tr['psi_gap'][j]:.2e} grad={tr['grad'][j]:.2e} "
                f"F={tr['F'][j]:.2e} dist={tr['dist'][j]:.2e} da={tr['dist_alpha'][j]:.2e} "
                f"db={tr['dist_b'][j]:.2e} tau={tr['tau'][j]:.2e} t={tr['time'][j]:.1f}s"
            )
    print("\nfirst step / wall time with |z - z*| below a threshold (SMO: first tolerance):")
    smo = read_csv("smo.csv") if path("smo.csv").exists() else []
    for thr in (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6):
        parts = []
        for m, tr in traces.items():
            st, tm = first_below(tr, "dist", thr)
            parts.append(f"{m}: {st:.0f} steps / {tm:.1f}s")
        hit = [r for r in smo if float(r["dist"]) <= thr]
        if hit:
            parts.append(
                f"smo(tol={hit[0]['tol']}): {hit[0]['steps']} steps / {float(hit[0]['time']):.4f}s"
            )
        print(f"  {thr:.0e}  " + "; ".join(parts))
    print("\nSMO (dual, cold start, best of 3):")
    for r in smo:
        print(
            f"  tol={float(r['tol']):7.0e} steps={int(r['steps']):7d} time={float(r['time']):.4f}s "
            f"gap={float(r['psi_gap']):.2e} F={float(r['F']):.2e} dist={float(r['dist']):.2e}"
        )
    print("\nerror bound (records with |F| < 1e-2 and psi gap > 1e-11):")
    for m, tr in traces.items():
        sel = (tr["F"] < 1e-2) & (tr["psi_gap"] > 1e-11) & (tr["dist"] > 0)
        if sel.sum() < 6:
            print(f"  {m}: no asymptotic region reached")
            continue
        ratio = tr["F"][sel] / tr["dist"][sel]
        print(
            f"  {m}: F/dist in [{ratio.min():.3g}, {ratio.max():.3g}]; "
            f"grad/dist in [{(tr['grad'][sel] / tr['dist'][sel]).min():.3g}, "
            f"{(tr['grad'][sel] / tr['dist'][sel]).max():.3g}]; "
            f"grad_tilde/dist in [{(tr['grad_tilde'][sel] / tr['dist'][sel]).min():.3g}, "
            f"{(tr['grad_tilde'][sel] / tr['dist'][sel]).max():.3g}]"
        )
        f = tr["F"]
        print(
            f"     log-log slope vs F: gap {slope_fit(f, tr['psi_gap'], 1e-6, 1e-2):.2f} (2), "
            f"dist {slope_fit(f, tr['dist'], 1e-6, 1e-2):.2f} (1), "
            f"grad {slope_fit(f, tr['grad'], 1e-6, 1e-2):.2f}, "
            f"dist_alpha {slope_fit(f, tr['dist_alpha'], 1e-6, 1e-2):.2f}, "
            f"dist_b {slope_fit(f, tr['dist_b'], 1e-6, 1e-2):.2f}"
        )
        db = tr["dist_b"][sel] / tr["dist_alpha"][sel]
        print(
            f"     |b-b*|/|alpha-alpha*| in [{db.min():.3g}, {db.max():.3g}] "
            f"(median {np.median(db):.3g}); jitter of log(grad) {jitter(tr['grad'][sel]):.3g} "
            f"vs log(F) {jitter(tr['F'][sel]):.3g} vs log(dist) {jitter(tr['dist'][sel]):.3g}"
        )


def cmd_plot(args: argparse.Namespace) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    traces = {m: read_trace(m) for m in METHODS if path(f"{m}.csv.gz").exists()}
    smo = read_csv("smo.csv") if path("smo.csv").exists() else []
    names = {"gd": "gradient descent", "pgd": "preconditioned GD", "pgd_w1": "PGD with W = I"}
    series = [
        ("psi_gap", r"$\psi(z^k)-\psi(z^\star)$"),
        ("grad", r"$\|\nabla\psi(z^k)\|$"),
        ("F", r"$\|F(z^k)\|$"),
        ("dist", r"$\|z^k-z^\star\|$"),
    ]

    def pos(v: np.ndarray) -> np.ndarray:
        return np.where(v > 0, v, np.nan)

    for m, tr in traces.items():  # Fig. 10.3.1
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        for key, lab in series:
            ax.loglog(tr["step"] + 1, pos(tr[key]), label=lab, lw=1)
        ax.set(xlabel="step $k+1$", title=f"{names[m]} (cf. Fig. 10.3.1)")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / f"fig_10_3_1_{m}.png", dpi=150)
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))  # comparison with SMO
    for m, tr in traces.items():
        axes[0].loglog(tr["step"] + 1, pos(tr["dist"]), label=names[m], lw=1)
        axes[1].loglog(tr["time"] + 1e-3, pos(tr["dist"]), label=names[m], lw=1)
    if smo:
        axes[0].loglog(
            [int(r["steps"]) for r in smo],
            [float(r["dist"]) for r in smo],
            "ks-",
            ms=4,
            label="SMO (dual)",
        )
        axes[1].loglog(
            [float(r["time"]) for r in smo],
            [float(r["dist"]) for r in smo],
            "ks-",
            ms=4,
            label="SMO (dual)",
        )
    axes[0].set(xlabel="steps", ylabel=r"$\|z-z^\star\|$")
    axes[1].set(xlabel="wall time [s]", ylabel=r"$\|z-z^\star\|$")
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig_compare_methods_smo.png", dpi=150)
    plt.close(fig)
    xs = np.logspace(-9, 1, 3)
    for m, tr in traces.items():  # Fig. 10.3.2
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
        for key, lab in [
            ("psi_gap", r"$\psi-\psi^\star$"),
            ("grad", r"$\|\nabla\psi\|$"),
            ("grad_tilde", r"$\|\nabla\tilde\psi\|$"),
            ("dist", r"$\|z-z^\star\|$"),
        ]:
            axes[0].loglog(tr["F"], pos(tr[key]), ".", ms=2, label=lab)
        axes[0].set(xlabel=r"$\|F(z^k)\|$", title=f"(a) {names[m]}")
        for key, lab in [
            ("dist_alpha", r"$\|\alpha-\alpha^\star\|$"),
            ("dist_b", r"$|b-b^\star|$"),
        ]:
            axes[1].loglog(tr["F"], pos(tr[key]), ".", ms=2, label=lab)
        axes[1].set(xlabel=r"$\|F(z^k)\|$", title=f"(b) {names[m]}")
        for ax in axes:
            ax.loglog(xs, xs, "k--", lw=0.8, label="slope 1")
            if ax is axes[0]:
                ax.loglog(xs, xs**2, "k:", lw=0.8, label="slope 2")
            ax.legend(markerscale=4, fontsize=8)
            ax.grid(alpha=0.3)
            ax.set_xlim(1e-8, 1e1)
            ax.set_ylim(1e-14, 1e2)
        fig.tight_layout()
        fig.savefig(out / f"fig_10_3_2_{m}.png", dpi=150)
        plt.close(fig)
    print(f"plots written to {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run the methods and write the result files")
    r.add_argument("--steps", type=int, default=1_000_000)
    r.add_argument("--methods", default=",".join(METHODS))
    r.add_argument("--aux", action="store_true", help="compute reference, SMO study and timing")
    r.set_defaults(fn=cmd_run)
    sub.add_parser("summary", help="print the summary numbers").set_defaults(fn=cmd_summary)
    sub.add_parser("timing", help="re-measure the cost per step").set_defaults(fn=cmd_timing)
    pl = sub.add_parser("plot", help="write the figures to OUTDIR")
    pl.add_argument("outdir")
    pl.set_defaults(fn=cmd_plot)
    args = p.parse_args()
    with threadpool_limits(limits=min(4, os.cpu_count() or 1)):
        args.fn(args)


if __name__ == "__main__":
    main()

"""Study of the switch from SMO to Newton's method (Strasdat, Ch. 11: "when to switch").

``train`` runs SMO up to a rough tolerance ``tol`` and then refines with Newton's method. A
larger ``tol`` hands over earlier: less SMO work, but Newton needs more steps (each one a
Cholesky factorisation of the curvature set) and may fail. This script measures SMO and Newton
time as a function of ``tol`` for varying problem size, regularization, kernel width, loss,
smoothing and for cold starts as well as warm-started steps of a hyperparameter optimisation.

Usage::

    python benchmarks/smo_newton_switch.py --out switch.csv
    python benchmarks/smo_newton_switch.py --summary switch.csv
"""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
import time
from typing import Any

import numpy as np
from sklearn.datasets import make_classification, make_friedman1
from sklearn.preprocessing import StandardScaler

from optimized_svm import EpsilonInsensitiveLoss, HingeLoss, TrainingProblem, train
from optimized_svm.estimators import auto_scaling
from optimized_svm.losses import Loss

TOLS = [1e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0, 3.0, 1e9]  # 1e9: no SMO, Newton from zero
FIELDS = [
    "task", "scenario", "n", "lam", "s_factor", "smoothing", "degree", "max_asum", "tol",
    "t_smo", "t_newton", "t_total", "smo_steps", "newton_steps", "newton_failed", "n_curv",
    "residual",
]  # fmt: skip


def make_data(task: str, n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    if task == "svc":
        x, y = make_classification(n, 8, n_informative=5, flip_y=0.1, random_state=seed)
        return StandardScaler().fit_transform(x), np.where(y > 0, 1.0, -1.0)
    x, t = make_friedman1(n, noise=0.5, random_state=seed)
    return StandardScaler().fit_transform(x), (t - t.mean()) / t.std()


def make_loss(task: str, smoothing: float, degree: str) -> Loss:
    if task == "svc":
        return HingeLoss(1.0, smoothing, degree)  # type: ignore[arg-type]
    return EpsilonInsensitiveLoss(0.1, smoothing, degree)  # type: ignore[arg-type]


def one_run(
    prob: TrainingProblem, tol: float, alpha0: np.ndarray | None, reps: int, **smo_options: Any
) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for _ in range(reps):
        t0 = time.perf_counter()
        sol = train(prob, {"tol": tol, **smo_options}, None, alpha0)
        total = time.perf_counter() - t0
        row = {
            "t_smo": sol.time_smo, "t_newton": sol.time_newton, "t_total": total,
            "smo_steps": sol.smo_steps, "newton_steps": sol.newton_steps,
            "newton_failed": int(sol.info.get("newton") == "failed"),
            "residual": sol.residual,
        }  # fmt: skip
        if best is None or row["t_total"] < best["t_total"]:
            best = row
    assert best is not None
    return best


def settings(quick: bool) -> list[dict[str, Any]]:
    """One-factor-at-a-time variations around a base case plus an n x lam grid."""
    base = dict(
        task="svc", n=1000, lam=1e-2, s_factor=1.0, smoothing=0.5, degree="2", max_asum=None
    )
    out: list[dict[str, Any]] = []

    def add(**kw: Any) -> None:
        d = {**base, **kw}
        if d not in out:
            out.append(d)

    for task in ("svc", "svr"):
        add(task=task)
        for lam in (1e-1, 1e-3, 1e-4):
            add(task=task, lam=lam)
        for sf in (0.25, 0.5, 2.0, 4.0):
            add(task=task, s_factor=sf)
        for sm in (0.05, 0.2, 1.0):
            add(task=task, smoothing=sm)
        for deg in ("3", "6"):
            add(task=task, degree=deg)
        for n in (300, 3000):
            add(task=task, n=n)
        for n, lam in itertools.product((300, 1000, 3000), (1e-1, 1e-3)):
            add(task=task, n=n, lam=lam)
        add(task=task, max_asum=0.3 * 1000)  # active 1-norm bound (nu-type)
    return out[:12] if quick else out


def run(out_path: str, quick: bool, reps: int) -> None:
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, FIELDS)
        writer.writeheader()
        for st in settings(quick):
            x, y = make_data(st["task"], st["n"])
            s = auto_scaling(x) * st["s_factor"]
            loss = make_loss(st["task"], st["smoothing"], st["degree"])
            max_asum = st["max_asum"] if st["task"] == "svr" else None
            prob = TrainingProblem(x * s, y, loss, st["lam"], max_asum=max_asum)
            for scenario in ("cold", "warm_step"):
                alpha0 = None
                prob_run = prob
                if scenario == "warm_step":
                    # one step of a hyperparameter optimisation: neighbouring lam and scaling,
                    # warm-started with the accurate solution of the previous point
                    first = train(prob, {"tol": 1e-3})
                    alpha0 = first.a
                    prob_run = TrainingProblem(
                        x * s * 1.1, y, loss, st["lam"] * 1.3, max_asum=max_asum
                    )
                ref = train(prob_run, {"tol": 1e-3}, None, alpha0)
                dec = prob_run.decision(ref.b, ref.c, ref.ka)
                n_curv = int((prob_run.loss.d2(dec, prob_run.terms) > 0).sum())
                for tol in TOLS:
                    row = one_run(prob_run, tol, alpha0, reps)
                    rec = {
                        **{
                            k: st[k]
                            for k in (
                                "task",
                                "n",
                                "lam",
                                "s_factor",
                                "smoothing",
                                "degree",
                                "max_asum",
                            )
                        }
                    }
                    rec.update(scenario=scenario, tol=tol, n_curv=n_curv, **row)
                    writer.writerow(rec)
                    fh.flush()
            print("done", st, file=sys.stderr, flush=True)


#: candidate switching rules: (name, tol, steps per sample allowed to SMO or None)
RULES: list[tuple[str, float, float | None]] = [
    ("tol 0.1", 0.1, None),
    ("tol 0.3", 0.3, None),
    ("tol 1", 1.0, None),
    ("tol 0.1, <= 5n steps", 0.1, 5),
    ("tol 0.1, <= 20n steps", 0.1, 20),
    ("tol 0.3, <= 5n steps", 0.3, 5),
    ("tol 0.3, <= 20n steps", 0.3, 20),
    ("tol 0.03, <= 5n steps", 0.03, 5),
    ("tol 0.01, <= 5n steps", 0.01, 5),
    ("tol 0.01, <= 2n steps", 0.01, 2),
]


def run_rules(out_path: str, reps: int) -> None:
    """Fixed tolerances with and without a step budget for SMO (n_terms * budget steps)."""
    fields = ["task", "scenario", "n", "lam", "s_factor", "smoothing", "degree", "max_asum"]
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, [*fields, "rule", "t_total", "smo_steps", "newton_steps"])
        writer.writeheader()
        for st in settings(False):
            x, y = make_data(st["task"], st["n"])
            s = auto_scaling(x) * st["s_factor"]
            loss = make_loss(st["task"], st["smoothing"], st["degree"])
            max_asum = st["max_asum"] if st["task"] == "svr" else None
            prob = TrainingProblem(x * s, y, loss, st["lam"], max_asum=max_asum)
            first = train(prob, {"tol": 1e-3})
            neighbour = TrainingProblem(x * s * 1.1, y, loss, st["lam"] * 1.3, max_asum=max_asum)
            for scenario, pr, a0 in (("cold", prob, None), ("warm_step", neighbour, first.a)):
                for name, tol, budget in RULES:
                    opts = {} if budget is None else {"max_steps": int(budget * pr.size)}
                    row = one_run(pr, tol, a0, reps, **opts)
                    rec = {k: st[k] for k in fields if k != "scenario"}
                    rec.update(
                        scenario=scenario,
                        rule=name,
                        **{k: row[k] for k in ("t_total", "smo_steps", "newton_steps")},
                    )
                    writer.writerow(rec)
                    fh.flush()
            print("done", st, file=sys.stderr, flush=True)


def summary_rules(path: str) -> None:
    rows = list(csv.DictReader(open(path)))
    keys = ["task", "scenario", "n", "lam", "s_factor", "smoothing", "degree", "max_asum"]
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in keys), []).append(r)
    names = [n for n, _, _ in RULES]
    print("relative total time vs. the best rule of each setting (geometric mean / worst case)\n")
    for scenario in ("cold", "warm_step"):
        for task in ("svc", "svr"):
            sel = [g for k, g in groups.items() if k[0] == task and k[1] == scenario]
            print(f"{scenario} {task}")
            for name in names:
                rel = []
                for g in sel:
                    best = min(float(r["t_total"]) for r in g)
                    rel.append(float(next(r for r in g if r["rule"] == name)["t_total"]) / best)
                print(
                    f"  {name:24} mean {np.exp(np.mean(np.log(rel))):5.2f}   worst {max(rel):6.2f}"
                )


def summary(path: str) -> None:
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        for k in (
            "tol",
            "t_smo",
            "t_newton",
            "t_total",
            "smo_steps",
            "newton_steps",
            "newton_failed",
        ):
            r[k] = float(r[k])
    keys = ["task", "scenario", "n", "lam", "s_factor", "smoothing", "degree", "max_asum"]
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in keys), []).append(r)
    tols = sorted({r["tol"] for r in rows})
    print("relative total time (1.00 = best tol of the setting); geometric mean over settings\n")
    for scenario in ("cold", "warm_step"):
        for task in ("svc", "svr"):
            sel = [g for k, g in groups.items() if k[0] == task and k[1] == scenario]
            rel = {t: [] for t in tols}
            for g in sel:
                best = min(r["t_total"] for r in g)
                for r in g:
                    rel[r["tol"]].append(r["t_total"] / best)
            line = "  ".join(f"{t:>7.0e}:{np.exp(np.mean(np.log(v))):5.2f}" for t, v in rel.items())
            fails = {
                t: sum(r["newton_failed"] for g in sel for r in g if r["tol"] == t) for t in tols
            }
            print(f"{scenario:9} {task}  {line}   (failures: {sum(fails.values())})")
    print("\nbest tol per setting (cold | warm_step); times in ms at tol=0.1 and at the best tol\n")
    for k, g in groups.items():
        if k[1] != "cold":
            continue
        w = groups.get((k[0], "warm_step") + k[2:], [])

        def best(gr: list[dict]) -> str:
            if not gr:
                return "-"
            b = min(gr, key=lambda r: r["t_total"])
            at = next(r for r in gr if r["tol"] == 0.1)
            ms, ms_at = 1e3 * b["t_total"], 1e3 * at["t_total"]
            return f"{b['tol']:.0e} ({ms:.0f} ms, tol 0.1: {ms_at:.0f} ms)"

        label = f"{k[0]} n={k[2]} lam={k[3]} s={k[4]} d={k[5]} deg={k[6]} nu={k[7]}"
        print(f"{label}:  {best(g)}  |  {best(w)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="switch.csv")
    ap.add_argument("--summary", help="summarise an existing CSV instead of running")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--rules", action="store_true", help="compare switching rules (budgets)")
    ap.add_argument("--summary-rules", help="summarise a CSV written by --rules")
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    if a.summary:
        summary(a.summary)
    elif a.summary_rules:
        summary_rules(a.summary_rules)
    elif a.rules:
        run_rules(a.out, a.reps)
    else:
        run(a.out, a.quick, a.reps)

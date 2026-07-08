"""Experiment harness: matheuristic, ablation, and deliverables.

Per (instance, seed), see :mod:`ihtp.pipeline`:
    construct  ->  PAS MILP (admission core)  ->  ALNS refine  ->  exact OT + NRA polish
Official C++ validator has the last word. Nothing counts as feasible until it
reports Total violations = 0.

Outputs land in ``results/``:
    * ``<instance>.json``            best solution found (official schema)
    * ``<instance>.validator.txt``   official validator log for that solution
    * ``results.csv``                required CSV: one ``i01,<objective>`` (or ``,infeasible``)
    * ``summary.csv``                instance, feasibility, objective, best-known, gap, runtime
    * ``seed_runs.csv``              every (instance, seed): objective and runtime (distribution)
    * ``bounds.csv``                 certified lower bounds (with ``--bounds``)

Run: ``python -m ihtp.experiments --instances all --seeds 5 --jobs 0``
     ``python -m ihtp.experiments --instances all --bounds``   (certified lower bounds)
     ``python -m ihtp.experiments --ablation i04,i13,i16,i27`` (cumulative ablation)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

from . import config
from .io_instance import Instance, load_instance
from .objective import evaluate
from .pipeline import matheuristic
from .polish import polish
from .validate import run_validator
from .writer import write_solution, write_solution_dict


def budgets(inst: Instance) -> dict:
    """Budgets in iteration/round counts and solver WorkLimits, never wall-clock seconds.

    Fixed seed -> same solution on any machine with the same solver versions; only
    runtime tracks hardware. Round and iteration count is what admits optional patients,
    not longer single solves, so those counts are what we pin down."""
    return dict(
        pas_work=50.0,         # Gurobi work units, deterministic; small instances hit opt
        alns_iters=8000,       # ALNS iters; profiling says past 8k buys ~nothing on small instances
        # NRA WorkLimit. component-gap analysis: residual gap vs best-known is ~90% nurse
        # cost (S2/S3/S4). controlled run showed the final NRA MIP was starved at the old flat
        # 120. re-solve at 600 cut nurse cost up to 16% and aggregate gap 9.2% -> 7.9%. this is a
        # ceiling, not a target: easy NRA MIPs prove opt and stop early anyway. floor 600
        # (validated, free on easy ones), up to 800 for the biggest MIPs (rooms x days).
        nra_work=float(min(800, max(600, inst.R * inst.days))),
        ot_work=25.0,          # Gurobi work units per per-day OT MIP
        descent_passes=4,      # full descent sweeps
        lns_rounds=4,          # CP-SAT LNS rounds, the workhorse; past 4 diminishes (profiled)
        opt_window=8 if inst.P > 300 else None,
    )


def run_instance(inst: Instance, seed: int):
    b = budgets(inst)
    return matheuristic(inst, seed=seed, pas_work=b["pas_work"], alns_iters=b["alns_iters"],
                        nra_work=b["nra_work"], ot_work=b["ot_work"],
                        descent_passes=b["descent_passes"], lns_rounds=b["lns_rounds"],
                        opt_window=b["opt_window"])


def ablation(inst: Instance, seed: int) -> dict:
    """Cumulative ablation; each stage's marginal contribution."""
    b = budgets(inst)
    # construct -> +PAS -> +ALNS. stage_costs come from the no-polish pipeline run.
    r = matheuristic(inst, seed=seed, pas_work=b["pas_work"], alns_iters=b["alns_iters"],
                     nra_work=b["nra_work"], ot_work=b["ot_work"], descent_passes=b["descent_passes"],
                     lns_rounds=b["lns_rounds"], opt_window=b["opt_window"], do_polish=False)
    stages = {"construct": r.stage_costs.get("construct"),
              "+pas": r.stage_costs.get("pas"),
              "+alns": r.stage_costs.get("alns"),
              "+descent": r.stage_costs.get("descent"),
              "+lns": r.stage_costs.get("lns")}
    # +exact OT
    s_ot, _ = polish(inst, r.state, ot_work=b["ot_work"], nra_work=b["nra_work"], do_nra=False)
    c_ot = evaluate(inst, s_ot)
    stages["+exact_ot"] = c_ot.total_cost if c_ot.total_violations == 0 else None
    # +exact NRA, full polish
    s_full, _ = polish(inst, r.state, ot_work=b["ot_work"], nra_work=b["nra_work"])
    c_full = evaluate(inst, s_full)
    stages["+exact_nra"] = c_full.total_cost if c_full.total_violations == 0 else None
    return stages


def _solve_task(task):
    """solve one (instance, seed); returns a picklable tuple.

    Own process, so instance-level parallelism scales to all cores. Each task is
    independent with a fixed seed and deterministic solver config, so the result
    reproduces."""
    name, seed = task
    inst = load_instance(config.instance_path(name))
    t0 = time.time()
    r = run_instance(inst, seed)
    return (name, seed, int(r.cost), int(r.violations),
            write_solution_dict(inst, r.state), time.time() - t0)


def _bound_task(name):
    """certified lower bound for one instance; own process, deterministic solver"""
    from .bounds import patient_side_lower_bound
    inst = load_instance(config.instance_path(name))
    t0 = time.time()
    r = patient_side_lower_bound(inst)
    return (name, r["lower_bound"], r["relax_obj"], r["status"], time.time() - t0)


def run_bounds(instances, out_dir: str, jobs: int | None = None) -> None:
    """Certified lower bounds for every instance, in parallel, to ``bounds.csv``.

    Bound is Gurobi's proven dual bound on a valid relaxation (see :mod:`ihtp.bounds`),
    deterministic (WorkLimit-bounded). Lets Task 3 report a certified optimality gap
    instead of a gap to the non-optimal competition best-known."""
    os.makedirs(out_dir, exist_ok=True)
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(_v, "1")
    jobs = jobs or os.cpu_count() or 1
    # Gurobi status code -> label (stable across versions). 2 = relaxation proven optimal.
    # 16 = hit the deterministic WorkLimit; bound still valid, just not tight.
    grb_status = {2: "OPTIMAL", 9: "TIME_LIMIT", 13: "SUBOPTIMAL", 16: "WORK_LIMIT",
                  3: "INFEASIBLE", 4: "INF_OR_UNBD", 5: "UNBOUNDED"}
    rows = []
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        for name, lb, relax, status, secs in ex.map(_bound_task, instances):
            rows.append([name, lb, relax if relax is not None else "", grb_status.get(status, str(status))])
            print(f"  [{name}] lower_bound={lb} relax={relax} status={status} ({secs:.0f}s)", flush=True)
    rows.sort(key=lambda r: r[0])
    with open(os.path.join(out_dir, "bounds.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["instance", "lower_bound", "relax_obj", "gurobi_status"])
        w.writerows(rows)
    print(f"wrote {os.path.join(out_dir, 'bounds.csv')}")


def _write_environment(out_dir: str, wall_seconds: float, jobs: int, seeds) -> None:
    """Snapshot the machine that ran this (CPU, cores, RAM, OS, solver versions) plus
    wall-clock, so the deliverable records env and timing truthfully for whatever box ran
    it. Objectives depend only on solver versions, not hardware; these env/timing numbers
    are the machine-specific part."""
    import platform
    import subprocess

    def sysctl(key):
        try:
            return subprocess.run(["sysctl", "-n", key], capture_output=True, text=True).stdout.strip()
        except Exception:
            return ""
    import gurobipy
    import numpy
    import ortools
    mem = sysctl("hw.memsize")
    lines = [
        "Computational environment (auto-captured by experiments.py):",
        f"  machine     : {sysctl('machdep.cpu.brand_string') or platform.processor()}",
        f"  CPU cores   : {os.cpu_count()} (physical {sysctl('hw.physicalcpu') or '?'})",
        f"  RAM         : {round(int(mem)/1e9,1)} GB" if mem else "  RAM         : ?",
        f"  OS          : {platform.platform()}",
        f"  Python      : {platform.python_version()}",
        f"  solvers     : gurobipy {'.'.join(map(str, gurobipy.gurobi.version()))}, "
        f"ortools {ortools.__version__}, numpy {numpy.__version__}",
        f"  parallelism : {jobs} processes; seeds {list(seeds)}",
        f"  wall-clock  : {wall_seconds:.1f} s total",
    ]
    with open(os.path.join(out_dir, "environment.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


def run_all(instances, seeds, out_dir: str, jobs: int | None = None):
    """Solve every (instance, seed) across ``jobs`` processes (default all cores), keep
    the best validated solution per instance, write the deliverables."""
    os.makedirs(out_dir, exist_ok=True)
    # one BLAS thread per process (children inherit): no oversubscription when instances
    # run in parallel, and numpy stays deterministic
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(_v, "1")
    jobs = jobs or os.cpu_count() or 1
    wall0 = time.time()
    tasks = [(n, s) for n in instances for s in seeds]

    best: dict = {}       # name -> (cost, sol_dict, seconds, seed), best-of-k incumbent
    seed_rows = []        # per-seed rows: name, seed, feasibility, objective, runtime_s
    total_secs: dict = {} # name -> total solver-time over all seeds. winning-seed time alone
                          # under-reports a best-of-k method, so sum all seeds for honest effort.
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        for name, seed, cost, viol, sol_dict, secs in ex.map(_solve_task, tasks):
            feasible = viol == 0
            seed_rows.append([name, seed, "feasible" if feasible else "infeasible",
                              cost if feasible else "", f"{secs:.1f}"])
            total_secs[name] = total_secs.get(name, 0.0) + secs
            if feasible and (name not in best or cost < best[name][0]):
                best[name] = (cost, sol_dict, secs, seed)
            print(f"  [{name} seed {seed}] cost={cost} viol={viol} ({secs:.0f}s)", flush=True)

    # per-seed results feed the Task-3 variance analysis. each seed is deterministic, so
    # the whole distribution (and the best-of-k below) reproduces exactly.
    with open(os.path.join(out_dir, "seed_runs.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["instance", "seed", "feasibility", "objective", "runtime_s"])
        w.writerows(seed_rows)

    rows_csv, rows_summary = [], []
    for name in instances:
        ip = config.instance_path(name)
        bk = config.BEST_KNOWN[name]
        if name not in best:
            rows_csv.append([name, "infeasible"])
            rows_summary.append([name, "infeasible", "", bk, "", f"{total_secs.get(name, 0.0):.1f}"])
            print(f"{name}: INFEASIBLE")
            continue
        cost, sol_dict, secs, bseed = best[name]
        sol_path = os.path.join(out_dir, f"{name}.json")
        with open(sol_path, "w") as fh:
            json.dump(sol_dict, fh)
        vr = run_validator(ip, sol_path)             # the authoritative feasibility/quality gate
        with open(os.path.join(out_dir, f"{name}.validator.txt"), "w") as fh:
            fh.write(vr.raw)
        if vr.feasible:
            gap = 100.0 * (vr.cost - bk) / bk
            rows_csv.append([name, vr.cost])
            rows_summary.append([name, "feasible", vr.cost, bk, f"{gap:.2f}", f"{total_secs[name]:.1f}"])
            print(f"{name}: feasible cost={vr.cost} best={bk} gap={gap:+.1f}%  "
                  f"({total_secs[name]:.0f}s over {len(seeds)} seeds)")
        else:
            rows_csv.append([name, "infeasible"])
            rows_summary.append([name, "infeasible", "", bk, "", f"{total_secs[name]:.1f}"])
            print(f"{name}: INFEASIBLE (validator viol={vr.violations})")

    with open(os.path.join(out_dir, "results.csv"), "w", newline="") as fh:
        csv.writer(fh).writerows(rows_csv)
    with open(os.path.join(out_dir, "summary.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        # runtime_s_total = solver-time summed over all seeds for this instance (honest best-of-k
        # effort). per-seed times in seed_runs.csv; whole-campaign wall-clock in environment.txt.
        w.writerow(["instance", "feasibility", "objective", "best_known", "gap_pct", "runtime_s_total"])
        w.writerows(rows_summary)

    feas = [r for r in rows_summary if r[1] == "feasible"]
    tot = sum(r[2] for r in feas)
    tot_bk = sum(config.BEST_KNOWN[r[0]] for r in feas)
    if tot_bk:
        print(f"\nfeasible {len(feas)}/{len(instances)}  sum_obj={tot}  sum_best={tot_bk}  "
              f"overall_gap={100*(tot-tot_bk)/tot_bk:.1f}%")
    _write_environment(out_dir, time.time() - wall0, jobs, seeds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=1, help="best-of-k over seeds 1..k")
    ap.add_argument("--instances", default="all", help="'all' or comma list e.g. i01,i13")
    ap.add_argument("--out", default=config.RESULTS_DIR)
    ap.add_argument("--jobs", type=int, default=0, help="parallel processes (0 = all CPU cores)")
    ap.add_argument("--ablation", default="", help="comma list of instances to ablate instead")
    ap.add_argument("--bounds", action="store_true", help="compute certified lower bounds instead of solving")
    args = ap.parse_args()

    instances = config.PUBLIC_INSTANCES if args.instances == "all" else args.instances.split(",")

    if args.ablation:
        for name in args.ablation.split(","):
            inst = load_instance(config.instance_path(name))
            st = ablation(inst, seed=1)
            print(name, st)
        return

    if args.bounds:
        run_bounds(instances, args.out, jobs=(args.jobs or None))
        return

    seeds = list(range(1, args.seeds + 1))
    run_all(instances, seeds, args.out, jobs=(args.jobs or None))


if __name__ == "__main__":
    main()

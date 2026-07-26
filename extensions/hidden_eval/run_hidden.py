#!/usr/bin/env python3
"""Held-out evaluation on the competition's hidden instances (m01 to m30).

The protocol is PROTOCOL_HIDDEN.md in this folder, registered before any run.
The frozen pipeline runs unchanged, with the production budgets and the same
five-seed best-of-five policy as the public campaign. Every result is scored
by the official validator. This script is resumable: a finished
(instance, seed) pair is skipped on relaunch.

Usage:
    run_hidden.py                 all 30 x 5 with 10 workers
    run_hidden.py m01 --seed 1    a single run (smoke test)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

VENV = os.path.join(ROOT, ".venv", "bin", "python")
INST_DIR = os.path.join(ROOT, "data", "instances_hidden")
PAIRS = [(f"m{i:02d}", s) for i in range(1, 31) for s in range(1, 6)]

ENV = dict(os.environ)
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS"):
    ENV[var] = "1"


def result_path(name: str, seed: int) -> str:
    return os.path.join(HERE, f"result_{name}_s{seed}.json")


def solve_one(name: str, seed: int) -> None:
    from ihtp.experiments import budgets
    from ihtp.io_instance import load_instance
    from ihtp.objective import SOFT_NAMES, evaluate
    from ihtp.pipeline import matheuristic
    from ihtp.validate import validate_state
    from ihtp.writer import write_solution

    inst_path = os.path.join(INST_DIR, f"{name}.json")
    inst = load_instance(inst_path)
    b = budgets(inst)
    t0 = time.time()
    res = matheuristic(inst, seed=seed, pas_work=b["pas_work"], alns_iters=b["alns_iters"],
                       nra_work=b["nra_work"], ot_work=b["ot_work"],
                       descent_passes=b["descent_passes"], lns_rounds=b["lns_rounds"],
                       opt_window=b["opt_window"])
    secs = time.time() - t0
    val = validate_state(inst, res.state, inst_path)
    comps = evaluate(inst, res.state).weighted_components()
    write_solution(inst, res.state, os.path.join(HERE, f"sol_{name}_s{seed}.json"))
    rec = dict(instance=name, seed=seed, objective=val.cost, violations=val.violations,
               runtime_s=round(secs, 1), stage_costs=res.stage_costs,
               components={k: comps[k] for k in SOFT_NAMES})
    with open(result_path(name, seed), "w") as fh:
        json.dump(rec, fh, indent=1)
    print(f"[{name} s{seed}] objective={val.cost} violations={val.violations} {secs:.0f}s",
          flush=True)


def run_subprocess(pair):
    name, seed = pair
    if os.path.exists(result_path(name, seed)):
        return f"{name} s{seed}: cached"
    import subprocess
    log = os.path.join(HERE, f"log_{name}_s{seed}.txt")
    with open(log, "w") as fh:
        rc = subprocess.run([VENV, os.path.abspath(__file__), name, "--seed", str(seed)],
                            stdout=fh, stderr=subprocess.STDOUT, env=ENV, cwd=HERE).returncode
    ok = rc == 0 and os.path.exists(result_path(name, seed))
    return f"{name} s{seed}: {'ok' if ok else f'FAILED rc={rc}'}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("instance", nargs="?", default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    if args.instance and args.seed is not None:
        solve_one(args.instance, args.seed)
        return

    todo = [p for p in PAIRS if not os.path.exists(result_path(*p))]
    print(f"hidden campaign: {len(PAIRS)} pairs, {len(todo)} to run", flush=True)
    with ThreadPoolExecutor(max_workers=10) as ex:
        for msg in ex.map(run_subprocess, PAIRS):
            print(msg, flush=True)
    with open(os.path.join(HERE, "HIDDEN_DONE.txt"), "w") as fh:
        fh.write("done\n")
    print("HIDDEN_DONE", flush=True)


if __name__ == "__main__":
    main()

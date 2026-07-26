#!/usr/bin/env python3
"""Budget-slack probe (PROBE.md): re-solve the NRA-MIP at work budget 3000 on
the stored public layouts, layout untouched, one instance per process."""
from __future__ import annotations
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
from ihtp import config
from ihtp.exact_nra import optimize_nra
from ihtp.io_instance import load_instance
from ihtp.objective import evaluate
from ihtp.validate import validate_state
from ihtp.writer import read_solution

def probe(name: str) -> None:
    inst_path = config.instance_path(name)
    inst = load_instance(inst_path)
    st = read_solution(inst, os.path.join(config.RESULTS_DIR, f"{name}.json"))
    before = evaluate(inst, st); bc = before.weighted_components()
    t0 = time.time()
    optimize_nra(inst, st, work_limit=3000.0)
    secs = time.time() - t0
    val = validate_state(inst, st, inst_path)
    ac = evaluate(inst, st).weighted_components()
    rec = dict(instance=name, before=int(before.total_cost), after=val.cost,
               violations=val.violations, runtime_s=round(secs, 1),
               nurse_before={k: bc[k] for k in ("RoomSkillLevel","ContinuityOfCare","ExcessiveNurseWorkload")},
               nurse_after={k: ac[k] for k in ("RoomSkillLevel","ContinuityOfCare","ExcessiveNurseWorkload")})
    with open(os.path.join(HERE, f"probe_{name}.json"), "w") as fh:
        json.dump(rec, fh, indent=1)
    print(f"[{name}] {rec['before']} -> {rec['after']} ({rec['after']-rec['before']:+d}) "
          f"viol {val.violations} {secs:.0f}s", flush=True)

if __name__ == "__main__":
    probe(sys.argv[1])

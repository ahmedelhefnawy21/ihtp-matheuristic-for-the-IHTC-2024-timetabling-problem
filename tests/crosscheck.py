"""Python evaluator vs compiled C++ validator, per component.

reference solutions across sizes/costs. hits paths i01 misses: surgeon transfer,
excess workload, near-horizon truncation, many OTs+surgeons, ~500 patients.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ihtp import config, evaluate, load_instance, read_solution  # noqa: E402
from ihtp.objective import HARD_NAMES, SOFT_NAMES  # noqa: E402

# keep paths portable, nothing absolute/session-specific: refs from the bundled
# snapshot, validator in bin/, instances through the config resolver
PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF_DIR = os.path.join(PKG_ROOT, "data", "reference_solutions")
VALIDATOR = os.path.join(PKG_ROOT, "bin", "IHTP_Validator")
INSTANCES = ["01", "04", "10", "13", "17", "24", "27", "29"]


def parse_validator(out: str):
    hard, soft = {}, {}
    for name in HARD_NAMES:
        m = re.search(rf"^{re.escape(name)}\.+\s*(-?\d+)\s*$", out, re.M)
        hard[name] = int(m.group(1)) if m else None
    for name in SOFT_NAMES:
        # raw count only: "RoomAgeMix............15 (  5 X   3)" -> 3
        m = re.search(rf"{re.escape(name)}\.+\d+\s*\(\s*\d+\s*X\s*(\d+)\s*\)", out)
        soft[name] = int(m.group(1)) if m else None
    tv = re.search(r"Total violations = (\d+)", out)
    tc = re.search(r"Total cost = (\d+)", out)
    return hard, soft, int(tv.group(1)), int(tc.group(1))


def main() -> int:
    all_ok = True
    for i in INSTANCES:
        inst_path = config.instance_path(f"i{i}")
        sol_path = os.path.join(REF_DIR, f"sol_i{i}.json")
        out = subprocess.run([VALIDATOR, inst_path, sol_path],
                             capture_output=True, text=True).stdout
        v_hard, v_soft, v_tv, v_tc = parse_validator(out)

        inst = load_instance(inst_path)
        costs = evaluate(inst, read_solution(inst, sol_path))
        my_hard = costs.hard_components()
        my_soft = {n: int(costs.soft[k]) for k, n in enumerate(SOFT_NAMES)}

        diffs = []
        for name in HARD_NAMES:
            if my_hard[name] != v_hard[name]:
                diffs.append(f"{name}: py={my_hard[name]} cc={v_hard[name]}")
        for name in SOFT_NAMES:
            if my_soft[name] != v_soft[name]:
                diffs.append(f"{name}: py={my_soft[name]} cc={v_soft[name]}")
        if costs.total_cost != v_tc:
            diffs.append(f"TOTAL: py={costs.total_cost} cc={v_tc}")
        if costs.total_violations != v_tv:
            diffs.append(f"VIOL: py={costs.total_violations} cc={v_tv}")

        status = "PASS ✓" if not diffs else "FAIL ✗"
        if diffs:
            all_ok = False
        print(f"i{i}: cost={v_tc:<7} viol={v_tv}  P={inst.P:<3} R={inst.R:<2} "
              f"N={inst.N:<2} OT={inst.T:<2} U={inst.U:<2} days={inst.days:<2} -> {status}")
        for d in diffs:
            print(f"      {d}")

    print("\nOVERALL:", "ALL PASS ✓" if all_ok else "FAILURES ✗")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

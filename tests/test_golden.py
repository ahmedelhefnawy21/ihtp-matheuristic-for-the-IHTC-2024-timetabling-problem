"""Golden test: Python evaluator vs the official C++ validator.

Runs the published i01 reference solution through ``objective.evaluate``,
demands an exact match (0 violations, cost 3842). If it passes, the tricky
indexing is right: relative vs absolute shift indices, occupants, horizon
truncation, continuity counting.
"""

from __future__ import annotations

import os
import sys

# lets "python tests/test_golden.py" run without installing the package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ihtp import config, evaluate, load_instance, read_solution  # noqa: E402
from ihtp.objective import SOFT_NAMES  # noqa: E402

# portable paths: instance via the config resolver, i01 reference from the
# bundled snapshot (data/reference_solutions/). nothing absolute or
# session-specific, so it runs on any clean checkout
PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INSTANCE = config.instance_path("i01")
SOLUTION = os.path.join(PKG_ROOT, "data", "reference_solutions", "sol_i01.json")

# raw soft-cost counts from the validator "X count" column, SOFT_NAMES order
EXPECTED_SOFT = {
    "RoomAgeMix": 3,
    "RoomSkillLevel": 19,
    "ContinuityOfCare": 127,
    "ExcessiveNurseWorkload": 0,
    "OpenOperatingTheater": 8,
    "SurgeonTransfer": 0,
    "PatientDelay": 47,
    "ElectiveUnscheduledPatients": 8,
}
EXPECTED_TOTAL_COST = 3842


def main() -> int:
    inst = load_instance(INSTANCE)
    st = read_solution(inst, SOLUTION)
    costs = evaluate(inst, st)

    ok = True
    print(f"instance: {inst.name}  (P={inst.P}, O={inst.O}, R={inst.R}, N={inst.N}, "
          f"days={inst.days}, OT={inst.T}, surgeons={inst.U})")
    print(f"Total violations = {costs.total_violations}")
    print("Hard components:", costs.hard_components())
    print("Soft (weight X count -> weighted):")
    for i, name in enumerate(SOFT_NAMES):
        raw = int(costs.soft[i])
        w = int(costs.weights[i])
        exp = EXPECTED_SOFT[name]
        flag = "OK" if raw == exp else f"MISMATCH exp={exp}"
        if raw != exp:
            ok = False
        print(f"  {name:<28} {w:>4} X {raw:<5} = {w*raw:<7} [{flag}]")
    print(f"Total cost = {costs.total_cost} (expected {EXPECTED_TOTAL_COST})")

    if costs.total_violations != 0:
        print("FAIL: solution reported infeasible by our evaluator")
        ok = False
    if costs.total_cost != EXPECTED_TOTAL_COST:
        print("FAIL: total cost mismatch")
        ok = False

    print("\nRESULT:", "PASS ✓" if ok else "FAIL ✗")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

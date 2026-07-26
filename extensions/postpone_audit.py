#!/usr/bin/env python3
"""Bonus extension: a nurse-aware admission audit (the "right of repentance").

STANDALONE. This script imports the ihtp package read-only and does not modify
the pipeline, any pipeline stage, or any file in ADVANCED_MODELLING/results/.
It consumes finished solutions and writes its own output to extension_results/.

--- The bottleneck this probes -------------------------------------------------
The residual gap to best-known is ~90% nurse cost (S2 skill, S3 continuity,
S4 workload), yet we admit MORE patients than best-known (+62). The diagnosis:
admission is nurse-blind. Every in-or-out decision in the pipeline compares a
patient's *patient-side* cost against the unscheduled penalty W8:

  - PAS-MIP objective is literally W8*unscheduled + W7*delay  (exact_pas.py)
  - construct/ALNS admit an optional iff its placement cost < W8
  - CP-SAT LNS prices W8 against patient-side cost; its honest full-objective
    check is per-ROUND, so one bad admission can hide in a net-positive bundle

None of these sees the staffing bill the admission creates. The descent is the one stage
that DOES score the full 8-term objective with a fresh nurse solve. It only sweeps
patients with adm != -1, so it can RELOCATE an admitted patient but never POSTPONE one
(local_search.py:45). The pipeline is therefore a one-way ratchet. An optional who saves
350 in penalty but costs 400 to staff is admitted, and no later stage can un-admit it on
true cost.

--- The fix being tested -------------------------------------------------------
The fix adds exactly one candidate to the descent's neighbourhood: "leave p out". The
descent already builds that state (it calls unplace(p) as a step
between relocations) but never scores it as an outcome. Here we score it,
with the same evaluator, and accept it under the same strict-improvement rule.

This adds no model, no stage, no loop and no tuned parameter. It removes an
asymmetry: the search could already admit greedily and score truthfully; it
just could not change its mind.

--- Protocol (fair comparison) -------------------------------------------------
The saved baseline was finished with exact OT-MIP and NRA-MIP solves, so scoring a
swept plan under only a greedy nurse fill would be an unfair comparison. Hence:

  1. load the finished solution, validate it (must reproduce the headline)
  2. run the postpone sweep      (decision rule: greedy NRA, as the descent uses)
  3. re-solve exactly            (OT-MIP then NRA-MIP, production work budgets)
  4. re-validate with the official C++ binary; report ITS number

Step 2's decision rule (greedy nurse) is cheaper than step 3's scorer (exact
nurse), so a postponement judged good under greedy could be neutral or bad once
nurses are re-optimised. The validator in step 4 is the arbiter, and a negative
result is reported as such; see REPORTING below.

Deterministic: fixed sweep seed, deterministic solver work budgets, no
wall-clock stopping. Feasibility is preserved by construction (removing an
OPTIONAL patient cannot violate anything; only a missing MANDATORY patient is a
violation, and mandatory patients are never swept).

REPORTING: both outcomes are informative. Postponements found => measured,
validator-certified evidence of the admission/nurse coupling plus a bonus
improvement. Nothing found => the coupling loss does NOT sit in marginal
admissions, which sharpens the diagnosis toward roster/layout structure
(continuity, S3) rather than admission volume.

Usage:
    python3 postpone_audit.py                     # the 5 hardest + i01 sanity
    python3 postpone_audit.py i01                 # a single instance
    python3 postpone_audit.py --no-repolish i01   # sweep only, no exact re-solve
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)  # repo root, which contains the ihtp/ package
sys.path.insert(0, PKG)

import numpy as np  # noqa: E402

from ihtp import config  # noqa: E402
from ihtp.construct import layout_from_state  # noqa: E402
from ihtp.exact_nra import optimize_nra  # noqa: E402
from ihtp.exact_ot import optimize_ot  # noqa: E402
from ihtp.io_instance import load_instance  # noqa: E402
from ihtp.model import SolutionState  # noqa: E402
from ihtp.nra import greedy_nra  # noqa: E402
from ihtp.objective import SOFT_NAMES, evaluate  # noqa: E402
from ihtp.validate import validate_state  # noqa: E402
from ihtp.writer import read_solution, write_solution  # noqa: E402

# the five worst instances by gap to best-known, plus i01 as a fast sanity check
HARDEST = ["i27", "i17", "i29", "i02", "i21"]
DEFAULT_SET = ["i01"] + HARDEST

OUT_DIR = os.path.join(HERE, "extension_results")


def postpone_sweep(inst, state: SolutionState, max_passes: int = 4,
                   seed: int = 0) -> tuple[SolutionState, int, list[int], int]:
    """First-improvement sweep whose only move is 'postpone this optional patient'.

    The sweep mirrors layout_descent exactly: same evaluator, same fresh greedy NRA
    per candidate, same strict-improvement acceptance, same seeded shuffle, same early
    bail. The one difference is that the candidate scored is the state with p left OUT,
    rather than p relocated.

    Mandatory patients are never considered: leaving one out is a hard
    violation (H5), so they are not eligible and feasibility is preserved.
    """
    rng = np.random.default_rng(seed)
    lay = layout_from_state(inst, state)

    def true_cost() -> int:
        greedy_nra(inst, lay.st)
        return int(evaluate(inst, lay.st).total_cost)

    best = true_cost()
    postponed: list[int] = []
    improved, passes = True, 0
    while improved and passes < max_passes:
        passes += 1
        improved = False
        admitted = np.nonzero(lay.st.adm != -1)[0]
        order = [int(p) for p in admitted if not inst.p_mandatory[p]]
        rng.shuffle(order)
        for p in order:
            d0, r0, o0 = int(lay.st.adm[p]), int(lay.st.room[p]), int(lay.st.ot[p])
            lay.unplace(p)
            c = true_cost()                     # score the plan with p LEFT OUT
            if c < best - 1e-9:
                best = c
                improved = True
                postponed.append(p)
            else:
                lay.place(p, d0, r0, o0)        # restore exactly
    return lay.st.copy(), best, postponed, passes


def margins(name: str, verbose: bool = True) -> dict:
    """Quantify the null result: how far is each admitted optional from being worth postponing?

    For every admitted optional p, margin(p) = cost(plan without p) - cost(plan with p),
    both scored with a fresh greedy nurse solve. margin > 0 means keeping p is cheaper
    (postponing would cost us that much); margin < 0 would mean p is a net loser and the
    sweep should have postponed them.

    This turns "the sweep found nothing" into "the marginal patient clears the bar by N", and it
    also shows the sweep is not a no-op: the margins vary from patient to patient.
    """
    inst_path = config.instance_path(name)
    inst = load_instance(inst_path)
    base = read_solution(inst, os.path.join(config.RESULTS_DIR, f"{name}.json"))
    lay = layout_from_state(inst, base)

    def true_cost() -> int:
        greedy_nra(inst, lay.st)
        return int(evaluate(inst, lay.st).total_cost)

    w8 = int(inst.weights[7])
    c_in = true_cost()
    vals = []
    optionals = [int(p) for p in np.nonzero(lay.st.adm != -1)[0]
                 if not inst.p_mandatory[p]]
    for p in optionals:
        d0, r0, o0 = int(lay.st.adm[p]), int(lay.st.room[p]), int(lay.st.ot[p])
        lay.unplace(p)
        vals.append(true_cost() - c_in)
        lay.place(p, d0, r0, o0)
    a = np.array(sorted(vals)) if vals else np.array([0])
    rec = dict(instance=name, w8_penalty=w8, n_optional=len(vals),
               n_negative=int((a < 0).sum()),
               min=int(a.min()), p10=int(np.percentile(a, 10)),
               median=int(np.median(a)), max=int(a.max()))
    if verbose:
        print(f"[{name}] W8={w8} | {len(vals)} admitted optionals | "
              f"margin: min={rec['min']} p10={rec['p10']} median={rec['median']} "
              f"max={rec['max']} | net-losers={rec['n_negative']}")
    return rec


def audit(name: str, max_passes: int = 4, seed: int = 0,
          repolish: bool = True, verbose: bool = True) -> dict:
    """Run the audit on one instance and return a record of what moved."""
    inst_path = config.instance_path(name)
    sol_path = os.path.join(config.RESULTS_DIR, f"{name}.json")
    inst = load_instance(inst_path)
    base = read_solution(inst, sol_path)

    # --- 1. baseline, and an integrity check that our scorer == the validator
    base_costs = evaluate(inst, base)
    base_val = validate_state(inst, base, inst_path)
    n_optional_admitted = int(sum(1 for p in range(inst.P)
                                  if base.adm[p] != -1 and not inst.p_mandatory[p]))
    if verbose:
        agree = "OK" if int(base_costs.total_cost) == base_val.cost else "MISMATCH"
        print(f"  baseline: validator={base_val.cost} violations={base_val.violations} "
              f"| our scorer={int(base_costs.total_cost)} [{agree}] "
              f"| optional admitted={n_optional_admitted}")

    # --- 2. the sweep
    t0 = time.time()
    swept, internal, postponed, passes = postpone_sweep(inst, base, max_passes, seed)
    sweep_secs = time.time() - t0
    if verbose:
        print(f"  sweep: {len(postponed)} postponed in {passes} pass(es), "
              f"{sweep_secs:.1f}s")

    # --- 3. exact re-solve so the comparison is like-for-like with the baseline
    #
    # The sweep scores candidates with greedy_nra, so the state it hands back
    # carries a GREEDY nurse roster. That roster would otherwise silently replace
    # the baseline's exact NRA-MIP roster and show up as a phantom nurse-cost
    # regression. Two cases:
    #   nothing postponed -> the plan is unchanged, so return the baseline
    #                        untouched (exact roster intact), delta exactly 0
    #   something postponed -> re-solve nurses (and theatres) exactly, so the
    #                        comparison is exact-roster vs exact-roster
    polished = "none"
    if not postponed:
        swept = base                                  # unchanged: keep the exact roster
        polished = "n/a (no change)"
    elif repolish:
        b = dict(nra_work=float(min(800, max(600, inst.R * inst.days))), ot_work=25.0)
        try:
            optimize_ot(inst, swept, per_day_work=b["ot_work"])
            optimize_nra(inst, swept, work_limit=b["nra_work"])
            polished = "exact (OT-MIP + NRA-MIP)"
        except Exception as exc:                      # noqa: BLE001 report, do not crash
            greedy_nra(inst, swept)
            polished = f"greedy fallback ({type(exc).__name__}: {str(exc)[:80]})"
        if verbose:
            print(f"  re-polish: {polished}")

    # --- 4. the official binary is the arbiter
    ext_costs = evaluate(inst, swept)
    ext_val = validate_state(inst, swept, inst_path)

    delta = ext_val.cost - base_val.cost
    comp_before = base_costs.weighted_components()
    comp_after = ext_costs.weighted_components()
    moved = {k: comp_after[k] - comp_before[k]
             for k in SOFT_NAMES if comp_after[k] != comp_before[k]}

    rec = dict(
        instance=name,
        baseline_cost=base_val.cost, baseline_violations=base_val.violations,
        extension_cost=ext_val.cost, extension_violations=ext_val.violations,
        delta=delta, improved=bool(delta < 0 and ext_val.violations == 0),
        n_postponed=len(postponed), postponed_patients=postponed,
        optional_admitted_before=n_optional_admitted,
        passes=passes, sweep_seconds=round(sweep_secs, 1), repolish=polished,
        scorer_matches_validator=bool(int(base_costs.total_cost) == base_val.cost),
        components_before=comp_before, components_after=comp_after,
        components_moved=moved,
    )
    # an improved plan is saved beside this script, never into results/
    if rec["improved"]:
        os.makedirs(OUT_DIR, exist_ok=True)
        out_sol = os.path.join(OUT_DIR, f"{name}.json")
        write_solution(inst, swept, out_sol)
        rec["saved_solution"] = out_sol

    if verbose:
        verdict = ("IMPROVED" if rec["improved"]
                   else "no change" if delta == 0 else "WORSE")
        print(f"  result: {base_val.cost} -> {ext_val.cost} "
              f"({delta:+d}, {delta / base_val.cost * 100:+.2f}%)  [{verdict}]"
              f"  violations={ext_val.violations}")
        if moved:
            print("  moved: " + ", ".join(f"{k} {v:+d}" for k, v in moved.items()))
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("instances", nargs="*", default=None,
                    help="instance names (default: i01 sanity + the 5 hardest)")
    ap.add_argument("--passes", type=int, default=4, help="max sweep passes (default 4)")
    ap.add_argument("--seed", type=int, default=0, help="sweep shuffle seed (default 0)")
    ap.add_argument("--no-repolish", action="store_true",
                    help="skip the exact OT/NRA re-solve (sweep only)")
    ap.add_argument("--margins", action="store_true",
                    help="report how far each admitted optional sits from the threshold")
    args = ap.parse_args()

    names = args.instances if args.instances else DEFAULT_SET
    os.makedirs(OUT_DIR, exist_ok=True)

    if args.margins:
        print("Margin distribution: cost(without p) - cost(with p), per admitted optional")
        print("positive => keeping the patient is cheaper (postponing would cost that much)\n")
        recs = [margins(n) for n in names]
        # merge by instance, so a partial run updates its own record without
        # dropping the records of the instances it did not touch
        mpath = os.path.join(OUT_DIR, "postpone_margins.json")
        merged: dict[str, dict] = {}
        if os.path.exists(mpath):
            try:
                with open(mpath) as fh:
                    merged = {r.get("instance"): r for r in json.load(fh)}
            except (json.JSONDecodeError, OSError):
                merged = {}
        for r in recs:
            merged[r.get("instance")] = r
        with open(mpath, "w") as fh:
            json.dump(list(merged.values()), fh, indent=1)
        print(f"\nwrote {os.path.join(OUT_DIR, 'postpone_margins.json')}")
        return

    print("Nurse-aware admission audit (bonus extension; pipeline untouched)")
    print(f"instances: {', '.join(names)} | passes={args.passes} seed={args.seed} "
          f"| repolish={not args.no_repolish}\n")

    records = []
    for name in names:
        print(f"[{name}]")
        try:
            rec = audit(name, args.passes, args.seed, not args.no_repolish)
            records.append(rec)
        except Exception as exc:                      # noqa: BLE001
            print(f"  FAILED: {type(exc).__name__}: {exc}")
            records.append(dict(instance=name, error=f"{type(exc).__name__}: {exc}"))
        print()

    # merge by instance so a partial run (say, one instance) updates its own
    # record without dropping the records of the instances it did not touch
    out_path = os.path.join(OUT_DIR, "postpone_audit.json")
    merged: dict[str, dict] = {}
    if os.path.exists(out_path):
        try:
            with open(out_path) as fh:
                merged = {r.get("instance"): r for r in json.load(fh)}
        except (json.JSONDecodeError, OSError):
            merged = {}
    for r in records:
        merged[r.get("instance")] = r
    with open(out_path, "w") as fh:
        json.dump(list(merged.values()), fh, indent=1)

    ok = [r for r in records if "error" not in r]
    print("=" * 74)
    print(f"{'instance':<10}{'baseline':>10}{'extension':>11}{'delta':>9}"
          f"{'%':>8}{'postponed':>11}")
    print("-" * 74)
    for r in ok:
        pct = r["delta"] / r["baseline_cost"] * 100
        print(f"{r['instance']:<10}{r['baseline_cost']:>10}{r['extension_cost']:>11}"
              f"{r['delta']:>+9d}{pct:>+8.2f}{r['n_postponed']:>11}")
    if ok:
        tb = sum(r["baseline_cost"] for r in ok)
        te = sum(r["extension_cost"] for r in ok)
        print("-" * 74)
        print(f"{'TOTAL':<10}{tb:>10}{te:>11}{te - tb:>+9d}"
              f"{(te - tb) / tb * 100:>+8.2f}"
              f"{sum(r['n_postponed'] for r in ok):>11}")
    print("=" * 74)
    print(f"\nwrote {os.path.join(OUT_DIR, 'postpone_audit.json')}")


if __name__ == "__main__":
    main()

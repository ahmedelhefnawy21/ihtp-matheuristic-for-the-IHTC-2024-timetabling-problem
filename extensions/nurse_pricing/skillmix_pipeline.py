#!/usr/bin/env python3
"""Nurse-aware upper layer, v2: supply-aware pricing. The script lives under extensions/, outside the ihtp package. It imports the package read-only and monkey-patches at runtime, so no pipeline file is modified. Results are written next to this script only.

=== Question ==================================================================
Our gap to best-known is ~90% nurse cost and the upper layer prices none of it.
v1 tested a generic skill-mix SPREAD term (age-mix analog). On the development instances
it helped the nurse-heavy ones and hurt the small ones. A pure mix penalty is SUPPLY-
BLIND: it taxes mixing even when plenty of nurses can cover it. Those v1 runs were not
archived, so no figures are quoted here. v2 prices what the layout forces on the FIXED
nurse roster. The roster is a known input: who is on duty each shift, at what skill,
with what load cap.

  --k-load   S4 workload, an EXACT per-shift lower bound, not a proxy:
             any assignment of a shift's total patient workload W_s to nurses
             with total capacity C_s overflows by at least max(0, W_s - C_s)
             (caps fill first; indivisibility only raises it). We price
             k_load * W4 * that overflow. ZERO whenever capacity is slack, so
             it cannot distort easy instances (v1's failure mode).

  --k-scarce S2 under-skill, priced as SCARCITY not spread: what costs is the
             number of rooms demanding TOP-skill coverage in a day exceeding
             the top-skill nurses on duty. We price k_scarce * W2 *
             max(0, top-demanding rooms - top nurses on duty), per day, with
             supply = the day's minimum over its three shifts. Assumption
             (stated): one top room per top nurse before strain; the workload
             term separately prices load pressure. Also zero under slack.

  --k-spread v1's per-room-day spread of daily skill levels (kept for
             comparison; default off in v2).

Continuity S3 stays unpriced BY DESIGN: our audit showed it is roster-structural (which
nurses'
shift patterns tile a stay); we found no static price for it without the admission-nurse
feedback
loop.

=== Wiring (same verified setup as v1) ===================================
Patient day-level lvl(p,k) = max skill required over stay-day k's three shifts
(max-over-stay is DEGENERATE on these instances: on i01 every patient's max is
2). The patched Layout methods are __init__ (caches seeded with occupants),
place/unplace (incremental costs), age_delta (placement scoring; via
best_placement this is the cost the construct admit-threshold compares
against W8), and upper_cost (the ALNS surrogate). All of construct,
ALNS, and descent enumeration flow through them, and alns snapshot/revert
calls place/unplace, so caches stay in sync. NOT patched: PAS-MIP, CP-SAT
re-pack, and every honest gate (descent/LNS/final-stage acceptance, final
validator); they score the true official objective. The unpatched control
reproduces seed_runs.csv exactly.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(HERE, "..", "..")
sys.path.insert(0, os.path.abspath(PKG))

import numpy as np  # noqa: E402

from ihtp import config  # noqa: E402
from ihtp import layout as L  # noqa: E402
from ihtp.experiments import budgets  # noqa: E402
from ihtp.io_instance import load_instance  # noqa: E402
from ihtp.objective import SOFT_NAMES, evaluate  # noqa: E402
from ihtp.pipeline import matheuristic  # noqa: E402
from ihtp.validate import validate_state  # noqa: E402
from ihtp.writer import read_solution, write_solution  # noqa: E402

K_SPREAD = 0.0   # v1 term (per-room-day daily-level spread), off by default in v2
K_LOAD = 1.0     # exact S4 per-shift overflow lower bound
K_SCARCE = 1.0   # top-skill room demand vs top-skill nurse supply, per day
PLACEBO = None   # permutation control: shuffle the roster supply vectors in time

_orig_init = L.Layout.__init__
_orig_place = L.Layout.place
_orig_unplace = L.Layout.unplace
_orig_age_delta = L.Layout.age_delta
_orig_upper = L.Layout.upper_cost


def _day_levels(arr: np.ndarray) -> np.ndarray:
    """per-day level = max requirement over that day's three shifts."""
    return np.array([int(arr[k:k + 3].max()) for k in range(0, len(arr), 3)],
                    dtype=np.int64)


def _patched_init(self, inst, st=None):
    _orig_init(self, inst, st)
    spd = inst.shifts_per_day
    self._sp_w = int(round(K_SPREAD * int(inst.weights[1])))
    self._ld_w = int(round(K_LOAD * int(inst.weights[3])))
    self._sc_w = int(round(K_SCARCE * int(inst.weights[1])))
    self._top = inst.skill_levels - 1

    # per-day skill-level profiles (patients relative, occupants absolute)
    self._sk_pday = [_day_levels(a) for a in inst.p_skillreq]
    o_day = [_day_levels(a) for a in inst.o_skillreq]

    # skill-level buckets per room-day, seeded with occupants
    self._sk_cnt = np.zeros((inst.R, inst.days, inst.skill_levels), dtype=np.int64)
    for r in range(inst.R):
        for d in range(inst.days):
            for o in inst.occ_in_room_day[r][d]:
                self._sk_cnt[r, d, int(o_day[o][d])] += 1

    # workload demand per global shift, seeded with occupants; supply from roster
    self._w_dem = np.zeros(inst.shifts, dtype=np.int64)
    for o in range(inst.O):
        w = inst.o_workload[o]
        self._w_dem[:len(w)] += w
    self._w_cap = inst.nurse_max_load.sum(axis=0).astype(np.int64)   # 0 if absent

    # top-skill supply per day = min over the day's shifts of on-duty top nurses
    is_top = inst.nurse_skill == self._top
    on_duty_top = ((inst.nurse_max_load > 0) & is_top[:, None]).sum(axis=0)
    self._k_top = np.array([int(on_duty_top[d * spd:(d + 1) * spd].min())
                            for d in range(inst.days)], dtype=np.int64)
    if PLACEBO is not None:
        # PERMUTATION CONTROL (PROTOCOL_PLACEBO.md): identical terms, identical
        # magnitudes multiset, but nurse supply no longer aligned with the shifts
        # and days it belongs to. Deterministic: same seed + same lengths -> same
        # permutation on every Layout construction within a run.
        prng = np.random.default_rng(PLACEBO)
        self._w_cap = prng.permutation(self._w_cap)
        self._k_top = prng.permutation(self._k_top)
    # rooms whose current top bucket is the top level, per day
    self._top_cnt = np.zeros(inst.days, dtype=np.int64)
    for r in range(inst.R):
        for d in range(inst.days):
            if self._room_top(r, d) == self._top:
                self._top_cnt[d] += 1

    # running costs
    self._sp_cost = 0
    for r in range(inst.R):
        for d in range(inst.days):
            self._sp_cost += self._sp_w * self._sk_spread(r, d)
    self._ld_cost = self._ld_w * int(np.maximum(0, self._w_dem - self._w_cap).sum())
    self._sc_cost = self._sc_w * int(np.maximum(0, self._top_cnt - self._k_top).sum())


def _sk_spread(self, r: int, d: int) -> int:
    nz = np.nonzero(self._sk_cnt[r, d])[0]
    if nz.size == 0:
        return 0
    return int(nz[-1] - nz[0])


def _room_top(self, r: int, d: int) -> int:
    nz = np.nonzero(self._sk_cnt[r, d])[0]
    return int(nz[-1]) if nz.size else -1


def _apply_day(self, p: int, d0: int, r: int, dd: int, sign: int) -> None:
    """update skill buckets + spread/scarcity costs for one stay day (sign=+1/-1)."""
    lvl = int(self._sk_pday[p][dd - d0])
    old_sp = self._sk_spread(r, dd)
    old_top = self._room_top(r, dd) == self._top
    self._sk_cnt[r, dd, lvl] += sign
    if self._sp_w:
        self._sp_cost += self._sp_w * (self._sk_spread(r, dd) - old_sp)
    if self._sc_w:
        new_top = self._room_top(r, dd) == self._top
        if new_top != old_top:
            over_old = max(0, int(self._top_cnt[dd]) - int(self._k_top[dd]))
            self._top_cnt[dd] += 1 if new_top else -1
            over_new = max(0, int(self._top_cnt[dd]) - int(self._k_top[dd]))
            self._sc_cost += self._sc_w * (over_new - over_old)


def _apply_load(self, p: int, d: int, sign: int) -> None:
    """update per-shift workload demand + exact overflow cost (sign=+1/-1)."""
    if not self._ld_w:
        return
    w = self.inst.p_workload[p]
    base = d * self.inst.shifts_per_day
    last = min(self.inst.shifts, base + len(w))
    for j in range(last - base):
        s = base + j
        over_old = max(0, int(self._w_dem[s]) - int(self._w_cap[s]))
        self._w_dem[s] += sign * int(w[j])
        over_new = max(0, int(self._w_dem[s]) - int(self._w_cap[s]))
        self._ld_cost += self._ld_w * (over_new - over_old)


def _patched_place(self, p: int, d: int, r: int, t: int) -> None:
    _orig_place(self, p, d, r, t)
    last = min(self.inst.days, d + int(self.inst.p_los[p]))
    for dd in range(d, last):
        _apply_day(self, p, d, r, dd, +1)
    _apply_load(self, p, d, +1)


def _patched_unplace(self, p: int) -> None:
    if self.st.adm[p] == -1:
        return _orig_unplace(self, p)
    d, r = int(self.st.adm[p]), int(self.st.room[p])
    _orig_unplace(self, p)
    last = min(self.inst.days, d + int(self.inst.p_los[p]))
    for dd in range(d, last):
        _apply_day(self, p, d, r, dd, -1)
    _apply_load(self, p, d, -1)


def _patched_age_delta(self, p: int, r: int, d: int) -> int:
    """placement score = true age delta + the priced nurse-aware terms."""
    base = _orig_age_delta(self, p, r, d)
    before = self._sp_cost + self._sc_cost + self._ld_cost
    last = min(self.inst.days, d + int(self.inst.p_los[p]))
    for dd in range(d, last):
        _apply_day(self, p, d, r, dd, +1)
    _apply_load(self, p, d, +1)
    after = self._sp_cost + self._sc_cost + self._ld_cost
    _apply_load(self, p, d, -1)
    for dd in range(d, last):
        _apply_day(self, p, d, r, dd, -1)
    return base + (after - before)


def _patched_upper(self) -> int:
    return _orig_upper(self) + self._sp_cost + self._ld_cost + self._sc_cost


def install_patch() -> None:
    L.Layout._sk_spread = _sk_spread
    L.Layout._room_top = _room_top
    L.Layout.__init__ = _patched_init
    L.Layout.place = _patched_place
    L.Layout.unplace = _patched_unplace
    L.Layout.age_delta = _patched_age_delta
    L.Layout.upper_cost = _patched_upper


def selftest(inst) -> None:
    """place/unplace round-trip must restore every cache and cost exactly."""
    lay = L.Layout(inst)
    snap = (lay._sp_cost, lay._ld_cost, lay._sc_cost,
            lay._sk_cnt.copy(), lay._w_dem.copy(), lay._top_cnt.copy())
    placed = 0
    for p in range(inst.P):
        c, move = lay.best_placement(p)
        if move is not None:
            lay.place(p, *move)
            placed += 1
        if placed == 12:
            break
    assert placed > 0, "selftest placed nobody"
    for p in np.nonzero(lay.st.adm != -1)[0]:
        lay.unplace(int(p))
    assert (lay._sp_cost, lay._ld_cost, lay._sc_cost) == snap[:3], "cost drifted"
    assert (lay._sk_cnt == snap[3]).all() and (lay._w_dem == snap[4]).all() \
        and (lay._top_cnt == snap[5]).all(), "cache drifted"
    print(f"  selftest ok | initial costs: load={snap[1]} scarce={snap[2]} spread={snap[0]}")


def baseline_from_seed_runs(name: str, seed: int):
    path = os.path.join(config.RESULTS_DIR, "seed_runs.csv")
    with open(path) as fh:
        for row in csv.DictReader(fh):
            if row["instance"] == name and int(row["seed"]) == seed:
                return int(row["objective"]), float(row["runtime_s"])
    return None, None


def tag() -> str:
    base = f"sp{K_SPREAD}_ld{K_LOAD}_sc{K_SCARCE}"
    return base + (f"_pl{PLACEBO}" if PLACEBO is not None else "")


def run(name: str, seed: int) -> dict:
    inst_path = config.instance_path(name)
    inst = load_instance(inst_path)
    if L.Layout.__init__ is _patched_init:
        selftest(inst)

    b = budgets(inst)
    t0 = time.time()
    res = matheuristic(inst, seed=seed, pas_work=b["pas_work"], alns_iters=b["alns_iters"],
                       nra_work=b["nra_work"], ot_work=b["ot_work"],
                       descent_passes=b["descent_passes"], lns_rounds=b["lns_rounds"],
                       opt_window=b["opt_window"])
    secs = time.time() - t0

    val = validate_state(inst, res.state, inst_path)
    comps = evaluate(inst, res.state).weighted_components()
    base_obj, base_secs = baseline_from_seed_runs(name, seed)
    headline = read_solution(inst, os.path.join(config.RESULTS_DIR, f"{name}.json"))
    head_val = validate_state(inst, headline, inst_path)
    head_comps = evaluate(inst, headline).weighted_components()

    out_sol = os.path.join(HERE, f"{name}_s{seed}_{tag()}.json")
    write_solution(inst, res.state, out_sol)

    rec = dict(instance=name, seed=seed, k_spread=K_SPREAD, k_load=K_LOAD,
    k_scarce=K_SCARCE,
               placebo=PLACEBO,
               patched_objective=val.cost, patched_violations=val.violations,
               patched_runtime_s=round(secs, 1), stage_costs=res.stage_costs,
               baseline_same_seed=base_obj,
               delta_vs_same_seed=(val.cost - base_obj) if base_obj is not None else
               None,
               headline_best_of_5=head_val.cost, delta_vs_headline=val.cost -
               head_val.cost,
               components_patched=comps, components_headline=head_comps,
               solution_file=out_sol)
    d = rec["delta_vs_same_seed"]
    print(f"[{name} seed {seed} {tag()}] patched={val.cost} (viol {val.violations}, "
          f"{secs:.0f}s) | same-seed baseline={base_obj} -> delta {d:+d}"
          f" | best-of-5 headline={head_val.cost} -> delta {val.cost - head_val.cost:+d}")
    moved = {k: comps[k] - head_comps[k] for k in SOFT_NAMES if comps[k] !=
    head_comps[k]}
    print(f"  components vs headline: {moved}")
    return rec


def main() -> None:
    global K_SPREAD, K_LOAD, K_SCARCE, PLACEBO
    ap = argparse.ArgumentParser()
    ap.add_argument("instances", nargs="+")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--k-spread", type=float, default=0.0)
    ap.add_argument("--k-load", type=float, default=1.0)
    ap.add_argument("--k-scarce", type=float, default=1.0)
    ap.add_argument("--no-patch", action="store_true")
    ap.add_argument("--placebo", type=int, default=None,
                    help="permutation control: shuffle roster supply in time with this seed")
    args = ap.parse_args()
    K_SPREAD, K_LOAD, K_SCARCE = args.k_spread, args.k_load, args.k_scarce
    PLACEBO = args.placebo

    if args.no_patch:
        K_SPREAD = K_LOAD = K_SCARCE = 0.0
        print("running UNPATCHED (baseline reproduction control)")
    else:
        install_patch()
        kind = ("PLACEBO permutation control (roster shuffled in time, seed "
                f"{PLACEBO})" if PLACEBO is not None else "supply-aware patch")
        print(f"{kind} installed: k_load={K_LOAD} (exact S4 shift-overflow LB), "
              f"k_scarce={K_SCARCE} (top-skill rooms vs top nurses/day), k_spread={K_SPREAD}")

    recs = []
    for name in args.instances:
        try:
            recs.append(run(name, args.seed))
        except Exception as exc:                      # noqa: BLE001
            print(f"[{name}] FAILED: {type(exc).__name__}: {exc}")
            recs.append(dict(instance=name, seed=args.seed, error=str(exc)))
    out = os.path.join(HERE, f"results_{'_'.join(args.instances)}_s{args.seed}_{tag()}.json")
    json.dump(recs, open(out, "w"), indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

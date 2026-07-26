"""CP-SAT patient LNS with exact repair.

S8 (unscheduled optionals) dominates the objective. The one-shot exact PAS model gets
most of the gap but times out on big instances. A pure metaheuristic cannot re-pack a
saturated surgeon/room schedule to free capacity for one more optional. So the method is
LNS with an exact repair (Shaw 1998; Ropke & Pisinger 2006). A slice is freed and re-
optimised exactly, and the rest stays frozen. The slice goes straight at S8. The freed
set is every unscheduled optional plus everyone admitted in a random day window. The
rest is frozen, and the frozen patients' resource use is subtracted as residual
capacities. The unscheduled optionals stay in the freed set every round. That gives the
exact solver a fresh shot at admitting them without paying for the full-size model.
Window width scales to hold the freed set near ``target_free``.

Freed placement is a CP-SAT model. Reified/implication constraints do gender exclusivity
and the open-OT + surgeon-transfer indicators. Only minimises the patient-side costs it
owns: S7 delay, S8 unscheduled, S5 open OTs, S6 surgeon transfers. Room age-mix (S1) and
nurse costs (S2-S4) belong to the descent and exact NRA stages.

Acceptance is the caller's call, on the true full objective, strict improvement only. So
every kept move lowers the validated objective; nothing regresses even transiently.
"""

from __future__ import annotations

from ortools.sat.python import cp_model

import numpy as np

from .io_instance import GENDER_A, Instance
from .model import SolutionState
from .objective import evaluate
from .solvers import configure_cpsat


def patient_side_cost(inst: Instance, costs) -> int:
    """weighted patient-side soft costs the LNS owns: S5, S6, S7, S8"""
    w = inst.weights
    return int(costs.soft[4] * w[4] + costs.soft[5] * w[5]
              + costs.soft[6] * w[6] + costs.soft[7] * w[7])


def _patch_nurses(inst: Instance, st: SolutionState) -> None:
    """cover newly-occupied room-shifts, leave every other nurse assignment alone so a
    prior exact NRA solution survives. a room-shift that emptied out keeps its nurse for
    free (empty entity, zero nurse cost)."""
    spd = inst.shifts_per_day
    # room-day occupancy from patients + occupants
    occ = np.zeros((inst.R, inst.days), dtype=bool)
    occ |= inst.occ_room_day_count > 0
    for p in np.nonzero(st.adm != -1)[0]:
        r, ad = int(st.room[p]), int(st.adm[p])
        occ[r, ad:min(inst.days, ad + int(inst.p_los[p]))] = True
    # rooms each nurse already covers per shift, for balancing the new picks
    load = {}
    for r in range(inst.R):
        for s in range(inst.shifts):
            n = int(st.cover[r, s])
            if n != -1:
                load[(n, s)] = load.get((n, s), 0) + 1
    for r in range(inst.R):
        for s in range(inst.shifts):
            if st.cover[r, s] != -1 or not occ[r, s // spd]:
                continue
            avail = inst.available_nurses[s]
            if avail.size:
                n = int(min(avail, key=lambda nn: load.get((int(nn), s), 0)))
                st.cover[r, s] = n
                load[(n, s)] = load.get((n, s), 0) + 1


def lns_improve(inst: Instance, state: SolutionState, rounds: int = 6,
                seed: int = 0, batch: int = 6, det_time: float = 8.0,
                nra_work: float = 120.0, target_free: int = 110):
    """propose/repair/accept loop, monotone on the full objective.

    Fixed round count, not a wall-clock budget, so it's deterministic on a given OR-Tools
    version. Per round: explore a batch of CP-SAT moves accepting on the patient-side cost
    the LNS owns, then re-optimise nurses exactly (NRA-MIP). Keep the round only if the full
    validated objective strictly improves, else revert to incumbent. Catches
    capacity-creating moves whose payoff only shows up after the nurse re-solve. Returns
    ``(best_state, best_cost)``.
    """
    from .exact_nra import optimize_nra

    best = state.copy()
    best_cost = int(evaluate(inst, best).total_cost)
    work = best.copy()
    for rnd in range(rounds):
        ps_work = patient_side_cost(inst, evaluate(inst, work))
        for step in range(batch):
            cand, _status = cpsat_lns_step(inst, work, seed=seed + rnd * 97 + step,
                                           det_time=det_time, target_free=target_free)
            if cand is None:
                continue
            c = evaluate(inst, cand)
            if c.total_violations == 0:
                ps = patient_side_cost(inst, c)
                if ps < ps_work:
                    work, ps_work = cand, ps
        optimize_nra(inst, work, work_limit=nra_work)  # exact nurse repair, deterministic
        f = evaluate(inst, work)
        if f.total_violations == 0 and f.total_cost < best_cost:
            best, best_cost = work.copy(), int(f.total_cost)   # accept on full objective
        work = best.copy()                             # restart from incumbent
    return best, best_cost


def cpsat_lns_step(inst: Instance, state: SolutionState, seed: int = 0,
                   det_time: float = 8.0, target_free: int = 150,
                   workers: int = 1, free_all: bool = False):
    """one CP-SAT LNS step. Returns ``(new_state, status_name)`` or ``(None, status)``.

    ``new_state`` is a fresh :class:`SolutionState`: re-optimised placement plus a greedy
    nurse cover. Caller evaluates and accepts only on strict improvement. ``free_all=True``
    re-solves the whole placement, doubles as diversification and a feasibility fallback.
    """
    import random
    rng = random.Random(seed)
    P, R, T, U, days = inst.P, inst.R, inst.T, inst.U, inst.days
    spd = inst.shifts_per_day

    adm, room, ot = state.adm, state.room, state.ot
    unsched = [p for p in range(P) if adm[p] == -1]

    # pick the freed set
    if free_all:
        freed = set(range(P))
    else:
        width = max(2, min(days, round(target_free * days / max(1, P))))
        start = rng.randrange(0, days - width + 1)
        window = range(start, start + width)
        freed = set(unsched) | {p for p in range(P) if adm[p] in window}
        if len(freed) > target_free:
            # never drop an unscheduled optional, and keep all freed mandatory.
            # only the already-admitted window optionals get sampled down.
            keep = set(unsched) | {p for p in freed if inst.p_mandatory[p]}
            rest = [p for p in freed if p not in keep]
            rng.shuffle(rest)
            freed = keep | set(rest[:max(0, target_free - len(keep))])
    frozen = [p for p in range(P) if p not in freed]
    freed = sorted(freed)

    # residual capacities + fixed indicators from occupants and frozen patients
    room_used = [[int(inst.occ_room_day_count[r, d]) for d in range(days)] for r in range(R)]
    room_gender = [[None] * days for _ in range(R)]
    for r in range(R):
        for d in range(days):
            if inst.occ_room_day_a[r, d] > 0:
                room_gender[r][d] = GENDER_A
            elif inst.occ_room_day_b[r, d] > 0:
                room_gender[r][d] = 1 - GENDER_A
    surg_used = [[0] * days for _ in range(U)]
    ot_used = [[0] * days for _ in range(T)]
    ot_open_fixed = [[False] * days for _ in range(T)]
    surg_ot_fixed = set()
    for p in frozen:
        if adm[p] == -1:
            continue
        d0, r, t = int(adm[p]), int(room[p]), int(ot[p])
        g = int(inst.p_gender[p])
        for d in range(d0, min(days, d0 + int(inst.p_los[p]))):
            room_used[r][d] += 1
            if room_gender[r][d] is None:
                room_gender[r][d] = g
        surg_used[int(inst.p_surgeon[p])][d0] += int(inst.p_duration[p])
        ot_used[t][d0] += int(inst.p_duration[p])
        ot_open_fixed[t][d0] = True
        surg_ot_fixed.add((int(inst.p_surgeon[p]), t, d0))

    # freed patients sitting on a surgeon-day: leaving gives those minutes back
    freed_surg_day = {(int(inst.p_surgeon[p]), int(adm[p])) for p in freed if adm[p] != -1}

    m = cp_model.CpModel()
    x, y, a = {}, {}, {}                 # x[p,d] day, y[p,r] room, a[p,t,d] theatre

    for p in freed:
        u, dur = int(inst.p_surgeon[p]), int(inst.p_duration[p])
        days_p = []
        for d in range(int(inst.p_release[p]), int(inst.p_last[p]) + 1):
            if inst.surgeon_max_time[u, d] - surg_used[u][d] < dur and (u, d) not in freed_surg_day:
                continue
            days_p.append(d)
        if not days_p:
            if inst.p_mandatory[p]:
                return None, "PRUNED_INFEASIBLE"
            continue
        for d in days_p:
            x[p, d] = m.NewBoolVar(f"x{p}_{d}")
            ots = [t for t in range(T)
                   if inst.ot_availability[t, d] - ot_used[t][d] >= dur or ot_open_fixed[t][d]]
            for t in ots:
                a[p, t, d] = m.NewBoolVar(f"a{p}_{t}_{d}")
            m.Add(sum(a[p, t, d] for t in ots) == x[p, d])
        rooms_p = [r for r in range(R) if not inst.p_incompatible[p, r]]
        for r in rooms_p:
            y[p, r] = m.NewBoolVar(f"y{p}_{r}")
        sched = sum(x[p, d] for d in days_p)
        m.Add(sched == 1) if inst.p_mandatory[p] else m.Add(sched <= 1)
        m.Add(sum(y[p, r] for r in rooms_p) == sched)

    pat_days_x = {p: [d for d in range(int(inst.p_release[p]), int(inst.p_last[p]) + 1)
                      if (p, d) in x] for p in freed}

    # presence literals w[p,r,d] on room-days that can take p; impossible combos turn into cuts
    w = {}
    for p in freed:
        los, g = int(inst.p_los[p]), int(inst.p_gender[p])
        rooms_p = [r for r in range(R) if (p, r) in y]
        pres_days = sorted({d for d0 in pat_days_x[p] for d in range(d0, min(days, d0 + los))})
        for r in rooms_p:
            for d in pres_days:
                pres_terms = [x[p, d0] for d0 in pat_days_x[p] if d0 <= d < d0 + los]
                if not pres_terms:
                    continue
                blocked = (inst.room_capacity[r] - room_used[r][d] <= 0
                           or (room_gender[r][d] is not None and room_gender[r][d] != g))
                if blocked:
                    m.Add(y[p, r] + sum(pres_terms) <= 1)
                else:
                    w[p, r, d] = m.NewBoolVar(f"w{p}_{r}_{d}")
                    m.Add(w[p, r, d] <= y[p, r])
                    m.Add(w[p, r, d] <= sum(pres_terms))
                    m.Add(w[p, r, d] >= y[p, r] + sum(pres_terms) - 1)

    # residual room capacity + gender exclusivity
    for r in range(R):
        for d in range(days):
            here = [p for p in freed if (p, r, d) in w]
            if not here:
                continue
            cap = int(inst.room_capacity[r]) - room_used[r][d]
            m.Add(sum(w[p, r, d] for p in here) <= max(0, cap))
            fg = room_gender[r][d]
            if fg is not None:
                for p in here:
                    if int(inst.p_gender[p]) != fg:
                        m.Add(w[p, r, d] == 0)
            else:
                ga = [p for p in here if int(inst.p_gender[p]) == GENDER_A]
                gb = [p for p in here if int(inst.p_gender[p]) != GENDER_A]
                if ga and gb:
                    gv = m.NewBoolVar(f"g{r}_{d}")
                    for p in ga:
                        m.Add(w[p, r, d] == 0).OnlyEnforceIf(gv.Not())
                    for p in gb:
                        m.Add(w[p, r, d] == 0).OnlyEnforceIf(gv)

    # residual surgeon daily capacity
    for u in range(U):
        for d in range(days):
            terms = [(int(inst.p_duration[p]), x[p, d]) for p in freed
                     if int(inst.p_surgeon[p]) == u and (p, d) in x]
            if terms:
                m.Add(sum(dur * v for dur, v in terms) <= int(inst.surgeon_max_time[u, d]) - surg_used[u][d])

    # residual theatre capacity, open-OT indicators, surgeon-transfer indicators
    open_terms, transfer_terms = [], []
    for t in range(T):
        for d in range(days):
            terms = [(int(inst.p_duration[p]), a[p, t, d]) for p in freed if (p, t, d) in a]
            if terms:
                m.Add(sum(dur * v for dur, v in terms) <= int(inst.ot_availability[t, d]) - ot_used[t][d])
                if not ot_open_fixed[t][d]:
                    ov = m.NewBoolVar(f"o{t}_{d}")
                    for _, v in terms:
                        m.AddImplication(v, ov)
                    open_terms.append(ov)
    for u in range(U):
        for d in range(days):
            cnt = []
            for t in range(T):
                terms = [a[p, t, d] for p in freed if int(inst.p_surgeon[p]) == u and (p, t, d) in a]
                if (u, t, d) in surg_ot_fixed:
                    cnt.append(None)                     # frozen surgery already opened (u,t,d)
                elif terms:
                    mv = m.NewBoolVar(f"mm{u}_{t}_{d}")
                    for v in terms:
                        m.AddImplication(v, mv)
                    cnt.append(mv)
            fixed_cnt = sum(1 for c in cnt if c is None)
            var_cnt = [c for c in cnt if c is not None]
            if var_cnt:
                tv = m.NewIntVar(0, T, f"tr{u}_{d}")
                m.Add(tv >= sum(var_cnt) + fixed_cnt - 1)
                transfer_terms.append(tv)

    # objective: patient-side only (S7 delay, S8 unscheduled, S5 open, S6 transfer)
    w_delay, w_unsched = int(inst.weights[6]), int(inst.weights[7])
    w_open, w_transfer = int(inst.weights[4]), int(inst.weights[5])
    obj = []
    for p in freed:
        for d in pat_days_x[p]:
            c = w_delay * (d - int(inst.p_release[p]))
            if c:
                obj.append(c * x[p, d])
        if not inst.p_mandatory[p]:
            obj.append(w_unsched * (1 - sum(x[p, d] for d in pat_days_x[p])))
    obj.append(w_open * sum(open_terms))
    obj.append(w_transfer * sum(transfer_terms))
    m.Minimize(sum(obj))

    for p in freed:
        if adm[p] != -1:
            if (p, int(adm[p])) in x:
                m.AddHint(x[p, int(adm[p])], 1)
            if (p, int(room[p])) in y:
                m.AddHint(y[p, int(room[p])], 1)
            if (p, int(ot[p]), int(adm[p])) in a:
                m.AddHint(a[p, int(ot[p]), int(adm[p])], 1)

    solver = cp_model.CpSolver()
    configure_cpsat(solver, det_time=det_time, seed=seed, workers=workers)  # deterministic
    status = solver.Solve(m)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None, solver.StatusName(status)

    # write the new placement onto a fresh state, then re-solve NRA
    new = state.copy()
    for p in freed:
        new.adm[p] = new.room[p] = new.ot[p] = -1
    for (p, d), var in x.items():
        if solver.Value(var):
            new.adm[p] = d
    for (p, r), var in y.items():
        if solver.Value(var) and new.adm[p] != -1:
            new.room[p] = r
    for (p, t, d), var in a.items():
        if solver.Value(var) and new.adm[p] == d:
            new.ot[p] = t
    _patch_nurses(inst, new)     
    return new, solver.StatusName(status)

"""Exact admission MILP for the dominant objective term.

Unscheduled optional patients dominate the objective (weight 300 to 500 each), and heuristic
admission plateaus below the reference. What binds is surgeon-day and theatre-day capacity, a
bin-packing that local search chokes on but a MILP handles fine. Splitting off a MILP
admission core is the standard decomposition here 
(see the report)

Decides only the admission day per patient (surgery that day). That's where the admission vs
capacity trade-off lives. Rooms get assigned later by the heuristic layer; ALNS and the exact
NRA/OT polish then refine everything. So this is one piece of the matheuristic, not the whole
thing.

Model:
``x[p,d] in {0,1}`` : p admitted on day d (feasible (p,d) only).
    * ``sum_d x[p,d] == 1`` mandatory p (H5);  ``<= 1`` optional p.
    * surgeon-day capacity (H3), per surgeon u, day δ:
          sum_{p: surgeon u} x[p,δ] * duration_p  <=  surgeon_max_time[u, δ].
    * theatre-day capacity (H4, summed over theatres), per day δ:
          sum_p x[p,δ] * duration_p  <=  sum_t availability[t, δ].
    * bed capacity (H7, summed over rooms), per day δ:
          occupants_present[δ] + sum_{p,d : d<=δ<d+los_p} x[p,d]  <=  total_beds.
Objective: minimise  w_unsched * (#optional not admitted)  +  w_delay * total delay.

Room and gender feasibility (H1, H2) relaxed here, left to the heuristic room assignment. So
the MILP is an optimistic target: the heuristic realises as much as the exact room packing
allows, ALNS mops up the rest.
"""

from __future__ import annotations

import numpy as np

from .io_instance import Instance
from .layout import Layout
from .model import SolutionState
from .solvers import configure_gurobi, gurobi_env


def solve_admission(inst: Instance, work_limit: float = 60.0,
                    warm: SolutionState | None = None) -> dict[int, int]:
    """{patient: admission_day} for the patients the MILP admits."""
    import gurobipy as gp
    from gurobipy import GRB

    env = gurobi_env()
    m = gp.Model(env=env)
    configure_gurobi(m, work_limit=work_limit, mip_gap=0.0)   # deterministic

    days = inst.days
    total_beds = int(inst.room_capacity.sum())
    ot_day_cap = inst.ot_availability.sum(axis=0)
    max_ot_day = inst.ot_availability.max(axis=0)          # one surgery has to fit in a single theatre
    occ_present = inst.occ_room_day_count.sum(axis=0)

    x: dict[tuple[int, int], gp.Var] = {}
    feas_days: dict[int, list[int]] = {}
    for p in range(inst.P):
        u = int(inst.p_surgeon[p]); dur = int(inst.p_duration[p])
        ds = []
        for d in range(int(inst.p_release[p]), int(inst.p_last[p]) + 1):
            if int(inst.surgeon_max_time[u, d]) >= dur and int(max_ot_day[d]) >= dur:
                x[(p, d)] = m.addVar(vtype=GRB.BINARY)
                ds.append(d)
        feas_days[p] = ds

    # H5: mandatory once, optional at most once
    for p in range(inst.P):
        ds = feas_days[p]
        if not ds:
            continue
        expr = gp.quicksum(x[(p, d)] for d in ds)
        if inst.p_mandatory[p]:
            m.addConstr(expr == 1)
        else:
            m.addConstr(expr <= 1)

    # surgeon-day capacity (H3)
    for u in range(inst.U):
        for d in range(days):
            terms = [int(inst.p_duration[p]) * x[(p, d)]
                     for p in range(inst.P)
                     if int(inst.p_surgeon[p]) == u and (p, d) in x]
            if terms:
                m.addConstr(gp.quicksum(terms) <= int(inst.surgeon_max_time[u, d]))

    # theatre-day aggregate capacity (H4)
    for d in range(days):
        terms = [int(inst.p_duration[p]) * x[(p, d)] for p in range(inst.P) if (p, d) in x]
        if terms:
            m.addConstr(gp.quicksum(terms) <= int(ot_day_cap[d]))

    # bed aggregate capacity (H7): occupants + admitted patients present on day delta
    for delta in range(days):
        terms = []
        for p in range(inst.P):
            los = int(inst.p_los[p])
            for d in feas_days[p]:
                if d <= delta < d + los:
                    terms.append(x[(p, d)])
        if terms:
            m.addConstr(gp.quicksum(terms) <= total_beds - int(occ_present[delta]))

    # objective: unscheduled-optional penalty + admission delay
    w_uns = int(inst.weights[7]); w_delay = int(inst.weights[6])
    unsched = gp.quicksum((1 - gp.quicksum(x[(p, d)] for d in feas_days[p]))
                          for p in range(inst.P)
                          if not inst.p_mandatory[p] and feas_days[p])
    delay = gp.quicksum((d - int(inst.p_release[p])) * x[(p, d)]
                        for (p, d) in x)
    m.setObjective(w_uns * unsched + w_delay * delay, GRB.MINIMIZE)

    # warm start off an existing solution's admission days
    if warm is not None:
        for (p, d), var in x.items():
            var.Start = 1.0 if int(warm.adm[p]) == d else 0.0

    m.optimize()
    if m.SolCount == 0:
        return {}
    return {p: d for (p, d), var in x.items() if var.X > 0.5}


def solve_pas_full(inst: Instance, work_limit: float = 120.0,
                   warm: SolutionState | None = None, mip_gap: float = 0.0,
                   opt_window: int | None = None) -> dict[int, tuple[int, int]]:
    """Full PAS MILP: pick admission day and room per patient, exactly.

    :func:`solve_admission` plus room assignment and exact gender, capacity, compatibility, so
    the plan is directly realisable, no room-packing shortfall. Gender and capacity go through
    two constraints per room-day, keyed on a gender indicator ``g[r,delta]``:

        A_present(r,delta) <= capacity_r * (1 - g[r,delta])
        B_present(r,delta) <= capacity_r *      g[r,delta]

    So a room-day holds at most ``capacity_r`` patients, all one gender. Occupants pin ``g`` on
    the days they're present. Returns ``{patient: (room, day)}``.
    """
    import gurobipy as gp
    from gurobipy import GRB
    from .io_instance import GENDER_A

    env = gurobi_env()
    m = gp.Model(env=env)
    configure_gurobi(m, work_limit=work_limit, mip_gap=mip_gap)   # deterministic

    days = inst.days
    ot_day_cap = inst.ot_availability.sum(axis=0)
    max_ot_day = inst.ot_availability.max(axis=0)

    # x[p,r,d]: admit p to room r on day d; feasible triples only
    x: dict[tuple[int, int, int], gp.Var] = {}
    triples: dict[int, list[tuple[int, int]]] = {p: [] for p in range(inst.P)}
    for p in range(inst.P):
        u = int(inst.p_surgeon[p]); dur = int(inst.p_duration[p])
        last = int(inst.p_last[p])
        # clip the wide optional window so the model stays small on large instances.
        # mandatory windows are already tight (bounded by the due date)
        if opt_window is not None and not inst.p_mandatory[p]:
            last = min(last, int(inst.p_release[p]) + opt_window)
        for d in range(int(inst.p_release[p]), last + 1):
            if int(inst.surgeon_max_time[u, d]) < dur or int(max_ot_day[d]) < dur:
                continue
            for r in range(inst.R):
                if inst.p_incompatible[p, r]:
                    continue
                x[(p, r, d)] = m.addVar(vtype=GRB.BINARY)
                triples[p].append((r, d))

    # H5: mandatory once, optional at most once
    for p in range(inst.P):
        if not triples[p]:
            continue
        expr = gp.quicksum(x[(p, r, d)] for (r, d) in triples[p])
        m.addConstr(expr == 1 if inst.p_mandatory[p] else expr <= 1)

    # per room-day gender indicator + capacity; occupants pin the gender
    g = {(r, delta): m.addVar(vtype=GRB.BINARY) for r in range(inst.R) for delta in range(days)}
    # index which (p,r,d) occupy (r, delta)
    occupy: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
    for (p, r, d) in x:
        los = int(inst.p_los[p])
        for delta in range(d, min(days, d + los)):
            occupy.setdefault((r, delta), []).append((p, r, d))
    for r in range(inst.R):
        cap = int(inst.room_capacity[r])
        for delta in range(days):
            occ_a = int(inst.occ_room_day_a[r, delta])
            occ_b = int(inst.occ_room_day_b[r, delta])
            a_terms = [x[key] for key in occupy.get((r, delta), []) if inst.p_gender[key[0]] == GENDER_A]
            b_terms = [x[key] for key in occupy.get((r, delta), []) if inst.p_gender[key[0]] != GENDER_A]
            if a_terms or occ_a:
                m.addConstr(gp.quicksum(a_terms) + occ_a <= cap * (1 - g[(r, delta)]))
            if b_terms or occ_b:
                m.addConstr(gp.quicksum(b_terms) + occ_b <= cap * g[(r, delta)])

    # surgeon-day (H3), theatre-day aggregate (H4)
    for u in range(inst.U):
        for d in range(days):
            terms = [int(inst.p_duration[p]) * x[(p, r, d)]
                     for (p, r, dd) in x if dd == d and int(inst.p_surgeon[p]) == u]
            if terms:
                m.addConstr(gp.quicksum(terms) <= int(inst.surgeon_max_time[u, d]))
    for d in range(days):
        terms = [int(inst.p_duration[p]) * x[(p, r, d)] for (p, r, dd) in x if dd == d]
        if terms:
            m.addConstr(gp.quicksum(terms) <= int(ot_day_cap[d]))

    # objective: unscheduled-optional penalty + delay
    w_uns = int(inst.weights[7]); w_delay = int(inst.weights[6])
    unsched = gp.quicksum((1 - gp.quicksum(x[(p, r, d)] for (r, d) in triples[p]))
                          for p in range(inst.P) if not inst.p_mandatory[p] and triples[p])
    delay = gp.quicksum((d - int(inst.p_release[p])) * x[(p, r, d)] for (p, r, d) in x)
    m.setObjective(w_uns * unsched + w_delay * delay, GRB.MINIMIZE)

    if warm is not None:
        for (p, r, d), var in x.items():
            var.Start = 1.0 if (int(warm.adm[p]) == d and int(warm.room[p]) == r) else 0.0

    m.optimize()
    if m.SolCount == 0:
        return {}
    return {p: (r, d) for (p, r, d), var in x.items() if var.X > 0.5}


def realize_pas(inst: Instance, plan: dict[int, tuple[int, int]]) -> Layout:
    """Turn a full-PAS plan ``{p:(room,day)}`` into a Layout, picking theatres."""
    lay = Layout(inst)
    order = sorted(plan.keys(), key=lambda p: (int(plan[p][1]), -int(inst.p_duration[p])))
    for p in order:
        r, d = plan[p]
        t, _ = lay.best_ot(p, d)
        if t != -1 and lay.surgeon_ok(p, d) and lay.ot_ok(p, d, t) and lay.room_ok(p, r, d):
            lay.place(p, d, r, t)
        else:
            _c, move = lay.best_placement(p)
            if move is not None:
                lay.place(p, *move)
    return lay


def realize_admission(inst: Instance, admit_days: dict[int, int]) -> Layout:
    """MILP admission-day plan -> feasible :class:`Layout`, assigning rooms and OTs.

    Placed hardest-first at or near the planned day. If that day has no feasible room (gender
    or capacity), fall back to the cheapest feasible placement in the patient's window. Failing
    that, leave the patient for ALNS."""
    lay = Layout(inst)
    # hardest-first: mandatory, then tight windows, then long surgeries
    order = sorted(admit_days.keys(),
                   key=lambda p: (0 if inst.p_mandatory[p] else 1,
                                  int(inst.p_last[p]) - int(inst.p_release[p]),
                                  -int(inst.p_duration[p])))
    for p in order:
        d = admit_days[p]
        r, _ = lay.best_room(p, d)
        t, _ = lay.best_ot(p, d)
        if r != -1 and t != -1 and lay.surgeon_ok(p, d) and lay.ot_ok(p, d, t):
            lay.place(p, d, r, t)
        else:
            _c, move = lay.best_placement(p)      # fall back to any feasible day in the window
            if move is not None:
                lay.place(p, *move)
    return lay

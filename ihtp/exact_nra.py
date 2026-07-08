"""Exact post-processing (b): nurse-to-room assignment (NRA) re-optimisation.

Patient layout is fixed here (admission days, rooms, theatres), so the nurse layer splits
off as its own problem. One working nurse per occupied room-shift, minimising
w2*S2 (skill) + w3*S3 (continuity) + w4*S4 (workload). H8: each occupied room-shift covered
by exactly one on-duty nurse.

Greedy NRA is myopic, fixes each shift in turn. Solve the whole horizon at once and the
solver can trade skill/workload against continuity globally. gap analysis
showed real S2/S3/S4 left on the table after the heuristic.

Model:
x[r,s,n] in {0,1}: nurse n (on duty at s) covers room r at shift s, exactly one per
occupied room-shift.
z[n,s] >= 0: workload overflow of nurse n at shift s (S4).
y[e,n] in {0,1}: nurse n touches entity e at least once during its stay. S3 = sum y[e,n],
linked by y[e,n] >= x[r_e,s,n].

skill_gap and wlreq are precomputed constants so S2 and workload stay linear in x. Warm
start comes from greedy NRA. Big instances get a deterministic WorkLimit (not wall-clock).
Worst case returns the best incumbent inside the budget, never worse than the greedy start.
"""

from __future__ import annotations

import numpy as np

from .io_instance import Instance
from .model import SolutionState
from .nra import greedy_nra
from .objective import _build_occupancy
from .solvers import configure_gurobi, gurobi_env


def optimize_nra(inst: Instance, st: SolutionState, work_limit: float = 200.0,
                 warm_start: bool = True) -> None:
    """Re-solve NRA for the fixed layout in st, write result to st.cover."""
    import gurobipy as gp
    from gurobipy import GRB

    spd, P = inst.shifts_per_day, inst.P
    w_skill = int(inst.weights[1]); w_cont = int(inst.weights[2]); w_work = int(inst.weights[3])

    if warm_start:
        greedy_nra(inst, st)                    
    warm_cover = st.cover.copy()

    room_day_entities, _, _ = _build_occupancy(inst, st)

    # per occupied (room, shift): who's present, total workload demand
    occ_rs: dict[tuple[int, int], list[int]] = {}
    wlreq: dict[tuple[int, int], int] = {}
    for r in range(inst.R):
        for s in range(inst.shifts):
            d = s // spd
            ents = room_day_entities[r][d]
            if not ents:
                continue
            load = 0
            for e in ents:
                if e < P:
                    load += int(inst.p_workload[e][s - int(st.adm[e]) * spd])
                else:
                    load += int(inst.o_workload[e - P][s])
            occ_rs[(r, s)] = ents
            wlreq[(r, s)] = load

    env = gurobi_env()
    m = gp.Model(env=env)
    configure_gurobi(m, work_limit=work_limit)  # deterministic run: WorkLimit, fixed Seed/Threads
    m.setParam("MIPFocus", 1)                   # chase feasible solutions over the bound

    # x[r,s,n]: one per occupied (r,s) crossed with the nurses on duty at s
    x: dict[tuple[int, int, int], gp.Var] = {}
    for (r, s), ents in occ_rs.items():
        nurses = inst.available_nurses[s]
        for n in nurses:
            x[(r, s, int(n))] = m.addVar(vtype=GRB.BINARY)

    # H8: exactly one nurse per occupied room-shift
    for (r, s), ents in occ_rs.items():
        m.addConstr(gp.quicksum(x[(r, s, int(n))] for n in inst.available_nurses[s]) == 1)

    # S2 skill cost: gap is a constant per (r,s,n), so the term is linear
    skill_terms = []
    for (r, s), ents in occ_rs.items():
        for n in inst.available_nurses[s]:
            n = int(n)
            skill_n = int(inst.nurse_skill[n])
            gap = 0
            for e in ents:
                if e < P:
                    req = int(inst.p_skillreq[e][s - int(st.adm[e]) * spd])
                else:
                    req = int(inst.o_skillreq[e - P][s])
                if req > skill_n:
                    gap += req - skill_n
            if gap:
                skill_terms.append(gap * x[(r, s, n)])

    # S4: z[n,s] catches workload over the nurse's cap
    z = {}
    nurse_shift_rooms: dict[tuple[int, int], list[int]] = {}
    for (r, s) in occ_rs:
        for n in inst.available_nurses[s]:
            nurse_shift_rooms.setdefault((int(n), s), []).append(r)
    for (n, s), rooms in nurse_shift_rooms.items():
        z[(n, s)] = m.addVar(lb=0.0)
        m.addConstr(z[(n, s)] >= gp.quicksum(wlreq[(r, s)] * x[(r, s, n)] for r in rooms)
                    - int(inst.nurse_max_load[n, s]))

    # S3 continuity: y[e,n] >= x[room_e, s, n] over every shift of e's stay
    y = {}
    # patients
    for p in np.nonzero(st.adm != -1)[0]:
        p = int(p); r = int(st.room[p]); ad = int(st.adm[p])
        last = min(inst.days, ad + int(inst.p_los[p]))
        for d in range(ad, last):
            for s in range(d * spd, (d + 1) * spd):
                if (r, s) not in occ_rs:
                    continue
                for n in inst.available_nurses[s]:
                    n = int(n)
                    key = (p, n)
                    if key not in y:
                        y[key] = m.addVar(vtype=GRB.BINARY)
                    m.addConstr(y[key] >= x[(r, s, n)])
    # occupants
    for o in range(inst.O):
        e = P + o; r = int(inst.o_room[o])
        for d in range(min(int(inst.o_los[o]), inst.days)):
            for s in range(d * spd, (d + 1) * spd):
                if (r, s) not in occ_rs:
                    continue
                for n in inst.available_nurses[s]:
                    n = int(n)
                    key = (e, n)
                    if key not in y:
                        y[key] = m.addVar(vtype=GRB.BINARY)
                    m.addConstr(y[key] >= x[(r, s, n)])

    m.setObjective(w_skill * gp.quicksum(skill_terms)
                   + w_work * gp.quicksum(z.values())
                   + w_cont * gp.quicksum(y.values()), GRB.MINIMIZE)

    # seed x from the greedy cover
    for (r, s, n), var in x.items():
        var.Start = 1.0 if int(warm_cover[r, s]) == n else 0.0

    m.optimize()
    if m.SolCount == 0:
        st.cover = warm_cover                   # nothing found, fall back to greedy
        return

    st.cover.fill(-1)
    for (r, s, n), var in x.items():
        if var.X > 0.5:
            st.cover[r, s] = n

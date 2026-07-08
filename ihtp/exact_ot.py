"""Exact post-pass: re-assign operating theatres and surgeons per day.

Admission days and rooms are frozen, so S1/S7/S8 and the hard room/surgeon
constraints can't move. Only free variable left in the SCP layer: which theatre
each surgery lands in. Re-optimising that hits S5 (OpenOperatingTheater) + S6
(SurgeonTransfer) under H4 (theatre daily minute capacity).

Decomposes per day, a day-d surgery only touches other day-d surgeries. Each day
is a tiny assignment MILP, optimal in ms. gap analysis showed the
heuristic leaves S5/S6 on the table, and this is the cheapest exact way to grab it.

"""

from __future__ import annotations

import numpy as np

from .io_instance import Instance
from .model import SolutionState
from .solvers import configure_gurobi, gurobi_env


def optimize_ot(inst: Instance, st: SolutionState, per_day_work: float = 30.0) -> int:
    """Day-by-day theatre re-assignment to cut S5+S6. Mutates st.ot.
    Per-day models are tiny, solved deterministically via WorkLimit."""
    import gurobipy as gp
    from gurobipy import GRB

    env = gurobi_env()
    w_open = int(inst.weights[4])
    w_transfer = int(inst.weights[5])
    spd = inst.shifts_per_day  # noqa: F841 (indexed by day directly, spd unused)

    total_gain = 0
    adm, ot = st.adm, st.ot
    for d in range(inst.days):
        P_d = [int(p) for p in np.nonzero(adm == d)[0]]        # day-d surgeries
        if not P_d:
            continue
        # theatres open on day d
        Ts = [t for t in range(inst.T) if int(inst.ot_availability[t, d]) > 0]
        if not Ts:
            continue
        surgeons_d = sorted({int(inst.p_surgeon[p]) for p in P_d})

        m = gp.Model(env=env)
        configure_gurobi(m, work_limit=per_day_work)   # fixed work budget -> reproducible
        # a[p,t]: surgery p in theatre t
        a = {(p, t): m.addVar(vtype=GRB.BINARY) for p in P_d for t in Ts}
        o = {t: m.addVar(vtype=GRB.BINARY) for t in Ts}                       # t is open
        g = {(u, t): m.addVar(vtype=GRB.BINARY) for u in surgeons_d for t in Ts}
        tr = {u: m.addVar(lb=0.0) for u in surgeons_d}                        # surgeon u's extra OTs

        for p in P_d:
            m.addConstr(gp.quicksum(a[p, t] for t in Ts) == 1)               # exactly one theatre
        for t in Ts:
            # H4 capacity
            m.addConstr(gp.quicksum(int(inst.p_duration[p]) * a[p, t] for p in P_d)
                        <= int(inst.ot_availability[t, d]))
            for p in P_d:
                m.addConstr(o[t] >= a[p, t])                                  # used implies open
                m.addConstr(g[int(inst.p_surgeon[p]), t] >= a[p, t])         # flag surgeon in t
        for u in surgeons_d:
            m.addConstr(tr[u] >= gp.quicksum(g[u, t] for t in Ts) - 1)       # every OT past the first is a transfer

        m.setObjective(w_open * gp.quicksum(o[t] for t in Ts)
                       + w_transfer * gp.quicksum(tr[u] for u in surgeons_d), GRB.MINIMIZE)

        # warm start off the current assignment
        for p in P_d:
            cur = int(ot[p])
            for t in Ts:
                a[p, t].Start = 1.0 if t == cur else 0.0

        m.optimize()
        if m.SolCount == 0:
            continue
        for p in P_d:
            for t in Ts:
                if a[p, t].X > 0.5:
                    ot[p] = t
                    break

    return total_gain  # always 0; caller re-scores for the real gain, return stays for signature parity

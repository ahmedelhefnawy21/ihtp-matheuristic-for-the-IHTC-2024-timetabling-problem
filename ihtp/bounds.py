"""Certified lower bounds, so the gap is to the true optimum, and not to best known alone.

Competition best-known values are upper bounds the field found, not proven optima. Gap to
them proves nothing about distance to optimal. So compute a valid lower bound on the true
optimum and the gap becomes a certificate.

Relaxation keeps admission plus the two binding capacity resources (surgeon-day,
theatre-day), the two patient-side soft costs those resources drive (S5 open OTs, S6 surgeon
transfers), S7 delay and S8 unscheduled. Drops room/gender assignment (H1, H2, H7) and the
whole nurse layer (S2 to S4).

Dropping constraints only lowers the optimum; dropped soft terms are all >= 0. So
optimum(full) >= optimum(relaxation), and Gurobi's proven dual bound here is a valid lower
bound on the full optimum. Loose (ignores S1 to S4 and room packing) but a certificate: true
optimum can't sit below it.
"""

from __future__ import annotations

import numpy as np

from .io_instance import Instance
from .solvers import configure_gurobi, gurobi_env


def patient_side_lower_bound(inst: Instance, work_limit: float = 120.0) -> dict:
    """Valid lower bound on the full optimum via the surgeon/OT relaxation.

    Returns proven lower_bound, best relaxation solution relax_obj, and status.
    """
    import gurobipy as gp
    from gurobipy import GRB

    env = gurobi_env()
    m = gp.Model(env=env)
    configure_gurobi(m, work_limit=work_limit)   # WorkLimit not TimeLimit, so the bound reproduces across machines
    days = inst.days
    max_ot_day = inst.ot_availability.max(axis=0)

    # x[p,d]: admit p on day d. a[p,t,d]: its theatre. only feasible triples get vars
    x, a = {}, {}
    feas_days = {}
    for p in range(inst.P):
        u, dur = int(inst.p_surgeon[p]), int(inst.p_duration[p])
        ds = []
        for d in range(int(inst.p_release[p]), int(inst.p_last[p]) + 1):
            if int(inst.surgeon_max_time[u, d]) >= dur and int(max_ot_day[d]) >= dur:
                x[p, d] = m.addVar(vtype=GRB.BINARY)
                for t in range(inst.T):
                    if int(inst.ot_availability[t, d]) >= dur:
                        a[p, t, d] = m.addVar(vtype=GRB.BINARY)
                m.addConstr(gp.quicksum(a[p, t, d] for t in range(inst.T) if (p, t, d) in a) == x[p, d])
                ds.append(d)
        feas_days[p] = ds
        if not ds:
            continue
        expr = gp.quicksum(x[p, d] for d in ds)
        m.addConstr(expr == 1 if inst.p_mandatory[p] else expr <= 1)

    for u in range(inst.U):
        for d in range(days):
            terms = [int(inst.p_duration[p]) * x[p, d] for p in range(inst.P)
                     if int(inst.p_surgeon[p]) == u and (p, d) in x]
            if terms:
                m.addConstr(gp.quicksum(terms) <= int(inst.surgeon_max_time[u, d]))
    # S5 open-OT and S6 surgeon-transfer indicators
    o, g = {}, {}
    for t in range(inst.T):
        for d in range(days):
            terms = [(int(inst.p_duration[p]), a[p, t, d]) for p in range(inst.P) if (p, t, d) in a]
            if terms:
                m.addConstr(gp.quicksum(dur * v for dur, v in terms) <= int(inst.ot_availability[t, d]))
                o[t, d] = m.addVar(vtype=GRB.BINARY)
                for _, v in terms:
                    m.addConstr(o[t, d] >= v)
    tr = {}
    for u in range(inst.U):
        for d in range(days):
            ots = []
            for t in range(inst.T):
                terms = [a[p, t, d] for p in range(inst.P)
                         if int(inst.p_surgeon[p]) == u and (p, t, d) in a]
                if terms:
                    gv = m.addVar(vtype=GRB.BINARY)
                    for v in terms:
                        m.addConstr(gv >= v)
                    g[u, t, d] = gv
                    ots.append(gv)
            if ots:
                tr[u, d] = m.addVar(lb=0.0)
                m.addConstr(tr[u, d] >= gp.quicksum(ots) - 1)

    w_uns, w_delay = int(inst.weights[7]), int(inst.weights[6])
    w_open, w_transfer = int(inst.weights[4]), int(inst.weights[5])
    unsched = gp.quicksum((1 - gp.quicksum(x[p, d] for d in feas_days[p]))
                          for p in range(inst.P) if not inst.p_mandatory[p] and feas_days[p])
    delay = gp.quicksum((d - int(inst.p_release[p])) * x[p, d] for (p, d) in x)
    m.setObjective(w_uns * unsched + w_delay * delay
                   + w_open * gp.quicksum(o.values()) + w_transfer * gp.quicksum(tr.values()),
                   GRB.MINIMIZE)
    m.optimize()

    # need a finite proven dual bound. under WorkLimit status is OPTIMAL (tight) or WORK_LIMIT
    # (partial, still valid). proved nothing in budget -> ObjBound is the -1e100 no-bound
    # sentinel; on INFEASIBLE/UNBOUNDED it may not exist at all. emit a bound only when ObjBound
    # is finite, else lower_bound=None and the instance drops from the certified-gap table. no
    # bogus bounds. relaxation sheds the hardest constraints so a finite root dual bound almost
    # always lands inside budget anyway.
    GRB_INF = 1e30
    try:
        bound = m.ObjBound
    except Exception:
        bound = None
    lb = int(np.floor(bound)) if (bound is not None and -GRB_INF < bound < GRB_INF) else None
    relax_obj = m.ObjVal if m.SolCount > 0 else None
    return {"lower_bound": lb, "relax_obj": relax_obj, "status": int(m.Status)}

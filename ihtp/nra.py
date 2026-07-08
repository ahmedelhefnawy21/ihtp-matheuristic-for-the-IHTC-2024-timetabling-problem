"""Nurse-to-Room Assignment (NRA), lower layer.

Patient layout is fixed. Pin one working nurse to each occupied (room, shift)
so H8 holds, minimising the three nurse softs: S2 RoomSkillLevel,
S3 ContinuityOfCare, S4 ExcessiveNurseWorkload.

Greedy sweeps shifts in time order so continuity can favour a nurse the patient
already saw. Rooms within a shift go hardest first (highest required skill).
Each room picks the on-duty nurse minimising weighted marginal cost:

    w_skill * skill_gap + w_work * workload_overflow_delta + w_cont * new_distinct

new_distinct = patients in the room this nurse hasn't carried yet, i.e. the S3
the pick would add. Score uses the real instance weights, so minimising it hits
the objective. One nurse can cover several rooms in a shift; the overflow term
keeps us from dumping everything on one of them.
"""

from __future__ import annotations

import numpy as np

from .io_instance import Instance
from .model import SolutionState
from .objective import _build_occupancy


def greedy_nra(inst: Instance, st: SolutionState) -> int:
    """Fill st.cover greedily. Returns uncovered occupied room-shifts;
    0 means H8 holds."""
    spd, P = inst.shifts_per_day, inst.P
    w_skill = int(inst.weights[1]); w_cont = int(inst.weights[2]); w_work = int(inst.weights[3])

    room_day_entities, _, _ = _build_occupancy(inst, st)
    st.cover.fill(-1)

    # nurses each entity has seen; carried across shifts for continuity
    seen: list[set[int]] = [set() for _ in range(P + inst.O)]
    uncovered = 0

    for s in range(inst.shifts):
        d = s // spd
        avail = inst.available_nurses[s]
        # occupied rooms this shift + their demand: entities, workload, max req
        rooms = []
        for r in range(inst.R):
            ents = room_day_entities[r][d]
            if not ents:
                continue
            reqs, wls = [], 0
            max_req = 0
            for e in ents:
                if e < P:
                    rel = s - int(st.adm[e]) * spd
                    req = int(inst.p_skillreq[e][rel]); wl = int(inst.p_workload[e][rel])
                else:
                    o = e - P
                    req = int(inst.o_skillreq[o][s]); wl = int(inst.o_workload[o][s])
                reqs.append((e, req)); wls += wl
                if req > max_req:
                    max_req = req
            rooms.append((max_req, r, ents, reqs, wls))

        if not rooms:
            continue
        if avail.size == 0:                      # rooms occupied, nobody on duty -> uncovered
            uncovered += len(rooms)
            continue

        rooms.sort(reverse=True)                 # hardest first (max required skill)
        nload = {int(n): 0 for n in avail}       # per-nurse workload, this shift only

        for _max_req, r, ents, reqs, wls in rooms:
            best_n, best_score = -1, None
            for n in avail:
                n = int(n)
                skill_n = int(inst.nurse_skill[n])
                gap = 0
                for _e, req in reqs:
                    if req > skill_n:
                        gap += req - skill_n
                maxload = int(inst.nurse_max_load[n, s])
                before = max(0, nload[n] - maxload)
                after = max(0, nload[n] + wls - maxload)
                overflow_delta = after - before
                new_distinct = sum(1 for e in ents if n not in seen[e])
                score = w_skill * gap + w_work * overflow_delta + w_cont * new_distinct
                if best_score is None or score < best_score:
                    best_score, best_n = score, n
            st.cover[r, s] = best_n
            nload[best_n] += wls
            for e in ents:
                seen[e].add(best_n)

    return uncovered

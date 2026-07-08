"""Read/write IHTC-2024 solution JSON.

schema (matches reference solutions on the competition site)::

    {"patients": [{"id","admission_day","room","operating_theater"}, ...],
     "nurses":   [{"id","assignments":[{"day","shift","rooms":[...]}, ...]}, ...]}

admission_day is an int, or "none" for a postponed patient.
"""

from __future__ import annotations

import json

from .io_instance import Instance
from .model import SolutionState


def write_solution_dict(inst: Instance, st: SolutionState) -> dict:
    """SolutionState to official solution dict.

    emits every room-shift with a nurse (cover[r,s] != -1), rooms grouped per
    (day, shift) per schema. can include a room-shift that a preserved assignment
    left non-empty in cover after it went patient-empty, e.g. an LNS move that frees
    a room. harmless: validator ignores nurse assignments on empty room-shifts (no
    patients -> no skill/continuity/workload cost). dropping them would just shrink
    the file, same validator result.
    """
    spd = inst.shifts_per_day

    patients = []
    for p in range(inst.P):
        if st.adm[p] == -1:
            patients.append({"id": inst.patient_ids[p], "admission_day": "none"})
        else:
            patients.append({
                "id": inst.patient_ids[p],
                "admission_day": int(st.adm[p]),
                "room": inst.room_ids[int(st.room[p])],
                "operating_theater": inst.ot_ids[int(st.ot[p])],
            })

    # invert cover[r, s] -> nurse -> day -> shift -> rooms
    per_nurse: dict[int, dict[int, dict[int, list[int]]]] = {}
    R, S = inst.R, inst.shifts
    for r in range(R):
        for s in range(S):
            n = int(st.cover[r, s])
            if n == -1:
                continue
            d, sh = divmod(s, spd)
            per_nurse.setdefault(n, {}).setdefault(d, {}).setdefault(sh, []).append(r)

    nurses = []
    for n in range(inst.N):
        assignments = []
        for d in sorted(per_nurse.get(n, {})):
            for sh in sorted(per_nurse[n][d]):
                rooms = [inst.room_ids[r] for r in sorted(per_nurse[n][d][sh])]
                assignments.append({
                    "day": d,
                    "shift": inst.shift_names[sh],
                    "rooms": rooms,
                })
        nurses.append({"id": inst.nurse_ids[n], "assignments": assignments})

    return {"patients": patients, "nurses": nurses}


def write_solution(inst: Instance, st: SolutionState, path: str) -> None:
    with open(path, "w") as fh:
        json.dump(write_solution_dict(inst, st), fh, indent=1)


def read_solution(inst: Instance, path: str) -> SolutionState:
    """official solution JSON -> SolutionState.

    mainly to replay reference solutions through the evaluator (golden test).
    """
    with open(path) as fh:
        j = json.load(fh)
    return solution_from_dict(inst, j)


def solution_from_dict(inst: Instance, j: dict) -> SolutionState:
    spd = inst.shifts_per_day
    p_index = {pid: i for i, pid in enumerate(inst.patient_ids)}
    r_index = {rid: i for i, rid in enumerate(inst.room_ids)}
    t_index = {tid: i for i, tid in enumerate(inst.ot_ids)}
    n_index = {nid: i for i, nid in enumerate(inst.nurse_ids)}
    sh_index = {name: i for i, name in enumerate(inst.shift_names)}

    st = SolutionState(inst)
    for jp in j["patients"]:
        if jp.get("admission_day", "none") == "none":
            continue
        p = p_index[jp["id"]]
        st.adm[p] = int(jp["admission_day"])
        st.room[p] = r_index[jp["room"]]
        st.ot[p] = t_index[jp["operating_theater"]]

    for jn in j["nurses"]:
        n = n_index[jn["id"]]
        for a in jn.get("assignments", []):
            s = int(a["day"]) * spd + sh_index[a["shift"]]
            for rid in a["rooms"]:
                st.cover[r_index[rid], s] = n
    return st

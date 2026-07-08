"""IHTC-2024 instance JSON -> one frozen Instance, derived tables in it.

Field names, index conventions, and the 8-weight order mirror IHTP_Validator.cc
so objective.py matches the validator bit-for-bit.

Two entity kinds share room occupancy: patients (0..P-1) and occupants
(pre-admitted, global index o + P like the validator). Occupants have a fixed room
and stay, and never touch OT or surgeon decisions.

A patient's per-shift workload_produced and
skill_level_required are relative to admission (s - admission_day * shifts_per_day);
an occupant's are absolute within its stay. Raw arrays stay here; the offset math
is in objective.py and model.py.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

# weight order per IHTP_Validator.cc lines 263-270; costs[i] gets multiplied by WEIGHT_KEYS[i] downstream
WEIGHT_KEYS = [
    "room_mixed_age",           # w0 -> S1 RoomAgeMix
    "room_nurse_skill",         # w1 -> S2 RoomSkillLevel
    "continuity_of_care",       # w2 -> S3 ContinuityOfCare
    "nurse_eccessive_workload", # w3 -> S4 ExcessiveNurseWorkload  (sic: 'eccessive')
    "open_operating_theater",   # w4 -> S5 OpenOperatingTheater
    "surgeon_transfer",         # w5 -> S6 SurgeonTransfer
    "patient_delay",            # w6 -> S7 PatientDelay
    "unscheduled_optional",     # w7 -> S8 ElectiveUnscheduledPatients
]

GENDER_A, GENDER_B = 0, 1


@dataclass(frozen=True)
class Instance:
    """Parsed IHTC-2024 instance, derived tables precomputed. Read-only: state,
    evaluator, constructive, ALNS, exact models all treat it as immutable while solving.
    """

    name: str

    # global scalars
    days: int
    shifts_per_day: int            # len(shift_types), always 3: early/late/night
    shifts: int                    # = days * shifts_per_day
    skill_levels: int
    n_age_groups: int
    weights: np.ndarray            # int[8], WEIGHT_KEYS order

    # index -> id string, needed when writing the solution JSON back out
    shift_names: list[str]
    age_group_names: list[str]
    room_ids: list[str]
    ot_ids: list[str]
    surgeon_ids: list[str]
    nurse_ids: list[str]
    patient_ids: list[str]

    # rooms
    room_capacity: np.ndarray      # int[R]

    # surgeons and operating theatres
    surgeon_max_time: np.ndarray   # int[U, days] minutes; 0 means unavailable that day
    ot_availability: np.ndarray    # int[T, days] minutes; 0 means closed that day

    # nurses
    nurse_skill: np.ndarray        # int[N]
    nurse_working: np.ndarray      # bool[N, S]  nurse n on duty in shift s?
    nurse_max_load: np.ndarray     # int[N, S]   max workload in shift s, 0 if absent
    nurse_working_shifts: list[np.ndarray]   # per nurse: global shift indices worked
    available_nurses: list[np.ndarray]       # per shift s: nurse indices on duty

    # patients
    p_mandatory: np.ndarray        # bool[P]
    p_gender: np.ndarray           # int[P]  (0=A, 1=B)
    p_age: np.ndarray              # int[P]  age-group ordinal
    p_los: np.ndarray              # int[P]  length of stay (days)
    p_release: np.ndarray          # int[P]  earliest admission day
    p_due: np.ndarray              # int[P]  latest admission day if mandatory, else -1
    p_last: np.ndarray             # int[P]  last admission day: due if mandatory, else days-1
    p_duration: np.ndarray         # int[P]  surgery duration, minutes
    p_surgeon: np.ndarray          # int[P]  surgeon index
    p_incompatible: np.ndarray     # bool[P, R]  True where room is incompatible with patient
    p_workload: list[np.ndarray]   # per patient: int[los*spd] relative-indexed
    p_skillreq: list[np.ndarray]   # per patient: int[los*spd] relative-indexed

    # occupants (pre-admitted; fixed room, absolute-indexed arrays)
    o_gender: np.ndarray           # int[O]
    o_age: np.ndarray              # int[O]
    o_los: np.ndarray              # int[O]
    o_room: np.ndarray             # int[O]  fixed assigned room
    o_workload: list[np.ndarray]   # per occupant: int[los*spd] absolute-indexed
    o_skillreq: list[np.ndarray]   # per occupant: int[los*spd] absolute-indexed

    # fixed occupant contribution to room-day occupancy
    # occ_room_day_count[r, d]: occupants in room r on day d
    occ_room_day_count: np.ndarray  # int[R, days]
    occ_room_day_a: np.ndarray      # int[R, days] gender-A occupant count
    occ_room_day_b: np.ndarray      # int[R, days] gender-B occupant count
    # occ_in_room_day[r][d]: occupant indices present, for age/skill/continuity
    occ_in_room_day: list = field(default_factory=list)

    # sizes
    @property
    def R(self) -> int: return len(self.room_ids)
    @property
    def T(self) -> int: return len(self.ot_ids)
    @property
    def U(self) -> int: return len(self.surgeon_ids)
    @property
    def N(self) -> int: return len(self.nurse_ids)
    @property
    def P(self) -> int: return len(self.patient_ids)
    @property
    def O(self) -> int: return len(self.o_los)


def load_instance(path: str) -> Instance:
    """IHTC-2024 instance JSON -> Instance."""
    with open(path) as fh:
        j = json.load(fh)

    name = path.split("/")[-1].replace(".json", "")

    days = int(j["days"])
    shift_names = list(j["shift_types"])
    spd = len(shift_names)
    shifts = days * spd
    skill_levels = int(j["skill_levels"])
    age_group_names = list(j["age_groups"])
    n_age = len(age_group_names)
    age_index = {name: i for i, name in enumerate(age_group_names)}

    weights = np.array([int(j["weights"][k]) for k in WEIGHT_KEYS], dtype=np.int64)

    # parse rooms before patients; patient incompatibilities reference room ids
    room_ids = [r["id"] for r in j["rooms"]]
    room_index = {rid: i for i, rid in enumerate(room_ids)}
    room_capacity = np.array([int(r["capacity"]) for r in j["rooms"]], dtype=np.int64)
    R = len(room_ids)

    # surgeons
    surgeon_ids = [s["id"] for s in j["surgeons"]]
    surgeon_index = {sid: i for i, sid in enumerate(surgeon_ids)}
    surgeon_max_time = np.array(
        [[int(x) for x in s["max_surgery_time"]] for s in j["surgeons"]], dtype=np.int64
    )

    # operating theatres
    ot_ids = [t["id"] for t in j["operating_theaters"]]
    ot_availability = np.array(
        [[int(x) for x in t["availability"]] for t in j["operating_theaters"]], dtype=np.int64
    )

    # occupants (fixed room and stay)
    o_gender, o_age, o_los, o_room = [], [], [], []
    o_workload, o_skillreq = [], []
    for f in j["occupants"]:
        o_gender.append(GENDER_A if f["gender"] == "A" else GENDER_B)
        o_age.append(age_index[f["age_group"]])
        los = int(f["length_of_stay"])
        o_los.append(los)
        o_room.append(room_index[f["room_id"]])
        o_workload.append(np.array([int(x) for x in f["workload_produced"]], dtype=np.int64))
        o_skillreq.append(np.array([int(x) for x in f["skill_level_required"]], dtype=np.int64))
    O = len(o_los)

    # occupant room-day occupancy is fixed, it is once here
    occ_room_day_count = np.zeros((R, days), dtype=np.int64)
    occ_room_day_a = np.zeros((R, days), dtype=np.int64)
    occ_room_day_b = np.zeros((R, days), dtype=np.int64)
    occ_in_room_day = [[[] for _ in range(days)] for _ in range(R)]
    for o in range(O):
        r = o_room[o]
        # validator loops d in [0, LOS) over room_day arrays sized [R][days].
        for d in range(min(o_los[o], days)):
            occ_room_day_count[r, d] += 1
            if o_gender[o] == GENDER_A:
                occ_room_day_a[r, d] += 1
            else:
                occ_room_day_b[r, d] += 1
            occ_in_room_day[r][d].append(o)

    # patients
    patient_ids = [p["id"] for p in j["patients"]]
    P = len(patient_ids)
    p_mandatory = np.zeros(P, dtype=bool)
    p_gender = np.zeros(P, dtype=np.int64)
    p_age = np.zeros(P, dtype=np.int64)
    p_los = np.zeros(P, dtype=np.int64)
    p_release = np.zeros(P, dtype=np.int64)
    p_due = np.full(P, -1, dtype=np.int64)
    p_duration = np.zeros(P, dtype=np.int64)
    p_surgeon = np.zeros(P, dtype=np.int64)
    p_incompatible = np.zeros((P, R), dtype=bool)
    p_workload, p_skillreq = [], []
    for p, jp in enumerate(j["patients"]):
        p_mandatory[p] = bool(jp["mandatory"])
        p_gender[p] = GENDER_A if jp["gender"] == "A" else GENDER_B
        p_age[p] = age_index[jp["age_group"]]
        p_los[p] = int(jp["length_of_stay"])
        p_release[p] = int(jp["surgery_release_day"])
        if p_mandatory[p]:                      # due day only exists for mandatory patients
            p_due[p] = int(jp["surgery_due_day"])
        p_duration[p] = int(jp["surgery_duration"])
        p_surgeon[p] = surgeon_index[jp["surgeon_id"]]
        inc = jp.get("incompatible_room_ids") or []
        for rid in inc:
            p_incompatible[p, room_index[rid]] = True
        p_workload.append(np.array([int(x) for x in jp["workload_produced"]], dtype=np.int64))
        p_skillreq.append(np.array([int(x) for x in jp["skill_level_required"]], dtype=np.int64))
    # latest admission: due date if mandatory, else last horizon day
    p_last = np.where(p_mandatory, p_due, days - 1).astype(np.int64)

    # nurses
    nurse_ids = [n["id"] for n in j["nurses"]]
    N = len(nurse_ids)
    shift_index = {name: i for i, name in enumerate(shift_names)}
    nurse_skill = np.zeros(N, dtype=np.int64)
    nurse_working = np.zeros((N, shifts), dtype=bool)
    nurse_max_load = np.zeros((N, shifts), dtype=np.int64)
    nurse_working_shifts = []
    available = [[] for _ in range(shifts)]
    for n, jn in enumerate(j["nurses"]):
        nurse_skill[n] = int(jn["skill_level"])
        worked = []
        for ws in jn["working_shifts"]:
            s = int(ws["day"]) * spd + shift_index[ws["shift"]]   # flatten to global shift index
            nurse_working[n, s] = True
            nurse_max_load[n, s] = int(ws["max_load"])
            available[s].append(n)
            worked.append(s)
        nurse_working_shifts.append(np.array(sorted(worked), dtype=np.int64))
    available_nurses = [np.array(a, dtype=np.int64) for a in available]

    return Instance(
        name=name,
        days=days, shifts_per_day=spd, shifts=shifts,
        skill_levels=skill_levels, n_age_groups=n_age, weights=weights,
        shift_names=shift_names, age_group_names=age_group_names,
        room_ids=room_ids, ot_ids=ot_ids, surgeon_ids=surgeon_ids,
        nurse_ids=nurse_ids, patient_ids=patient_ids,
        room_capacity=room_capacity,
        surgeon_max_time=surgeon_max_time, ot_availability=ot_availability,
        nurse_skill=nurse_skill, nurse_working=nurse_working,
        nurse_max_load=nurse_max_load, nurse_working_shifts=nurse_working_shifts,
        available_nurses=available_nurses,
        p_mandatory=p_mandatory, p_gender=p_gender, p_age=p_age, p_los=p_los,
        p_release=p_release, p_due=p_due, p_last=p_last, p_duration=p_duration,
        p_surgeon=p_surgeon, p_incompatible=p_incompatible,
        p_workload=p_workload, p_skillreq=p_skillreq,
        o_gender=o_gender_arr(o_gender), o_age=np.array(o_age, dtype=np.int64),
        o_los=np.array(o_los, dtype=np.int64), o_room=np.array(o_room, dtype=np.int64),
        o_workload=o_workload, o_skillreq=o_skillreq,
        occ_room_day_count=occ_room_day_count,
        occ_room_day_a=occ_room_day_a, occ_room_day_b=occ_room_day_b,
        occ_in_room_day=occ_in_room_day,
    )


def o_gender_arr(vals: list[int]) -> np.ndarray:
    """int64 array; keeps dtype int64 even when the occupant list is empty."""
    return np.array(vals, dtype=np.int64) if vals else np.zeros(0, dtype=np.int64)

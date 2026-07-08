"""Objective and feasibility for the IHTP.

Port of IHTP_Validator.cc. Single source of truth for solution quality: nine hard
checks, eight weighted soft costs, validator index conventions. Golden test
(tests/test_golden.py) pins it to the validator on i01's published best (cost 3842,
0 violations).

Full from-scratch eval only. ALNS's incremental delta builds on these definitions
and gets checked against this so it can't drift.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .io_instance import GENDER_A, Instance
from .model import SolutionState

# soft components, validator costs order (IHTP_Validator.cc PrintCosts, lines 1104-1112)
# cost[i] weighted by weights[i]
SOFT_NAMES = [
    "RoomAgeMix", "RoomSkillLevel", "ContinuityOfCare", "ExcessiveNurseWorkload",
    "OpenOperatingTheater", "SurgeonTransfer", "PatientDelay", "ElectiveUnscheduledPatients",
]
HARD_NAMES = [
    "RoomGenderMix", "PatientRoomCompatibility", "SurgeonOvertime", "OperatingTheaterOvertime",
    "MandatoryUnscheduledPatients", "AdmissionDay", "RoomCapacity", "NursePresence", "UncoveredRoom",
]


@dataclass
class Costs:
    """Cost and feasibility breakdown."""
    hard: np.ndarray          # int[9] raw hard-violation counts, HARD_NAMES order
    soft: np.ndarray          # int[8] raw soft-cost counts, SOFT_NAMES order
    weights: np.ndarray       # int[8] instance weights, copied so callers don't alias the instance

    @property
    def total_violations(self) -> int:
        return int(self.hard.sum())

    @property
    def total_cost(self) -> int:
        """sum_i weights[i] * soft[i], the objective to minimise."""
        return int((self.soft * self.weights).sum())

    @property
    def feasible(self) -> bool:
        return self.total_violations == 0

    def weighted_components(self) -> dict[str, int]:
        return {name: int(self.soft[i] * self.weights[i]) for i, name in enumerate(SOFT_NAMES)}

    def hard_components(self) -> dict[str, int]:
        return {name: int(self.hard[i]) for i, name in enumerate(HARD_NAMES)}


def _build_occupancy(inst: Instance, st: SolutionState):
    """Rebuild the validator's room-day occupancy.

    room_day_entities: entities in room r, day d. a_cnt / b_cnt: gender-A and gender-B
    head-counts per room-day. patients keep index p (< P), occupants get global P + o,
    same as validator.
    """
    R, days, P = inst.R, inst.days, inst.P
    # start from the fixed occupant contribution
    room_day_entities = [[list(inst.occ_in_room_day[r][d]) for d in range(days)] for r in range(R)]
    for r in range(R):
        for d in range(days):
            # occupant lists hold local indices; shift to global P + o
            room_day_entities[r][d] = [P + o for o in room_day_entities[r][d]]
    a_cnt = inst.occ_room_day_a.copy()
    b_cnt = inst.occ_room_day_b.copy()

    for p in np.nonzero(st.adm != -1)[0]:
        ad, r = int(st.adm[p]), int(st.room[p])
        last = min(days, ad + int(inst.p_los[p]))     # clamp to horizon (validator line 623)
        ga = inst.p_gender[p] == GENDER_A
        for d in range(ad, last):
            room_day_entities[r][d].append(int(p))
            if ga:
                a_cnt[r, d] += 1
            else:
                b_cnt[r, d] += 1
    return room_day_entities, a_cnt, b_cnt


def evaluate(inst: Instance, st: SolutionState) -> Costs:
    """From-scratch eval of st, mirroring the C++ validator."""
    R, T, U, N = inst.R, inst.T, inst.U, inst.N
    days, spd, P, O = inst.days, inst.shifts_per_day, inst.P, inst.O
    adm, room, ot, cover = st.adm, st.room, st.ot, st.cover

    room_day_entities, a_cnt, b_cnt = _build_occupancy(inst, st)

    hard = np.zeros(9, dtype=np.int64)
    soft = np.zeros(8, dtype=np.int64)

    # surgeon and OT day loads, plus open/transfer counts; patients only
    surg_load = np.zeros((U, days), dtype=np.int64)
    ot_load = np.zeros((T, days), dtype=np.int64)
    ot_cnt = np.zeros((T, days), dtype=np.int64)
    surg_ot_cnt = np.zeros((U, days, T), dtype=np.int64)
    for p in np.nonzero(adm != -1)[0]:
        d, t, u = int(adm[p]), int(ot[p]), int(inst.p_surgeon[p])
        dur = int(inst.p_duration[p])
        surg_load[u, d] += dur
        ot_load[t, d] += dur
        ot_cnt[t, d] += 1
        surg_ot_cnt[u, d, t] += 1

    # H3 SurgeonOvertime, H4 OTOvertime, S5 OpenOT, S6 SurgeonTransfer
    hard[2] = np.maximum(0, surg_load - inst.surgeon_max_time).sum()          # SurgeonOvertime
    hard[3] = np.maximum(0, ot_load - inst.ot_availability).sum()             # OTOvertime
    soft[4] = int((ot_cnt > 0).sum())                                        # OpenOperatingTheater
    distinct_ot = (surg_ot_cnt > 0).sum(axis=2)                              # shape [U, days]
    soft[5] = int(np.maximum(0, distinct_ot - 1).sum())                      # SurgeonTransfer

    # per-patient scalar costs
    scheduled = adm != -1
    mandatory = inst.p_mandatory
    hard[4] = int(np.count_nonzero((~scheduled) & mandatory))                 # MandatoryUnscheduled
    soft[7] = int(np.count_nonzero((~scheduled) & (~mandatory)))             # ElectiveUnscheduled
    # AdmissionDay: scheduled but admitted before release or past last allowed day
    bad_adm = scheduled & ((adm < inst.p_release) | (adm > inst.p_last))
    hard[5] = int(np.count_nonzero(bad_adm))                                  # AdmissionDay
    delay = np.where(scheduled, np.maximum(0, adm - inst.p_release), 0)
    soft[6] = int(delay.sum())                                               # PatientDelay
    # scheduled into an incompatible room
    compat_viol = 0
    for p in np.nonzero(scheduled)[0]:
        if inst.p_incompatible[p, int(room[p])]:
            compat_viol += 1
    hard[1] = compat_viol                                                     # PatientRoomCompatibility

    # room-day costs: gender mix (H1), capacity (H7), age mix (S1)
    room_gender = 0
    room_cap = 0
    room_age = 0
    for r in range(R):
        cap = int(inst.room_capacity[r])
        for d in range(days):
            ents = room_day_entities[r][d]
            n_here = len(ents)
            room_gender += min(int(a_cnt[r, d]), int(b_cnt[r, d]))
            if n_here > cap:
                room_cap += n_here - cap
            if n_here > 0:
                ages = [inst.p_age[e] if e < P else inst.o_age[e - P] for e in ents]
                room_age += max(ages) - min(ages)
    hard[0] = room_gender                                                     # RoomGenderMix
    hard[6] = room_cap                                                        # RoomCapacity
    soft[0] = room_age                                                        # RoomAgeMix

    # nurse presence (H8a) and uncovered rooms (H8b)
    nurse_presence = 0
    uncovered = 0
    for r in range(R):
        for s in range(inst.shifts):
            n = int(cover[r, s])
            d = s // spd
            occupied = len(room_day_entities[r][d]) > 0
            if n != -1:
                if not inst.nurse_working[n, s]:
                    nurse_presence += 1
            elif occupied:
                uncovered += 1
    hard[7] = nurse_presence                                                  # NursePresence
    hard[8] = uncovered                                                       # UncoveredRoom

    # room skill level (S2)
    room_skill = 0
    for r in range(R):
        for s in range(inst.shifts):
            n = int(cover[r, s])
            if n == -1:
                continue                 # uncovered room-shift: no skill cost, hard viol already counted
            skill_n = int(inst.nurse_skill[n])
            d = s // spd
            for e in room_day_entities[r][d]:
                if e < P:
                    req = int(inst.p_skillreq[e][s - int(adm[e]) * spd])
                else:
                    req = int(inst.o_skillreq[e - P][s])
                if req > skill_n:
                    room_skill += req - skill_n
    soft[1] = room_skill                                                      # RoomSkillLevel

    # excessive nurse workload (S4)
    # per nurse per worked shift: sum workload of entities in the rooms it covers,
    # charge the overflow above max_load
    excessive = 0
    for n in range(N):
        for s in inst.nurse_working_shifts[n]:
            s = int(s)
            d = s // spd
            load = 0
            for r in range(R):
                if int(cover[r, s]) != n:
                    continue
                for e in room_day_entities[r][d]:
                    if e < P:
                        load += int(inst.p_workload[e][s - int(adm[e]) * spd])
                    else:
                        load += int(inst.o_workload[e - P][s])
            over = load - int(inst.nurse_max_load[n, s])
            if over > 0:
                excessive += over
    soft[3] = excessive                                                       # ExcessiveNurseWorkload

    # continuity of care (S3): distinct nurses per entity over its stay
    continuity = 0
    # occupants
    for o in range(O):
        r = int(inst.o_room[o])
        limit = int(inst.o_los[o]) * spd
        seen = set()
        for s in range(limit):
            n = int(cover[r, s])
            if n != -1:
                seen.add(n)
        continuity += len(seen)
    # patients
    for p in np.nonzero(scheduled)[0]:
        ad, r = int(adm[p]), int(room[p])
        limit = min(int(inst.p_los[p]) * spd, (days - ad) * spd)   # horizon clamp
        seen = set()
        for s1 in range(limit):
            n = int(cover[r, ad * spd + s1])
            if n != -1:
                seen.add(n)
        continuity += len(seen)
    soft[2] = continuity                                                      # ContinuityOfCare

    return Costs(hard=hard, soft=soft, weights=inst.weights.copy())

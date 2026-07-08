"""Incremental upper-layer (PAS + SCP) bookkeeping.

Everything needed to test hard feasibility and score the upper-layer soft cost of a
candidate placement (admission_day, room, OT), kept in O(affected) per edit.

Soft terms here (validator names bracketed):
S1 RoomAgeMix, S5 OpenOperatingTheater, S6 SurgeonTransfer, S7 PatientDelay.
Hard terms held at zero by only ever offering feasible placements:
H1 gender mix, H2 room compatibility, H3 surgeon overtime, H4 OT overtime,
H6 admission window, H7 room capacity.
S8 unscheduled and the four nurse-layer terms live elsewhere (constructor/experiment
layer, nra.py).

Occupant occupancy is constant, so it is in once at init. ALNS reuses one Layout;
place/unplace are the only mutators and both keep every cache and the running weighted
cost exact.
"""

from __future__ import annotations

import numpy as np

from .io_instance import GENDER_A, Instance
from .model import SolutionState


class Layout:
    def __init__(self, inst: Instance, st: SolutionState | None = None):
        self.inst = inst
        # caches below are seeded from OCCUPANTS ONLY, never replayed from `st`. a populated
        # `st` would desync caches vs placements. pass empty/None and add via the Layout API.
        # to rebuild from a populated state use construct.layout_from_state (it replays).
        if st is not None and bool((st.adm != -1).any()):
            raise ValueError(
                "Layout(inst, st) does not replay st's placements into the incremental caches; "
                "use construct.layout_from_state(inst, st) to build a populated layout.")
        self.st = st if st is not None else SolutionState(inst)
        R, T, U, days, n_age = inst.R, inst.T, inst.U, inst.days, inst.n_age_groups

        # int weights, pulled from the instance
        w = inst.weights
        self.w_age = int(w[0]); self.w_open = int(w[4])
        self.w_transfer = int(w[5]); self.w_delay = int(w[6])
        self.w_unsched = int(w[7])

        # surgeon/OT caches, patients only (occupants never have surgery)
        self.surg_load = np.zeros((U, days), dtype=np.int64)
        self.ot_load = np.zeros((T, days), dtype=np.int64)
        self.ot_cnt = np.zeros((T, days), dtype=np.int64)
        self.surg_ot_cnt = np.zeros((U, days, T), dtype=np.int64)
        self.distinct_ot = np.zeros((U, days), dtype=np.int64)   # distinct OTs per surgeon per day

        # room-day occupancy caches, seeded with occupants
        self.count = np.zeros((R, days), dtype=np.int64)
        self.a_cnt = inst.occ_room_day_a.copy()
        self.b_cnt = inst.occ_room_day_b.copy()
        self.count += inst.occ_room_day_count
        self.age_cnt = np.zeros((R, days, n_age), dtype=np.int64)
        for r in range(R):
            for d in range(days):
                for o in inst.occ_in_room_day[r][d]:
                    self.age_cnt[r, d, inst.o_age[o]] += 1

        # running weighted upper-layer soft cost, one accumulator per term
        self.cost_age = 0
        self.cost_open = 0
        self.cost_transfer = 0
        self.cost_delay = 0
        # age cost starts from occupant-only occupancy; patients add to it once placed
        for r in range(R):
            for d in range(days):
                sp = self._spread(r, d)
                self.cost_age += self.w_age * sp

    # helpers
    def _spread(self, r: int, d: int) -> int:
        """max age group minus min among entities in (r, d); 0 if fewer than 2 groups."""
        buckets = self.age_cnt[r, d]
        nz = np.nonzero(buckets)[0]
        if nz.size == 0:
            return 0
        return int(nz[-1] - nz[0])

    # feasibility
    def surgeon_ok(self, p: int, d: int) -> bool:
        u = int(self.inst.p_surgeon[p])
        return self.surg_load[u, d] + int(self.inst.p_duration[p]) <= int(self.inst.surgeon_max_time[u, d])

    def ot_ok(self, p: int, d: int, t: int) -> bool:
        return self.ot_load[t, d] + int(self.inst.p_duration[p]) <= int(self.inst.ot_availability[t, d])

    def room_ok(self, p: int, r: int, d: int) -> bool:
        """H2 compatibility + H1 gender + H7 capacity, checked across the whole stay."""
        inst = self.inst
        if inst.p_incompatible[p, r]:
            return False
        cap = int(inst.room_capacity[r])
        last = min(inst.days, d + int(inst.p_los[p]))
        is_a = inst.p_gender[p] == GENDER_A
        for dd in range(d, last):
            if self.count[r, dd] + 1 > cap:
                return False
            if is_a:
                if self.b_cnt[r, dd] > 0:       # opposite gender already in the room
                    return False
            elif self.a_cnt[r, dd] > 0:
                return False
        return True

    # cost deltas
    def delay_cost(self, p: int, d: int) -> int:
        return self.w_delay * max(0, d - int(self.inst.p_release[p]))

    def age_delta(self, p: int, r: int, d: int) -> int:
        """weighted S1 change if p's age group occupies room r starting day d."""
        inst = self.inst
        a = int(inst.p_age[p])
        last = min(inst.days, d + int(inst.p_los[p]))
        delta = 0
        for dd in range(d, last):
            old = self._spread(r, dd)
            self.age_cnt[r, dd, a] += 1
            new = self._spread(r, dd)
            self.age_cnt[r, dd, a] -= 1        # revert: query only, no mutation
            delta += new - old
        return self.w_age * delta

    def ot_delta(self, p: int, d: int, t: int) -> int:
        """weighted S5 + S6 change if p's surgery lands in theatre t on day d."""
        u = int(self.inst.p_surgeon[p])
        cost = 0
        if self.ot_cnt[t, d] == 0:                       # OT would open fresh
            cost += self.w_open
        if self.surg_ot_cnt[u, d, t] == 0 and self.distinct_ot[u, d] >= 1:
            cost += self.w_transfer                      # surgeon picks up an extra OT this day
        return cost

    # best sub-choices
    def best_room(self, p: int, d: int):
        """cheapest feasible room for p admitted day d, by S1 age-mix delta."""
        best_r, best_c = -1, None
        for r in range(self.inst.R):
            if self.room_ok(p, r, d):
                c = self.age_delta(p, r, d)
                if best_c is None or c < best_c:
                    best_c, best_r = c, r
        return best_r, best_c

    def best_ot(self, p: int, d: int):
        """cheapest feasible theatre for p on day d, by S5+S6 delta."""
        best_t, best_c = -1, None
        for t in range(self.inst.T):
            if self.ot_ok(p, d, t):
                c = self.ot_delta(p, d, t)
                if best_c is None or c < best_c:
                    best_c, best_t = c, t
        return best_t, best_c

    def _day_placements(self, p: int) -> list[tuple[int, tuple[int, int, int]]]:
        """every feasible one-per-day placement of p, with weighted upper cost.

        room and OT only interact through the chosen day, so pick each independently.
        keeps it O(days*(rooms+OTs)) per patient instead of the full product.
        """
        inst = self.inst
        out = []
        for d in range(int(inst.p_release[p]), int(inst.p_last[p]) + 1):
            if not self.surgeon_ok(p, d):
                continue
            r, rc = self.best_room(p, d)
            if r == -1:
                continue
            t, tc = self.best_ot(p, d)
            if t == -1:
                continue
            out.append((self.delay_cost(p, d) + rc + tc, (d, r, t)))
        return out

    def best_placement(self, p: int):
        """cheapest feasible (day, room, OT); (None, None) if p can't be placed."""
        ps = self._day_placements(p)
        if not ps:
            return None, None
        cost, move = min(ps, key=lambda x: x[0])
        return cost, move

    def best_two_placements(self, p: int):
        """best placement plus second-best cost, for regret-k insertion.

        returns (cost1, move1, cost2). cost2 None when only one day is feasible;
        all three None when p can't be placed at all.
        """
        ps = self._day_placements(p)
        if not ps:
            return None, None, None
        ps.sort(key=lambda x: x[0])
        c2 = ps[1][0] if len(ps) > 1 else None
        return ps[0][0], ps[0][1], c2

    def sampled_placement(self, p: int, rng, k: int = 3):
        """random pick among the k cheapest placements, for randomized greedy."""
        ps = self._day_placements(p)
        if not ps:
            return None, None
        ps.sort(key=lambda x: x[0])
        j = int(rng.integers(0, min(k, len(ps))))
        return ps[j]

    # mutators
    def place(self, p: int, d: int, r: int, t: int) -> None:
        inst = self.inst
        u = int(inst.p_surgeon[p])
        dur = int(inst.p_duration[p])
        a = int(inst.p_age[p])
        is_a = inst.p_gender[p] == GENDER_A
        last = min(inst.days, d + int(inst.p_los[p]))

        # surgery-day side: S5 open, S6 transfer, H3/H4 loads
        self.surg_load[u, d] += dur
        self.ot_load[t, d] += dur
        if self.ot_cnt[t, d] == 0:
            self.cost_open += self.w_open
        self.ot_cnt[t, d] += 1
        if self.surg_ot_cnt[u, d, t] == 0:
            old_tr = max(0, int(self.distinct_ot[u, d]) - 1)
            self.distinct_ot[u, d] += 1
            new_tr = max(0, int(self.distinct_ot[u, d]) - 1)
            self.cost_transfer += self.w_transfer * (new_tr - old_tr)
        self.surg_ot_cnt[u, d, t] += 1

        # room-day side: S1 age, H1 gender, H7 capacity
        for dd in range(d, last):
            old = self._spread(r, dd)
            self.age_cnt[r, dd, a] += 1
            new = self._spread(r, dd)
            self.cost_age += self.w_age * (new - old)
            self.count[r, dd] += 1
            if is_a:
                self.a_cnt[r, dd] += 1
            else:
                self.b_cnt[r, dd] += 1

        self.cost_delay += self.delay_cost(p, d)
        self.st.place_patient(p, d, r, t)

    def unplace(self, p: int) -> None:
        inst = self.inst
        if self.st.adm[p] == -1:
            return
        d, r, t = int(self.st.adm[p]), int(self.st.room[p]), int(self.st.ot[p])
        u = int(inst.p_surgeon[p])
        dur = int(inst.p_duration[p])
        a = int(inst.p_age[p])
        is_a = inst.p_gender[p] == GENDER_A
        last = min(inst.days, d + int(inst.p_los[p]))

        self.surg_load[u, d] -= dur
        self.ot_load[t, d] -= dur
        self.ot_cnt[t, d] -= 1
        if self.ot_cnt[t, d] == 0:
            self.cost_open -= self.w_open
        self.surg_ot_cnt[u, d, t] -= 1
        if self.surg_ot_cnt[u, d, t] == 0:
            old_tr = max(0, int(self.distinct_ot[u, d]) - 1)
            self.distinct_ot[u, d] -= 1
            new_tr = max(0, int(self.distinct_ot[u, d]) - 1)
            self.cost_transfer += self.w_transfer * (new_tr - old_tr)

        for dd in range(d, last):
            old = self._spread(r, dd)
            self.age_cnt[r, dd, a] -= 1
            new = self._spread(r, dd)
            self.cost_age += self.w_age * (new - old)
            self.count[r, dd] -= 1
            if is_a:
                self.a_cnt[r, dd] -= 1
            else:
                self.b_cnt[r, dd] -= 1

        self.cost_delay -= self.delay_cost(p, d)
        self.st.remove_patient(p)

    # reporting
    def upper_cost(self) -> int:
        """weighted S1+S5+S6+S7, i.e. upper-layer soft cost minus unscheduled."""
        return self.cost_age + self.cost_open + self.cost_transfer + self.cost_delay

    def unscheduled_cost(self) -> int:
        """weighted S8: w7 times the number of optional patients postponed right now."""
        adm = self.st.adm
        postponed_optional = np.count_nonzero((adm == -1) & (~self.inst.p_mandatory))
        return self.w_unsched * int(postponed_optional)

    def mandatory_unplaced(self) -> int:
        """mandatory patients still postponed; H5 needs this at 0 to be feasible."""
        return int(np.count_nonzero((self.st.adm == -1) & self.inst.p_mandatory))

    def search_objective(self, penalty: int) -> int:
        """what ALNS minimises.

        placements are hard-feasible by construction (H1,H2,H3,H4,H6,H7) and greedy NRA
        always covers occupied room-shifts (H8), so the only hard violation left is an
        unadmitted mandatory patient (H5). big penalty forces it to 0:

            objective = S1 + S5 + S6 + S7 (upper soft) + S8 (optional unscheduled)
                        + penalty * (mandatory unplaced)

        nurse-layer soft (S2,S3,S4) is handled apart: greedy NRA at checkpoints, exact
        NRA in post-processing. secondary to the upper terms, folded in only when the
        true objective is reported.
        """
        return (self.upper_cost() + self.unscheduled_cost()
                + penalty * self.mandatory_unplaced())

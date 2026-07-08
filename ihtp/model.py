"""IHTP solution representation.

SolutionState = the four decisions. adm[p] admission day, room[p] room for the
whole stay, ot[p] operating theatre, cover[r, s] the one nurse on room r in
shift s. -1 = postponed (or no nurse).

Occupants live in the instance, not here; they're fixed and folded into the
evaluator directly.

objective does the full eval. ALNS delta caches sit on top of this later.
"""

from __future__ import annotations

import numpy as np

from .io_instance import Instance


class SolutionState:
    """Mutable holder for the four IHTP decisions."""

    __slots__ = ("inst", "adm", "room", "ot", "cover")

    def __init__(self, inst: Instance):
        self.inst = inst
        P = inst.P
        self.adm = np.full(P, -1, dtype=np.int64)     # -1 = postponed
        self.room = np.full(P, -1, dtype=np.int64)
        self.ot = np.full(P, -1, dtype=np.int64)
        # nurse on room r, shift s; -1 = none
        self.cover = np.full((inst.R, inst.shifts), -1, dtype=np.int64)

    # queries
    def is_scheduled(self, p: int) -> bool:
        return self.adm[p] != -1

    def scheduled_patients(self) -> np.ndarray:
        return np.nonzero(self.adm != -1)[0]

    # edits
    def place_patient(self, p: int, day: int, room: int, ot: int) -> None:
        """Give p an admission day, room and OT."""
        self.adm[p] = day
        self.room[p] = room
        self.ot[p] = ot

    def remove_patient(self, p: int) -> None:
        """Postpone p, wiping its placement."""
        self.adm[p] = -1
        self.room[p] = -1
        self.ot[p] = -1

    def set_cover(self, r: int, s: int, nurse: int) -> None:
        self.cover[r, s] = nurse

    def clear_cover(self) -> None:
        self.cover.fill(-1)

    # copy
    def copy(self) -> "SolutionState":
        other = SolutionState.__new__(SolutionState)
        other.inst = self.inst
        other.adm = self.adm.copy()
        other.room = self.room.copy()
        other.ot = self.ot.copy()
        other.cover = self.cover.copy()
        return other

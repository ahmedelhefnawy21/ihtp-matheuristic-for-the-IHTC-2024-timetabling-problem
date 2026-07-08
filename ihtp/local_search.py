"""True-cost local search over the layout (rooms and admission days).

ALNS only touches the upper-layer surrogate. It never sees nurse costs S2/S3/S4,
just re-solves them at checkpoints. On small instances the leftover gap lives in
the secondary terms tied to where/when a patient sits: age-mix S1, nurse skill
S2, workload S4. So relocate one patient at a time to another feasible room or
day, keep the move only if the full objective drops (fresh greedy NRA included).
Every eval re-runs NRA greedily, so we land on rooms/days that are nurse-friendly,
not just cheap to admit.

Per-candidate eval is O(instance): cheap on small instances, which is where the
secondary gap sits anyway. Big ones just fit fewer improving moves into the pass
budget (max_passes, deterministic, no wall-clock).
"""

from __future__ import annotations

import numpy as np

from .construct import layout_from_state
from .io_instance import Instance
from .model import SolutionState
from .nra import greedy_nra
from .objective import evaluate


def layout_descent(inst: Instance, state: SolutionState, max_passes: int = 6,
                   seed: int = 0) -> tuple[SolutionState, int]:
    """First-improvement descent on the true objective, room/day relocations.

    At most max_passes sweeps; bails early once a pass finds no improving move."""
    rng = np.random.default_rng(seed)
    lay = layout_from_state(inst, state)

    def true_cost() -> int:
        greedy_nra(inst, lay.st)
        return int(evaluate(inst, lay.st).total_cost)

    best = true_cost()
    improved = True
    passes = 0
    while improved and passes < max_passes:
        passes += 1
        improved = False
        order = [int(p) for p in np.nonzero(lay.st.adm != -1)[0]]
        rng.shuffle(order)
        for p in order:
            d0, r0, o0 = int(lay.st.adm[p]), int(lay.st.room[p]), int(lay.st.ot[p])
            lay.unplace(p)
            cands = lay._day_placements(p)                 # feasible (day, room, OT) triples
            found = False
            for _uc, (d, r, t) in cands:
                if (d, r, t) == (d0, r0, o0):
                    continue
                lay.place(p, d, r, t)
                c = true_cost()
                if c < best - 1e-9:
                    best = c
                    improved = found = True
                    break
                lay.unplace(p)
            if not found:
                lay.place(p, d0, r0, o0)
    return lay.st.copy(), best

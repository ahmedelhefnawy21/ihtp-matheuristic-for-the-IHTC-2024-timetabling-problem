"""Constructive heuristic: build a feasible starting solution.

Upper layer (PAS + SCP) is greedy insertion, hardest patient first to protect
feasibility. Order: mandatory before optional, then ascending window slack (tight
windows first), then descending surgery duration and LOS (long stays take the most
resources). Each patient lands in its cheapest feasible (day, room, OT) by upper
soft cost. Optionals only get admitted if that cost undercuts the
unscheduled_optional penalty, otherwise postponed. ALNS insert/remove operators
rebalance this later.

Lower layer (NRA): layout fixed, greedy_nra assigns nurses to occupied room-shifts.

Out comes a complete SolutionState. The validator checks feasibility.
"""

from __future__ import annotations

import numpy as np

from .io_instance import Instance
from .layout import Layout
from .model import SolutionState
from .nra import greedy_nra


def patient_order(inst: Instance, rng=None, jitter: float = 0.0) -> list[int]:
    """Hardest-first insertion order (see module docstring).

    Gets us out of tight packings where one greedy order boxes a mandatory
    patient out.
    """
    slack = inst.p_last - inst.p_release           # window width
    keys = []
    for p in range(inst.P):
        noise = rng.random() * jitter if (rng is not None and jitter > 0) else 0.0
        keys.append((
            0 if inst.p_mandatory[p] else 1,       # mandatory first
            float(slack[p]) + noise,                # tighter window first, plus jitter
            -int(inst.p_duration[p]),               # longer surgery first
            -int(inst.p_los[p]),                    # longer stay first
            p,
        ))
    return [k[-1] for k in sorted(keys)]


def construct_upper(inst: Instance, order: list[int] | None = None,
                    st: SolutionState | None = None) -> Layout:
    """Greedy PAS+SCP placement, returns the filled Layout."""
    if order is None:
        order = patient_order(inst)
    lay = Layout(inst, st)
    for p in order:
        cost, move = lay.best_placement(p)
        if move is None:
            continue                                # nowhere to go; _repair_mandatory picks it up if mandatory
        if inst.p_mandatory[p] or cost < lay.w_unsched:
            lay.place(p, *move)
    _repair_mandatory(inst, lay)
    return lay


def _blocker_candidates(inst: Instance, lay: Layout, p: int) -> list[int]:
    """Placed patients whose relocation could unblock p.

    Two ways p gets blocked. (a) no compatible room with free capacity and matching
    gender on any window day. (b) surgeon out of minutes on every window day. So a
    useful blocker either sits in a room p could use on an overlapping day, or shares
    p's surgeon and is admitted inside p's window (freeing surgeon time). Optionals
    and wide-slack mandatory patients re-home easier, try them first.
    """
    st = lay.st
    rel, last = int(inst.p_release[p]), int(inst.p_last[p])
    los_p = int(inst.p_los[p])
    surg_p = int(inst.p_surgeon[p])
    cands = []
    for q in range(inst.P):
        if st.adm[q] == -1 or q == p:
            continue
        dq = int(st.adm[q])
        # (b) same surgeon, admitted in p's window -> frees surgeon capacity
        surgeon_blocker = (int(inst.p_surgeon[q]) == surg_p and rel <= dq <= last)
        # (a) room-overlap in a room p is allowed to use
        room_blocker = (not inst.p_incompatible[p, int(st.room[q])]
                        and not (dq + int(inst.p_los[q]) <= rel or dq >= last + los_p))
        if surgeon_blocker or room_blocker:
            cands.append(q)
    cands.sort(key=lambda q: (0 if not inst.p_mandatory[q] else 1,
                              -(int(inst.p_last[q]) - int(inst.p_release[q]))))
    return cands


def _place_or_eject(inst: Instance, lay: Layout, p: int, depth: int,
                    max_blockers: int = 40) -> bool:
    """Place p. If blocked, eject a placed patient, place p, then re-home the ejected
    one. Bounded ejection chain, full revert on failure."""
    _cost, move = lay.best_placement(p)
    if move is not None:
        lay.place(p, *move)
        return True
    if depth <= 0:
        return False
    st = lay.st
    for q in _blocker_candidates(inst, lay, p)[:max_blockers]:
        save = (int(st.adm[q]), int(st.room[q]), int(st.ot[q]))
        lay.unplace(q)
        _cost, move = lay.best_placement(p)
        if move is not None:
            lay.place(p, *move)
            if _place_or_eject(inst, lay, q, depth - 1, max_blockers):
                return True
            lay.unplace(p)                          # q wouldn't re-home; back out
        lay.place(q, *save)                         # put q back exactly
    return False


def _repair_mandatory(inst: Instance, lay: Layout) -> None:
    """Admit every mandatory patient (H5) via ejection chains.

    H5 dominates the objective, so shoving other patients around to fit a mandatory
    one always pays off.
    """
    for p in range(inst.P):
        if inst.p_mandatory[p] and lay.st.adm[p] == -1:
            _place_or_eject(inst, lay, p, depth=3)


def layout_from_state(inst: Instance, state: SolutionState) -> Layout:
    """Rebuild a Layout (caches and all) by replaying a state's placements.

    Used to roll search back to a snapshot when a restructuring step is rejected, e.g.
    an ejection chain that churned many patients."""
    lay = Layout(inst)
    for p in np.nonzero(state.adm != -1)[0]:
        lay.place(int(p), int(state.adm[p]), int(state.room[p]), int(state.ot[p]))
    return lay


def construct(inst: Instance, order: list[int] | None = None) -> SolutionState:
    """One greedy pass to a complete feasible solution (upper layer + NRA)."""
    lay = construct_upper(inst, order)
    greedy_nra(inst, lay.st)
    return lay.st


def n_mandatory_unplaced(inst: Instance, st: SolutionState) -> int:
    return int(np.count_nonzero((st.adm == -1) & inst.p_mandatory))


def construct_best(inst: Instance, attempts: int = 8, seed: int = 0) -> SolutionState:
    """Multi-start construction: one deterministic order, then randomized restarts.
    Returns the best feasible solution, or the least infeasible one if none feasible.

    Restarts are cheap (a construction is a fraction of a second) and reliably crack
    the tight instances where one greedy order leaves a mandatory patient unplaced.
    """
    from .objective import evaluate

    best_st, best_key = None, None
    for a in range(attempts):
        if a == 0:
            order = patient_order(inst)
        else:
            rng = np.random.default_rng(seed + a)
            order = patient_order(inst, rng, jitter=3.0)
        lay = construct_upper(inst, order)
        # cheap infeasibility screen before paying for NRA + full evaluate
        mand_un = n_mandatory_unplaced(inst, lay.st)
        greedy_nra(inst, lay.st)
        c = evaluate(inst, lay.st)
        key = (c.total_violations, c.total_cost)
        if best_key is None or key < best_key:
            best_key, best_st = key, lay.st.copy()
        if c.total_violations == 0 and a >= 1:
            break                                   # got a feasible restart, done
    return best_st

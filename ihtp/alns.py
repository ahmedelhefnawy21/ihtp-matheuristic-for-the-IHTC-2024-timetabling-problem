"""ALNS for the upper layer (PAS + SCP).

Why ALNS + SA acceptance:
single-patient moves can't escape a tight packing. Ruin-and-recreate can: empty a day,
theatre, or a cluster of related patients, then rebuild. Destroy/repair big chunks,
learn per-instance which operators pay off via adaptive weights (Ropke & Pisinger 2006;
Shaw 1998). SA acceptance (Kirkpatrick et al. 1983) takes worsening rebuilds early,
tightens later. No tabu lists to hand-tune.

Constraint handling:
every placement is hard-feasible by construction (H1,H2,H3,H4,H6,H7). Greedy NRA always
covers occupied room-shifts (H8). The only violation the search can produce is an
unadmitted mandatory (H5), penalised. Minimised value is
Layout.search_objective(penalty) = upper soft (S1+S5+S6+S7) + optional-unscheduled (S8)
+ big penalty per unadmitted mandatory. Pulls infeasible starts to feasibility, stays
feasible after.

Nurse soft costs (S2,S3,S4) aren't optimised in the loop: secondary to the upper terms,
handled by greedy NRA at checkpoints and exact NRA polish in post. True objective (with
NRA) is recomputed at every new upper best so the returned solution is scored honestly.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from .construct import _place_or_eject, construct_upper, layout_from_state, patient_order
from .io_instance import Instance
from .layout import Layout
from .model import SolutionState
from .nra import greedy_nra
from .objective import evaluate

PENALTY = 10_000_000        # per unadmitted mandatory; swamps any soft trade


# moves
def _snapshot_and_remove(lay: Layout, D: list[int]) -> dict[int, tuple[int, int, int]]:
    """save each patient's placement in D, unplace the placed ones.

    D may carry postponed patients (saved as (-1,-1,-1)) so repair can reconsider
    admitting them. Lets one path cover relocation and optional add/remove.
    """
    saved = {}
    st = lay.st
    for p in D:
        saved[p] = (int(st.adm[p]), int(st.room[p]), int(st.ot[p]))
        if st.adm[p] != -1:
            lay.unplace(p)
    return saved


def _revert(lay: Layout, saved: dict[int, tuple[int, int, int]]) -> None:
    """undo _snapshot_and_remove; localised, no full-state rebuild."""
    for p in saved:
        if lay.st.adm[p] != -1:
            lay.unplace(p)
    for p, (d, r, t) in saved.items():
        if d != -1:
            lay.place(p, d, r, t)


# destroy ops
def destroy_random(lay: Layout, rng, k: int) -> list[int]:
    """k random placed patients + a couple postponed optionals to retry."""
    st = lay.st
    placed = np.nonzero(st.adm != -1)[0]
    if placed.size == 0:
        return []
    k = min(k, placed.size)
    chosen = list(rng.choice(placed, size=k, replace=False))
    # give a couple postponed optionals another shot at admission
    postponed = np.nonzero((st.adm == -1) & (~lay.inst.p_mandatory))[0]
    if postponed.size:
        extra = rng.choice(postponed, size=min(2, postponed.size), replace=False)
        chosen += list(extra)
    return [int(p) for p in chosen]


def destroy_worst(lay: Layout, rng, k: int) -> list[int]:
    """k placed patients with the biggest admission delay, to relocate."""
    st = lay.st
    placed = np.nonzero(st.adm != -1)[0]
    if placed.size == 0:
        return []
    delays = np.maximum(0, st.adm[placed] - lay.inst.p_release[placed])
    # noise breaks up the deterministic order (Shaw-style)
    noise = rng.random(placed.size)
    order = np.lexsort((noise, -delays))
    return [int(placed[i]) for i in order[:min(k, placed.size)]]


def destroy_ruin_day(lay: Layout, rng, k: int) -> list[int]:
    """empty a random occupied day: drop everyone admitted that day."""
    st = lay.st
    days_used = np.unique(st.adm[st.adm != -1])
    if days_used.size == 0:
        return []
    d = int(rng.choice(days_used))
    return [int(p) for p in np.nonzero(st.adm == d)[0]]


def destroy_ruin_ot(lay: Layout, rng, k: int) -> list[int]:
    """empty a random open (theatre, day) by clearing its surgeries. Aims at S5 open-OT."""
    st = lay.st
    placed = np.nonzero(st.adm != -1)[0]
    if placed.size == 0:
        return []
    p0 = int(rng.choice(placed))
    t0, d0 = int(st.ot[p0]), int(st.adm[p0])
    return [int(p) for p in placed if int(st.ot[p]) == t0 and int(st.adm[p]) == d0]


def destroy_shaw(lay: Layout, rng, k: int) -> list[int]:
    """seed patient + its k-1 most related placed patients.

    relatedness = same surgeon + overlapping stay + same room. Pulling a whole cluster
    lets repair re-pack them together (Shaw 1998)."""
    st, inst = lay.st, lay.inst
    placed = np.nonzero(st.adm != -1)[0]
    if placed.size == 0:
        return []
    seed = int(rng.choice(placed))
    ds, rs, us = int(st.adm[seed]), int(st.room[seed]), int(inst.p_surgeon[seed])
    los_s = int(inst.p_los[seed])
    rel = []
    for p in placed:
        p = int(p)
        if p == seed:
            continue
        dp = int(st.adm[p])
        overlap = not (dp + int(inst.p_los[p]) <= ds or dp >= ds + los_s)
        score = (int(inst.p_surgeon[p]) == us) + (int(st.room[p]) == rs) + overlap
        rel.append((score + rng.random(), p))
    rel.sort(reverse=True)
    return [seed] + [p for _s, p in rel[:max(0, k - 1)]]


# repair ops
def repair_regret(lay: Layout, D: list[int], rng) -> None:
    """regret-2 insertion: place the patient that would most regret waiting.

    mandatory get a big boost so feasibility comes back first. An optional goes in only
    if its best cost beats the unscheduled penalty.
    """
    inst = lay.inst
    pending = set(D)
    BIG = 10 ** 9
    while pending:
        scored, drop = [], []
        for p in pending:
            c1, m1, c2 = lay.best_two_placements(p)
            if m1 is None:
                drop.append(p); continue
            if (not inst.p_mandatory[p]) and c1 >= lay.w_unsched:
                drop.append(p); continue                 # not worth admitting this one
            regret = (c2 - c1) if c2 is not None else BIG
            key = regret + (10 ** 12 if inst.p_mandatory[p] else 0)
            scored.append((key, p, m1))
        for p in drop:
            pending.discard(p)
        if not scored:
            break
        _key, p, m1 = max(scored, key=lambda x: x[0])
        lay.place(p, *m1)
        pending.discard(p)


def repair_greedy(lay: Layout, D: list[int], rng) -> None:
    """greedy cheapest-insertion, mandatory first."""
    inst = lay.inst
    pending = set(D)
    while pending:
        best = None
        drop = []
        for p in pending:
            c1, m1 = lay.best_placement(p)
            if m1 is None:
                drop.append(p); continue
            if (not inst.p_mandatory[p]) and c1 >= lay.w_unsched:
                drop.append(p); continue
            eff = c1 - (10 ** 12 if inst.p_mandatory[p] else 0)
            if best is None or eff < best[0]:
                best = (eff, p, m1)
        for p in drop:
            pending.discard(p)
        if best is None:
            break
        lay.place(best[1], *best[2])
        pending.discard(best[1])


def repair_random_greedy(lay: Layout, D: list[int], rng) -> None:
    """randomised greedy: random insertion order, sampled near-best slot."""
    inst = lay.inst
    order = list(D)
    # mandatory first, rest shuffled
    order.sort(key=lambda p: (0 if inst.p_mandatory[p] else 1, rng.random()))
    for p in order:
        c1, m1 = lay.sampled_placement(p, rng, k=3)
        if m1 is None:
            continue
        if (not inst.p_mandatory[p]) and c1 >= lay.w_unsched:
            continue
        lay.place(p, *m1)


DESTROY_OPS = [
    ("random", destroy_random), ("worst", destroy_worst),
    ("ruin_day", destroy_ruin_day), ("ruin_ot", destroy_ruin_ot), ("shaw", destroy_shaw),
]
REPAIR_OPS = [("regret", repair_regret), ("greedy", repair_greedy), ("rand_greedy", repair_random_greedy)]


# engine
@dataclass
class ALNSResult:
    state: SolutionState        # best found (with NRA), scored by true objective
    cost: int                   # true weighted objective, all 8 soft terms
    violations: int             # want 0
    iterations: int
    op_weights: dict = field(default_factory=dict)
    history: list = field(default_factory=list)   # (iter, true_cost) checkpoints


def _sigmoid_weights(scores, counts, weights, lam):
    for i in range(len(weights)):
        if counts[i] > 0:
            weights[i] = (1 - lam) * weights[i] + lam * (scores[i] / counts[i])
        scores[i] = 0.0
        counts[i] = 0


def admit_optionals(inst: Instance, lay: Layout, rng, max_passes: int = 3) -> Layout:
    """admit every postponed optional that pays for itself.

    unscheduled_optional weight is large, so admitting almost always beats leaving out;
    the objective is mostly a headcount of who we fit. Per postponed optional:
    (a) grab a directly-feasible slot if it undercuts the unscheduled penalty, else
    (b) try an ejection chain to relocate blockers and make room, kept only if the true
    search objective drops. Repeat passes until no admission helps.
    """
    for _ in range(max_passes):
        changed = False
        postponed = [p for p in range(inst.P)
                     if not inst.p_mandatory[p] and lay.st.adm[p] == -1]
        # scarcest window first, hardest to fit later
        postponed.sort(key=lambda p: int(inst.p_last[p]) - int(inst.p_release[p]))
        for p in postponed:
            c, m = lay.best_placement(p)
            if m is not None:
                if c < lay.w_unsched:                     # slot that pays for itself
                    lay.place(p, *m)
                    changed = True
                continue
            # no direct slot: eject to make room, keep only if it improves
            before = lay.search_objective(PENALTY)
            snap = lay.st.copy()
            if _place_or_eject(inst, lay, p, depth=3) and lay.search_objective(PENALTY) < before:
                changed = True
            else:
                lay = layout_from_state(inst, snap)       # roll back the restructuring
        if not changed:
            break
    return lay


def feasibility_phase(inst: Instance, lay: Layout, rng, max_iters: int):
    """drive mandatory-unscheduled (H5) to zero via ruin + ejection, hill-climbing on the
    violation count.

    one greedy order can box a mandatory out of a surgeon's schedule. Fix: repeatedly
    empty the congested region around an unplaced mandatory (its window days, same-surgeon
    patients) and rebuild with ejection chains that re-home blockers. Keep only rearrangements
    that cut violations; else rebuild from the best snapshot. Fixed max_iters, deterministic.
    Returns the (maybe new) layout and best snapshot.
    """
    best_state = lay.st.copy()
    best_v = lay.mandatory_unplaced()
    if best_v == 0:
        return lay, best_state
    it = 0
    while best_v > 0 and it < max_iters:
        it += 1
        unplaced = [p for p in range(inst.P) if inst.p_mandatory[p] and lay.st.adm[p] == -1]
        seed = int(rng.choice(unplaced))
        rel, last = int(inst.p_release[seed]), int(inst.p_last[seed])
        us = int(inst.p_surgeon[seed])
        # ruin: seed's window days, same surgeon, plus some random others
        D = [q for q in range(inst.P)
             if lay.st.adm[q] != -1 and rel <= int(lay.st.adm[q]) <= last
             and (int(inst.p_surgeon[q]) == us or rng.random() < 0.3)]
        _snapshot_and_remove(lay, D)
        # rebuild: force every unplaced mandatory in via ejection, then optionals
        order = sorted(D, key=lambda x: (0 if inst.p_mandatory[x] else 1,
                                         int(inst.p_last[x]) - int(inst.p_release[x])))
        for p in order:
            if inst.p_mandatory[p]:
                _place_or_eject(inst, lay, p, depth=3)
            else:
                c, m = lay.best_placement(p)
                if m is not None and c < lay.w_unsched:
                    lay.place(p, *m)
        for p in [q for q in range(inst.P) if inst.p_mandatory[q] and lay.st.adm[q] == -1]:
            _place_or_eject(inst, lay, p, depth=3)
        v = lay.mandatory_unplaced()
        if v < best_v:
            best_v, best_state = v, lay.st.copy()
        else:
            lay = layout_from_state(inst, best_state)          # roll back the restructuring
    lay = layout_from_state(inst, best_state)
    return lay, best_state


def solve(inst: Instance, max_iters: int = 20000, seed: int = 0,
          init_state: SolutionState | None = None, feas_iters: int = 400,
          nra_ckpt_every: int = 1500, verbose: bool = False) -> ALNSResult:
    """run ALNS for a fixed max_iters. Iteration-based, deterministic, no wall-clock."""
    rng = np.random.default_rng(seed)

    # warm start
    if init_state is not None:
        lay = layout_from_state(inst, init_state)
    else:
        lay = construct_upper(inst, patient_order(inst))

    # feasibility phase only if the warm start left mandatories out
    if lay.mandatory_unplaced() > 0:
        lay, _ = feasibility_phase(inst, lay, rng, feas_iters)

    # admit every optional that pays for itself (dominant objective term)
    lay = admit_optionals(inst, lay, rng)

    cur_obj = lay.search_objective(PENALTY)
    best_obj = cur_obj
    best_upper_state = lay.st.copy()             # best upper layout so far

    def eval_state(state: SolutionState) -> tuple[int, int]:
        s = state.copy()
        greedy_nra(inst, s)
        c = evaluate(inst, s)
        return c.total_cost, c.total_violations

    best_true_cost, best_true_viol = eval_state(best_upper_state)
    best_state = best_upper_state.copy()
    greedy_nra(inst, best_state)                 # attach NRA so the snapshot is complete
    history = [(0, best_true_cost)]

    # SA temperature calibrated from probe moves
    probe = []
    for _ in range(30):
        D = destroy_random(lay, rng, max(1, inst.P // 20))
        saved = _snapshot_and_remove(lay, D)
        repair_greedy(lay, D, rng)
        probe.append(abs(lay.search_objective(PENALTY) - cur_obj))
        _revert(lay, saved)
    pos = [g for g in probe if g > 0]
    avg_gap = max(1.0, float(np.mean(pos)) if pos else 1.0)
    T0 = -avg_gap / math.log(0.4)                # start accepting ~40% of worsening probes
    T_min = max(1e-3, T0 * 1e-3)
    alpha = 0.9995

    # adaptive operator weights
    n_d, n_r = len(DESTROY_OPS), len(REPAIR_OPS)
    dw, rw = np.ones(n_d), np.ones(n_r)
    dscore, rscore = np.zeros(n_d), np.zeros(n_r)
    dcount, rcount = np.zeros(n_d), np.zeros(n_r)
    sigma = (33.0, 9.0, 13.0)                    # new-best, improved, accepted-worse
    lam, segment = 0.8, 100

    T = T0
    admitted0 = int((lay.st.adm != -1).sum())
    kmax = max(2, min(admitted0 // 7, 25))       # cap destroy size, keeps big instances fast
    it = 0
    while it < max_iters:
        it += 1
        di = int(rng.choice(n_d, p=dw / dw.sum()))
        ri = int(rng.choice(n_r, p=rw / rw.sum()))
        k = int(rng.integers(1, kmax + 1))
        D = DESTROY_OPS[di][1](lay, rng, k)
        if not D:
            continue
        saved = _snapshot_and_remove(lay, D)
        REPAIR_OPS[ri][1](lay, D, rng)
        new_obj = lay.search_objective(PENALTY)
        delta = new_obj - cur_obj

        accept = delta <= 0 or rng.random() < math.exp(-delta / max(T, 1e-9))
        outcome = 2
        if accept:
            cur_obj = new_obj
            if new_obj < best_obj - 1e-9:
                best_obj, outcome = new_obj, 0
                best_upper_state = lay.st.copy()             # stash the new best upper layout
            elif delta < 0:
                outcome = 1
        else:
            _revert(lay, saved)

        dscore[di] += sigma[outcome]; dcount[di] += 1
        rscore[ri] += sigma[outcome]; rcount[ri] += 1
        if it % segment == 0:
            _sigmoid_weights(dscore, dcount, dw, lam)
            _sigmoid_weights(rscore, rcount, rw, lam)
        T = T * alpha if T > T_min else T0            # cool, reheat once we hit T_min

        # true-cost checkpoint every nra_ckpt_every iters; keeps NRA off the hot path
        if it % nra_ckpt_every == 0:
            # push admissions with ejection, the dominant lever
            lay = admit_optionals(inst, lay, rng)
            cur_obj = lay.search_objective(PENALTY)
            if cur_obj < best_obj - 1e-9:
                best_obj = cur_obj
                best_upper_state = lay.st.copy()
            # score the best-surrogate layout, not the current accepted one. Scoring current
            # instead moves no final objective (I checked it empirically): downstream true-cost
            # stages (descent, CP-SAT LNS, exact polish) re-optimise whatever ALNS hands
            # back, so the pipeline doesn't care about the exact return state.
            tc, tv = eval_state(best_upper_state)
            if (tv, tc) < (best_true_viol, best_true_cost):
                best_true_cost, best_true_viol = tc, tv
                best_state = best_upper_state.copy(); greedy_nra(inst, best_state)
            history.append((it, best_true_cost))
            if verbose:
                print(f"  it={it} best_upper={best_obj} best_true={best_true_cost} T={T:.1f}")

    # last checkpoint on the best upper layout
    tc, tv = eval_state(best_upper_state)
    if (tv, tc) < (best_true_viol, best_true_cost):
        best_true_cost, best_true_viol = tc, tv
        best_state = best_upper_state.copy(); greedy_nra(inst, best_state)

    return ALNSResult(
        state=best_state, cost=best_true_cost, violations=best_true_viol, iterations=it,
        op_weights={"destroy": {DESTROY_OPS[i][0]: float(dw[i]) for i in range(n_d)},
                    "repair": {REPAIR_OPS[i][0]: float(rw[i]) for i in range(n_r)}},
        history=history,
    )

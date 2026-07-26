"""Top-level matheuristic pipeline.

Stages, with the evidence behind each:
  1. Construct: greedy warm start to seed the MILP.
  2. PAS MILP: exact admission core (day, room, gender, capacity). Gap analysis
     put most of the objective on unscheduled optionals. Binding constraints are
     surgeon/theatre scheduling and room/gender packing, which a MILP beats local
     search on. Decides who gets admitted, when, into which room.
  3. ALNS: cleans up secondary terms the MILP leaves loose (age-mix S1, delay S7,
     theatre choice S5/S6), and mops up optionals the time-limited MILP dropped.
     Feasibility stays protected.
  3b. Descent: true-cost local search (layout_descent). One-patient moves are
      judged on the full objective with the nurses re-assigned.
  3c. CP-SAT LNS (lns_improve): frees the unscheduled optionals plus a day
      window and re-packs the slice exactly. This is the capacity-creating step.
  4. Final exact solves: per-day OT/surgeon MILP (S5/S6), then NRA MILP (S2/S3/S4).

Adoption rules. PAS is adopted when feasible, and ALNS hands its result forward
directly, since both re-decide the dominant admission terms. From the descent
onward a stage's output replaces the incumbent only when the full evaluated
objective strictly improves. Every returned solution goes through the official
C++ validator.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .alns import solve
from .construct import construct_best
from .exact_lns import lns_improve
from .exact_pas import realize_pas, solve_pas_full
from .io_instance import Instance
from .local_search import layout_descent
from .model import SolutionState
from .nra import greedy_nra
from .objective import evaluate
from .polish import polish


@dataclass
class Result:
    state: SolutionState
    cost: int
    violations: int
    seconds: float
    stage_costs: dict = field(default_factory=dict)


def _feasible_cost(inst: Instance, st: SolutionState):
    c = evaluate(inst, st)
    return (c.total_cost if c.total_violations == 0 else None), c.total_violations


def matheuristic(inst: Instance, seed: int = 0, pas_work: float = 100.0,
                 alns_iters: int = 15000, nra_work: float = 200.0, ot_work: float = 30.0,
                 descent_passes: int = 6, lns_rounds: int = 8, use_pas: bool = True,
                 opt_window: int | None = None, do_polish: bool = True) -> Result:
    """Deterministic matheuristic. Stopping criteria are iteration/round counts or solver
    WorkLimits, never wall-clock, so a given seed reproduces the same solution on any machine
    with matching solver versions. Only runtime moves with the hardware."""
    t0 = time.time()
    stage_costs: dict = {}

    # 1. warm start
    warm = construct_best(inst, attempts=4, seed=seed)
    stage_costs["construct"], _ = _feasible_cost(inst, warm)
    init = warm

    # 2. PAS MILP, the exact admission core
    if use_pas:
        plan = solve_pas_full(inst, work_limit=pas_work, warm=warm, opt_window=opt_window)
        if plan:
            lay = realize_pas(inst, plan)
            greedy_nra(inst, lay.st)
            c, viol = _feasible_cost(inst, lay.st)
            if viol == 0:
                init = lay.st
                stage_costs["pas"] = c

    # 3. ALNS, seeded from the MILP layout
    res = solve(inst, max_iters=alns_iters, seed=seed, init_state=init)
    best_state, best_cost, best_viol = res.state, res.cost, res.violations
    stage_costs["alns"] = best_cost if best_viol == 0 else None

    # 3b. true-cost local search: room/day relocations against the full objective (NRA included).
    #     Hits S1/S2/S4, which the upper-layer surrogate can't see.
    if descent_passes > 0 and best_viol == 0:
        ds, dc = layout_descent(inst, best_state, max_passes=descent_passes, seed=seed)
        dv = evaluate(inst, ds).total_violations
        if dv == 0 and dc < best_cost:
            best_state, best_cost = ds, dc
        stage_costs["descent"] = best_cost

    # 3c. CP-SAT capacity-creating LNS, monotone on the full objective. Releases unscheduled
    #     optionals plus a day window, re-solves placement exactly, keeps the round only on a
    #     strict validated improvement. Closes the leftover admission gap on the hard instances.
    if lns_rounds > 0 and best_viol == 0:
        ls, lc = lns_improve(inst, best_state, rounds=lns_rounds, seed=seed,
                             batch=6, nra_work=min(nra_work, 150.0))
        if lc < best_cost:
            best_state, best_cost = ls, lc
        stage_costs["lns"] = best_cost

    # 4. final exact solves (OT then NRA)
    if do_polish and best_viol == 0:
        polished, _ = polish(inst, best_state, ot_work=ot_work, nra_work=nra_work)
        pc = evaluate(inst, polished)
        stage_costs["polish"] = pc.total_cost if pc.total_violations == 0 else None
        if pc.total_violations == 0 and pc.total_cost < best_cost:
            best_state, best_cost, best_viol = polished, pc.total_cost, 0

    return Result(state=best_state, cost=best_cost, violations=best_viol,
                  seconds=time.time() - t0, stage_costs=stage_costs)

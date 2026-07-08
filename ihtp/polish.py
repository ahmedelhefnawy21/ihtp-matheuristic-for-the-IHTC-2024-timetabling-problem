"""Exact post-processing.

Two exact sub-models over a heuristic solution. Order matters: it keeps them independent.

 1. per-day OT/surgeon re-assignment (exact_ot). hits S5 open-OT and S6 transfer.
    only touches ot, so room occupancy and the NRA are left alone.
 2. NRA re-opt (exact_nra). hits S2 skill, S3 continuity, S4 workload against the
    now-fixed layout.

Every stage gets scored by the official Python evaluator, so we can report what the
polish buys us. that's the case for keeping it. warm start is already feasible, so
polish only ever lowers cost or leaves it flat.
"""

from __future__ import annotations

from .exact_nra import optimize_nra
from .exact_ot import optimize_ot
from .io_instance import Instance
from .model import SolutionState
from .objective import evaluate


def polish(inst: Instance, state: SolutionState, ot_work: float = 30.0,
           nra_work: float = 200.0, do_ot: bool = True, do_nra: bool = True) -> tuple[SolutionState, list]:
    """Polished copy of state, plus (stage, total_cost) checkpoints.

    Both exact stages run under deterministic WorkLimits.
    """
    s = state.copy()
    stages = [("input", int(evaluate(inst, s).total_cost))]

    if do_ot:
        optimize_ot(inst, s, per_day_work=ot_work)
        stages.append(("after_ot", int(evaluate(inst, s).total_cost)))

    if do_nra:
        optimize_nra(inst, s, work_limit=nra_work)
        stages.append(("after_nra", int(evaluate(inst, s).total_cost)))

    return s, stages

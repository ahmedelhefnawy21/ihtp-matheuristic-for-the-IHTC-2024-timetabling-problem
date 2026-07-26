"""Solver backend settings for the exact models.

The exact sub-models (PAS, OT, NRA, and the lower bound) run on Gurobi and
require a license. OR-Tools CP-SAT serves the LNS re-pack stage only
(exact_lns.py). The helpers here hold the deterministic solver settings
(threads, seed, work limits) and the availability checks.
"""

from __future__ import annotations

import os

from . import config

_ENV = None


def gurobi_available() -> bool:
    try:
        import gurobipy  
        return True
    except Exception:
        return False


def gurobi_env():
    """cached silent Gurobi env off the project license file"""
    global _ENV
    if _ENV is None:
        os.environ.setdefault("GRB_LICENSE_FILE", config.GUROBI_LICENSE)
        import gurobipy as gp
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.start()
        _ENV = env
    return _ENV


def configure_gurobi(model, work_limit: float | None = None,
                     mip_gap: float | None = None, threads: int = 1) -> None:
    """Gurobi params for reproducible solving.

    No wall-clock TimeLimit; its incumbent moves with machine speed. WorkLimit caps
    deterministic work units instead, stable across machines. Fixed MIPGap/Seed/Threads
    (Gurobi's parallel MIP is deterministic) pin the same solution everywhere. Only
    runtime tracks the hardware.
    """
    model.setParam("OutputFlag", 0)
    model.setParam("Threads", threads)
    model.setParam("Seed", 0)
    if work_limit is not None:
        model.setParam("WorkLimit", work_limit)
    if mip_gap is not None:
        model.setParam("MIPGap", mip_gap)


def configure_cpsat(solver, det_time: float, seed: int = 0, workers: int = 1) -> None:
    """CP-SAT params for deterministic solving.

    num_workers=1 on purpose. Single worker reproduces; the parallel portfolio can
    hand back a different incumbent when a sub-problem runs out of budget, which broke
    determinism for us in testing. Fixed random_seed plus max_deterministic_time (a
    machine-independent budget, unlike wall-clock max_time_in_seconds) gives the same
    solution on every machine. Saturate cores with one process per instance instead.
    """
    solver.parameters.max_deterministic_time = det_time
    solver.parameters.random_seed = seed
    solver.parameters.num_workers = workers
    solver.parameters.relative_gap_limit = 0.01


def ortools_available() -> bool:
    try:
        from ortools.sat.python import cp_model  # noqa: F401
        return True
    except Exception:
        return False


def default_backend() -> str:
    return "gurobi" if gurobi_available() else ("cpsat" if ortools_available() else "none")

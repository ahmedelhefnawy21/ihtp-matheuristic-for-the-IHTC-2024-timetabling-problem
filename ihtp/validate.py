"""Wrapper around the official C++ validator.

Re-check every claimed solution with the official IHTP_Validator binary, not
just our own Python evaluator. Independent feasibility+quality gate for Task 2d.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from dataclasses import dataclass

from . import config
from .io_instance import Instance
from .model import SolutionState
from .writer import write_solution


@dataclass
class ValidatorResult:
    violations: int
    cost: int
    raw: str

    @property
    def feasible(self) -> bool:
        return self.violations == 0


def run_validator(instance_path: str, solution_path: str,
                  validator_bin: str = config.VALIDATOR_BIN) -> ValidatorResult:
    """Run the validator binary on an instance+solution and parse the totals.

    Binary is the authority. Fail loud on anything that smells like a bad run:
    non-zero exit (crash, unreadable file) or unparseable output. A crashed or
    error-printing validator that we trust would wave an invalid solution
    straight through the feasibility gate."""
    proc = subprocess.run([validator_bin, instance_path, solution_path],
                          capture_output=True, text=True)
    out, err = proc.stdout, proc.stderr
    tv = re.search(r"Total violations = (\d+)", out)
    tc = re.search(r"Total cost = (\d+)", out)
    if proc.returncode != 0 or tv is None or tc is None:
        raise RuntimeError(
            f"validator did not run cleanly (exit code {proc.returncode}) on\n"
            f"  instance={instance_path}\n  solution={solution_path}\n"
            f"--- stdout ---\n{out}\n--- stderr ---\n{err}")
    return ValidatorResult(violations=int(tv.group(1)), cost=int(tc.group(1)), raw=out)


def validate_state(inst: Instance, st: SolutionState, instance_path: str) -> ValidatorResult:
    """Dump state to a temp file, hand it to the official binary."""
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        tmp = fh.name
    try:
        write_solution(inst, st, tmp)
        return run_validator(instance_path, tmp)
    finally:
        import os
        os.remove(tmp)

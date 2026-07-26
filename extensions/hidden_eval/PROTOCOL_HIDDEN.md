# Pre-specified protocol: held-out evaluation on the competition's hidden instances

Registered: **2026-07-25 18:44 CEST**, before any hidden-instance run was started.
This file is written first and not edited afterwards.

## Why this evaluation exists

Every number reported so far was produced on the 30 public instances, which we
also used for development and tuning. The report states this as its main
limitation. The competition kept a second, hidden set of 30 instances
(m01 to m30) for exactly this reason, and has since published both the
instances and the best solutions found during the final evaluation. A single
run of the frozen pipeline on that set measures generalization directly.

## Design under test (frozen)

The pipeline exactly as committed in `ihtp/` at registration time. No code
change, no budget change, no tuning of any kind. Budgets come from
`ihtp.experiments.budgets(inst)`, the same function that produced every public
result. Seeds 1 to 5, best of five reported, the same policy as the public
campaign. The project's own environment (`.venv`: Python 3.12.13, gurobipy
13.0.2, ortools 9.15.6755, numpy 2.5.1).

## Data

- Instances: `data/instances_hidden/m01.json` to `m30.json`, downloaded from
  the competition site on the registration date.
- Benchmarks: the competition's published best solutions for the hidden set,
  re-scored by the official validator
  (`data/reference_solutions_hidden/best_known_hidden.csv`; all 30 are
  feasible, and the values match the UB column of the competition paper on
  every instance spot-checked).

## Reporting (declared now)

This is an estimation protocol, not a hypothesis test. We will report,
regardless of outcome:

1. Feasibility on every instance and seed, judged by the official validator.
2. The total-based aggregate gap to the published best solutions over all 30
   hidden instances, the same convention as the public results.
3. The per-instance best-of-five objective, benchmark, and gap.
4. The comparison the reader needs: the public-set aggregate gap (7.8%) next
   to the hidden-set aggregate gap, with the difference stated plainly.

Failures, infeasibilities, or a worse-than-public outcome are reported exactly
as they land. No re-runs, no seed selection, no post-hoc exclusions.

## Result-set discipline

A separate result set. The public-set headline numbers are unchanged. Nothing
from this evaluation feeds back into any method choice in this paper.

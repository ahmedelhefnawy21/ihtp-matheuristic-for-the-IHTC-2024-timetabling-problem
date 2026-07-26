# Pre-specified protocol: confirmatory campaign for the supply-aware upper layer

Registered: **2026-07-24 19:05 CEST**, before any campaign run was started.
This file is written first and not edited afterwards; the campaign results are
interpreted against exactly what is declared here.

## Design under test (frozen)

The v2 supply-aware patch in `skillmix_pipeline.py`, exactly as committed to
disk at registration time:

- `k_load = 1.0` — exact per-shift S4 lower bound: max(0, patient+occupant
  workload demand − total on-duty nurse capacity), priced at W4.
- `k_scarce = 1.0` — per-day top-skill scarcity: max(0, rooms whose top daily
  requirement is the top level − top-skill nurses on duty, min over the day's
  three shifts), priced at W2.
- `k_spread = 0.0` (the v1 spread term stays off).

**No tuning of any kind during or after the campaign**: no kappa adjustment, no
seed selection, no instance exclusion, no post-hoc variant. Failures, if any,
are reported.

## Runs

All 30 public instances × all 5 seeds (1–5) = 150 runs, production budgets
(`experiments.budgets`), the project's own environment
(`ADVANCED_MODELLING/.venv`: Python 3.12.13, gurobipy 13.0.2, ortools
9.15.6755, numpy 2.5.1 — identical to `results/environment.txt`), one process
per run, solver threads pinned as in the main experiments. Every reported
number is the official validator's, zero violations required.

## Instance split (declared now, before results exist)

- **Development set (5): i01, i02, i17, i21, i29.** These instances influenced
  the design and may carry selection effects; they are reported separately and
  claim nothing. Exposure record: v1's failure was diagnosed on i01/i02; the
  skill-data degeneracy scan used i01/i21/i29; v2 (this design) was evaluated
  on all five before this registration.
- **Held-out set (25): the remaining instances.** No design decision, data
  inspection, or evaluation ever touched them under any variant of this patch.
  All confirmatory claims rest on this set alone.

## Comparisons (same conventions as the report)

1. Same-seed: patched(i, s) vs `results/seed_runs.csv` (i, s) — 150 exact
   deterministic pairs (the unpatched control reproduces the baseline
   bit-for-bit, verified).
2. Best-of-5 per instance: patched vs baseline.
3. Aggregate gap to best-known, total-based (Σours − Σbk)/Σbk over the frozen
   snapshot in `ihtp/config.py`, computed per set (held-out / development /
   all).

## Success criteria (declared now)

The design is **confirmed** iff, on the held-out 25 alone:
(a) the total-based best-of-5 aggregate gap improves versus baseline, and
(b) same-seed deltas do not worsen on strict majority of the 125 held-out
    pairs.
Secondary description (no claim): the improvement is expected to concentrate on
nurse-heavy instances and to be small or absent on instances whose rosters are
slack — that is the mechanism's own prediction, since both terms vanish under
slack.

Anything short of (a)+(b) is reported as **not confirmed**, with the same
tables.

## Result-set discipline

This campaign is a separate result set. It does not modify, and is never mixed
with, the report's headline results, which remain those of the frozen pipeline.

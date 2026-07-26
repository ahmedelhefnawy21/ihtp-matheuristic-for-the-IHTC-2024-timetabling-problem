# Pre-specified protocol: placebo (permutation) control for the supply-aware upper layer

Registered: **2026-07-25 07:00 CEST**, before the placebo was implemented or any
placebo run was started. Written first and not edited afterwards.

## Why this control exists

The confirmatory campaign (PROTOCOL.md) met its declared criteria: on the 25
held-out instances the best-of-5 aggregate gap improved 6.78% -> 6.39%
(-0.39 pp). Post-hoc analysis, however, showed the effect is carried by a few
large improvements rather than a broad tendency: per-instance 13 better / 12
worse (Wilcoxon p = 0.68), per-run 56 better / 62 worse (sign test p = 0.65),
and the pre-declared concentration prediction was not supported by rank
correlation (Spearman +0.20, wrong direction).

That leaves one causal question unresolved:

> Does pricing **true** nurse-supply pressure improve the aggregate, or would
> **any** comparable perturbation of the search's guidance signal reshuffle
> trajectories so that a few instances land better?

This control answers it.

## The placebo

A **permutation control**: the identical two terms, identical wiring, identical
kappa values, computed against a **temporally shuffled roster**.

Concretely, inside the patched `Layout.__init__`, after the true supply vectors
are built, both are permuted with a fixed placebo seed:

- `_w_cap` (total on-duty nurse capacity per shift) is permuted across shifts;
- `_k_top` (top-skill nurses on duty per day) is permuted across days.

Everything else is untouched: the demand side is real, the functional form is
identical, and the multiset of capacities is preserved exactly (so total roster
capacity is unchanged). What is destroyed is the **alignment** between nurse
supply and the shifts and days that supply actually belongs to. The permutation
is deterministic given the placebo seed and the instance, so every `Layout`
built during a run sees the same shuffled roster, and runs remain reproducible.

Placebo seed: **7**. kappa unchanged: `k_load = 1.0`, `k_scarce = 1.0`,
`k_spread = 0.0`. No tuning of any kind.

**Stated limitation of this design (declared now):** the placebo is an
*alignment* control, not an information-free one. Demand remains real, so the
placebo term still penalizes demand concentration to some degree, and
misalignment may make it a *stronger* perturbation than the true term. Both
initial term magnitudes are recorded per instance so the comparison is
transparent. An information-free control (a random room-day price of matched
magnitude) is named as the further control, not run here.

## Runs

All 30 instances x 5 seeds = 150 runs, production budgets, the project's own
environment, official validator on every result -- identical to the confirmatory
campaign in every respect except the shuffled roster.

## Sets

Unchanged from PROTOCOL.md: development = {i01, i02, i17, i21, i29}, held-out =
the other 25. All inference below uses the held-out 25 only.

## Decision rule (declared now)

Let `D_real = -0.392 pp` be the held-out best-of-5 aggregate gap change already
measured for the true patch, and `D_plac` the same quantity for the placebo.

- **Causal attribution supported** iff `D_plac` is materially smaller in
  magnitude than `D_real` -- specifically, `D_plac > D_real / 2` (i.e. the
  placebo recovers less than half the improvement) **and** the bootstrap
  distribution of `D_real - D_plac` over held-out instances excludes zero.
- **Causal attribution fails** iff the placebo reproduces a comparable
  improvement (`D_plac <= D_real / 2`), in which case the aggregate gain is
  attributed to trajectory perturbation rather than to supply-awareness, and
  will be reported as such.
- Any intermediate outcome is reported as **inconclusive**, with both numbers.

Secondary, no claim attached: per-instance and per-run sign counts and the
initial term magnitudes for real vs placebo.

## Result-set discipline

A separate result set again. Neither this control nor the confirmatory campaign
modifies the report's headline results, which remain those of the frozen
pipeline.

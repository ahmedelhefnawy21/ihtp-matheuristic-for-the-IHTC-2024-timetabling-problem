# Budget-slack probe: how much of the nurse gap is the NRA solve itself?

Registered: 2026-07-25 19:35 CEST, after two exploratory re-solves (i22, i17;
recorded below) and before the systematic probe ran.

## Why

The structural comparison in this folder found that the published best
solutions win on all three nurse terms at layouts whose structure matches
ours. That points at the nurse solve rather than the layout. The report infers
the opposite: it treats the NRA-MIP as exact given the layout, and assigns
the residual nurse gap to the admission-nurse coupling. The two exploratory
re-solves contradict that premise on large instances: with the layout frozen
and only the NRA work budget raised from its production ceiling (600 to 800)
to 3000, i17 improved by 2,940 (continuity -2,010) and i22 by 665.

## Design

For each probe instance, the stored best-of-five public solution is loaded,
the layout is left untouched, and the NRA-MIP is re-solved once at work
budget 3000. The official validator scores the result. Instances: the six
largest-gap nurse-heavy instances (i27, i17, i29, i21, i22, i19) plus i02 as
a small control, where the production budget should already be sufficient and
the probe should find nothing. Both exploratory runs are included in the
report of results, marked as exploratory.

## Reporting (declared now)

Per instance: objective before and after, the three nurse components before
and after, runtime, and the share of the instance's gap to best-known that
the re-solve closes. The result feeds a correction to the report's diagnosis,
whatever the direction: if the probe closes little, the coupling inference
stands; if it closes much, the diagnosis must split the nurse gap into a
solver-budget share and a residual coupling share.

## Result-set discipline

A separate result set. The headline numbers stay frozen; no pipeline change.

# Structure comparison and the nurse-budget probe

The component analysis places most of our remaining gap in the three nurse
terms. This folder measures where that gap sits, in two steps.

Step one compares the structure of our public-set solutions with the
competition's published best solutions on the same instances
(compare_structure.py, structure_diff.csv). The published best solutions do
not win through stay alignment or room consolidation; those metrics match
ours. They win on all three nurse terms at layouts of similar structure,
which points at the nurse solve rather than the layout.

Step two tests that reading directly (PROBE.md, probe_nra_budget.py,
probe_summary.csv): with each stored layout frozen, the NRA-MIP is re-solved
once at work budget 3000 instead of the production 600 to 800. On the six largest-gap
nurse-heavy instances this alone closes between 6% and 63% of the gap to best-known,
mostly in continuity. The pooled closure over the seven probed instances is 21.9%. The
small control instance (i02) closes nothing, as the protocol predicted. The probe was
registered in PROBE.md before the systematic runs;
two exploratory runs (i17, i22) came first and are disclosed there.

- compare_structure.py    the structural metrics, ours vs published best
- structure_diff.csv      one row per instance and side
- PROBE.md                the registered probe protocol, with the two
                          exploratory runs disclosed
- probe_nra_budget.py     one re-solve per instance, layout untouched
- probe_iXX.json          per-instance probe records (validator-scored) for
                          the five post-registration runs; the exploratory
                          i17 and i22 runs appear only in probe_summary.csv
- probe_run_iXX.log       the solver log of each post-registration run
- probe_summary.csv       the probe table, all seven rows: gap closure
                          percentages and continuity reductions (cost units)

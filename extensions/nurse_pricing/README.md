# Nurse-aware upper layer: pre-registered test + placebo control

This experiment is reported in Appendix F of the paper. It runs the unmodified
pipeline with two supply-aware terms monkey-patched into the upper-layer
surrogate (skillmix_pipeline.py); the pipeline package itself is not modified.

- PROTOCOL.md            pre-registered confirmatory protocol (design frozen,
                         dev/held-out split, success criteria)
- PROTOCOL_PLACEBO.md    pre-registered placebo (permutation) control
- skillmix_pipeline.py the patch and single-run script (--no-patch control
                         reproduces results/seed_runs.csv; verified bit-for-bit
                         on i01 seed 1, reproducible per pair via the deterministic script)
- run_campaign.py        30x5 campaign driver (resumable; --placebo N)
- summarize_campaign.py  aggregates the confirmatory campaign vs its criteria
- compare_placebo.py     applies the placebo decision rule
- campaign_summary.csv   per-instance outcome, real patch
- placebo_summary.csv    per-instance outcome, real vs placebo
- runs.csv               all 300 runs (both arms): objective, violations, and
                         the eight weighted cost components per run
- analyze_posthoc.py     reproduces every post-hoc number in Appendix F from
                         runs.csv (bootstrap CIs, Wilcoxon, Spearman,
                         component decomposition, patient deltas)
- record_term_magnitudes.py  rebuilds, from the instance data alone, the
                         per-instance initial term magnitudes of both arms
- term_magnitudes.csv    that record: one row per instance and arm, the
                         transparency record PROTOCOL_PLACEBO.md requires for
                         comparing the two perturbation strengths

Per-run outputs (150 result JSONs + logs per campaign) are regenerated
deterministically by run_campaign.py under the environment recorded in
results/environment.txt. analyze_posthoc.py works directly from the shipped runs.csv.
summarize_campaign.py and compare_placebo.py read the per-run JSONs, so the campaigns
must be re-run first. Both scripts refuse to overwrite the shipped summary CSVs when
those JSONs are absent.

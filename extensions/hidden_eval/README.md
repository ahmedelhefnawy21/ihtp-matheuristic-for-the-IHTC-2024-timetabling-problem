# Held-out evaluation on the competition's hidden instances

The public results of this project were produced on instances that also served
development and tuning. This evaluation runs the frozen pipeline once on the
competition's hidden set (m01 to m30), which was published after the competition. That
measures generalization directly. The protocol was registered before any run.

- PROTOCOL_HIDDEN.md      the pre-registered protocol (design frozen, reporting
                          declared, no tuning of any kind)
- run_hidden.py           campaign driver: 30 instances x 5 seeds, frozen
                          pipeline, production budgets, resumable
- summarize_hidden.py     aggregates the campaign against the published best
                          solutions; prints feasibility, per-instance gaps, and
                          the total-based aggregate gap
- hidden_summary.csv      the per-instance table behind Appendix H (also the
                          source of report/hidden_table.tex via report/make_hidden.py)
- result_mXX_sY.json      one record per run: validator objective, violations,
                          runtime, stage costs, cost components
- sol_mXX_sY.json         the solution behind each record
- log_mXX_sY.txt          the stdout of each run's subprocess (the first
                          smoke-test run, m01 seed 1, ran directly and has none)
- hidden_driver.log       the campaign driver's own log
- HIDDEN_DONE.txt         written by the driver when all 150 runs finished

Instances live in data/instances_hidden/. Benchmarks live in
data/reference_solutions_hidden/ (the competition's published best solutions,
re-scored by the official validator).

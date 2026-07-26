#!/usr/bin/env python3
"""Aggregate the hidden-instance campaign against PROTOCOL_HIDDEN.md.

The script reads the per-run records in this folder and the published
benchmarks in data/reference_solutions_hidden/best_known_hidden.csv. It prints
the declared report: feasibility on every run, the per-instance best-of-five
objective and gap, and the total-based aggregate gap, next to the public-set
aggregate for context. Every objective it reads is the official validator's.
"""
from __future__ import annotations

import csv
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
NAMES = [f"m{i:02d}" for i in range(1, 31)]
PUBLIC_AGG_GAP = 7.80  # Section 4 of the report, total-based over i01 to i30


def main() -> None:
    runs = {}
    for p in glob.glob(os.path.join(HERE, "result_m*_s*.json")):
        r = json.load(open(p))
        runs[(r["instance"], r["seed"])] = r

    bk = {}
    with open(os.path.join(ROOT, "data", "reference_solutions_hidden",
                           "best_known_hidden.csv")) as fh:
        for row in csv.DictReader(fh):
            bk[row["instance"]] = int(row["best_known_cost"])

    missing = [(n, s) for n in NAMES for s in range(1, 6) if (n, s) not in runs]
    if missing:
        print(f"INCOMPLETE: {len(missing)} runs missing, e.g. {missing[:5]}")

    per = []
    viol_total = 0
    for n in NAMES:
        seeds = [runs[(n, s)] for s in range(1, 6) if (n, s) in runs]
        if not seeds:
            continue
        viol_total += sum(r["violations"] for r in seeds)
        best = min(r["objective"] for r in seeds)
        per.append((n, best, bk[n], (best - bk[n]) / bk[n] * 100,
                    len(seeds), sum(r["runtime_s"] for r in seeds)))

    print(f"{'inst':6}{'best-of-5':>10}{'best-known':>11}{'gap%':>8}{'seeds':>7}{'runtime_s':>11}")
    for n, best, k, gap, ns, rt in per:
        print(f"{n:6}{best:>10}{k:>11}{gap:>8.2f}{ns:>7}{rt:>11.0f}")

    tot_ours = sum(p[1] for p in per)
    tot_bk = sum(p[2] for p in per)
    agg = (tot_ours - tot_bk) / tot_bk * 100
    print(f"\nfeasible runs: {sum(1 for k_ in runs.values() if k_['violations'] == 0)}"
          f"/{len(runs)} (violations total {viol_total})")
    print(f"hidden-set aggregate gap (total-based): {agg:.2f}%")
    print(f"public-set aggregate gap for context:   {PUBLIC_AGG_GAP:.2f}%")
    print(f"difference (hidden minus public):       {agg - PUBLIC_AGG_GAP:+.2f} points")

    with open(os.path.join(HERE, "hidden_summary.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["instance", "best_of_5", "best_known", "gap_pct", "seeds", "runtime_s_total"])
        for row in per:
            w.writerow([row[0], row[1], row[2], round(row[3], 2), row[4], round(row[5], 1)])
    print("\nwrote hidden_summary.csv")


if __name__ == "__main__":
    main()

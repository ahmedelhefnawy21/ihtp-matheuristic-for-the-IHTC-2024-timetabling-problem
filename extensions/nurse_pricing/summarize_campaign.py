#!/usr/bin/env python3
"""Aggregate the confirmatory campaign against PROTOCOL.md's declared criteria.

The script writes campaign_summary.csv, one row per instance, and prints the pre-specified
verdict: held-out aggregate gap (a) and held-out same-seed sign counts (b).
"""
import csv
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
from ihtp import config  # noqa: E402

TAG = "sp0.0_ld1.0_sc1.0"
DEV = {"i01", "i02", "i17", "i21", "i29"}
ALL = [f"i{i:02d}" for i in range(1, 31)]


def load_campaign():
    rows = {}
    for path in glob.glob(os.path.join(HERE, f"results_i*_s*_{TAG}.json")):
        for rec in json.load(open(path)):
            if "error" in rec:
                print(f"!! failed run in {os.path.basename(path)}: {rec['error']}")
                continue
            rows[(rec["instance"], rec["seed"])] = rec
    return rows


def load_baseline_seeds():
    base = {}
    with open(os.path.join(config.RESULTS_DIR, "seed_runs.csv")) as fh:
        for row in csv.DictReader(fh):
            base[(row["instance"], int(row["seed"]))] = int(row["objective"])
    return base


def main() -> None:
    camp = load_campaign()
    if not camp:
        sys.exit(f"no results_i*_s*_{TAG}.json found in {HERE}: this folder ships "
                 "the aggregated summaries, not the per-run records; run "
                 "run_campaign.py first (see README.md). Refusing to overwrite "
                 "campaign_summary.csv.")
    base = load_baseline_seeds()
    bk = config.BEST_KNOWN

    missing = [(n, s) for n in ALL for s in range(1, 6) if (n, s) not in camp]
    if missing:
        print(f"INCOMPLETE: {len(missing)} pairs missing, e.g. {missing[:5]}")

    per_inst = []
    for n in ALL:
        seeds = [camp[(n, s)] for s in range(1, 6) if (n, s) in camp]
        if not seeds:
            continue
        viol = sum(r["patched_violations"] for r in seeds)
        best_p = min(r["patched_objective"] for r in seeds)
        best_b = min(base[(n, s)] for s in range(1, 6))
        deltas = [r["patched_objective"] - base[(n, r["seed"])] for r in seeds]
        per_inst.append(dict(
            instance=n, dev=n in DEV, n_seeds=len(seeds), violations=viol,
            best_patched=best_p, best_baseline=best_b, best_delta=best_p - best_b,
            same_seed_deltas=deltas,
            n_improved=sum(1 for d in deltas if d < 0),
            n_equal=sum(1 for d in deltas if d == 0),
            n_worse=sum(1 for d in deltas if d > 0),
            best_known=bk[n],
        ))

    with open(os.path.join(HERE, "campaign_summary.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["instance", "set", "best_patched", "best_baseline", "best_delta",
                    "gap_patched_%", "gap_baseline_%", "seeds_improved", "seeds_equal",
                    "seeds_worse", "violations"])
        for r in per_inst:
            w.writerow([r["instance"], "dev" if r["dev"] else "held-out",
                        r["best_patched"], r["best_baseline"], r["best_delta"],
                        round((r["best_patched"] - r["best_known"]) / r["best_known"] * 100, 2),
                        round((r["best_baseline"] - r["best_known"]) / r["best_known"] * 100, 2),
                        r["n_improved"], r["n_equal"], r["n_worse"], r["violations"]])

    for label, sel in [("HELD-OUT (25), the confirmatory set", [r for r in per_inst if not r["dev"]]), ("DEVELOPMENT (5), reported separately, no claims", [r for r in per_inst if r["dev"]])]:
        tot_p = sum(r["best_patched"] for r in sel)
        tot_b = sum(r["best_baseline"] for r in sel)
        tot_k = sum(r["best_known"] for r in sel)
        pairs = [d for r in sel for d in r["same_seed_deltas"]]
        imp = sum(1 for d in pairs if d < 0)
        eq = sum(1 for d in pairs if d == 0)
        wor = sum(1 for d in pairs if d > 0)
        print(f"\n=== {label} ===")
        print(f"  instances better/equal/worse on best-of-5: "
              f"{sum(1 for r in sel if r['best_delta'] < 0)}/"
              f"{sum(1 for r in sel if r['best_delta'] == 0)}/"
              f"{sum(1 for r in sel if r['best_delta'] > 0)}")
        print(f"  same-seed pairs improved/equal/worse: {imp}/{eq}/{wor} of {len(pairs)}")
        print(f"  aggregate gap (total-based): baseline {(tot_b - tot_k) / tot_k * 100:.2f}%  ->  "
              f"patched {(tot_p - tot_k) / tot_k * 100:.2f}%   "
              f"(sum objective {tot_b} -> {tot_p}, delta {tot_p - tot_b:+d})")
        print(f"  total violations across runs: {sum(r['violations'] for r in sel)}")
        if "HELD-OUT" in label:
            crit_a = tot_p < tot_b
            crit_b = imp + eq > len(pairs) / 2
            print(f"\n  PROTOCOL criterion (a) held-out aggregate improves: "
                  f"{'MET' if crit_a else 'NOT MET'}")
            print(f"  PROTOCOL criterion (b) majority of held-out pairs non-worsening: "
                  f"{'MET' if crit_b else 'NOT MET'}  ({imp}+{eq} vs {wor})")
            print(f"  VERDICT: {'CONFIRMED' if crit_a and crit_b else 'NOT CONFIRMED'}")

    allr = per_inst
    tot_p = sum(r["best_patched"] for r in allr)
    tot_k = sum(r["best_known"] for r in allr)
    tot_b = sum(r["best_baseline"] for r in allr)
    print(f"\n=== ALL 30 (context only) ===")
    print(f"  aggregate gap: baseline {(tot_b - tot_k) / tot_k * 100:.2f}%  ->  "
          f"patched {(tot_p - tot_k) / tot_k * 100:.2f}%")
    print(f"\nwrote campaign_summary.csv")


if __name__ == "__main__":
    main()

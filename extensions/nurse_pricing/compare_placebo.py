#!/usr/bin/env python3
"""Apply PROTOCOL_PLACEBO.md's declared decision rule to the placebo campaign.

D_real / D_plac = held-out best-of-5 aggregate gap change (percentage points)
for the true patch and the permutation control. Causal attribution is supported
iff the placebo recovers less than half the improvement AND the bootstrap
distribution of (D_real - D_plac) over held-out instances excludes zero.
"""
import csv
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
from ihtp import config  # noqa: E402

REAL_TAG = "sp0.0_ld1.0_sc1.0"
PLAC_TAG = "sp0.0_ld1.0_sc1.0_pl7"
DEV = {"i01", "i02", "i17", "i21", "i29"}
NAMES = [f"i{i:02d}" for i in range(1, 31)]


def load(tag):
    out = {}
    for path in glob.glob(os.path.join(HERE, f"results_i*_s*_{tag}.json")):
        if tag == REAL_TAG and path.endswith("_pl7.json"):
            continue
        for rec in json.load(open(path)):
            if "error" not in rec:
                out[(rec["instance"], rec["seed"])] = rec["patched_objective"]
    return out


def main() -> None:
    real, plac = load(REAL_TAG), load(PLAC_TAG)
    base = {}
    with open(os.path.join(config.RESULTS_DIR, "seed_runs.csv")) as fh:
        for row in csv.DictReader(fh):
            base[(row["instance"], int(row["seed"]))] = int(row["objective"])
    bk = config.BEST_KNOWN

    miss = [(n, s) for n in NAMES for s in range(1, 6) if (n, s) not in plac]
    if miss:
        print(f"INCOMPLETE placebo: {len(miss)} missing, e.g. {miss[:4]}")

    ho = [n for n in NAMES if n not in DEV]
    ho = [n for n in ho if all((n, s) in plac for s in range(1, 6))]
    if not ho:
        sys.exit(f"no complete placebo per-run records (results_i*_s*_{PLAC_TAG}.json) "
                 f"found in {HERE}: this folder ships the aggregated summaries, not "
                 "the per-run records; re-run both campaigns first (see README.md). "
                 "Refusing to overwrite placebo_summary.csv.")

    def best(d, n):
        return min(d[(n, s)] for s in range(1, 6))

    B = np.array([best(base, n) for n in ho], float)
    R = np.array([best(real, n) for n in ho], float)
    P = np.array([best(plac, n) for n in ho], float)
    K = np.array([bk[n] for n in ho], float)

    def gap(v, idx):
        return (v[idx].sum() - K[idx].sum()) / K[idx].sum() * 100

    allidx = np.arange(len(ho))
    d_real = gap(R, allidx) - gap(B, allidx)
    d_plac = gap(P, allidx) - gap(B, allidx)
    print(f"held-out instances used: {len(ho)}")
    print(f"  baseline aggregate gap : {gap(B, allidx):.3f}%")
    print(f"  real patch             : {gap(R, allidx):.3f}%   D_real = {d_real:+.3f} pp")
    print(f"  placebo (roster shuffled): {gap(P, allidx):.3f}%   D_plac = {d_plac:+.3f} pp")

    rng = np.random.default_rng(0)
    diffs = []
    for _ in range(20000):
        idx = rng.integers(0, len(ho), len(ho))
        diffs.append((gap(R, idx) - gap(B, idx)) - (gap(P, idx) - gap(B, idx)))
    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    obs = d_real - d_plac
    print(f"\n  D_real - D_plac = {obs:+.3f} pp   95% CI [{lo:+.3f}, {hi:+.3f}]"
          f"   P(real better) = {(diffs < 0).mean():.3f}")

    crit1 = d_plac > d_real / 2          # placebo recovers less than half
    crit2 = hi < 0                        # bootstrap excludes zero, real better
    print(f"\n  criterion 1 (placebo recovers < half of the gain): "
          f"{'MET' if crit1 else 'NOT MET'}  (D_plac {d_plac:+.3f} vs D_real/2 {d_real / 2:+.3f})")
    print(f"  criterion 2 (bootstrap of D_real-D_plac excludes 0): "
          f"{'MET' if crit2 else 'NOT MET'}")
    if crit1 and crit2:
        verdict = "CAUSAL ATTRIBUTION SUPPORTED"
    elif d_plac <= d_real / 2:
        verdict = "CAUSAL ATTRIBUTION FAILS: placebo reproduces the gain (trajectory effect)"
    else:
        verdict = "INCONCLUSIVE"
    print(f"\n  VERDICT: {verdict}")

    pr = np.array([real[(n, s)] - base[(n, s)] for n in ho for s in range(1, 6)], float)
    pp = np.array([plac[(n, s)] - base[(n, s)] for n in ho for s in range(1, 6)], float)
    print(f"\n  secondary (no claim): same-seed pairs improved/worse, "
          f"real {int((pr < 0).sum())}/{int((pr > 0).sum())}, "
          f"placebo {int((pp < 0).sum())}/{int((pp > 0).sum())}")
    print(f"  per-instance better/worse, real {int((R < B).sum())}/{int((R > B).sum())}, "
          f"placebo {int((P < B).sum())}/{int((P > B).sum())}")

    with open(os.path.join(HERE, "placebo_summary.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["instance", "baseline_best", "real_best", "placebo_best",
                    "real_delta", "placebo_delta"])
        for i, n in enumerate(ho):
            w.writerow([n, int(B[i]), int(R[i]), int(P[i]),
                        int(R[i] - B[i]), int(P[i] - B[i])])
    print("\nwrote placebo_summary.csv")


if __name__ == "__main__":
    main()

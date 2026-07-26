#!/usr/bin/env python3
"""Reproduce every post-hoc number in Appendix F from runs.csv.

Reads: runs.csv (both arms, per-run objective/violations/components),
../../results/seed_runs.csv (per-seed baselines), ../../results/iXX.json
(baseline best-of-5 solutions, re-scored for components), and the frozen
best-known snapshot in ihtp.config. Prints: aggregate gaps and criteria,
same-seed and per-instance sign counts, wins/losses sums, bootstrap CIs
(rng seed 0, 20,000 percentile resamples), Wilcoxon (normal approximation,
relative per-instance changes), Spearman vs baseline nurse-cost share,
component decompositions, and postponed-patient deltas.
"""
import csv
import os
import sys
from math import erf, sqrt

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
from ihtp import config  # noqa: E402
from ihtp.io_instance import load_instance  # noqa: E402
from ihtp.objective import evaluate  # noqa: E402
from ihtp.writer import read_solution  # noqa: E402

SOFT = ["RoomAgeMix", "RoomSkillLevel", "ContinuityOfCare", "ExcessiveNurseWorkload",
        "OpenOperatingTheater", "SurgeonTransfer", "PatientDelay", "ElectiveUnscheduledPatients"]
DEV = {"i01", "i02", "i17", "i21", "i29"}
NAMES = [f"i{i:02d}" for i in range(1, 31)]
HO = [n for n in NAMES if n not in DEV]


def main() -> None:
    runs = {}
    with open(os.path.join(HERE, "runs.csv")) as fh:
        for r in csv.DictReader(fh):
            runs[(r["instance"], int(r["seed"]), r["arm"])] = r
    base = {}
    with open(os.path.join(config.RESULTS_DIR, "seed_runs.csv")) as fh:
        for r in csv.DictReader(fh):
            base[(r["instance"], int(r["seed"]))] = int(r["objective"])
    bk = config.BEST_KNOWN

    viol = sum(int(r["violations"]) for r in runs.values())
    print(f"runs: {len(runs)} (violations total: {viol})")

    def best_run(n, arm):
        return min((runs[(n, s, arm)] for s in range(1, 6)),
                   key=lambda r: int(r["objective"]))

    B = np.array([min(base[(n, s)] for s in range(1, 6)) for n in HO], float)
    R = np.array([int(best_run(n, "real")["objective"]) for n in HO], float)
    P = np.array([int(best_run(n, "placebo")["objective"]) for n in HO], float)
    K = np.array([bk[n] for n in HO], float)

    def gap(v, idx):
        return (v[idx].sum() - K[idx].sum()) / K[idx].sum() * 100

    idx = np.arange(len(HO))
    d_real, d_plac = gap(R, idx) - gap(B, idx), gap(P, idx) - gap(B, idx)
    print(f"\nheld-out aggregate gap: baseline {gap(B, idx):.2f}%  real {gap(R, idx):.2f}%  "
          f"placebo {gap(P, idx):.2f}%   D_real {d_real:+.3f} pp  D_plac {d_plac:+.3f} pp")

    pd = np.array([int(runs[(n, s, "real")]["objective"]) - base[(n, s)]
                   for n in HO for s in range(1, 6)], float)
    print(f"same-seed pairs (real): {int((pd < 0).sum())} improved, "
          f"{int((pd == 0).sum())} unchanged, {int((pd > 0).sum())} worse")
    deltas = R - B
    print(f"per-instance: {int((deltas < 0).sum())} better, {int((deltas > 0).sum())} worse; "
          f"wins sum {int(deltas[deltas < 0].sum())}, losses sum +{int(deltas[deltas > 0].sum())}")

    rng = np.random.default_rng(0)
    bs = np.array([gap(R, i) - gap(B, i)
                   for i in (rng.integers(0, len(HO), len(HO)) for _ in range(20000))])
    print(f"bootstrap CI D_real: [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}]")
    rng = np.random.default_rng(0)
    bs2 = np.array([(gap(R, i) - gap(B, i)) - (gap(P, i) - gap(B, i))
                    for i in (rng.integers(0, len(HO), len(HO)) for _ in range(20000))])
    print(f"bootstrap CI D_real - D_plac: "
          f"[{np.percentile(bs2, 2.5):+.3f}, {np.percentile(bs2, 97.5):+.3f}]")

    rel = (R - B) / B * 100
    x = rel[rel != 0]
    rr = np.argsort(np.argsort(np.abs(x))) + 1
    W = min(rr[x < 0].sum(), rr[x > 0].sum())
    nn = len(x)
    z = (W - nn * (nn + 1) / 4) / sqrt(nn * (nn + 1) * (2 * nn + 1) / 24)
    print(f"Wilcoxon (relative changes, normal approx): p = "
          f"{2 * 0.5 * (1 + erf(-abs(z) / sqrt(2))):.2f}")

    head_comps, shares = {}, []
    for n in HO:
        inst = load_instance(config.instance_path(n))
        c = evaluate(inst, read_solution(
            inst, os.path.join(config.RESULTS_DIR, f"{n}.json"))).weighted_components()
        head_comps[n] = c
        nurse = c["RoomSkillLevel"] + c["ContinuityOfCare"] + c["ExcessiveNurseWorkload"]
        shares.append(nurse / max(1, sum(c.values())) * 100)
    shares = np.array(shares)
    ra, rb = np.argsort(np.argsort(rel)), np.argsort(np.argsort(shares))
    print(f"Spearman (relative change vs nurse share): {np.corrcoef(ra, rb)[0, 1]:+.2f}")

    for arm in ("real", "placebo"):
        d = {k: 0 for k in SOFT}
        pats = 0
        for n in HO:
            br = best_run(n, arm)
            inst_w8 = int(load_instance(config.instance_path(n)).weights[7])
            for k in SOFT:
                d[k] += int(br[k]) - head_comps[n][k]
            pats += (int(br["ElectiveUnscheduledPatients"])
                     - head_comps[n]["ElectiveUnscheduledPatients"]) // inst_w8
        recoup = d["PatientDelay"] + d["RoomSkillLevel"] + d["ContinuityOfCare"] \
            + d["ExcessiveNurseWorkload"]
        print(f"\n{arm}: component deltas {d}")
        print(f"  unscheduled {d['ElectiveUnscheduledPatients']:+} "
              f"(~{pats:+} patients), delay+nurse recoup {recoup:+}, "
              f"total {sum(d.values()):+}")


if __name__ == "__main__":
    main()

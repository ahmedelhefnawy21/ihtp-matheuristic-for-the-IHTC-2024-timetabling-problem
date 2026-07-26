#!/usr/bin/env python3
"""Structural comparison of our public-set solutions with the published best.

The component analysis locates our remaining gap in the nurse terms, with
continuity (S3) the largest share. This script measures where that advantage
lives in the published best solutions, instance by instance, using only the
two solution files and the instance data. Each metric targets one candidate
mechanism:

- distinct nurses per patient: the direct S3 driver
- roommate stay overlap: aligned stays let fewer nurses cover a room
- room churn (patients per used room) and occupied room-days: how
  consolidated the layout is
- workload per shift (mean and peak vs capacity): census smoothness, the S4
  driver
- top-skill room-days per top-skill nurse on duty: skill demand versus
  supply, the S2 driver
- admitted optionals and mean admission delay: the patient-side trade

The output is one row per instance and side in structure_diff.csv, and a
comparison table for the nurse-heaviest instances. Every metric comes from
re-scoring the stored solutions; nothing is re-solved.
"""
from __future__ import annotations

import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

from ihtp import config  # noqa: E402
from ihtp.io_instance import load_instance  # noqa: E402
from ihtp.objective import evaluate  # noqa: E402
from ihtp.writer import read_solution  # noqa: E402

NAMES = [f"i{i:02d}" for i in range(1, 31)]
FOCUS = ["i17", "i27", "i19", "i21", "i29", "i22"]


def day_levels(arr, k):
    return int(arr[3 * k:3 * k + 3].max())


def metrics(inst, st):
    P, R, D = inst.P, inst.R, inst.days
    # distinct nurses per patient (patients only; occupants excluded for a
    # like-for-like patient-side view)
    distinct = []
    for p in range(P):
        d, r = int(st.adm[p]), int(st.room[p])
        if d < 0:
            continue
        nurses = set()
        for dd in range(d, min(D, d + int(inst.p_los[p]))):
            for sh in range(3):
                n = st.cover[r, dd * 3 + sh]
                if n >= 0:
                    nurses.add(int(n))
        distinct.append(len(nurses))
    # room occupancy structure
    occ_days = np.zeros((R, D), dtype=bool)
    room_patients = [[] for _ in range(R)]
    for p in range(P):
        d, r = int(st.adm[p]), int(st.room[p])
        if d < 0:
            continue
        occ_days[r, d:min(D, d + int(inst.p_los[p]))] = True
        room_patients[r].append(p)
    for r in range(R):
        for d in range(D):
            if inst.occ_in_room_day[r][d]:
                occ_days[r, d] = True
    # roommate stay overlap (patients sharing a room, pairwise, Jaccard on days)
    overlaps = []
    for r in range(R):
        ps = room_patients[r]
        for a in range(len(ps)):
            for b in range(a + 1, len(ps)):
                p, q = ps[a], ps[b]
                s1 = set(range(int(st.adm[p]), min(D, int(st.adm[p]) + int(inst.p_los[p]))))
                s2 = set(range(int(st.adm[q]), min(D, int(st.adm[q]) + int(inst.p_los[q]))))
                if s1 | s2:
                    overlaps.append(len(s1 & s2) / len(s1 | s2))
    # workload demand per shift vs capacity
    dem = np.zeros(inst.shifts)
    for p in range(P):
        d = int(st.adm[p])
        if d < 0:
            continue
        w = inst.p_workload[p]
        base = d * 3
        dem[base:min(inst.shifts, base + len(w))] += w[:max(0, min(inst.shifts, base + len(w)) - base)]
    for o in range(inst.O):
        w = inst.o_workload[o]
        dem[:len(w)] += w
    cap = inst.nurse_max_load.sum(axis=0)
    # top-skill demand vs supply per day
    top = inst.skill_levels - 1
    is_top = inst.nurse_skill == top
    top_sup = ((inst.nurse_max_load > 0) & is_top[:, None]).sum(axis=0)
    top_rooms = 0
    for r in range(R):
        for d in range(D):
            lv = -1
            for p in room_patients[r]:
                k = d - int(st.adm[p])
                if 0 <= k < int(inst.p_los[p]):
                    lv = max(lv, day_levels(inst.p_skillreq[p], k))
            for o in inst.occ_in_room_day[r][d]:
                lv = max(lv, day_levels(inst.o_skillreq[o], d))
            if lv == top:
                top_rooms += 1
    top_sup_days = sum(int(top_sup[d * 3:(d + 1) * 3].min()) for d in range(D))
    # patient-side
    adm = [p for p in range(P) if st.adm[p] >= 0]
    opt_admitted = sum(1 for p in adm if not inst.p_mandatory[p])
    delay = float(np.mean([int(st.adm[p]) - int(inst.p_release[p]) for p in adm])) if adm else 0.0
    c = evaluate(inst, st)
    comps = c.weighted_components()
    return dict(
        nurses_per_patient=round(float(np.mean(distinct)), 2),
        stay_overlap=round(float(np.mean(overlaps)), 3) if overlaps else 0.0,
        patients_per_used_room=round(len(adm) / max(1, int((occ_days.any(axis=1)).sum())), 2),
        occupied_room_days=int(occ_days.sum()),
        workload_peak_ratio=round(float((dem / np.maximum(cap, 1)).max()), 2),
        workload_mean_ratio=round(float((dem.sum() / max(1, cap.sum()))), 2),
        top_rooms_per_top_nurse_day=round(top_rooms / max(1, top_sup_days), 2),
        optionals_admitted=opt_admitted,
        mean_delay=round(delay, 2),
        S2=comps["RoomSkillLevel"], S3=comps["ContinuityOfCare"],
        S4=comps["ExcessiveNurseWorkload"], total=c.total_cost,
    )


def main() -> None:
    rows = []
    for n in NAMES:
        inst = load_instance(config.instance_path(n))
        ours = metrics(inst, read_solution(
            inst, os.path.join(config.RESULTS_DIR, f"{n}.json")))
        best = metrics(inst, read_solution(
            inst, os.path.join(ROOT, "data", "reference_solutions", f"sol_{n}.json")))
        rows.append((n, "ours", ours))
        rows.append((n, "best", best))

    keys = list(rows[0][2].keys())
    with open(os.path.join(HERE, "structure_diff.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["instance", "side"] + keys)
        for n, side, m in rows:
            w.writerow([n, side] + [m[k] for k in keys])

    print("nurse-heavy focus instances, ours vs published best:")
    show = ["nurses_per_patient", "stay_overlap", "patients_per_used_room",
            "workload_peak_ratio", "top_rooms_per_top_nurse_day",
            "optionals_admitted", "mean_delay", "S2", "S3", "S4"]
    for n in FOCUS:
        a = next(m for x, s, m in rows if x == n and s == "ours")
        b = next(m for x, s, m in rows if x == n and s == "best")
        print(f"\n[{n}]")
        for k in show:
            print(f"  {k:28}{a[k]:>10}{b[k]:>10}")

    # aggregate view over all 30
    print("\naggregates over 30 instances (ours vs best):")
    for k in show:
        va = np.mean([m[k] for x, s, m in rows if s == "ours"])
        vb = np.mean([m[k] for x, s, m in rows if s == "best"])
        print(f"  {k:28}{va:>10.2f}{vb:>10.2f}")
    print("\nwrote structure_diff.csv")


if __name__ == "__main__":
    main()

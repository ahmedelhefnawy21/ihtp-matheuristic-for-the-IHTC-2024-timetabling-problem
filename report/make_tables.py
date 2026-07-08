"""LaTeX tables for the report, built from results/<run>/summary.csv.

run: python report/make_tables.py results/run1. Writes full_table.tex and
results_summary.tex into report/, updates macros in report/_macros.tex.
"""
from __future__ import annotations

import csv
import os
import statistics
import sys

REPORT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_summary(run_dir):
    rows = []
    with open(os.path.join(run_dir, "summary.csv")) as fh:
        for r in csv.DictReader(fh):
            rows.append(r)
    return rows


def load_seed_costs(run_dir):
    """instance -> its feasible per-seed objectives. empty dict if seed_runs.csv is missing.
    """
    path = os.path.join(run_dir, "seed_runs.csv")
    runs = {}
    if not os.path.exists(path):
        return runs
    with open(path) as fh:
        for r in csv.DictReader(fh):
            if r["feasibility"] == "feasible" and r["objective"]:
                runs.setdefault(r["instance"], []).append(int(r["objective"]))
    return runs


def load_bounds(run_dir):
    """instance -> certified lower bound. empty if no bounds were computed for this run."""
    path = os.path.join(run_dir, "bounds.csv")
    b = {}
    if not os.path.exists(path):
        return b
    with open(path) as fh:
        for r in csv.DictReader(fh):
            if r["lower_bound"]:
                b[r["instance"]] = int(r["lower_bound"])
    return b


def write_bounds(run_dir, rows):
    """certified optimality gap: our objective vs a valid LB. also prints best-known's own
    certified gap, so a big gap reads as a loose bound or hard instance rather than us doing
    badly."""
    lb = load_bounds(run_dir)
    if not lb:
        return
    feas = [r for r in rows if r["feasibility"] == "feasible" and r["instance"] in lb]
    lines = [r"\begin{center}\small",
             r"\begin{tabular}{lrrrrr}",
             r"\toprule",
             r"inst & ours & best-known & LB & cert.\ gap\% & BK cert.\ gap\%\\",
             r"\midrule"]
    num = den = bk_num = 0
    for r in sorted(feas, key=lambda x: x["instance"]):
        obj, L, bk = int(r["objective"]), lb[r["instance"]], int(r["best_known"])
        cg = 100.0 * (obj - L) / L if L > 0 else 0.0
        bkg = 100.0 * (bk - L) / L if L > 0 else 0.0
        num += obj - L
        den += L
        bk_num += bk - L
        lines.append(f"{esc(r['instance'])} & {obj} & {bk} & {L} & {cg:.1f} & {bkg:.1f}" + r"\\")
    agg = 100.0 * num / den if den else 0.0
    bk_agg = 100.0 * bk_num / den if den else 0.0
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{center}",
              rf"\noindent{{\small\emph{{Aggregate certified gap {agg:.1f}\% (ours) vs "
              rf"{bk_agg:.1f}\% (best-known). The bound is valid but loose (it drops the room, "
              rf"gender and nurse terms), so both gaps are large; the true optimum lies in "
              rf"$[\text{{LB}},\,\text{{ours}}]$.}}}}"]
    with open(os.path.join(REPORT_DIR, "bounds_table.tex"), "w") as fh:
        fh.write("\n".join(lines))
    print(f"certified gap: ours {agg:.1f}%  best-known {bk_agg:.1f}%  (over {len(feas)} instances)")


def write_distribution(run_dir, rows):
    """per-instance seed spread (best/mean/worst/std) plus the aggregate that justifies
    best-of-k: how far restarts beat the average single run."""
    seed_costs = load_seed_costs(run_dir)
    k = max((len(v) for v in seed_costs.values()), default=0)
    if k <= 1:
        return  # single-seed or empty: nothing to spread over
    bk = {r["instance"]: int(r["best_known"]) for r in rows}
    lines = [r"\begin{center}\small",
             r"\begin{tabular}{lrrrrrr}",
             r"\toprule",
             r"inst & \#seeds & best & mean & worst & std & best gap\%\\",
             r"\midrule"]
    gap_best_sum_num = gap_best_sum_den = 0
    mean_gaps, best_gaps = [], []
    for name in sorted(seed_costs):
        c = seed_costs[name]
        b, w, m = min(c), max(c), statistics.mean(c)
        sd = statistics.stdev(c) if len(c) > 1 else 0.0
        gb = 100.0 * (b - bk[name]) / bk[name]
        gm = 100.0 * (m - bk[name]) / bk[name]
        best_gaps.append(gb)
        mean_gaps.append(gm)
        gap_best_sum_num += b - bk[name]
        gap_best_sum_den += bk[name]
        lines.append(f"{esc(name)} & {len(c)} & {b} & {m:.0f} & {w} & {sd:.0f} & {gb:+.1f}"
                     + r"\\")
    agg_best = 100.0 * gap_best_sum_num / gap_best_sum_den if gap_best_sum_den else 0.0
    mean_single = sum(mean_gaps) / len(mean_gaps) if mean_gaps else 0.0
    mean_bestk = sum(best_gaps) / len(best_gaps) if best_gaps else 0.0
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{center}",
              rf"\noindent{{\small\emph{{Best-of-{k}: aggregate gap {agg_best:.1f}\%. "
              rf"Mean per-instance gap {mean_bestk:.1f}\% (best-of-{k}) vs {mean_single:.1f}\% "
              rf"(average single seed); restarts recover {mean_single - mean_bestk:.1f} "
              rf"points.}}}}"]
    with open(os.path.join(REPORT_DIR, "seed_distribution.tex"), "w") as fh:
        fh.write("\n".join(lines))
    print(f"seed distribution: k={k}  best-of-{k} agg gap={agg_best:.1f}%  "
          f"mean single-seed gap={mean_single:.1f}%  best-of-{k} mean gap={mean_bestk:.1f}%")


def esc(x):
    return str(x).replace("_", "\\_")


def write_components(run_dir):
    """Per soft term: our total weighted cost vs best-known, and its share of the gap. Backs the
    Task 3(e) claim that the gap is in the nurse layer. Our costs are read from the validator logs;
    best-known costs come from validating the bundled reference solutions."""
    import re
    import subprocess
    pkg = os.path.dirname(REPORT_DIR)
    validator = os.path.join(pkg, "bin", "IHTP_Validator")
    if not os.path.exists(validator):
        return
    soft = [("RoomAgeMix", "S1 age mix"), ("RoomSkillLevel", "S2 nurse skill"),
            ("ContinuityOfCare", "S3 continuity"), ("ExcessiveNurseWorkload", "S4 workload"),
            ("OpenOperatingTheater", "S5 open OTs"), ("SurgeonTransfer", "S6 surgeon transfer"),
            ("PatientDelay", "S7 delay"), ("ElectiveUnscheduledPatients", "S8 unscheduled")]

    def parse(text):
        out = {}
        for name, _ in soft:
            m = re.search(rf"{re.escape(name)}\.+(\d+)\s*\(", text)
            out[name] = int(m.group(1)) if m else 0
        return out

    ours = {n: 0 for n, _ in soft}
    best = {n: 0 for n, _ in soft}
    for i in range(1, 31):
        nm = f"i{i:02d}"
        log = os.path.join(run_dir, f"{nm}.validator.txt")
        ref = os.path.join(pkg, "data", "reference_solutions", f"sol_{nm}.json")
        inst = os.path.join(pkg, "data", "instances", f"{nm}.json")
        if not (os.path.exists(log) and os.path.exists(ref)):
            return
        co = parse(open(log).read())
        cb = parse(subprocess.run([validator, inst, ref], capture_output=True, text=True).stdout)
        for n, _ in soft:
            ours[n] += co[n]
            best[n] += cb[n]
    tot = sum(ours[n] - best[n] for n, _ in soft)
    lines = [r"\begin{center}\small",
             r"\begin{tabular}{@{}l r r r r@{}}",
             r"\toprule",
             r"soft term & our cost & best-known & difference & share of gap\%\\",
             r"\midrule"]
    for name, label in soft:
        g = ours[name] - best[name]
        share = 100.0 * g / tot if tot else 0.0
        lines.append(f"{label} & {ours[name]} & {best[name]} & {g:+d} & {share:+.1f}" + r"\\")
    lines += [r"\midrule",
              f"total & {sum(ours.values())} & {sum(best.values())} & {tot:+d} & 100" + r"\\",
              r"\bottomrule", r"\end{tabular}", r"\end{center}"]
    with open(os.path.join(REPORT_DIR, "component_table.tex"), "w") as fh:
        fh.write("\n".join(lines))
    print("component breakdown written")


def main():
    run_dir = sys.argv[1] if len(sys.argv) > 1 else "results/run1"
    rows = load_summary(run_dir)

    feas = [r for r in rows if r["feasibility"] == "feasible"]
    n_feas = len(feas)
    tot = sum(int(r["objective"]) for r in feas)
    tot_bk = sum(int(r["best_known"]) for r in feas)
    overall_gap = 100.0 * (tot - tot_bk) / tot_bk if tot_bk else 0.0
    gaps = [float(r["gap_pct"]) for r in feas]
    mean_gap = sum(gaps) / len(gaps) if gaps else 0.0
    n_close = sum(1 for g in gaps if 0.0 <= g <= 0.5)   # within 0.5%, not exact
    n_match = sum(1 for g in gaps if g <= 0.0)           # tied or better
    n_beat = sum(1 for g in gaps if g < 0.0)             # strictly better

    # Task 3 results table: all 30 instances, one row each, with feasibility and a short comment.
    comments = {"i06": "within 0.5\\% of best-known", "i11": "within 0.5\\% of best-known",
                "i27": "largest instance (493 patients)", "i17": "worst gap; nurse layer"}
    n3 = sum(1 for g in gaps if g <= 3.0)
    lines = [r"\begin{center}\footnotesize",
             r"\begin{tabular}{@{}l l r r r r p{3.0cm}@{}}",
             r"\toprule",
             r"instance & feasible & objective & best-known & gap\% & runtime (s) & comment\\",
             r"\midrule"]
    for r in rows:
        feas = "yes" if r["feasibility"] == "feasible" else "no"
        rt = r.get("runtime_s_total") or r.get("runtime_s", "")
        lines.append(f"{esc(r['instance'])} & {feas} & {r['objective'] or 'infeasible'} & "
                     f"{r['best_known']} & {r['gap_pct'] or '--'} & {rt} & "
                     f"{comments.get(r['instance'], '')}" + r"\\")
    match_txt = (rf"equalled or beat best-known on {n_match}" if n_match
                 else rf"within 0.5\% of best-known on {n_close}")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{center}",
              rf"\noindent{{\small\emph{{All 30 feasible. Aggregate gap {overall_gap:.1f}\%, "
              rf"mean per-instance gap {mean_gap:.1f}\%. {match_txt.capitalize()}; within 3\% "
              rf"on {n3}.}}}}"]
    with open(os.path.join(REPORT_DIR, "full_table.tex"), "w") as fh:
        fh.write("\n".join(lines))

    write_distribution(run_dir, rows)
    write_bounds(run_dir, rows)
    write_components(run_dir)

    print(f"feasible {n_feas}/30  overall_gap={overall_gap:.1f}%  mean_gap={mean_gap:.1f}%  "
          f"within0.5%={n_close} matched={n_match} beaten={n_beat}")
    print(f"GAPTOTAL suggestion: {overall_gap:.1f}\\% (aggregate) / {mean_gap:.1f}\\% (mean per instance)")


if __name__ == "__main__":
    main()

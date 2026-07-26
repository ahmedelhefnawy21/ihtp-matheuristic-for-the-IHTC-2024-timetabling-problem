"""Per-instance optional-admission counts (ours vs reference) -> admissions_table.tex.

Backs the Task 3(e) claim by counting admitted optional patients directly from the solution
JSONs, rather than inferring counts from the weighted S8 cost (whose weight varies by instance).
The weighted-S8 totals implied by these counts reconcile exactly with the official validator's
ElectiveUnscheduledPatients column in component_table.tex (409,750 ours / 417,950 reference).

Run: python report/make_admissions.py
"""
from __future__ import annotations

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT_DIR = os.path.dirname(os.path.abspath(__file__))


def _admitted(sp) -> bool:
    """A patient is admitted iff admission_day is an integer; 'none'/null/absent is unscheduled."""
    ad = sp.get("admission_day")
    if isinstance(ad, bool):
        return False
    if isinstance(ad, int):
        return True
    return isinstance(ad, str) and ad.isdigit()


def _counts(name):
    inst = json.load(open(os.path.join(ROOT, "data", "instances", f"{name}.json")))
    opt = {p["id"] for p in inst["patients"] if not p.get("mandatory")}
    our = {p["id"]: p for p in json.load(open(os.path.join(ROOT, "results", f"{name}.json")))["patients"]}
    rp = os.path.join(ROOT, "data", "reference_solutions", f"sol_{name}.json")
    ref = {p["id"]: p for p in json.load(open(rp))["patients"]}
    oa = sum(1 for pid in opt if pid in our and _admitted(our[pid]))
    ra = sum(1 for pid in opt if pid in ref and _admitted(ref[pid]))
    return len(opt), oa, ra


def _fmt(v):
    return f"{v:,}".replace(",", "\\,")


def main():
    names = [f"i{k:02d}" for k in range(1, 31)]
    data = {n: _counts(n) for n in names}
    tot_opt = sum(v[0] for v in data.values())
    tot_our = sum(v[1] for v in data.values())
    tot_ref = sum(v[2] for v in data.values())

    per = 10  # three side-by-side panels of 10 rows keeps the table to a few lines
    panels = [names[0:per], names[per:2 * per], names[2 * per:3 * per]]
    lines = [r"\begin{center}\footnotesize",
             r"\captionof{table}{Optional patients admitted per instance (opt $=$ optional patients; ours vs the "
             r"reference solutions), counted directly rather than from the weighted \Sc8.}"
             r"\label{tab:adm}",
             r"\begin{tabular}{@{}lrrr@{\quad}lrrr@{\quad}lrrr@{}}",
             r"\toprule",
             r"inst & opt & ours & ref & inst & opt & ours & ref & inst & opt & ours & ref\\",
             r"\midrule"]
    for i in range(per):
        cells = []
        for panel in panels:
            if i < len(panel):
                n = panel[i]
                o, oa, ra = data[n]
                cells.append(f"{n} & {_fmt(o)} & {_fmt(oa)} & {_fmt(ra)}")
            else:
                cells.append(" & & & ")
        lines.append(" & ".join(cells) + r"\\")
    lines += [r"\midrule",
              rf"\multicolumn{{12}}{{@{{}}l}}{{Total {_fmt(tot_our)} of {_fmt(tot_opt)} optionals "
              rf"admitted, versus {_fmt(tot_ref)} in the reference solutions ({tot_our - tot_ref:+d}).}}\\",
              r"\bottomrule", r"\end{tabular}", r"\end{center}"]
    with open(os.path.join(REPORT_DIR, "admissions_table.tex"), "w") as fh:
        fh.write("\n".join(lines))
    print(f"wrote admissions_table.tex: admitted {tot_our} ours vs {tot_ref} ref ({tot_our - tot_ref:+d})")


if __name__ == "__main__":
    main()

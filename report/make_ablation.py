"""cumulative ablation on representative instances -> ablation_table.tex.

Usage: python report/make_ablation.py i04 i13 i16 i27
run this after the main experiment, on a free box: it re-runs the whole pipeline. budgets come
from ``ihtp.experiments.budgets`` (deterministic, no time argument).
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ihtp import load_instance                         # noqa: E402
from ihtp import config                                # noqa: E402
from ihtp.experiments import ablation                  # noqa: E402

REPORT_DIR = os.path.dirname(os.path.abspath(__file__))
COLS = ["construct", "+pas", "+alns", "+descent", "+lns", "+exact_ot", "+exact_nra"]
HEAD = ["construct", "+PAS", "+ALNS", "+descent", "+LNS", "+OT", "+NRA", "best-known"]


def fmt(v):
    return f"{v:,}".replace(",", "\\,") if isinstance(v, int) else "--"


def main():
    names = sys.argv[1:] or ["i04", "i13", "i16", "i27"]

    rows = []
    for name in names:
        inst = load_instance(config.instance_path(name))
        st = ablation(inst, seed=1)
        rows.append((name, st, config.BEST_KNOWN[name]))
        print(name, st)

    lines = [r"\begin{center}\small",
             r"\captionof{table}{Cumulative ablation, adding the stages in pipeline order on four instances "
             r"that span the size range from the small i04 to the largest i27 (seed 1, so the "
             r"best-of-5 headline objective can be marginally lower). Each kept stage lowers the "
             r"overall cost: the PAS-MIP gives the largest drop (admissions), the CP-SAT LNS and "
             r"the NRA polish trim every row, and descent and the OT polish improve the instances "
             r"where their cost term is not already minimal. Construction alone is infeasible on "
             r"i16, so it has no scored construct objective there.}\label{tab:abl}",
             r"\begin{tabular}{l" + "r" * (len(HEAD)) + "}",
             r"\toprule",
             "instance & " + " & ".join(HEAD) + r"\\",
             r"\midrule"]
    for name, st, bk in rows:
        cells = [fmt(st.get(c)) for c in COLS] + [fmt(bk)]
        lines.append(f"{name} & " + " & ".join(cells) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{center}"]
    with open(os.path.join(REPORT_DIR, "ablation_table.tex"), "w") as fh:
        fh.write("\n".join(lines))
    print("wrote ablation_table.tex")


if __name__ == "__main__":
    main()

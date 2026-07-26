"""LaTeX table for the held-out evaluation, built from
extensions/hidden_eval/hidden_summary.csv.

run: python report/make_hidden.py. The script writes hidden_table.tex into report/.
"""
import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "extensions", "hidden_eval", "hidden_summary.csv")

rows = list(csv.DictReader(open(SRC)))
# A table float keeps the caption and the tabular on the same page; the #
# earlier center+captionof form left the caption stranded at a page break.
lines = [r"\begin{table}[!ht]",
         r"\caption{The held-out evaluation: the frozen pipeline, run once on the "
         r"competition's hidden instances under the pre-registered protocol of "
         r"\texttt{extensions/hidden\_eval}, best of five seeds per instance, scored by the "
         r"official validator. The benchmark is the best published solution per instance, "
         r"re-scored by the same validator. Runtime is the solver time summed over the five "
         r"seeds.}\label{tab:hidden}",
         r"\centering\small",
         r"\begin{tabular}{@{}lrrrr@{}}",
         r"\toprule",
         r"instance & best-of-5 & published best & gap\% & runtime (s)\\",
         r"\midrule"]
for r in rows:
    lines.append(f"{r['instance']} & {int(r['best_of_5']):,} & {int(r['best_known']):,} & "
                 f"{float(r['gap_pct']):.2f} & {float(r['runtime_s_total']):,.0f}\\\\"
                 .replace(",", "\\,"))
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
open(os.path.join(HERE, "hidden_table.tex"), "w").write("\n".join(lines) + "\n")
print("wrote hidden_table.tex,", len(rows), "rows")

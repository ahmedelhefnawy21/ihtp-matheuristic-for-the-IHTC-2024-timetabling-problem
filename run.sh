#!/usr/bin/env bash
# one command, deterministic, self-contained run of the IHTC-2024 solver.
#
# writes to results/: one solution JSON per instance + its official validator log,
# results.csv, summary.csv, environment.txt (CPU, cores, RAM, OS, solver versions,
# wall-clock for this machine).
#
# determinism: PYTHONHASHSEED pinned. every solver stops on a fixed rule (iteration
# counts, solver WorkLimit). same seed -> identical objectives on any machine with the
# same solver versions; Gurobi and CP-SAT are deterministic within a version. runtime
# is the only thing that moves with hardware.
#
# parallelism: all CPU cores by default (--jobs 0). --jobs N to cap.
#
# usage:  ./run.sh                 # full repro: all 30, best-of-5 seeds, bounds, tables
#         ./run.sh --jobs 8        # cap parallel processes (default: all cores)
#         ./run.sh --seeds 1       # faster single seed, skips the distribution
set -euo pipefail
cd "$(dirname "$0")"

export PYTHONHASHSEED=0                                   # stable set/dict iteration
# one BLAS thread per process. we already fan out one process per instance, so threaded
# BLAS just oversubscribes the CPU. also pins determinism.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
# GRB_LICENSE_FILE set but pointing at nothing -> fall back to ~/gurobi.lic instead of
# dying. catches a placeholder path someone pasted by mistake.
if [ -n "${GRB_LICENSE_FILE:-}" ] && [ ! -f "${GRB_LICENSE_FILE}" ]; then
  echo ">> note: GRB_LICENSE_FILE='${GRB_LICENSE_FILE}' not found; falling back to \$HOME/gurobi.lic"
  unset GRB_LICENSE_FILE
fi
export GRB_LICENSE_FILE="${GRB_LICENSE_FILE:-$HOME/gurobi.lic}"
PY=./.venv/bin/python

# 1. deps, first run only
if [ ! -x "$PY" ]; then
  echo ">> creating venv + installing requirements"
  python3.12 -m venv .venv
  ./.venv/bin/pip install -q -r requirements.txt
fi

# 2. compile the official validator if missing (source + json.hpp vendored in bin/)
if [ ! -x bin/IHTP_Validator ]; then
  echo ">> compiling official validator"
  g++ -O2 -std=c++17 -I bin -o bin/IHTP_Validator bin/IHTP_Validator.cc
fi

# 3. correctness gate: Python objective has to match the official C++ validator
echo ">> golden test (evaluator vs official validator)"
"$PY" tests/test_golden.py | tail -1
"$PY" tests/crosscheck.py | tail -1

# 3b. bail early on a bad Gurobi license, before the long solve
echo ">> checking Gurobi license ($GRB_LICENSE_FILE)"
if ! "$PY" -c "import gurobipy; gurobipy.Env()" >/dev/null 2>&1; then
  echo "" >&2
  echo ">> ERROR: could not initialise Gurobi with GRB_LICENSE_FILE='$GRB_LICENSE_FILE'." >&2
  echo "   Fix: put your gurobi.lic at \$HOME/gurobi.lic and run './run.sh' (no env var)," >&2
  echo "   or run 'GRB_LICENSE_FILE=/full/path/to/gurobi.lic ./run.sh' with the REAL path." >&2
  exit 1
fi

# 4. solve everything, deterministic + parallel. best-of-5 seeds is what the report numbers use.
echo ">> solving all 30 (deterministic, best-of-5 seeds, all cores)"
"$PY" -m ihtp.experiments --instances all --seeds 5 --out results "$@"

# 5. certified lower bounds, deterministic Gurobi dual bounds -> results/bounds.csv
echo ">> computing certified lower bounds"
"$PY" -m ihtp.experiments --instances all --bounds --out results "$@"

# 6. cumulative ablation on the representative instances -> report/ablation_table.tex
echo ">> ablation (representative instances)"
"$PY" report/make_ablation.py i04 i13 i16 i27

# 7. rebuild the report tables from results/: summary, distribution, bounds, full, admissions
echo ">> generating report tables"
"$PY" report/make_tables.py results
"$PY" report/make_admissions.py

echo ">> done. Outputs in results/ (results.csv, summary.csv, seed_runs.csv, bounds.csv,"
echo "   environment.txt) and report/ (*.tex tables). Compile report/report.tex for the PDF."

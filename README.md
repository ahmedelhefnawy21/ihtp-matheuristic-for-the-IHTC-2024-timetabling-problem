# ihtp: a matheuristic for the IHTC-2024 timetabling problem

`ihtp` plans a hospital's admissions, surgeries, and nurse duty for a few weeks at once. It solves
the Integrated Healthcare Timetabling Problem (IHTP) from the
[IHTC-2024 competition](https://ihtc2024.github.io/) (IHTC is the Integrated Healthcare Timetabling
Competition).

It is feasible on all 30 public instances, with a total cost **7.8% above the best result the
competition field has published**. Every number here is reproducible from this repo alone.

## The problem

A hospital plans a schedule over a fixed number of days. Each day is split into three shifts:
early, late, and night. Patients are waiting for surgery. For every patient, the hospital decides
three things:

- **which day to admit them** (some can wait, some cannot),
- **which room** they stay in for their whole stay,
- **which operating theatre** does their surgery (surgery happens on the admission day).

At the same time, for **every room in every shift that has a patient in it**, the hospital puts
**one nurse on duty** to cover it.

Everything competes for limited resources. A room has a fixed number of beds. A surgeon can operate
only so many minutes a day. An operating theatre has limited hours. A nurse can carry only so much
work per shift and works only certain shifts. Patients come in two kinds: **mandatory** ones that
must be admitted by a deadline, and **optional** ones that can be postponed.

A valid plan respects the hard limits: no room over its beds, no surgeon or theatre over its daily
time, no mixed genders in a room, every mandatory patient admitted on time, and every occupied
room-shift staffed by one on-duty nurse. Among all valid plans, I want the cheapest. The cost adds
up penalties for:

- **an optional patient left unscheduled** (weight 150 to 500, far above the rest, so admitting
  people is the main goal),
- a patient made to wait past the day they became ready,
- too many open operating theatres, or a surgeon sent between theatres in one day,
- one patient seen by many different nurses across their stay,
- a room staffed by a nurse below the skill a patient needs, or past their workload limit,
- different age groups mixed in one room.

IHTP is the name for solving all of this at once. It fuses three problems that share the same beds,
staff, and theatres:

- **PAS, Patient Admission Scheduling:** the admission day and room per patient.
- **SCP, Surgical Case Planning:** the theatre and day per surgery.
- **NRA, Nurse-to-Room Assignment:** the nurse per room-shift.

If the three are solved separately, the plans are worse: a good admission plan can be impossible to
staff well, and a nurse-friendly plan can waste beds. The point of the competition, and of this
solver, is to treat them as one problem.

## Results

- Feasible on all 30 public instances. The official validator confirms zero violations.
- Total cost 7.8% above the competition best-known, using the best of 5 random seeds.
- Within 3% of best-known on 7 instances, within 5% on 12.
- A certified lower bound per instance, so the distance to the true optimum is bounded, not guessed.

## Repository map

```
THIS_REPO/
├── README.md                  this file
├── run.sh                     the whole pipeline in one command
├── requirements.txt           the three Python dependencies
│
├── ihtp/                      the solver (one Python package, 21 modules)
│
├── data/
│   ├── instances/             the 30 competition instances (i01.json .. i30.json)
│   ├── reference_solutions/   the 30 published best-known solutions (for comparison only)
│   └── best_known_ihtc2024.csv  best-known cost per instance, a dated snapshot
│
├── bin/
│   ├── IHTP_Validator.cc      the official competition validator (source)
│   └── json.hpp               its one dependency; run.sh compiles the binary here
│
├── tests/
│   ├── test_golden.py         my scorer must match the official validator on i01
│   └── crosscheck.py          the same check on 8 instances across the size range
│
├── report/
│   ├── report.pdf             the write-up: model, method, results
│   ├── report.tex             its source
│   └── make_tables.py, make_ablation.py   report tables, rebuilt from results/
│
└── results/                   run.sh output: solutions, validator logs, CSVs
```

Everything needed to run and check the solver is here. The reference solutions and best-known
values are used only to measure the gap. The solver never reads them.

## Requirements

- Python 3.12.
- A Gurobi license. Gurobi solves the admission, theatre, and nurse models. The license file goes at
  `~/gurobi.lic`, or `GRB_LICENSE_FILE` points to it.
- `g++`, to compile the validator, only if the bundled binary does not run on your machine.
- macOS or Linux. Windows needs WSL (Windows Subsystem for Linux).

Python packages (`requirements.txt`): `numpy`, `gurobipy`, `ortools`.

## Quick start

```bash
python3.12 -m venv .venv
./.venv/bin/pip install -r requirements.txt
# license: gurobi.lic at ~/gurobi.lic, or GRB_LICENSE_FILE=/full/path/gurobi.lic
./run.sh
```

This folder can go anywhere. Every path is worked out relative to the code, so nothing needs
configuring.

`run.sh` runs the whole pipeline in order: it compiles the validator, checks my scorer against it,
verifies the Gurobi license, solves all 30 instances (best of 5 seeds, across all cores), computes
the lower bounds, runs the ablation, and rebuilds the report tables. The full run took about 5.8
hours on a 10-core Apple M5. `./run.sh --seeds 1` runs a single seed and is about five times faster.

## Outputs

Everything lands in `results/`:

| File | What it holds |
|---|---|
| `iNN.json` | the best solution for instance NN, in the official format |
| `iNN.validator.txt` | the official validator's log for that solution |
| `results.csv` | one line per instance: `i01,<cost>` (or `i04,infeasible`) |
| `summary.csv` | per instance: feasibility, cost, best-known, gap, total runtime |
| `seed_runs.csv` | every (instance, seed): cost and runtime, the full spread |
| `bounds.csv` | certified lower bound and solver status per instance |
| `environment.txt` | the machine and solver versions that produced this run |

## How it works

The solver is a **matheuristic**: a heuristic (fast, approximate search) wrapped around exact
**MIP** models (Mixed-Integer Programs, the standard tool for scheduling and packing). The
heuristic explores broadly and cheaply. The MIPs solve the parts that must be exact.

The split is on purpose. The cost is dominated by unscheduled optional patients, and admitting more
of them is a packing problem: fit as many surgeries as possible into the fixed surgeon time,
theatre time, and beds. Local search alone stalls on packing, because the rearrangement that frees
space for one more patient can be large and it never tries it. A MIP is suitable for this. But
a single MIP over the whole problem, nurses included, is too large to solve. So I use a MIP where
it pays (admission) and lighter tools everywhere else.

The plan passes through seven stages. Each one is here because turning it off makes the result
worse. The report has the ablation that shows this.

1. **Construction.** A first valid plan, built fast: patients are admitted hardest-first, each into
   its cheapest open spot. It is rough, but a real starting point.

2. **PAS-MIP (admission).** The admission decision, solved exactly: for each patient, the day and
   the room, honoring beds, gender, surgeon time, and theatre time. This stage sets who gets in and
   when, which drives most of the cost.

3. **ALNS (Adaptive Large Neighborhood Search).** The plan improves by repeated tear-and-rebuild: a
   whole day, one theatre, or a group of related patients gets emptied and re-inserted a better way.
   "Adaptive" means it tracks which moves pay off on this instance and favors them. Some worse plans
   are accepted early (a simulated-annealing rule), so it does not get trapped, then it tightens up.

4. **Descent.** Small, safe moves: one patient shifts to a better room or day, and the move stays
   only if the full cost drops. This trims the secondary costs, including the nurse costs, since it
   re-checks the whole score.

5. **CP-SAT LNS (Large Neighborhood Search).** The capacity-creating step, and the one that admits
   the last stubborn optionals. Every still-unscheduled optional, plus everyone admitted inside a
   random window of days, is freed and re-packed exactly with **CP-SAT** (the constraint solver in
   Google OR-Tools). An exact re-pack of a slice can open room that no small move would find.

6. **OT-MIP (theatres).** With days and rooms fixed, this stage picks the operating theatre (OT) for
   each surgery, to use as few theatres as possible and keep each surgeon in one theatre per day. It
   is tiny, one exact solve per day.

7. **NRA-MIP (nurses).** With the full layout fixed, this stage assigns nurses to minimize
   under-skilling, workload overflow, and the number of different nurses a patient sees. It is
   solved exactly.

One finding shaped the design. Once admission is solved well, almost all of the remaining gap is in
the nurse layer, not in who got admitted. So the nurse model (stage 7) gets a larger, size-scaled
budget than a naive setting would give it.

Every solution is re-checked by the official competition validator, and
a result counts as feasible only when the validator reports zero violations.

## Inside the package

`ihtp/` is one Python package. Its modules fall into four groups, and data flows down the list: an
instance is read and turned into fast tables, the heuristic core searches, the exact models sharpen
the parts that need it, and the orchestration layer runs it all and writes the output.

**The foundation (read and score).**

- `io_instance.py` reads an instance file and precomputes the lookup tables the search needs.
- `model.py` holds one solution: the day, room, and theatre per patient, and the nurse per
  room-shift.
- `layout.py` keeps running totals of cost and capacity, so testing a move costs time proportional
  to what the move touches, not to the whole instance. This is what makes the search fast enough.
- `objective.py` is the scorer, a faithful reimplementation of the official validator. The search
  optimizes exactly what the competition measures. A golden test pins the two together.

**Heuristic core (most of the work).**

- `construct.py` builds the first valid plan.
- `nra.py` does a quick greedy nurse assignment.
- `alns.py` is the ALNS engine: the tear-and-rebuild operators, the learning that reweights them,
  and the acceptance rule.
- `local_search.py` is the descent.

**Exact models (each aimed at one part of the cost).**

- `exact_pas.py` is the admission MIP (stage 2).
- `exact_lns.py` is the CP-SAT capacity-creating search (stage 5).
- `exact_ot.py` is the theatre MIP (stage 6).
- `exact_nra.py` is the nurse MIP (stage 7).
- `bounds.py` computes a certified lower bound by solving a relaxed version of the problem, which
  says how far below my cost the true optimum could still be.

**Orchestration and I/O.**

- `pipeline.py` runs the seven stages in order; `polish.py` runs the two final exact models.
- `solvers.py` is the thin layer over Gurobi and CP-SAT, with the settings that make every solve
  deterministic.
- `writer.py` turns a solution into the official JSON; `validate.py` runs the official validator on
  it.
- `experiments.py` drives the whole thing across instances and seeds, computes the bounds, and
  writes the CSVs.
- `config.py` holds the paths (all relative to the package) and loads the best-known snapshot.

## Reproducibility

The method is deterministic. Nothing stops on a clock. The heuristic runs a fixed number of
iterations, and every solver stops on a fixed amount of work (Gurobi's WorkLimit, CP-SAT's
deterministic time). For a fixed seed you get the same solution on any machine with the same solver
versions. Only the runtime changes with hardware. The 30 instances are independent, so they run in
parallel across all cores without breaking determinism. The best-known baseline is a dated snapshot
in the repo, so the reported gaps stay fixed even if the competition site changes later.

## Tests

```bash
./.venv/bin/python tests/test_golden.py   # my scorer == official validator on i01 (cost 3842)
./.venv/bin/python tests/crosscheck.py    # the same, on 8 instances across the size range
```

If the golden test passes, the fiddly details (shift indexing, existing occupants, counting
distinct nurses per patient) are right, and the search is optimizing the real objective.


"""IHTP solver config and shared paths.

Solver params live next to their use sites, exposed on the CLI by ``experiments.py``.
Only fixed paths and a few project-wide constants here.
"""

from __future__ import annotations

import os

PKG_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(PKG_DIR)                       # .../ADVANCED_MODELLING
REPO_DIR = os.path.dirname(PROJECT_DIR)                      # .../Advanced_Modeeling_final

# raw instances vendored in data/instances/ to keep the project self-contained
# no vendored copy? fall back to sibling ihtc2024_competition_instances/ so an
# unbundled checkout still works
_VENDORED_INSTANCES = os.path.join(PROJECT_DIR, "data", "instances")
INSTANCE_DIR = (_VENDORED_INSTANCES if os.path.isdir(_VENDORED_INSTANCES)
                else os.path.join(REPO_DIR, "ihtc2024_competition_instances"))
VALIDATOR_BIN = os.path.join(PROJECT_DIR, "bin", "IHTP_Validator")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")

# Gurobi academic license: GRB_LICENSE_FILE if set, else ~/gurobi.lic
# no hard-coded user path, runs on any machine with a valid license
GUROBI_LICENSE = os.environ.get("GRB_LICENSE_FILE", os.path.expanduser("~/gurobi.lic"))

# 30 public instances, in order
PUBLIC_INSTANCES = [f"i{n:02d}" for n in range(1, 31)]

# best-known objectives for the public set, from a dated version-controlled
# snapshot (data/best_known_ihtc2024.csv), not a moving external target. gaps stay
# reproducible if the competition site changes later. these are competition upper
# bounds, not proven optima. reporting only, never touched inside the search
BEST_KNOWN_SNAPSHOT = os.path.join(PROJECT_DIR, "data", "best_known_ihtc2024.csv")
BEST_KNOWN_SOURCE = "https://ihtc2024.github.io/ (ph1_results.csv row-minima)"
BEST_KNOWN_SNAPSHOT_DATE = "2026-07-06"


def _load_best_known(path: str) -> dict:
    import csv as _csv
    out = {}
    with open(path) as fh:
        for row in _csv.reader(fh):
            if not row or row[0].startswith("#") or row[0] == "instance":
                continue
            out[row[0]] = int(row[1])
    return out


BEST_KNOWN = _load_best_known(BEST_KNOWN_SNAPSHOT)


def instance_path(name: str) -> str:
    return os.path.join(INSTANCE_DIR, f"{name}.json")

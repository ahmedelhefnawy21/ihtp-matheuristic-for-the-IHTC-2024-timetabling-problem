#!/usr/bin/env python3
"""Initial priced-term magnitudes of both campaign arms, recorded to a CSV.

PROTOCOL_PLACEBO.md requires a transparency record for comparing the two
perturbation strengths, and Appendix F describes the placebo as an
equal-magnitude perturbation. The campaign logs that printed these values (each
run's selftest line) are not part of the repo, so this script rebuilds the
record deterministically. For every instance it constructs the patched Layout once per arm, under the campaign settings (k_load=1.0, k_scarce=1.0, spread off; placebo permutation seed 7). It then writes the initial weighted term costs to term_magnitudes.csv. Nothing is solved, and no recorded result is read or modified.

Usage:
    python3 record_term_magnitudes.py
"""
from __future__ import annotations

import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))

import skillmix_pipeline as sp  # noqa: E402 (the campaign script itself)

from ihtp import config  # noqa: E402
from ihtp.io_instance import load_instance  # noqa: E402

NAMES = [f"i{i:02d}" for i in range(1, 31)]
PLACEBO_SEED = 7  # the registered placebo permutation seed (PROTOCOL_PLACEBO.md)


def main() -> None:
    sp.install_patch()
    rows = []
    for name in NAMES:
        inst = load_instance(config.instance_path(name))
        for arm, seed in (("real", None), ("placebo", PLACEBO_SEED)):
            sp.PLACEBO = seed
            lay = sp.L.Layout(inst)
            rows.append([name, arm, lay._ld_cost, lay._sc_cost, lay._sp_cost])
            print(f"[{name}] {arm:8} load={lay._ld_cost} scarce={lay._sc_cost} "
                  f"spread={lay._sp_cost}")
    sp.PLACEBO = None

    out = os.path.join(HERE, "term_magnitudes.csv")
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["instance", "arm", "load_cost", "scarce_cost", "spread_cost"])
        w.writerows(rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

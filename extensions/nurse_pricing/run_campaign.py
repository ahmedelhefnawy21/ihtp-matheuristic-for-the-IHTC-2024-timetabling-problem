#!/usr/bin/env python3
"""Campaign driver: 30 instances x 5 seeds of skillmix_pipeline.py, 10 workers.

Resumable: a (instance, seed) pair is skipped when its results JSON already
exists. Each run gets its own log. The driver writes CAMPAIGN_DONE.txt when finished.
"""
import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
VENV = os.path.join(HERE, "..", "..", ".venv", "bin", "python")
AP = argparse.ArgumentParser()
AP.add_argument("--placebo", type=int, default=None)
ARGS = AP.parse_args()
TAG = "sp0.0_ld1.0_sc1.0" + (f"_pl{ARGS.placebo}" if ARGS.placebo is not None else "")
EXTRA = ["--placebo", str(ARGS.placebo)] if ARGS.placebo is not None else []
MARKER = "CAMPAIGN_DONE.txt" if ARGS.placebo is None else f"CAMPAIGN_DONE_pl{ARGS.placebo}.txt"
PREFIX = "campaign" if ARGS.placebo is None else f"campaign_pl{ARGS.placebo}"

ENV = dict(os.environ)
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS"):
    ENV[var] = "1"

PAIRS = [(f"i{i:02d}", s) for i in range(1, 31) for s in range(1, 6)]


def done(name: str, seed: int) -> bool:
    return os.path.exists(os.path.join(HERE, f"results_{name}_s{seed}_{TAG}.json"))


def run_one(pair):
    name, seed = pair
    if done(name, seed):
        return f"{name} s{seed}: cached"
    log = os.path.join(HERE, f"{PREFIX}_{name}_s{seed}.log")
    with open(log, "w") as fh:
        rc = subprocess.run(
            [VENV, os.path.join(HERE, "skillmix_pipeline.py"), name, "--seed", str(seed)] + EXTRA,
            stdout=fh, stderr=subprocess.STDOUT, env=ENV, cwd=HERE).returncode
    return f"{name} s{seed}: {'ok' if rc == 0 and done(name, seed) else f'FAILED rc={rc}'}"


def main() -> None:
    todo = [p for p in PAIRS if not done(*p)]
    print(f"campaign: {len(PAIRS)} pairs total, {len(todo)} to run", flush=True)
    with ThreadPoolExecutor(max_workers=10) as ex:
        for msg in ex.map(run_one, PAIRS):
            print(msg, flush=True)
    with open(os.path.join(HERE, MARKER), "w") as fh:
        fh.write("done\n")
    print(f"CAMPAIGN_DONE ({MARKER})", flush=True)


if __name__ == "__main__":
    main()

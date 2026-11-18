"""Map an integer job index to a job's arguments (useful for array schedulers; see
scripts/slurm_template.sh). Ordered job lists (stable -> indices are reproducible):
  encoders : (encoder,)                             len = 3    [Stage 1: extract + dump]
  cells    : (encoder, activation, variant, seed)   len = 54   [Stages 2-4]

CLI: python -m hypogen.jobs --kind cells --index $TASK_ID
     -> prints the args space-separated for `read` in bash; --count prints the list length.
"""

from __future__ import annotations

import argparse

from hypogen import config


def encoder_jobs():
    return [(enc,) for enc in config.ENCODERS]


def cell_jobs():
    return [(c.encoder, c.activation, c.variant, c.seed) for c in config.iter_cells()]


KINDS = {"encoders": encoder_jobs, "cells": cell_jobs}


def main():
    ap = argparse.ArgumentParser(description="Map a job index to its arguments.")
    ap.add_argument("--kind", required=True, choices=list(KINDS))
    ap.add_argument("--index", type=int, default=None)
    ap.add_argument("--count", action="store_true", help="print the number of jobs of this kind")
    args = ap.parse_args()
    jobs = KINDS[args.kind]()
    if args.count:
        print(len(jobs))
        return
    if args.index is None or not (0 <= args.index < len(jobs)):
        raise SystemExit(f"--index out of range [0,{len(jobs)}) for kind={args.kind}")
    print(" ".join(str(x) for x in jobs[args.index]))


if __name__ == "__main__":
    main()

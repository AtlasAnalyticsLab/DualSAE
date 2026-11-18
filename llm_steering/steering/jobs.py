"""Canonical unit-index -> unit decoder for the steering stages.

The planner (plan.py) and the stage drivers (run_stage.sh, run_all.sh) call THIS module, so they
can never disagree about what index i of a stage means.  Unit lists come from base.py
(arms.iter_arms order for SAEs; base.gen_units() for generation units).

KINDS (unit -> count)
  s0    SAE (arms.iter_arms('B'))                     81  rescore + D-43 measurements + D-44
  s1    layer (6, 12, 18)                              3  concept AUROC selection + DiffMean
  s2    Dual SAE (dual, dual_g0)                      54  analytic cancellation
  s4a   pilot generation                               1
  s4j   pilot judge (phi-4, concept only)              1
  s5    SAE units + armfree_L6/L12/L18                84  KL calibration (armfree_L12 owned by s4a)
  s6    SAE units + armfree x3 + null                 85  generation + judge-free metrics (s7m
                                                          merged in, D-51 1:1 rule)
  s7c   bge concept classifier                         1
  s7j   = s6 units                                    85  phi-4 judge
  s8j   InternVL3.5 second judge (optional)            1
  s9    report                                         1
JumpReLU units exist in every per-SAE kind (implemented fully) and are BLOCKED by the planner
while the Track B JumpReLU cells are held (base.held_reason).

CLI
  python -m steering.jobs --kind s0 --count
  python -m steering.jobs --kind s0 --index 7 --shell     # A16_* lines for `eval`
  python -m steering.jobs --selftest
"""

from __future__ import annotations

import argparse
import shlex
import sys
from typing import Dict, List, Tuple

from steering import base

# kind -> (python module, needs pyarrow or vLLM, argument template)
KIND_CMD: Dict[str, Tuple[str, bool, str]] = {
    "s0": ("steering.s0_rescore", False, "--index {i}"),
    "s1": ("steering.s1_select", True, "--index {i}"),     # amazon parquet -> pyarrow
    "s2": ("steering.s2_cancel", False, "--index {i}"),
    "s4a": ("steering.s4_pilot", False, "--generate"),
    "s4j": ("steering.s7_judge", True, "--pilot"),
    "s5": ("steering.calibrate", False, "--index {i}"),
    "s6": ("steering.s6_generate", False, "--index {i}"),
    "s7c": ("steering.s7_metrics", True, "--train-clf"),
    "s7j": ("steering.s7_judge", True, "--index {i}"),
    "s8j": ("steering.s8_handlabel", True, "--second-judge"),
    "s9": ("steering.s9_report", False, ""),
}
KINDS = tuple(KIND_CMD)


def units(kind: str) -> List[str]:
    if kind == "s0":
        return [r.tag for r in base.sae_refs()]
    if kind == "s1":
        return [f"L{l}" for l in base.LAYERS]
    if kind == "s2":
        return [r.tag for r in base.dual_refs()]
    if kind == "s5":
        return base.gen_units()[:-1]
    if kind in ("s6", "s7j"):
        return base.gen_units()
    if kind in ("s4a", "s4j", "s7c", "s8j", "s9"):
        return [kind]
    raise KeyError(kind)


EXPECTED_COUNTS = {"s0": 81, "s1": 3, "s2": 54, "s4a": 1, "s4j": 1, "s5": 84, "s6": 85, "s7c": 1,
                   "s7j": 85, "s8j": 1, "s9": 1}


def shell_env(kind: str, index: int) -> List[Tuple[str, str]]:
    us = units(kind)
    if not 0 <= index < len(us):
        raise IndexError(f"{kind}: index {index} outside [0, {len(us)})")
    mod, llm, tpl = KIND_CMD[kind]
    return [("A16_KIND", kind), ("A16_INDEX", str(index)), ("A16_UNIT", us[index]),
            ("A16_MODULE", mod), ("A16_ARGS", tpl.format(i=index)), ("A16_NEEDS_LLM_ENV", "1" if llm else "0")]


def selftest() -> int:
    bad = []
    for k, n in EXPECTED_COUNTS.items():
        if len(units(k)) != n:
            bad.append(f"{k}: {len(units(k))} units, expected {n}")
    if units("s5") != units("s6")[:84] or units("s7j") != units("s6"):
        bad.append("s5/s6/s7 unit lists are not index-aligned")
    if [u for u in units("s0")] != [u for u in units("s6")[:81]]:
        bad.append("s0 and s6 SAE indices differ")
    for k in KINDS:
        for i in (0, len(units(k)) - 1):
            shell_env(k, i)
        try:
            shell_env(k, len(units(k)))
            bad.append(f"{k}: out-of-range index accepted")
        except IndexError:
            pass
    for b in bad:
        print("  FAIL", b)
    print("A16 JOBS SELFTEST", "OK" if not bad else "FAILED")
    return 0 if not bad else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", choices=KINDS)
    ap.add_argument("--index", type=int)
    ap.add_argument("--count", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--shell", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.kind:
        ap.error("--kind required")
    if a.count:
        print(len(units(a.kind)))
    elif a.list:
        for i, u in enumerate(units(a.kind)):
            print(i, u)
    elif a.shell:
        for k, v in shell_env(a.kind, a.index):
            print(f"{k}={shlex.quote(v)}")
    else:
        print(units(a.kind)[a.index])
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/bin/bash
# run_all.sh -- the full steering cascade, sequential and resumable.
#
#   export LANGDUAL_ROOT=/path/to/language/data  A16_ROOT=/path/to/outputs
#   bash run_all.sh                  # every stage, in waves, until nothing is left to run
#   ONLY="s0 s1" bash run_all.sh     # restrict to some stages
#   A16_SECOND_JUDGE=1 bash run_all.sh   # also the optional InternVL second judge (s8)
#
# Each wave asks the completion-aware planner (steering/plan.py) which units are runnable NOW
# (all inputs on disk) and runs exactly those, one after the other; a killed run simply resumes
# at the next unfinished unit (every unit has a skip-guard and writes atomically).  Between
# waves the per-unit-free steps run: the P1 rule after the pilot judge (s4p), and, with
# A16_SECOND_JUDGE=1, the hand-label sheet export and the InternVL environment check.
# Then the post-processing: the quantitative table and the example pools (both directions).
# One GPU is used (CUDA_VISIBLE_DEVICES selects it).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/env.sh"
cd "$HERE"
ORDER=(s0 s1 s2 s4a s4j s5 s6 s7c s7j s8j s9)
ONLY_SET=" ${ONLY:-${ORDER[*]}} "

expand() {   # "0-3,7" -> "0 1 2 3 7"
  local out=() part a b
  IFS=',' read -ra parts <<< "$1"
  for part in "${parts[@]}"; do
    if [[ "$part" == *-* ]]; then a="${part%-*}"; b="${part#*-}"; out+=($(seq "$a" "$b"));
    elif [ -n "$part" ]; then out+=("$part"); fi
  done
  echo "${out[@]:-}"
}

wave=0
while :; do
  wave=$((wave + 1))
  ran=0
  for s in "${ORDER[@]}"; do
    [[ "$ONLY_SET" == *" $s "* ]] || continue
    if [ "$s" = "s8j" ] && [ "${A16_SECOND_JUDGE:-0}" != "1" ]; then continue; fi
    todo="$("$PY" -m steering.plan --stage "$s")"
    [ -z "$todo" ] && continue
    for i in $(expand "$todo"); do
      bash "$HERE/run_stage.sh" "$s" "$i"
      ran=$((ran + 1))
    done
  done
  # P1 rule once the pilot is judged (s4p; a decided P1 is never re-written).
  if [ ! -f "$A16_ROOT/s4/p1_primary_budget.json" ] && \
     "$PY" -c "import sys; from steering import s7_judge; sys.exit(0 if s7_judge.unit_done('pilot') else 1)"; then
    "$PY" -m steering.s4_pilot --rule
    ran=$((ran + 1))
  fi
  if [ "${A16_SECOND_JUDGE:-0}" = "1" ]; then
    [ -f "$A16_ROOT/s8/handlabel_sheet.csv" ] || { "$PY" -m steering.s8_handlabel --export && ran=$((ran + 1)) || true; }
    "$PY" -c "import json,sys; from steering import base; r=base.read_json(base.SECOND_JUDGE_ENV_RECORD); sys.exit(0 if r and r.get('passed') else 1)" \
      || { "$PY" -m steering.s8_handlabel --second-judge-smoke && ran=$((ran + 1)) || true; }
  fi
  echo "--- wave $wave: $ran unit(s) run"
  [ "$ran" -eq 0 ] && break
done
"$PY" -m steering.plan --summary
if [ -f "$A16_ROOT/s9/per_seed.csv" ]; then
  "$PY" -m steering.make_table
  "$PY" -m steering.make_table --gate strict
  "$PY" -m steering.select_examples --direction dual
  "$PY" -m steering.select_examples --direction baseline
fi

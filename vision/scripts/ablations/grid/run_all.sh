#!/bin/bash
# The whole ablation grid, in dependency order, sequentially on one machine (each stage skips
# finished work, so this script can be re-run after an interruption). For a cluster, run each
# stage as an array with ../slurm_template.sh and --index.
#
#   1. S, Z, R0 and the projector arms (joint vs. sequential + projector ablation), seeds 42-44
#   2. F (sequential training; needs finished S, Z, R0)
#   3. gamma arms G0 ... G10 (decorrelation-coefficient tables), seeds 42-44
#   4. expansion arms (x4, x8, x16 and x32; seed 42) + the two selection reruns
#   5. evaluation of every finished run (branch_metrics.py)
#   6. collectors and builders (tables and figures under $ABL_ROOT)
#
#   MAIN_MANIFEST=<json> MAIN_RESULTS=<dir>  main-table runs and their branch_metrics results
#   (scripts/ablations/make_run_manifest.py, scripts/ablations/branch_metrics/run_branch_metrics.py);
#   needed for the DualSAE points of the joint-vs-sequential figure / table.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"
G="$ABL_SCRIPTS_DIR/grid/run_grid.py"
"$PY" "$G" --stage train                                     # S, Z, R0, L1, B64, B256, B512, B2048, B512G, D3, D4
"$PY" "$G" --stage frozen                                    # F
"$PY" "$G" --stage train --arms G0 G1e-3 G1e-2 G1e-1 G1 G10
"$PY" "$G" --stage train --arms E4 E8 E16 E32 SE4 SE8 SE16 SE32 --seeds 42
"$PY" "$G" --stage reruns
"$PY" "$G" --stage eval

GRID="$ABL_ROOT/grid"
MAIN_ARGS=()
if [ -n "${MAIN_MANIFEST:-}" ]; then MAIN_ARGS=(--main-manifest "$MAIN_MANIFEST" --main-results "$MAIN_RESULTS"); fi
"$PY" "$ANALYSIS_DIR/collect_grid.py" --root "$GRID" ${MAIN_ARGS[@]+"${MAIN_ARGS[@]}"}
JOINT=R0; [ -n "${MAIN_MANIFEST:-}" ] && JOINT=published_dual
"$PY" "$ANALYSIS_DIR/make_grid_tables.py" --long-csv "$GRID/long.csv" --joint "$JOINT"
"$PY" "$ANALYSIS_DIR/plot_joint_vs_seq.py" "$GRID" "$GRID/long.csv" --dual "$JOINT"
"$PY" "$ANALYSIS_DIR/collect_span_ablation.py" --grid-root "$GRID" --checkpoint final
"$PY" "$ANALYSIS_DIR/make_gamma_tables.py"
"$PY" "$ANALYSIS_DIR/collect_scale_ablation.py" --grid-root "$GRID" --seeds 42
"$PY" "$ANALYSIS_DIR/build_expansion_data.py" --grid-root "$GRID" --seeds 42
"$PY" "$ANALYSIS_DIR/plot_expansion.py" "$ABL_ROOT/expansion/expansion_data.json" "$ABL_ROOT/expansion"
"$PY" "$ANALYSIS_DIR/plot_expansion_main.py" "$ABL_ROOT/expansion/expansion_data.json" "$ABL_ROOT/expansion"

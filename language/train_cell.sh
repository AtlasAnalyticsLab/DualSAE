#!/bin/bash
# ============================================================================
# train_cell.sh -- train ONE language cell (a main-grid or an ablation cell) by tag.
#
#     bash language/train_cell.sh B_L12_owt_topk_dual_seed42
#     bash language/train_cell.sh abl_warmupB1250_B_L12_owt_topk_dual_seed42
#
# Print the exact vision/main.py argv of a cell without training or writing anything:
#     LD_DRY_RUN=1 bash language/train_cell.sh B_L12_owt_jump_relu_baseline_seed42
#
# WHY IT GOES THROUGH vision/main.py AND NOT THE TRAINERS DIRECTLY
# Only main.py writes `training_config.json` into the run directory, and the evaluator rebuilds
# the model from that file -- so every arm dispatches through
# `main.py --model {topk,batch_topk,jumprelu,dual}`.
#
# WHY IT cd's INTO vision/
# The vision core imports its own modules by BARE NAME (`from models import ...`); they resolve
# with vision/ as the working directory.  language/ is put on PYTHONPATH so the core's dataset
# resolver can import the activation-shard reader (langdual.lmsae.resid_shards).
#
# WHY THE WARM-START GATE EXISTS
# A DualSAE cell whose Residual Branch never unfreezes is numerically a single-branch SAE.
# train_dual.py prints "WARM START COMPLETE - ENABLING POLY BRANCH" exactly when the branch is
# unfrozen, so that marker is grepped out of the captured stdout and the done-sentinel is REFUSED
# without it (whenever a warm-up was requested).
#
# Resumable: a cell whose sentinel, checkpoint and training_config.json are in place exits 0.
# ============================================================================
set -euo pipefail

LANG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$LANG_DIR/scripts/env.sh"

TAG="${1:-${LD_CELL_TAG:-}}"
if [ -z "$TAG" ]; then
  echo "FATAL: pass a cell tag (see: python -m langdual.jobs --list main|ablation)" >&2
  exit 2
fi

# LD_DRY_RUN is an interactive tool: it prints the argv and exits 0 without training, so inside
# a batch allocation it would mark work as done that never ran.  Refused there.
if [ "${LD_DRY_RUN:-0}" = "1" ] && [ -n "${SLURM_JOB_ID:-}" ]; then
  echo "FATAL: LD_DRY_RUN=1 inside a batch job; unset it and resubmit." >&2
  exit 2
fi

# ── Decode the cell ──────────────────────────────────────────────────────────
# langdual.jobs is the single decoder; every value below (paths, --model, --sparsity_mode,
# --normalize, --lr, the JumpReLU threshold init, ablation overrides) comes from that one call.
# Not inside a pipeline: under `set -e` a failure in a process substitution would leave the
# variables unset instead of aborting.
# Run from language/: `python -m` puts the working directory first on sys.path, so another
# checkout's `langdual` package in the caller's directory must not shadow this one.
CELL_ENV="$(cd "$LANG_DIR" && "$LD_PY" -m langdual.jobs --tag "$TAG" --shell)"
set -a
eval "$CELL_ENV"
set +a
: "${LD_TAG:?langdual.jobs did not emit LD_TAG -- decode failed}"

echo "=== train_cell : $LD_TAG"
echo "    corpus=$LD_CORPUS layer=$LD_KEY act=$LD_ACT var=$LD_VAR seed=$LD_SEED"
echo "    model=$LD_MODEL sparsity_mode=${LD_SPARSITY_MODE:-<n/a>} normalize=$LD_NORMALIZE lr=$LD_LR"
echo "    run_dir=$LD_RUN_DIR"

CKPT="$LD_RUN_DIR/best_val_monoscore_model.pt"
TCFG="$LD_RUN_DIR/training_config.json"
LOG="$LD_RUN_DIR/train_stdout.log"

# ── Skip-guard ───────────────────────────────────────────────────────────────
if [ "${LD_DRY_RUN:-0}" != "1" ] && [ -f "$LD_DONE_SENTINEL" ] && [ -f "$CKPT" ] && [ -f "$TCFG" ]; then
  echo "[skip] $LD_TAG is already complete (sentinel + checkpoint + training_config.json)"
  exit 0
fi

# ── Validate inputs before allocating anything ───────────────────────────────
# The shard directory must exist AND carry manifest.json: the extractor creates the directory
# before it extracts and writes the manifest last and atomically, so the manifest is the real
# completion marker.  Without it the core's dataset resolver would route the path to the LMDB
# reader and fail much later.
for p in "$LD_TRAIN_PATH" "$LD_VAL_PATH"; do
  if [ "${LD_DRY_RUN:-0}" != "1" ] && [ ! -f "$p/manifest.json" ]; then
    echo "FATAL: activation shards missing or incomplete (no manifest.json): $p" >&2
    echo "       run the extraction stage first (run_pipeline.sh --stage extract)" >&2
    exit 3
  fi
done
if [ "${LD_DRY_RUN:-0}" != "1" ] && [ ! -d "$LD_TEST_PATH" ]; then
  echo "[warn] test split not present yet: $LD_TEST_PATH (training proceeds; evaluation needs it)"
fi

# ── Opt-in knobs: validate BEFORE anything is created ────────────────────────
# Emitted EMPTY by langdual.jobs for every cell that does not use them; empty == flag not passed.
# A non-empty value on a cell that cannot use it is REFUSED (exit 2), never silently dropped.
LD_JUMP_THRESHOLD_INIT="${LD_JUMP_THRESHOLD_INIT:-}"
LD_TIED_WEIGHTS="${LD_TIED_WEIGHTS:-}"
if [ -n "$LD_JUMP_THRESHOLD_INIT" ]; then
  # Readers: the JumpReLU baseline (--model jumprelu) and a Dual whose Sparse Branch is JumpReLU.
  if [ "$LD_MODEL" != "jumprelu" ] && ! { [ "$LD_IS_DUAL" = "1" ] && [ "${LD_SPARSITY_MODE:-}" = "jump_relu" ]; }; then
    echo "FATAL: LD_JUMP_THRESHOLD_INIT='$LD_JUMP_THRESHOLD_INIT' on a non-JumpReLU cell ($LD_TAG)" >&2
    exit 2
  fi
  # theta itself (stored as log(theta)): must parse as a finite float > 0.
  if ! "$LD_PY" -c 'import math, sys; v = float(sys.argv[1]); sys.exit(0 if math.isfinite(v) and v > 0 else 1)' \
       "$LD_JUMP_THRESHOLD_INIT" 2>/dev/null; then
    echo "FATAL: LD_JUMP_THRESHOLD_INIT='$LD_JUMP_THRESHOLD_INIT' is not a finite float > 0 ($LD_TAG)" >&2
    exit 2
  fi
fi
case "$LD_TIED_WEIGHTS" in
  "") ;;        # default: no flag, i.e. main.py's tied TopK decoder
  untied)
    if [ "$LD_MODEL" != "topk" ] || [ "$LD_IS_DUAL" = "1" ]; then
      echo "FATAL: LD_TIED_WEIGHTS=untied applies only to the TopK baseline ($LD_TAG)" >&2
      exit 2
    fi
    ;;
  *)
    echo "FATAL: LD_TIED_WEIGHTS='$LD_TIED_WEIGHTS' is not understood ($LD_TAG); use '' or 'untied'" >&2
    exit 2
    ;;
esac

if [ "${LD_DRY_RUN:-0}" != "1" ]; then mkdir -p "$LD_RUN_DIR" "$LD_TB_DIR"; fi

# DataLoader workers (value used for the reported runs: 8).  The sampling order is drawn in the
# main process, so this changes throughput only.
LD_NUM_WORKERS="${LD_NUM_WORKERS:-8}"

# ── Build the argv ───────────────────────────────────────────────────────────
ARGS=(
  --model "$LD_MODEL"
  --train_path "$LD_TRAIN_PATH"
  --val_path "$LD_VAL_PATH"
  --exp_name "$LD_TAG"
  --output_dir "$LD_RUNS_DIR"          # the trainer appends exp_name -> $LD_RUN_DIR
  --n_latents "$LD_N_LATENTS"
  --topk_k "$LD_TOP_K"
  --batch_size "$LD_BATCH_SIZE"
  --num_epochs "$LD_NUM_EPOCHS"
  # A fixed STEP budget expressed as num_epochs x steps_per_epoch (31,250 steps = one pass over
  # the 64M-token stream), which also gives 50 validation-MonoScore selection points.
  --steps_per_epoch "$LD_STEPS_PER_EPOCH"
  --lr "$LD_LR"
  --eps "$LD_EPS"
  --l1_coef "$LD_L1_COEF"
  --bandwidth "$LD_BANDWIDTH"
  --dead_steps_threshold "$LD_DEAD_STEPS_THRESHOLD"
  --seed "$LD_SEED"
  --num_workers "$LD_NUM_WORKERS"
  --tb_dir "$LD_TB_DIR"
  # W&B is opt-in; an empty project name keeps it off in every trainer.
  --wandb_project ""
  # Skips the logging-only evals (per-epoch train R2/mono, end-of-run KNN/LP). Best-val
  # selection and the checkpoints are unaffected; evaluation rescoring is a separate stage.
  --skip_extra_eval
)
# Recorded in training_config.json; read later by the evaluator.  Always explicit: an empty
# value would make main.py fall back to its built-in vision dataset paths.
ARGS+=( --test_path "$LD_TEST_PATH" )
# Dataset identity recorded in training_config.json (main.py's defaults name a vision dataset).
ARGS+=( --dataset pythia410m --extractor resid )

if [ "${LD_MULTILABEL:-0}" = "1" ]; then
  ARGS+=( --multilabel )
else
  ARGS+=( --no_multilabel )
fi

# --force_normalize states input_unit_norm explicitly for EVERY model branch (including
# --model topk, whose 'auto' default is True).  The residual stream uses no per-sample
# normalisation (config.TRACKB_NORMALIZE), so every language cell passes 'off'.
if [ "$LD_NORMALIZE" = "1" ]; then
  ARGS+=( --normalize --force_normalize on )
else
  ARGS+=( --force_normalize off )
fi

if [ "$LD_IS_DUAL" = "1" ]; then
  # DualSAE arm.  --sparsity_mode selects the Sparse Branch activation, which is what makes
  # "Dual-TopK vs TopK" a within-activation comparison.  --span_coef is gamma (L_dcor).
  # --poly_layers 1 builds the Residual Branch as Linear(D,D)+GELU (--poly_hidden is inert on
  # this path and is not passed).
  ARGS+=(
    --sparsity_mode "$LD_SPARSITY_MODE"
    --span_coef "$LD_SPAN_COEF"
    --poly_layers "${LD_POLY_LAYERS:-1}"
    --warmup_steps "$LD_WARMUP_STEPS"
    --dead_check_interval "$LD_DEAD_CHECK_INTERVAL"
    # Per-epoch checkpoints are never read; only best-val and final are kept.
    --skip_epoch_checkpoints
  )
elif [ "$LD_MODEL" = "topk" ]; then
  # train_topk.py implements both AuxK and dead-neuron reinitialisation.
  ARGS+=( --auxk_coef "$LD_AUXK_COEF" --dead_check_interval "$LD_DEAD_CHECK_INTERVAL" )
else
  # train_other.py (batch_topk / jumprelu): AuxK and gradient clipping, but no dead-neuron
  # reinitialisation, so --dead_check_interval is not passed.
  ARGS+=( --auxk_coef "$LD_AUXK_COEF" --max_grad_norm "$LD_MAX_GRAD_NORM" )
fi

# Opt-in knobs, APPENDED only when set (main.py's argparse is order-insensitive for these).
if [ -n "$LD_JUMP_THRESHOLD_INIT" ]; then
  # Value used for the reported language JumpReLU runs (config.JUMP_THRESHOLD_INIT).
  ARGS+=( --jump_threshold_init "$LD_JUMP_THRESHOLD_INIT" )
fi
if [ "$LD_TIED_WEIGHTS" = "untied" ]; then
  ARGS+=( --no_tied_weights )
fi

# ── Dry run: print the argv and stop ─────────────────────────────────────────
if [ "${LD_DRY_RUN:-0}" = "1" ]; then
  echo "--- LD_DRY_RUN argv for $LD_TAG (one word per line):"
  printf '%q\n' main.py "${ARGS[@]}"
  exit 0
fi

# ── Train ────────────────────────────────────────────────────────────────────
cd "$LD_SAE_DIR"
echo "--- python main.py ${ARGS[*]}"
# stdout is teed into the run dir because the warm-start gate below reads it; PIPESTATUS keeps
# python's exit status rather than tee's.
set +e
"$LD_PY" main.py "${ARGS[@]}" 2>&1 | tee "$LOG"
RC=${PIPESTATUS[0]}
set -e
if [ "$RC" -ne 0 ]; then
  echo "FATAL: main.py exited $RC for $LD_TAG (log: $LOG)" >&2
  exit "$RC"
fi

# ── Completion gate ──────────────────────────────────────────────────────────
# At warmup_steps=0 the Residual Branch is never frozen (correct joint training from step 0)
# and the marker is never printed, so the gate applies only when a warm-up was requested.
export LD_WARM_MARKER=0
if [ "$LD_IS_DUAL" = "1" ] && [ "${LD_WARMUP_STEPS:-0}" -eq 0 ]; then
  echo "[note] warmup_steps=0: joint training from step 0, no WARM START marker expected."
  export LD_WARM_MARKER=-1
elif [ "$LD_IS_DUAL" = "1" ]; then
  if grep -q "WARM START COMPLETE" "$LOG"; then
    export LD_WARM_MARKER=1
  else
    echo "FATAL: $LD_TAG is a Dual cell but its log never printed 'WARM START COMPLETE';" >&2
    echo "       the Residual Branch never unfroze. Check that num_epochs x steps_per_epoch" >&2
    echo "       ($LD_TOTAL_STEPS) exceeds warmup_steps=$LD_WARMUP_STEPS. Log: $LOG" >&2
    exit 5
  fi
fi

for f in "$CKPT" "$TCFG"; do
  if [ ! -f "$f" ]; then
    echo "FATAL: training reported success but $f is missing" >&2
    exit 6
  fi
done

# ── Sentinel (written last, atomically: tmp + os.replace) ────────────────────
"$LD_PY" - "$LD_DONE_SENTINEL" <<'PYEOF'
import json, os, sys, time
out = sys.argv[1]
payload = {k: os.environ.get(v) for k, v in (
    ("tag", "LD_TAG"), ("corpus", "LD_CORPUS"), ("key", "LD_KEY"), ("activation", "LD_ACT"),
    ("variant", "LD_VAR"), ("seed", "LD_SEED"), ("model", "LD_MODEL"), ("lr", "LD_LR"),
    ("num_epochs", "LD_NUM_EPOCHS"), ("steps_per_epoch", "LD_STEPS_PER_EPOCH"),
    ("total_steps", "LD_TOTAL_STEPS"), ("warmup_steps", "LD_WARMUP_STEPS"))}
payload["sparsity_mode"] = os.environ.get("LD_SPARSITY_MODE") or None
payload["normalize"] = os.environ.get("LD_NORMALIZE") == "1"
# None for a baseline (the marker does not apply); True only when it was actually seen.
payload["warm_start_complete"] = True if os.environ.get("LD_WARM_MARKER") == "1" else None
for key, env_key in (("jump_threshold_init", "LD_JUMP_THRESHOLD_INIT"),
                     ("tied_weights", "LD_TIED_WEIGHTS")):
    if os.environ.get(env_key):
        payload[key] = os.environ[env_key]
payload["checkpoint"] = "best_val_monoscore_model.pt"
payload["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
os.makedirs(os.path.dirname(out), exist_ok=True)
tmp = out + ".tmp"
with open(tmp, "w") as fh:
    json.dump(payload, fh, indent=2)
os.replace(tmp, out)
print(f"sentinel written: {out}")
PYEOF

echo "=== train_cell : $LD_TAG DONE"

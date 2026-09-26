#!/usr/bin/env bash
# O'Prior simplified generation, one process per batch, like generate_rq4.sh.
# Same loop and seeds; only the options that o_prior_simplified removed are gone.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

GENLOAD_CMD="${GENLOAD_CMD:-python3 -u -m o_prior_simplified.prior.genload}"

NUM_BATCHES="${NUM_BATCHES:-1000}"
RESUME_FROM="${RESUME_FROM:-0}"
SAVE_FORMAT="${SAVE_FORMAT:-h5}"

NP_SEED="${NP_SEED:-42}"
TORCH_SEED="${TORCH_SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-50}"
BATCH_SIZE_PER_GP="${BATCH_SIZE_PER_GP:-4}"

# Core prior
PRIOR_TYPE="${PRIOR_TYPE:-mix_scm_hscm}"
MIN_FEATURES="${MIN_FEATURES:-1}"
MAX_FEATURES="${MAX_FEATURES:-99}"
MAX_CLASSES="${MAX_CLASSES:-10}"           # 0 for regression
MIN_SEQ_LEN="${MIN_SEQ_LEN:-1000}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-1000}"
SAMPLING="${SAMPLING:-beta}"
HYBRID_SAMPLING_STRATEGY="${HYBRID_SAMPLING_STRATEGY:-graph_aware}"
# Needed by export_ltm1.py, which reads the feature metadata of every batch.
RETURN_METADATA="${RETURN_METADATA:-True}"

# Extra generation controls
N_JOBS="${N_JOBS:--1}"
THREADS="${THREADS:-1}"
DEVICE="${DEVICE:-cpu}"
DATA_ROOT="${DATA_ROOT:-/path/to/directory}"
LOG_BASE="${LOG_BASE:-$DATA_ROOT/logs/simplified}"
RUN_TS="${RUN_TS:-$(date +%Y-%m-%d_%H-%M-%S)}"
LOG_DIR="$LOG_BASE/$RUN_TS"

mkdir -p "$DATA_ROOT" "$LOG_DIR"
STDOUT_LOG="$LOG_DIR/trace_log_${RUN_TS}.log"
STDERR_LOG="$LOG_DIR/trace_error_${RUN_TS}.log"

echo "Phase: O'Prior simplified"
echo "Save root: $DATA_ROOT"
echo "Logs: $LOG_DIR"
echo "Batches: $NUM_BATCHES (resume_from=$RESUME_FROM)"
echo "Device: $DEVICE"
echo

for ((i=0; i<NUM_BATCHES; i++)); do
  batch_idx=$((RESUME_FROM + i))
  echo "[O'Prior-simplified] Generating batch #$batch_idx ..."
  t0=$(date +%s)

  $GENLOAD_CMD \
    --save_dir "$DATA_ROOT" \
    --save_format "$SAVE_FORMAT" \
    --np_seed "$((NP_SEED + batch_idx))" \
    --torch_seed "$((TORCH_SEED + batch_idx))" \
    --num_batches 1 \
    --resume_from "$batch_idx" \
    --batch_size "$BATCH_SIZE" \
    --batch_size_per_gp "$BATCH_SIZE_PER_GP" \
    --prior_type "$PRIOR_TYPE" \
    --min_features "$MIN_FEATURES" \
    --max_features "$MAX_FEATURES" \
    --max_classes "$MAX_CLASSES" \
    --min_seq_len "$MIN_SEQ_LEN" \
    --max_seq_len "$MAX_SEQ_LEN" \
    --sampling "$SAMPLING" \
    --n_jobs "$N_JOBS" \
    --num_threads_per_generate "$THREADS" \
    --device "$DEVICE" \
    --hybrid_sampling_strategy "$HYBRID_SAMPLING_STRATEGY" \
    --return_metadata "$RETURN_METADATA" \
    > >(tee -a "$STDOUT_LOG") \
    2> >(tee -a "$STDERR_LOG" >&2)

  status=$?
  t1=$(date +%s)
  elapsed=$((t1 - t0))

  if [ "$status" -eq 0 ]; then
    echo "[O'Prior-simplified] Batch #$batch_idx done in ${elapsed}s."
  else
    echo "[O'Prior-simplified] Batch #$batch_idx FAILED after ${elapsed}s."
    exit "$status"
  fi
done

echo "[O'Prior-simplified] Done. Quick checks:"
echo "  find $DATA_ROOT -maxdepth 1 -name 'batch_*.'"$SAVE_FORMAT"' | wc -l"
echo "  du -sh $DATA_ROOT"
#!/usr/bin/env bash
# O'Prior realism-generation script (classification) with:
# - hybrid SCM mix
# - realism profile (hard)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

GENLOAD_CMD="${GENLOAD_CMD:-python3 -u -m o_prior.prior.genload}"

NUM_BATCHES="${NUM_BATCHES:-1000}"
RESUME_FROM="${RESUME_FROM:-0}"
SAVE_FORMAT="${SAVE_FORMAT:-h5}"

NP_SEED="${NP_SEED:-42}"
TORCH_SEED="${TORCH_SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-50}"
BATCH_SIZE_PER_GP="${BATCH_SIZE_PER_GP:-4}"

# Core prior
PRIOR_TYPE="${PRIOR_TYPE:-mix_scm_hscm}"   # SCM mix
MIN_FEATURES="${MIN_FEATURES:-2}"
MAX_FEATURES="${MAX_FEATURES:-50}"
MAX_CLASSES="${MAX_CLASSES:-10}"           # classification
MIN_SEQ_LEN="${MIN_SEQ_LEN:-1024}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-1024}"

# Global curriculum (existing)
USE_CURRICULUM="${USE_CURRICULUM:-False}"

# Realism curriculum (mild -> hard)
REALISM_PROFILE="${REALISM_PROFILE:-hard}"
USE_REALISM_CURRICULUM="${USE_REALISM_CURRICULUM:-False}"
CATEGORICAL_ATTR="${CATEGORICAL_ATTR:-True}"
APPLY_CROSS_SECTIONAL_RANK="${APPLY_CROSS_SECTIONAL_RANK:-True}"
APPLY_CENSORED_TARGETS="${APPLY_CENSORED_TARGETS:-False}" # mostly regression-oriented
USE_STRICTLY_POSITIVE_TARGET="${USE_STRICTLY_POSITIVE_TARGET:-False}"
ADD_SKEWNESS="${ADD_SKEWNESS:-True}"
ADD_SVD_FEATURES="${ADD_SVD_FEATURES:-True}"
SVD_ENCODE_CATEGORICAL_BEFORE_SVD="${SVD_ENCODE_CATEGORICAL_BEFORE_SVD:-True}"
SVD_MAX_ONEHOT_CARDINALITY="${SVD_MAX_ONEHOT_CARDINALITY:-16}"
ADD_FINGERPRINT_FEATURE="${ADD_FINGERPRINT_FEATURE:-True}"
FINGERPRINT_METHOD="${FINGERPRINT_METHOD:-projection}"
SAMPLING="${SAMPLING:-beta}"
HYBRID_SAMPLING_STRATEGY="${HYBRID_SAMPLING_STRATEGY:-graph_aware}"



# Shift stressors
APPLY_COVARIATE_SHIFT="${APPLY_COVARIATE_SHIFT:-False}"
APPLY_SEASONAL_DRIFT="${APPLY_SEASONAL_DRIFT:-False}"
APPLY_TEMPORAL_DRIFT="${APPLY_TEMPORAL_DRIFT:-False}"
TARGET_NORM_METHOD="${TARGET_NORM_METHOD:-}"                     # zscore|minmax or empty
TIME_LAGGED_LAG_ORDER="${TIME_LAGGED_LAG_ORDER:-}"               # int or empty
TIME_LAGGED_WEIGHT_SPARSITY="${TIME_LAGGED_WEIGHT_SPARSITY:-}"   # float or empty
TIME_LAGGED_OUTPUT_NOISE_STD="${TIME_LAGGED_OUTPUT_NOISE_STD:-}" # float or empty

# Extra generation controls
N_JOBS="${N_JOBS:--1}"
THREADS="${THREADS:-1}"
DEVICE="${DEVICE:-cpu}"
DATA_ROOT="${DATA_ROOT:-/directory/path}"
LOG_BASE="${LOG_BASE:-$DATA_ROOT/logs/rq2c_2}"
RUN_TS="${RUN_TS:-$(date +%Y-%m-%d_%H-%M-%S)}"
LOG_DIR="$LOG_BASE/$RUN_TS"

mkdir -p "$DATA_ROOT" "$LOG_DIR"
STDOUT_LOG="$LOG_DIR/trace_log_${RUN_TS}.log"
STDERR_LOG="$LOG_DIR/trace_error_${RUN_TS}.log"

EXTRA_ARGS=()
[[ -n "$TARGET_NORM_METHOD" ]] && EXTRA_ARGS+=(--target_norm_method "$TARGET_NORM_METHOD")
[[ -n "$TIME_LAGGED_LAG_ORDER" ]] && EXTRA_ARGS+=(--time_lagged_lag_order "$TIME_LAGGED_LAG_ORDER")
[[ -n "$TIME_LAGGED_WEIGHT_SPARSITY" ]] && EXTRA_ARGS+=(--time_lagged_weight_sparsity "$TIME_LAGGED_WEIGHT_SPARSITY")
[[ -n "$TIME_LAGGED_OUTPUT_NOISE_STD" ]] && EXTRA_ARGS+=(--time_lagged_output_noise_std "$TIME_LAGGED_OUTPUT_NOISE_STD")

echo "Phase: O'Prior realism classification"
echo "Save root: $DATA_ROOT"
echo "Logs: $LOG_DIR"
echo "Batches: $NUM_BATCHES (resume_from=$RESUME_FROM)"
echo "Device: $DEVICE"
echo

for ((i=0; i<NUM_BATCHES; i++)); do
  batch_idx=$((RESUME_FROM + i))
  echo "[o'prior-realism] Generating batch #$batch_idx ..."
  t0=$(date +%s)

  $GENLOAD_CMD \
    --save_dir "$DATA_ROOT" \
    --save_format "$SAVE_FORMAT" \
    --np_seed "$NP_SEED" \
    --torch_seed "$TORCH_SEED" \
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
    --use_curriculum "$USE_CURRICULUM" \
    --realism_profile "$REALISM_PROFILE" \
    --use_realism_curriculum "$USE_REALISM_CURRICULUM" \
    --sampling "$SAMPLING" \
    --n_jobs "$N_JOBS" \
    --num_threads_per_generate "$THREADS" \
    --device "$DEVICE" \
    --add_skewness "$ADD_SKEWNESS" \
    --add_svd_features "$ADD_SVD_FEATURES" \
    --svd_encode_categorical_before_svd "$SVD_ENCODE_CATEGORICAL_BEFORE_SVD" \
    --svd_max_onehot_cardinality "$SVD_MAX_ONEHOT_CARDINALITY" \
    --add_fingerprint_feature "$ADD_FINGERPRINT_FEATURE" \
    --fingerprint_method "$FINGERPRINT_METHOD" \
    --categorical_attr "$CATEGORICAL_ATTR" \
    --apply_cross_sectional_rank "$APPLY_CROSS_SECTIONAL_RANK" \
    --apply_censored_targets "$APPLY_CENSORED_TARGETS" \
    --use_strictly_positive_target "$USE_STRICTLY_POSITIVE_TARGET" \
    --apply_covariate_shift "$APPLY_COVARIATE_SHIFT" \
    --apply_seasonal_drift "$APPLY_SEASONAL_DRIFT" \
    --apply_temporal_drift "$APPLY_TEMPORAL_DRIFT" \
    --hybrid_sampling_strategy "$HYBRID_SAMPLING_STRATEGY" \
    "${EXTRA_ARGS[@]}" \
    > >(tee -a "$STDOUT_LOG") \
    2> >(tee -a "$STDERR_LOG" >&2)

  status=$?
  t1=$(date +%s)
  elapsed=$((t1 - t0))

  if [ "$status" -eq 0 ]; then
    echo "[o'prior-realism] Batch #$batch_idx done in ${elapsed}s."
  else
    echo "[o'prior-realism] Batch #$batch_idx FAILED after ${elapsed}s."
    exit "$status"
  fi
done

echo "[o'prior-realism] Done. Quick checks:"
echo "  find $DATA_ROOT -maxdepth 1 -name 'batch_*.'"$SAVE_FORMAT"' | wc -l"
echo "  du -sh $DATA_ROOT"

#!/usr/bin/env bash
# O'Prior full-generation script (classification) with:
# - hierarchical SCM mix
# - realism curriculum (mild -> hard)
# - explicit shift modules

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
PRIOR_TYPE="${PRIOR_TYPE:-mix_scm_hscm}"   # hierarchical SCM mix
MIN_FEATURES="${MIN_FEATURES:-2}"
MAX_FEATURES="${MAX_FEATURES:-100}"
MAX_CLASSES="${MAX_CLASSES:-10}"           # classification
MIN_SEQ_LEN="${MIN_SEQ_LEN:-1024}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-1024}"

# Global curriculum (existing)
USE_CURRICULUM="${USE_CURRICULUM:-True}"
CURRICULUM_SCHEDULE="${CURRICULUM_SCHEDULE:-cosine}" # linear | cosine
CURRICULUM_WARMUP_STEPS="${CURRICULUM_WARMUP_STEPS:-1000}"
CURRICULUM_MIN_RATIO="${CURRICULUM_MIN_RATIO:-0.2}"

# Realism curriculum (mild -> hard)
USE_REALISM_CURRICULUM="${USE_REALISM_CURRICULUM:-True}"
REALISM_PROFILE_START="${REALISM_PROFILE_START:-mild}"
REALISM_PROFILE_END="${REALISM_PROFILE_END:-hard}"
REALISM_SCHEDULE="${REALISM_SCHEDULE:-linear}"
REALISM_WARMUP_STEPS="${REALISM_WARMUP_STEPS:-1000}"
CATEGORICAL_ATTR="${CATEGORICAL_ATTR:-True}"
APPLY_CROSS_SECTIONAL_RANK="${APPLY_CROSS_SECTIONAL_RANK:-True}"
ADD_SKEWNESS="${ADD_SKEWNESS:-True}"
ADD_SVD_FEATURES="${ADD_SVD_FEATURES:-True}"
SVD_ENCODE_CATEGORICAL_BEFORE_SVD="${SVD_ENCODE_CATEGORICAL_BEFORE_SVD:-True}"
SVD_MAX_ONEHOT_CARDINALITY="${SVD_MAX_ONEHOT_CARDINALITY:-16}"
ADD_FINGERPRINT_FEATURE="${ADD_FINGERPRINT_FEATURE:-True}"
FINGERPRINT_METHOD="${FINGERPRINT_METHOD:-projection}"
SAMPLING="${SAMPLING:-beta}"
HYBRID_SAMPLING_STRATEGY="${HYBRID_SAMPLING_STRATEGY:-graph_aware}"
# Needed by export_ltm1.py to restore missingness as NaN.
RETURN_METADATA="${RETURN_METADATA:-True}"


# Shift stressors
APPLY_COVARIATE_SHIFT="${APPLY_COVARIATE_SHIFT:-True}"
APPLY_SEASONAL_DRIFT="${APPLY_SEASONAL_DRIFT:-True}"
APPLY_TEMPORAL_DRIFT="${APPLY_TEMPORAL_DRIFT:-True}"
TEMPORAL_DRIFT_TRANSITION="${TEMPORAL_DRIFT_TRANSITION:-mixed}"  # abrupt|gradual|mixed|none
TARGET_NORM_METHOD="${TARGET_NORM_METHOD:-}"                     # zscore|minmax or empty
TIME_LAGGED_LAG_ORDER="${TIME_LAGGED_LAG_ORDER:-}"               # int or empty
TIME_LAGGED_WEIGHT_SPARSITY="${TIME_LAGGED_WEIGHT_SPARSITY:-}"   # float or empty
TIME_LAGGED_OUTPUT_NOISE_STD="${TIME_LAGGED_OUTPUT_NOISE_STD:-}" # float or empty

# Extra generation controls
N_JOBS="${N_JOBS:--1}"
THREADS="${THREADS:-1}"
DEVICE="${DEVICE:-cpu}"
DATA_ROOT="${DATA_ROOT:-/path/to/directory}"
LOG_BASE="${LOG_BASE:-$DATA_ROOT/logs/rq4}"
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

echo "Phase: O'Prior full classification"
echo "Save root: $DATA_ROOT"
echo "Logs: $LOG_DIR"
echo "Batches: $NUM_BATCHES (resume_from=$RESUME_FROM)"
echo "Device: $DEVICE"
echo

for ((i=0; i<NUM_BATCHES; i++)); do
  batch_idx=$((RESUME_FROM + i))
  echo "[O'Prior-full] Generating batch #$batch_idx ..."
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
    --use_curriculum "$USE_CURRICULUM" \
    --curriculum_schedule "$CURRICULUM_SCHEDULE" \
    --curriculum_warmup_steps "$CURRICULUM_WARMUP_STEPS" \
    --curriculum_min_ratio "$CURRICULUM_MIN_RATIO" \
    --realism_profile "${REALISM_PROFILE:-$REALISM_PROFILE_START}" \
    --use_realism_curriculum "$USE_REALISM_CURRICULUM" \
    --realism_profile_start "$REALISM_PROFILE_START" \
    --realism_profile_end "$REALISM_PROFILE_END" \
    --realism_schedule "$REALISM_SCHEDULE" \
    --realism_warmup_steps "$REALISM_WARMUP_STEPS" \
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
    --apply_covariate_shift "$APPLY_COVARIATE_SHIFT" \
    --apply_seasonal_drift "$APPLY_SEASONAL_DRIFT" \
    --apply_temporal_drift "$APPLY_TEMPORAL_DRIFT" \
    --temporal_drift_transition "$TEMPORAL_DRIFT_TRANSITION" \
    --hybrid_sampling_strategy "$HYBRID_SAMPLING_STRATEGY" \
    --return_metadata "$RETURN_METADATA" \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
    > >(tee -a "$STDOUT_LOG") \
    2> >(tee -a "$STDERR_LOG" >&2)

  status=$?
  t1=$(date +%s)
  elapsed=$((t1 - t0))

  if [ "$status" -eq 0 ]; then
    echo "[O'Prior-full] Batch #$batch_idx done in ${elapsed}s."
  else
    echo "[O'Prior-full] Batch #$batch_idx FAILED after ${elapsed}s."
    exit "$status"
  fi
done

echo "[O'Prior-full] Done. Quick checks:"
echo "  find $DATA_ROOT -maxdepth 1 -name 'batch_*.'"$SAVE_FORMAT"' | wc -l"
echo "  du -sh $DATA_ROOT"

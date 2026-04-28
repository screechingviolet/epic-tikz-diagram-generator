#!/usr/bin/env bash
# Chain SFT + GRPO across multiple datasets and log everything to disk.
#
# Designed to be set running and walked away from. Each (dataset, step) pair
# is run independently; if any one fails (OOM, transient CUDA error, etc.)
# the script logs the failure and moves on to the next step instead of
# tearing the whole run down. The on-disk adapter at $ADAPTER_DIR represents
# the best work-so-far and is what every subsequent step resumes from.
#
# Usage:
#   bash finetuning/run_overnight.sh
#
#   # Just the constraint-specific cycle:
#   DATASETS="dataset_angle dataset_line_tangent dataset_circle_tangent \
#             dataset_parallel dataset_perpendicular dataset_point_on_circle" \
#       bash finetuning/run_overnight.sh
#
#   # Reduce SFT pass length on big datasets:
#   SFT_EPOCHS=0.3 bash finetuning/run_overnight.sh

set -u  # treat unset vars as errors

# ---------------------------------------------------------------------------
# Configuration (overridable via environment variables)
# ---------------------------------------------------------------------------
DATASETS=${DATASETS:-"\
    dataset_simple \
    dataset_medium \
    dataset_complex \
    dataset_angle \
    dataset_circle_tangent \
    dataset_line_tangent \
    dataset_parallel \
    dataset_perpendicular \
    dataset_point_on_circle \
"}

ADAPTER_DIR=${ADAPTER_DIR:-"Qwen2-0.5B-GRPO-geometry"}
SFT_EPOCHS=${SFT_EPOCHS:-1}
LOG_DIR=${LOG_DIR:-"overnight_logs/$(date +%Y%m%d_%H%M%S)"}

mkdir -p "$LOG_DIR"
MAIN_LOG="$LOG_DIR/main.log"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log_main () {
    echo "[overnight $(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$MAIN_LOG"
}

# Set RESUME_FLAG based on whether the adapter directory looks valid right
# now. Called before every step so a failed prior step doesn't poison the
# resume path for subsequent steps.
update_resume_flag () {
    if [ -f "$ADAPTER_DIR/adapter_model.safetensors" ]; then
        RESUME_FLAG="--resume-from $ADAPTER_DIR"
    else
        RESUME_FLAG=""
    fi
}

# Run a single training command, capture full output to its own log file,
# and append a one-line success/failure to the main log.
run_step () {
    local label="$1"; shift
    local logfile="$LOG_DIR/${label}.log"
    log_main "=== $label ==="
    log_main "cmd: $*"
    local start
    start=$(date +%s)
    if "$@" >"$logfile" 2>&1; then
        local elapsed=$(( $(date +%s) - start ))
        log_main "$label OK  (${elapsed}s)"
    else
        local elapsed=$(( $(date +%s) - start ))
        log_main "$label FAILED  (${elapsed}s)  — last 20 lines:"
        tail -n 20 "$logfile" | sed 's/^/    /' | tee -a "$MAIN_LOG" >/dev/null
        FAILED_STEPS+=("$label")
    fi
}

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
FAILED_STEPS=()

log_main "starting; logs at $LOG_DIR"
log_main "datasets: $DATASETS"
log_main "adapter dir: $ADAPTER_DIR"
log_main "SFT epochs: $SFT_EPOCHS"
update_resume_flag
if [ -n "$RESUME_FLAG" ]; then
    log_main "found existing adapter — first step will resume from it"
else
    log_main "no existing adapter — first SFT will start from the base model"
fi

for ds in $DATASETS; do
    update_resume_flag
    run_step "${ds}_sft" python finetuning/train_sft.py \
        --dataset "$ds" --epochs "$SFT_EPOCHS" $RESUME_FLAG

    update_resume_flag
    run_step "${ds}_grpo" python finetuning/train_grpo.py \
        --dataset "$ds" $RESUME_FLAG
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
log_main "=== summary ==="
if [ ${#FAILED_STEPS[@]} -eq 0 ]; then
    log_main "all steps OK"
else
    log_main "${#FAILED_STEPS[@]} failed step(s):"
    for s in "${FAILED_STEPS[@]}"; do
        log_main "  - $s"
    done
fi
log_main "done.  full logs: $LOG_DIR"

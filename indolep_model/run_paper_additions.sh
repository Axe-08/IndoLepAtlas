#!/bin/bash
# =============================================================================
# run_paper_additions.sh
# Paper Addition Experiments — Tasks 1, 2, 3
#
# Task 1: External baselines (ResNet-101, ViT-B/16, EfficientNet-B5)
# Task 2: Geotemporal fusion (Phase 5) + shuffled control
# Task 3: MLFI warmup fix (Phase 3 with --mlfi_warmup 10)
#
# Run from: indolep_model/
#   screen -S paper_exp bash run_paper_additions.sh
# =============================================================================

set -uo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
DATA_ROOT="/home/23uec552/Butterfree/indolepatlas_data/data/butterflies"
RUNS_DIR="./runs"
LOG_DIR="./paper_exp_logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MASTER_LOG="$LOG_DIR/master_${TIMESTAMP}.log"
SUMMARY_LOG="$LOG_DIR/summary_${TIMESTAMP}.log"

mkdir -p "$LOG_DIR" "$RUNS_DIR"

# ── Helpers ───────────────────────────────────────────────────────────────────
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$MASTER_LOG"
}

log_section() {
    local msg="$*"
    local line
    line=$(printf '=%.0s' {1..60})
    log ""
    log "$line"
    log "  $msg"
    log "$line"
}

# Find the most recently created run dir matching a prefix.
# train.py appends _YYYYMMDD_HHMMSS to exp_name, so we sort by name (lexicographic
# on the timestamp suffix is equivalent to time order).
find_latest_run() {
    local prefix="$1"
    # Sort descending so the newest timestamp comes first
    ls -d "$RUNS_DIR"/${prefix}_[0-9]* 2>/dev/null | sort -r | head -1
}

# Run training and stream output to both master log and a per-experiment log.
# Returns 0 on success, 1 on failure. Does NOT abort the whole script.
run_train() {
    local exp_name="$1"
    shift
    local exp_log="$LOG_DIR/${exp_name}.log"

    log_section "TRAIN | $exp_name"
    log "Command: python train.py --data_root $DATA_ROOT --exp_name $exp_name $*"
    log "Per-run log: $exp_log"

    # tee to both master log and dedicated log; capture train.py exit code
    python train.py \
        --data_root "$DATA_ROOT" \
        --exp_name  "$exp_name" \
        "$@" \
        2>&1 | tee -a "$MASTER_LOG" "$exp_log"

    local exit_code=${PIPESTATUS[0]}
    if [ "$exit_code" -ne 0 ]; then
        log "ERROR: training FAILED for $exp_name (exit $exit_code)"
        FAILED_RUNS+=("train:$exp_name")
        return 1
    fi

    log "SUCCESS: $exp_name training complete"
    return 0
}

# Run evaluation on the best checkpoint of a completed run.
# $1 = exp_name prefix used in run_train
# $2 = eval output subdirectory name (default: eval_results)
# remaining args = extra flags for evaluate.py (e.g. --shuffle_geo)
run_eval() {
    local exp_name="$1"
    local eval_subdir="${2:-eval_results}"
    shift 2
    local extra_flags=("$@")

    local run_dir
    run_dir=$(find_latest_run "$exp_name")

    if [ -z "$run_dir" ]; then
        log "ERROR: no run directory found for prefix '$exp_name'"
        FAILED_RUNS+=("eval:$exp_name")
        return 1
    fi

    local checkpoint="$run_dir/best_model.pth"
    if [ ! -f "$checkpoint" ]; then
        log "ERROR: checkpoint not found at $checkpoint"
        FAILED_RUNS+=("eval:$exp_name")
        return 1
    fi

    local eval_log="$LOG_DIR/${exp_name}_${eval_subdir}.log"
    log_section "EVAL  | $exp_name → $eval_subdir"
    log "Checkpoint : $checkpoint"
    log "Output dir : $run_dir/$eval_subdir"
    log "Extra flags: ${extra_flags[*]:-none}"
    log "Eval log   : $eval_log"

    python evaluate.py \
        --data_root      "$DATA_ROOT" \
        --checkpoint     "$checkpoint" \
        --output_dir     "$run_dir/$eval_subdir" \
        --sparse_threshold 50 \
        "${extra_flags[@]}" \
        2>&1 | tee -a "$MASTER_LOG" "$eval_log"

    local exit_code=${PIPESTATUS[0]}
    if [ "$exit_code" -ne 0 ]; then
        log "ERROR: eval FAILED for $exp_name/$eval_subdir (exit $exit_code)"
        FAILED_RUNS+=("eval:$exp_name/$eval_subdir")
        return 1
    fi

    log "SUCCESS: $exp_name/$eval_subdir eval complete"
    return 0
}

# ── Initialise failure tracker ────────────────────────────────────────────────
FAILED_RUNS=()

log "======================================================================"
log "  PAPER ADDITION EXPERIMENTS — IndoLepAtlas"
log "  Started   : $(date)"
log "  Data root : $DATA_ROOT"
log "  Runs dir  : $RUNS_DIR"
log "  Master log: $MASTER_LOG"
log "======================================================================"

# =============================================================================
# TASK 1 — External baselines
#
# Full end-to-end fine-tuning (backbone at 0.1x LR, head at 1x LR).
# CE + balanced sampling matches your best ablation config.
# No need to test focal here — these are comparison-table runs, not ablations.
# =============================================================================

log_section "TASK 1: External Baselines"

run_train "t1_baseline_resnet101" \
    --baseline          resnet101 \
    --loss              ce \
    --label_smoothing   0.1 \
    --balanced_sampling true \
    --epochs            50 \
    --batch_size        8 \
    --grad_accum_steps  4 \
    --lr                1e-4 \
    --weight_decay      0.05 \
    --warmup_epochs     5 \
    --img_size          224 \
    --num_workers       4 \
    --output_dir        "$RUNS_DIR"

run_eval "t1_baseline_resnet101"

# ── ViT-B/16 uses 224 as well; timm handles the patch embedding ───────────────
run_train "t1_baseline_vit" \
    --baseline          vit_base_patch16 \
    --loss              ce \
    --label_smoothing   0.1 \
    --balanced_sampling true \
    --epochs            50 \
    --batch_size        8 \
    --grad_accum_steps  4 \
    --lr                1e-4 \
    --weight_decay      0.05 \
    --warmup_epochs     5 \
    --img_size          224 \
    --num_workers       4 \
    --output_dir        "$RUNS_DIR"

run_eval "t1_baseline_vit"

# ── EfficientNet-B5 prefers 456px but we keep 224 for fair comparison ─────────
run_train "t1_baseline_efficientnet" \
    --baseline          efficientnet_b5 \
    --loss              ce \
    --label_smoothing   0.1 \
    --balanced_sampling true \
    --epochs            50 \
    --batch_size        8 \
    --grad_accum_steps  4 \
    --lr                1e-4 \
    --weight_decay      0.05 \
    --warmup_epochs     5 \
    --img_size          224 \
    --num_workers       4 \
    --output_dir        "$RUNS_DIR"

run_eval "t1_baseline_efficientnet"

# =============================================================================
# TASK 2 — Geotemporal fusion
#
# Phase 5 = CA + MLFI + Geo.  Same loss/sampling as Task 1 for consistency.
# Two evaluations on the SAME checkpoint:
#   (a) Normal geo  → proves geotemporal helps
#   (b) Shuffled geo → proves the signal is real and not an artifact
# =============================================================================

log_section "TASK 2: Geotemporal Fusion (Phase 5)"

run_train "t2_geo_phase5_ce_bal" \
    --phase             5 \
    --loss              ce \
    --label_smoothing   0.1 \
    --balanced_sampling true \
    --epochs            50 \
    --batch_size        8 \
    --grad_accum_steps  4 \
    --lr                1e-4 \
    --weight_decay      0.05 \
    --warmup_epochs     5 \
    --dropout           0.3 \
    --img_size          224 \
    --num_workers       4 \
    --output_dir        "$RUNS_DIR"

# Normal eval (geo features intact)
run_eval "t2_geo_phase5_ce_bal" "eval_results"

# Shuffled-geo eval — randomises zone/month across the batch at inference time.
# A drop in F1 relative to the normal eval confirms the geo features carry real signal.
run_eval "t2_geo_phase5_ce_bal" "eval_results_shuffled" --shuffle_geo

# =============================================================================
# TASK 3 — MLFI warmup fix
#
# The Phase 3 underperformance was diagnosed as an optimisation imbalance:
# freshly initialised MLFI branches start at full LR while the pretrained
# backbone is near-converged at 0.1x LR.  --mlfi_warmup 10 ramps the MLFI
# branches from 10% → 100% LR over the first 10 epochs, then cosine-decays.
#
# Compare this result against the original Phase 3 run (from existing ablation)
# to see if the 2-point gap closes.
# =============================================================================

log_section "TASK 3: MLFI Warmup Fix (Phase 3, mlfi_warmup=10)"

run_train "t3_phase3_mlfi_warmup10" \
    --phase             3 \
    --loss              ce \
    --label_smoothing   0.1 \
    --balanced_sampling true \
    --mlfi_warmup       10 \
    --epochs            50 \
    --batch_size        8 \
    --grad_accum_steps  4 \
    --lr                1e-4 \
    --weight_decay      0.05 \
    --warmup_epochs     5 \
    --dropout           0.3 \
    --img_size          224 \
    --num_workers       4 \
    --output_dir        "$RUNS_DIR"

run_eval "t3_phase3_mlfi_warmup10"

# =============================================================================
# Aggregate results
# =============================================================================

log_section "Collecting aggregate results"

cd analysis || { log "ERROR: analysis/ directory not found"; exit 1; }
python collect_results.py 2>&1 | tee -a "../$MASTER_LOG"
cd ..

# =============================================================================
# Final summary
# =============================================================================

log_section "EXPERIMENT RUN COMPLETE"
log "Finished: $(date)"
log ""

if [ ${#FAILED_RUNS[@]} -eq 0 ]; then
    log "All experiments completed successfully."
else
    log "The following experiments FAILED:"
    for f in "${FAILED_RUNS[@]}"; do
        log "  - $f"
    done
fi

log ""
log "Checkpoint locations:"
for prefix in t1_baseline_resnet101 t1_baseline_vit t1_baseline_efficientnet \
              t2_geo_phase5_ce_bal t3_phase3_mlfi_warmup10; do
    run_dir=$(find_latest_run "$prefix")
    if [ -n "$run_dir" ]; then
        log "  $prefix → $run_dir/best_model.pth"
    else
        log "  $prefix → NOT FOUND"
    fi
done

log ""
log "Results summary saved to: results/summary.csv"
log "Full log: $MASTER_LOG"

# Write a compact human-readable summary of eval metrics
{
    echo "Paper Additions — Eval Summary"
    echo "Generated: $(date)"
    echo ""
    for prefix in t1_baseline_resnet101 t1_baseline_vit t1_baseline_efficientnet \
                  t2_geo_phase5_ce_bal t3_phase3_mlfi_warmup10; do
        run_dir=$(find_latest_run "$prefix")
        summary_json="$run_dir/eval_results/summary.json"
        if [ -f "$summary_json" ]; then
            echo "=== $prefix ==="
            python -c "
import json, sys
d = json.load(open('$summary_json'))
print(f\"  Top-1 Acc : {d.get('top1_accuracy', 'N/A'):.4f}\")
print(f\"  Top-5 Acc : {d.get('top5_accuracy', 'N/A'):.4f}\")
print(f\"  Macro F1  : {d.get('macro_f1', 'N/A'):.4f}\")
print(f\"  Weighted F1: {d.get('weighted_f1', 'N/A'):.4f}\")
"
        else
            echo "=== $prefix === (eval results not found)"
        fi
        echo ""
    done

    # Geo shuffled comparison
    geo_dir=$(find_latest_run "t2_geo_phase5_ce_bal")
    for subdir in eval_results eval_results_shuffled; do
        summary_json="$geo_dir/$subdir/summary.json"
        if [ -f "$summary_json" ]; then
            echo "=== t2_geo_phase5_ce_bal / $subdir ==="
            python -c "
import json
d = json.load(open('$summary_json'))
print(f\"  Top-1 Acc : {d.get('top1_accuracy', 'N/A'):.4f}\")
print(f\"  Macro F1  : {d.get('macro_f1', 'N/A'):.4f}\")
"
            echo ""
        fi
    done
} | tee "$SUMMARY_LOG" | tee -a "$MASTER_LOG"

log "Compact summary written to: $SUMMARY_LOG"

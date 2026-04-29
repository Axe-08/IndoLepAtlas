#!/bin/bash
# ============================================================
# run_paper_experiments.sh
# Paper-quality experiments for IndoLepAtlas research paper.
#
# GPU Allocation (based on current DGX availability):
#   GPU 3: ~31 GB free  → ResNet-101 baseline, then MLFI warmup fix
#   GPU 4: ~32 GB free  → ViT-B/16 baseline, then EfficientNet-B5
#   GPU 2: ~18 GB free  → Geotemporal fusion (Phase 5)
#   GPU 0: ~16 GB free  → Geo-shuffled control experiment
#
# Usage:
#   screen -S paper_exp
#   bash run_paper_experiments.sh
# ============================================================

set -e  # Exit on error

eval "$(micromamba shell hook --shell=bash)"
micromamba activate indolep_env

DATA_ROOT="/home/23uec552/Butterfree/indolepatlas_data/data/butterflies"
EPOCHS=40
WORKERS=8

echo "============================================================"
echo "  IndoLepAtlas Paper Experiments"
echo "  Data: $DATA_ROOT"
echo "  Epochs: $EPOCHS"
echo "  $(date)"
echo "============================================================"

# ── GPU 3: ResNet-101 baseline, then MLFI warmup fix (sequential) ──────────
(
    echo "[GPU 3] Starting ResNet-101 baseline..."
    python train.py \
        --baseline resnet101 \
        --data_root "$DATA_ROOT" \
        --loss ce \
        --balanced_sampling true \
        --epochs $EPOCHS \
        --batch_size 64 \
        --lr 1e-4 \
        --warmup_epochs 5 \
        --num_workers $WORKERS \
        --gpu 3 \
        --exp_name paper_baseline_resnet101

    echo "[GPU 3] ResNet-101 done. Starting MLFI warmup fix..."
    python train.py \
        --phase 3 \
        --mlfi_warmup 10 \
        --data_root "$DATA_ROOT" \
        --loss ce \
        --balanced_sampling true \
        --epochs $EPOCHS \
        --batch_size 32 \
        --lr 1e-4 \
        --warmup_epochs 5 \
        --num_workers $WORKERS \
        --gpu 3 \
        --exp_name paper_mlfi_warmup_fix

    echo "[GPU 3] All jobs done."
) &

# ── GPU 4: ViT-B/16 baseline, then EfficientNet-B5 (sequential) ────────────
(
    echo "[GPU 4] Starting ViT-B/16 baseline (bs=16 for memory safety)..."
    python train.py \
        --baseline vit_base_patch16 \
        --data_root "$DATA_ROOT" \
        --loss ce \
        --balanced_sampling true \
        --epochs $EPOCHS \
        --batch_size 16 \
        --grad_accum_steps 2 \
        --lr 1e-4 \
        --warmup_epochs 5 \
        --num_workers $WORKERS \
        --gpu 4 \
        --exp_name paper_baseline_vit_b16

    echo "[GPU 4] ViT-B/16 done. Starting EfficientNet-B5..."
    python train.py \
        --baseline efficientnet_b5 \
        --data_root "$DATA_ROOT" \
        --loss ce \
        --balanced_sampling true \
        --epochs $EPOCHS \
        --batch_size 32 \
        --lr 1e-4 \
        --warmup_epochs 5 \
        --num_workers $WORKERS \
        --gpu 4 \
        --exp_name paper_baseline_effnet_b5

    echo "[GPU 4] All jobs done."
) &

# ── GPU 2: Geotemporal fusion Phase 5 ──────────────────────────────────────
(
    echo "[GPU 2] Starting Geotemporal Phase 5..."
    python train.py \
        --phase 5 \
        --data_root "$DATA_ROOT" \
        --loss ce \
        --balanced_sampling true \
        --epochs $EPOCHS \
        --batch_size 32 \
        --lr 1e-4 \
        --warmup_epochs 5 \
        --num_workers $WORKERS \
        --gpu 2 \
        --exp_name paper_geo_phase5

    echo "[GPU 2] Geotemporal Phase 5 done."
) &

# ── GPU 0: Geotemporal shuffled control ─────────────────────────────────────
(
    echo "[GPU 0] Starting Geo-shuffled control..."
    python train.py \
        --phase 5 \
        --shuffle_geo \
        --data_root "$DATA_ROOT" \
        --loss ce \
        --balanced_sampling true \
        --epochs $EPOCHS \
        --batch_size 32 \
        --lr 1e-4 \
        --warmup_epochs 5 \
        --num_workers $WORKERS \
        --gpu 0 \
        --exp_name paper_geo_phase5_shuffled

    echo "[GPU 0] Geo-shuffled control done."
) &

# ── Wait for all GPU groups ─────────────────────────────────────────────────
wait
echo ""
echo "============================================================"
echo "  All training complete! $(date)"
echo "  Running evaluation on all experiments..."
echo "============================================================"

# ── Evaluate all paper experiments ──────────────────────────────────────────
for EXP in paper_baseline_resnet101 paper_baseline_vit_b16 paper_baseline_effnet_b5 \
           paper_geo_phase5 paper_geo_phase5_shuffled paper_mlfi_warmup_fix; do

    # Find the most recent run folder for this experiment
    RUN_DIR=$(ls -dt runs/${EXP}_* 2>/dev/null | head -1)
    if [ -z "$RUN_DIR" ]; then
        echo "  [WARN] No run dir found for $EXP, skipping eval."
        continue
    fi

    CKPT="$RUN_DIR/best_model.pth"
    if [ ! -f "$CKPT" ]; then
        echo "  [WARN] No checkpoint found at $CKPT, skipping."
        continue
    fi

    echo ""
    echo "  Evaluating: $EXP"
    echo "  Checkpoint: $CKPT"

    EVAL_FLAGS=""
    if [[ "$EXP" == *"shuffled"* ]]; then
        EVAL_FLAGS="--shuffle_geo"
    fi

    python evaluate.py \
        --data_root "$DATA_ROOT" \
        --checkpoint "$CKPT" \
        --output_dir "$RUN_DIR/eval_results" \
        --sparse_threshold 50 \
        --gpu 4 \
        $EVAL_FLAGS
done

echo ""
echo "============================================================"
echo "  All evaluations complete! $(date)"
echo "  Check runs/paper_*/eval_results/ for results."
echo "============================================================"

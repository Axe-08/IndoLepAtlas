#!/bin/bash
# ============================================================
# dry_run_test.sh
# Validates all code changes compile and run for 1 epoch
# before committing to the full overnight batch.
#
# Run this FIRST on DGX after pulling the code.
# All tests use GPU 4 (32GB free, fully isolated).
#
# Expected runtime: ~5-10 minutes total
# Usage:
#   bash dry_run_test.sh
# ============================================================

set -e

eval "$(micromamba shell hook --shell=bash)"
micromamba activate indolep_env

DATA_ROOT="/home/23uec552/Butterfree/indolepatlas_data/data/butterflies"
GPU=4
PASS=0
FAIL=0

# Use a temp output dir for dry runs so they don't clutter real runs/
DRY_RUN_DIR="./runs/_dryrun_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$DRY_RUN_DIR"

run_test() {
    local NAME="$1"
    local CMD="$2"
    
    echo ""
    echo "──────────────────────────────────────────────────────"
    echo "  TEST: $NAME"
    echo "──────────────────────────────────────────────────────"
    
    if eval "$CMD"; then
        echo "  ✓ PASSED: $NAME"
        PASS=$((PASS + 1))
    else
        echo "  ✗ FAILED: $NAME"
        FAIL=$((FAIL + 1))
    fi
}

echo "============================================================"
echo "  IndoLepAtlas Dry-Run Validation Suite"
echo "  GPU: $GPU | Data: $DATA_ROOT"
echo "  $(date)"
echo "============================================================"

# ── Test 1: Import sanity ─────────────────────────────────────────────────
run_test "Python imports" \
    "python -c \"
from losses import build_loss
from models.backbone import build_model
from models.baselines import build_baseline_model
from dataset import create_dataloaders
print('  All imports OK')
\""

# ── Test 2: Losses module ─────────────────────────────────────────────────
run_test "Losses: CE build" \
    "python -c \"
from losses import build_loss
import torch
loss_fn = build_loss('ce')
logits = torch.randn(4, 10)
labels = torch.randint(0, 10, (4,))
loss = loss_fn(logits, labels)
assert loss.item() > 0
print(f'  CE loss: {loss.item():.4f} ✓')
\""

run_test "Losses: Focal build" \
    "python -c \"
from losses import build_loss
import torch
weights = torch.ones(10)
loss_fn = build_loss('focal', class_weights=weights, gamma=2.0)
logits = torch.randn(4, 10)
labels = torch.randint(0, 10, (4,))
loss = loss_fn(logits, labels)
assert loss.item() > 0
print(f'  Focal loss: {loss.item():.4f} ✓')
\""

# ── Test 3: Baseline model builds ────────────────────────────────────────
run_test "Baseline: ResNet-101 build" \
    "python -c \"
from models.baselines import build_baseline_model
import torch
m = build_baseline_model('resnet101', num_classes=966, pretrained=False)
x = torch.randn(2, 3, 224, 224)
out = m(x)
assert out.shape == (2, 966), f'Expected (2, 966), got {out.shape}'
print(f'  ResNet-101 output: {out.shape} ✓')
\""

run_test "Baseline: ViT-B/16 build" \
    "python -c \"
from models.baselines import build_baseline_model
import torch
m = build_baseline_model('vit_base_patch16', num_classes=966, pretrained=False)
x = torch.randn(2, 3, 224, 224)
out = m(x)
assert out.shape == (2, 966), f'Expected (2, 966), got {out.shape}'
print(f'  ViT-B/16 output: {out.shape} ✓')
\""

run_test "Baseline: EfficientNet-B5 build" \
    "python -c \"
from models.baselines import build_baseline_model
import torch
m = build_baseline_model('efficientnet_b5', num_classes=966, pretrained=False)
x = torch.randn(2, 3, 224, 224)
out = m(x)
assert out.shape == (2, 966), f'Expected (2, 966), got {out.shape}'
print(f'  EfficientNet-B5 output: {out.shape} ✓')
\""

# ── Test 4: Phase 5 model (geotemporal) ──────────────────────────────────
run_test "Phase 5: Geotemporal model build + forward" \
    "python -c \"
from models.backbone import build_model
import torch
m = build_model(num_classes=966, phase=5, pretrained=False)
x = torch.randn(2, 3, 224, 224)
zone_idx = torch.randint(0, 9, (2,))
month_enc = torch.randn(2, 2)
out = m(x, zone_idx, month_enc)
assert out.shape == (2, 966), f'Expected (2, 966), got {out.shape}'
print(f'  Phase 5 output: {out.shape} ✓')
\""

# ── Test 5: Full 1-epoch training runs ───────────────────────────────────
run_test "1-Epoch: ResNet-101 baseline training" \
    "python train.py \
        --baseline resnet101 \
        --data_root '$DATA_ROOT' \
        --loss ce \
        --balanced_sampling true \
        --epochs 1 \
        --batch_size 32 \
        --num_workers 4 \
        --gpu $GPU \
        --output_dir '$DRY_RUN_DIR' \
        --exp_name dry_resnet101"

run_test "1-Epoch: ViT-B/16 baseline training" \
    "python train.py \
        --baseline vit_base_patch16 \
        --data_root '$DATA_ROOT' \
        --loss ce \
        --balanced_sampling true \
        --epochs 1 \
        --batch_size 16 \
        --grad_accum_steps 2 \
        --num_workers 4 \
        --gpu $GPU \
        --output_dir '$DRY_RUN_DIR' \
        --exp_name dry_vit_b16"

run_test "1-Epoch: EfficientNet-B5 baseline training" \
    "python train.py \
        --baseline efficientnet_b5 \
        --data_root '$DATA_ROOT' \
        --loss ce \
        --balanced_sampling true \
        --epochs 1 \
        --batch_size 32 \
        --num_workers 4 \
        --gpu $GPU \
        --output_dir '$DRY_RUN_DIR' \
        --exp_name dry_effnet_b5"

run_test "1-Epoch: Geotemporal Phase 5 training" \
    "python train.py \
        --phase 5 \
        --data_root '$DATA_ROOT' \
        --loss ce \
        --balanced_sampling true \
        --epochs 1 \
        --batch_size 32 \
        --num_workers 4 \
        --gpu $GPU \
        --output_dir '$DRY_RUN_DIR' \
        --exp_name dry_geo_phase5"

run_test "1-Epoch: Geo-shuffled control training" \
    "python train.py \
        --phase 5 \
        --shuffle_geo \
        --data_root '$DATA_ROOT' \
        --loss ce \
        --balanced_sampling true \
        --epochs 1 \
        --batch_size 32 \
        --num_workers 4 \
        --gpu $GPU \
        --output_dir '$DRY_RUN_DIR' \
        --exp_name dry_geo_shuffled"

run_test "1-Epoch: MLFI warmup fix training" \
    "python train.py \
        --phase 3 \
        --mlfi_warmup 10 \
        --data_root '$DATA_ROOT' \
        --loss ce \
        --balanced_sampling true \
        --epochs 1 \
        --batch_size 32 \
        --num_workers 4 \
        --gpu $GPU \
        --output_dir '$DRY_RUN_DIR' \
        --exp_name dry_mlfi_warmup"

# ── Test 6: Evaluate on a dry-run checkpoint ─────────────────────────────
CKPT_PATH=$(ls "${DRY_RUN_DIR}/dry_resnet101"*/best_model.pth 2>/dev/null | head -1)
if [ -n "$CKPT_PATH" ]; then
    run_test "Evaluate: ResNet-101 dry-run checkpoint" \
        "python evaluate.py \
            --data_root '$DATA_ROOT' \
            --checkpoint '$CKPT_PATH' \
            --output_dir '${DRY_RUN_DIR}/eval_resnet101' \
            --sparse_threshold 50 \
            --gpu $GPU"
else
    echo "  [SKIP] Eval test: no dry-run checkpoint found (training test may have failed)"
fi

# ── Test 7: Shuffle-geo eval flag ────────────────────────────────────────
GEO_CKPT=$(ls "${DRY_RUN_DIR}/dry_geo_phase5"*/best_model.pth 2>/dev/null | head -1)
if [ -n "$GEO_CKPT" ]; then
    run_test "Evaluate: Geo-shuffled eval flag" \
        "python evaluate.py \
            --data_root '$DATA_ROOT' \
            --checkpoint '$GEO_CKPT' \
            --output_dir '${DRY_RUN_DIR}/eval_geo_shuffled' \
            --sparse_threshold 50 \
            --shuffle_geo \
            --gpu $GPU"
fi

# ── Summary ───────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  DRY RUN COMPLETE  — $(date)"
echo "  PASSED: $PASS"
echo "  FAILED: $FAIL"
echo "  Dry run outputs: $DRY_RUN_DIR"
if [ $FAIL -eq 0 ]; then
    echo ""
    echo "  ✓ All tests passed. Safe to run:"
    echo "    bash run_paper_experiments.sh"
else
    echo ""
    echo "  ✗ $FAIL test(s) failed. Fix issues before launching full experiments."
fi
echo "============================================================"

# Exit with failure code if any test failed (useful for CI)
[ $FAIL -eq 0 ]

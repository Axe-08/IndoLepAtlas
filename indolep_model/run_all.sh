#!/bin/bash
# ==============================================================================
# Indian Butterfly Classifier - Master Execution Script
# ==============================================================================
# Focuses on highest-accuracy configuration suited for scattered / small tail datasets:
#   - Phase 3 Architecture: ConvNeXt + Coordinate Attention + MLFI 
#   - Loss: Focal Loss (Counteracts extreme class imbalance dynamically)
#   - Pretrained: True 
#
# Generates progress.log continuously for visual inspection.

set -e

# Configuration
DATA_ROOT="/home/23uec552/Butterfree/indolepatlas_data/data/butterflies"
EPOCHS=50
BATCH_SIZE=32
LR=0.0001
PHASE=3
LOSS="focal"

echo ""
echo "========================================================="
echo "  1. Running Data Audit & Filtering"
echo "========================================================="
if [ ! -f "metadata_filtered.csv" ]; then
    echo "Running data audit to process raw metadata..."
    python data_audit.py --data_root "$DATA_ROOT"
else
    echo "Filtered metadata already exists. Skipping audit."
fi

echo ""
echo "========================================================="
echo "  2. Initiating Training Sequence (Phase $PHASE - $LOSS)"
echo "========================================================="
echo "  Watch real-time status dynamically in your runs/ directory."
echo "  Wait a few seconds for the run directory to populate."
echo "---------------------------------------------------------"

# Run training
python train.py \
    --data_root "$DATA_ROOT" \
    --phase $PHASE \
    --loss $LOSS \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --pretrained True

echo ""
echo "========================================================="
echo "  3. Evaluating Best Model Checkpoint"
echo "========================================================="

# Dynamically find the most recent run generated
LATEST_RUN=$(ls -td runs/* | head -1)

if [ -z "$LATEST_RUN" ] || [ ! -f "$LATEST_RUN/best_model.pth" ]; then
    echo "Error: Could not locate a successful training run or best_model.pth in $LATEST_RUN."
    exit 1
fi

echo "  Evaluating checkpoint -> $LATEST_RUN/best_model.pth"

python evaluate.py \
    --data_root "$DATA_ROOT" \
    --checkpoint "$LATEST_RUN/best_model.pth" \
    --output_dir "$LATEST_RUN/eval_results"

echo "========================================================="
echo "  PIPELINE COMPLETE."
echo "  Review Metrics: $LATEST_RUN/metrics.csv"
echo "  Review Graphs:  $LATEST_RUN/metrics_curve.png"
echo "  Review Log:     $LATEST_RUN/progress.log"
echo "========================================================="

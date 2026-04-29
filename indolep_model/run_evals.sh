#!/bin/bash
# run_evals.sh
# Separated script to run only the evaluation and analysis collection loop.
# It automatically targets the most recent run folder for each experiment prefix.

eval "$(micromamba shell hook --shell=bash)"
micromamba activate indolep_env

DATA_ROOT="/home/23uec552/Butterfree/indolepatlas_data/data/butterflies"

echo "=========================================="
echo " Starting Evaluation Loop                 "
echo "=========================================="

for RUN_PREFIX in unit1_ce_bal unit1_ce_unbal unit1_focal_bal unit1_focal_unbal unit2_phase1 unit2_phase2 unit2_phase3 unit3_freeze_none unit3_head_only unit3_freeze_early unit3_freeze_late; do
    echo ">> Hunting for latest $RUN_PREFIX..."
    
    # Grab all matching directories sorted by newest first
    MATCHING_DIRS=$(ls -td runs/${RUN_PREFIX}_* 2>/dev/null)
    
    LATEST_DIR=""
    for D in $MATCHING_DIRS; do
        if [ -f "$D/best_model.pth" ]; then
            LATEST_DIR="$D"
            break
        fi
    done
    
    if [ -z "$LATEST_DIR" ]; then
        echo "   [ERROR] No completed run (with best_model.pth) found for $RUN_PREFIX. Skipping!"
        continue
    fi
    
    echo "   Evaluating..."
    python evaluate.py \
      --gpu 3 \
      --data_root $DATA_ROOT \
      --checkpoint $LATEST_DIR/best_model.pth \
      --output_dir $LATEST_DIR/eval_results \
      --sparse_threshold 50
    echo "   Done!"
done

echo "=========================================="
echo " Aggregating Results...                   "
echo "=========================================="
cd analysis
python collect_results.py

echo "=========================================="
echo " Pipeline Complete."
echo " Check results/summary.csv and the plots! "
echo "=========================================="

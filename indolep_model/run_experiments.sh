#!/bin/bash
# run_experiments.sh
# This script runs all Unit I, II, and III experiments inside a screen session.

# Activate environment
source ~/.bashrc
conda activate indolep_venv

DATA_ROOT="/home/23uec552/Butterfree/indolepatlas_data/data/butterflies"

echo "=========================================="
echo " Starting Unit I: Long-Tail Loss Dynamics "
echo "=========================================="
python train.py --data_root $DATA_ROOT --phase 3 --loss ce --balanced_sampling true --exp_name unit1_ce_bal
python train.py --data_root $DATA_ROOT --phase 3 --loss ce --balanced_sampling false --exp_name unit1_ce_unbal
python train.py --data_root $DATA_ROOT --phase 3 --loss focal --balanced_sampling true --exp_name unit1_focal_bal
python train.py --data_root $DATA_ROOT --phase 3 --loss focal --balanced_sampling false --exp_name unit1_focal_unbal

echo "=========================================="
echo " Starting Unit II: Feature-Fusion Impact  "
echo "=========================================="
python train.py --data_root $DATA_ROOT --phase 1 --exp_name unit2_phase1
python train.py --data_root $DATA_ROOT --phase 2 --exp_name unit2_phase2
python train.py --data_root $DATA_ROOT --phase 3 --exp_name unit2_phase3

echo "=========================================="
echo " Starting Unit III: Layer Freezing Strat. "
echo "=========================================="
python train.py --data_root $DATA_ROOT --phase 3 --freeze_strategy none --exp_name unit3_freeze_none
python train.py --data_root $DATA_ROOT --phase 3 --freeze_strategy head_only --exp_name unit3_head_only
python train.py --data_root $DATA_ROOT --phase 3 --freeze_strategy freeze_early --exp_name unit3_freeze_early
python train.py --data_root $DATA_ROOT --phase 3 --freeze_strategy freeze_late --exp_name unit3_freeze_late

echo "=========================================="
echo " All Training Complete! Running Eval...   "
echo "=========================================="

for RUN in unit1_ce_bal unit1_ce_unbal unit1_focal_bal unit1_focal_unbal unit2_phase1 unit2_phase2 unit2_phase3 unit3_freeze_none unit3_head_only unit3_freeze_early unit3_freeze_late; do
    echo "Evaluating $RUN..."
    python evaluate.py \
      --data_root $DATA_ROOT \
      --checkpoint runs/$RUN/best_model.pth \
      --output_dir runs/$RUN/eval_results \
      --sparse_threshold 50
done

echo "=========================================="
echo " Generating Aggregated Results and Plots  "
echo "=========================================="
cd analysis
python collect_results.py

echo "Done! Check the results/ directory for summaries."

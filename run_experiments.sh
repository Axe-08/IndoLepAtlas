#!/bin/bash
# ==============================================================================
# IndoLepAtlas Phase-2 Master Experiment Script
# ==============================================================================
# Executes all experiments for Unit I, II, and III, followed by evaluation.

set -e

DATA_ROOT="/home/23uec552/Butterfree/indolepatlas_data/data/butterflies"

# Ensure we're in the right directory
cd indolep_model

echo "=========================================="
echo " Running Unit I: Long-Tail Loss Dynamics"
echo "=========================================="
# CE + Balanced
python train.py --data_root "$DATA_ROOT" --phase 3 --loss ce --balanced_sampling true --exp_name unit1_ce_bal
python evaluate.py --data_root "$DATA_ROOT" --checkpoint runs/unit1_ce_bal/best_model.pth --output_dir runs/unit1_ce_bal/eval_results --sparse_threshold 50

# CE + Unbalanced
python train.py --data_root "$DATA_ROOT" --phase 3 --loss ce --balanced_sampling false --exp_name unit1_ce_unbal
python evaluate.py --data_root "$DATA_ROOT" --checkpoint runs/unit1_ce_unbal/best_model.pth --output_dir runs/unit1_ce_unbal/eval_results --sparse_threshold 50

# Focal + Balanced
python train.py --data_root "$DATA_ROOT" --phase 3 --loss focal --balanced_sampling true --exp_name unit1_focal_bal
python evaluate.py --data_root "$DATA_ROOT" --checkpoint runs/unit1_focal_bal/best_model.pth --output_dir runs/unit1_focal_bal/eval_results --sparse_threshold 50

# Focal + Unbalanced
python train.py --data_root "$DATA_ROOT" --phase 3 --loss focal --balanced_sampling false --exp_name unit1_focal_unbal
python evaluate.py --data_root "$DATA_ROOT" --checkpoint runs/unit1_focal_unbal/best_model.pth --output_dir runs/unit1_focal_unbal/eval_results --sparse_threshold 50


echo "=========================================="
echo " Running Unit II: Feature-Fusion Impact"
echo "=========================================="
# Phase 1
python train.py --data_root "$DATA_ROOT" --phase 1 --exp_name unit2_phase1
python evaluate.py --data_root "$DATA_ROOT" --checkpoint runs/unit2_phase1/best_model.pth --output_dir runs/unit2_phase1/eval_results --sparse_threshold 50

# Phase 2
python train.py --data_root "$DATA_ROOT" --phase 2 --exp_name unit2_phase2
python evaluate.py --data_root "$DATA_ROOT" --checkpoint runs/unit2_phase2/best_model.pth --output_dir runs/unit2_phase2/eval_results --sparse_threshold 50

# Phase 3
python train.py --data_root "$DATA_ROOT" --phase 3 --exp_name unit2_phase3
python evaluate.py --data_root "$DATA_ROOT" --checkpoint runs/unit2_phase3/best_model.pth --output_dir runs/unit2_phase3/eval_results --sparse_threshold 50


echo "=========================================="
echo " Running Unit III: Layer Freezing Strategy"
echo "=========================================="
# none
python train.py --data_root "$DATA_ROOT" --phase 3 --freeze_strategy none --exp_name unit3_freeze_none
python evaluate.py --data_root "$DATA_ROOT" --checkpoint runs/unit3_freeze_none/best_model.pth --output_dir runs/unit3_freeze_none/eval_results --sparse_threshold 50

# head_only
python train.py --data_root "$DATA_ROOT" --phase 3 --freeze_strategy head_only --exp_name unit3_head_only
python evaluate.py --data_root "$DATA_ROOT" --checkpoint runs/unit3_head_only/best_model.pth --output_dir runs/unit3_head_only/eval_results --sparse_threshold 50

# freeze_early
python train.py --data_root "$DATA_ROOT" --phase 3 --freeze_strategy freeze_early --exp_name unit3_freeze_early
python evaluate.py --data_root "$DATA_ROOT" --checkpoint runs/unit3_freeze_early/best_model.pth --output_dir runs/unit3_freeze_early/eval_results --sparse_threshold 50

# freeze_late
python train.py --data_root "$DATA_ROOT" --phase 3 --freeze_strategy freeze_late --exp_name unit3_freeze_late
python evaluate.py --data_root "$DATA_ROOT" --checkpoint runs/unit3_freeze_late/best_model.pth --output_dir runs/unit3_freeze_late/eval_results --sparse_threshold 50

echo "=========================================="
echo " All experiments and evaluations complete!"
echo "=========================================="
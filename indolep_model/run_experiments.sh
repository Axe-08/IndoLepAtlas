#!/bin/bash
# run_experiments.sh
# This script runs all Unit I, II, and III experiments inside a screen session.

# Activate environment
eval "$(micromamba shell hook --shell=bash)"
micromamba activate indolep_env

DATA_ROOT="/home/23uec552/Butterfree/indolepatlas_data/data/butterflies"
EPOCHS=40

echo "=========================================="
echo " Starting Parallel Execution of Units     "
echo " (40 Epochs per run - ~3.5 Hours)          "
echo "=========================================="

# GPU 3 (~25.2 GB Free) -> 4 Jobs (22.4 GB req)
(
    echo "[GPU 3] Starting Unit I: Long-Tail Loss Dynamics"
    python train.py --epochs $EPOCHS --gpu 3 --data_root $DATA_ROOT --phase 3 --loss ce --balanced_sampling true --exp_name unit1_ce_bal &
    sleep 5
    python train.py --epochs $EPOCHS --gpu 3 --data_root $DATA_ROOT --phase 3 --loss ce --balanced_sampling false --exp_name unit1_ce_unbal &
    sleep 5
    python train.py --epochs $EPOCHS --gpu 3 --data_root $DATA_ROOT --phase 3 --loss focal --balanced_sampling true --exp_name unit1_focal_bal &
    sleep 5
    python train.py --epochs $EPOCHS --gpu 3 --data_root $DATA_ROOT --phase 3 --loss focal --balanced_sampling false --exp_name unit1_focal_unbal &
    wait
    echo "[GPU 3] Unit I Finished!"
) &

# GPU 2 (~20.4 GB Free) -> 3 Jobs (16.8 GB req)
(
    echo "[GPU 2] Starting Unit II: Feature-Fusion Impact"
    python train.py --epochs $EPOCHS --gpu 2 --data_root $DATA_ROOT --phase 1 --exp_name unit2_phase1 &
    sleep 5
    python train.py --epochs $EPOCHS --gpu 2 --data_root $DATA_ROOT --phase 2 --exp_name unit2_phase2 &
    sleep 5
    python train.py --epochs $EPOCHS --gpu 2 --data_root $DATA_ROOT --phase 3 --exp_name unit2_phase3 &
    wait
    echo "[GPU 2] Unit II Finished!"
) &

# GPU 6 (~19.2 GB Free) -> 3 Jobs (16.8 GB req)
(
    echo "[GPU 6] Starting Unit III: Layer Freezing Strat."
    python train.py --epochs $EPOCHS --gpu 6 --data_root $DATA_ROOT --phase 3 --freeze_strategy none --exp_name unit3_freeze_none &
    sleep 5
    python train.py --epochs $EPOCHS --gpu 6 --data_root $DATA_ROOT --phase 3 --freeze_strategy head_only --exp_name unit3_head_only &
    sleep 5
    python train.py --epochs $EPOCHS --gpu 6 --data_root $DATA_ROOT --phase 3 --freeze_strategy freeze_early --exp_name unit3_freeze_early &
    wait
    echo "[GPU 6] Unit III Finished!"
) &

# GPU 1 (~11.6 GB Free) -> 1 Job (5.6 GB req)
(
    echo "[GPU 1] Starting Unit III: Late Freezing"
    python train.py --epochs $EPOCHS --gpu 1 --data_root $DATA_ROOT --phase 3 --freeze_strategy freeze_late --exp_name unit3_freeze_late &
    wait
    echo "[GPU 1] Freezing Late Finished!"
) &

# Wait for all background GPU queues to finish
wait

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

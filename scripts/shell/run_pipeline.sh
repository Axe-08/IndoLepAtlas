#!/bin/bash
# run_pipeline.sh - Centralized script for IndoLepAtlas pipeline on DGX

# 1. Sync latest changes
echo ">>> Syncing with Git..."
git pull origin main

# 2. Setup Environment
echo ">>> Setting up environment..."
source ~/miniconda3/bin/activate ./conda_env
export GDINO_CONFIG="GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
export GDINO_WEIGHTS="groundingdino_swint_ogc.pth"

# 3. Choose GPU (Default 0 if not specified)
GPU_ID=${1:-0}
echo ">>> Using GPU ID: $GPU_ID"

# 4. Start Pipeline in Background
echo ">>> Launching pipeline processes in background..."

# Step 1: Image Processing (CPU)
echo "    - Starting Image Processing (TRIM)..."
nohup python3 process_images.py --dataset all --workers 8 > trim.log 2>&1 &

# Step 2: Metadata Extraction (OCR)
echo "    - Starting Metadata Enrichment..."
nohup python3 enrich_metadata.py --dataset all > metadata.log 2>&1 &

# Step 3: Annotation (GPU)
echo "    - Starting GroundingDINO Annotation..."
CUDA_VISIBLE_DEVICES=$GPU_ID nohup python3 generate_annotations.py --dataset all > annotate.log 2>&1 &

echo ">>> All processes launched."
echo ">>> To view process logs: tail -f annotate.log"
echo ">>> Launching progress monitor in 5 seconds..."
sleep 5

# 5. Launch Monitor in Screen (or just run directly)
# We will run the monitor directly so the user sees it immediately.
python3 monitor_progress.py

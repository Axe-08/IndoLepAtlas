#!/bin/bash
# ============================================================
# sync_to_dgx.sh
# Syncs new/modified paper experiment files to DGX server.
# Run from LOCAL machine inside the indolep_model directory.
# ============================================================

DGX="dgx-direct"

# CORRECTED PATH BASED ON PWD OUTPUT
DGX_BASE="/home/23uec552/Butterfree/model"

echo "Syncing paper experiment files to DGX..."
echo "  Local:  $(pwd)"
echo "  Remote: $DGX:$DGX_BASE"
echo ""

# ── New files ─────────────────────────────────────────────────
echo "[1/6] models/baselines.py"
scp "models/baselines.py" "$DGX:$DGX_BASE/models/baselines.py"

echo "[2/6] losses.py"
scp "losses.py" "$DGX:$DGX_BASE/losses.py"

echo "[3/6] run_paper_experiments.sh"
scp "run_paper_experiments.sh" "$DGX:$DGX_BASE/run_paper_experiments.sh"

echo "[4/6] dry_run_test.sh"
scp "dry_run_test.sh" "$DGX:$DGX_BASE/dry_run_test.sh"

# ── Modified files ────────────────────────────────────────────
echo "[5/6] train.py"
scp "train.py" "$DGX:$DGX_BASE/train.py"

echo "[6/6] evaluate.py"
scp "evaluate.py" "$DGX:$DGX_BASE/evaluate.py"

# ── Make scripts executable on DGX ────────────────────────────
echo ""
echo "Setting execute permissions on DGX..."
ssh "$DGX" "chmod +x $DGX_BASE/run_paper_experiments.sh $DGX_BASE/dry_run_test.sh"

echo ""
echo "✓ Sync complete! Now SSH into DGX and run:"
echo "  cd $DGX_BASE"
echo "  bash dry_run_test.sh"

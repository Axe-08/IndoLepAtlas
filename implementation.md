# DLCV Phase-2 Implementation Plan (IndoLepAtlas / Butterfree)

This document defines what data (plots/metrics) to generate, how to generate it, and a step-by-step execution plan for the three agreed research questions:

1. Unit I (Custom): Long-Tail Loss Dynamics
2. Unit II (Custom): Feature-Fusion Impact (CA + MLFI)
3. Unit III (PDF #14): Layer Freezing Strategy

All experiments use the existing dataset in:
/home/23uec552/Butterfree/indolepatlas_data/data/butterflies

---

## 0. Environment and Paths (DGX)

```bash
ssh dgx-direct
cd /home/23uec552/Butterfree/model
conda activate indolep_venv
```

Dataset root: `/home/23uec552/Butterfree/indolepatlas_data/data/butterflies`  
Metadata: `metadata_filtered.csv`  
Splits: train/val/test already present

---

## 1. Data to Generate (Plots and Numbers)

### A) Core Metrics (all experiments)
Numbers:
- Top-1 accuracy
- Macro-F1
- Macro precision
- Dense vs Sparse stratum accuracy/F1 (threshold=50)
- Overfitting gap: train_acc minus val_acc

Plots:
- Loss curves (train vs val)
- Metrics curves (acc, macro-F1)

### B) Unit-Specific Additions

Unit I (Loss Dynamics):
- Class-frequency vs accuracy plot
- Table: CE vs Focal x Balanced vs Unbalanced

Unit II (Feature-Fusion):
- Ablation table: Phase 1 vs Phase 2 vs Phase 3
- Confusion shift: top-20 most confused species pairs (before vs after MLFI)

Unit III (Layer Freezing):
- Accuracy + overfit gap vs freezing strategy
- Recommendation table

---

## 2. Required Code Updates (Minimal, High-Value)

### 2.1 Fix Boolean CLI Parsing in train.py
Currently `--balanced_sampling` uses `type=bool`, which mis-parses "False" as True.  
Update file: `train.py`

Add a `str2bool()` helper and use it for:
- `--balanced_sampling`
- `--pretrained` (optional)

### 2.2 Add Freezing Strategy Support
Update file: `train.py`

Add CLI arg: `--freeze_strategy` with values:
- `none`
- `head_only`
- `freeze_early` (stem + stage0 + stage1)
- `freeze_late` (stage2 + stage3)

Then implement:
```python
def apply_freeze_strategy(model, strategy):
    # Freeze by name containing "stem" or "stages.0" etc
```

### 2.3 Ensure Evaluation Outputs are Saved
Update file: `evaluate.py`

Add outputs:
```
eval_results/
  summary.json
  per_stratum.json
  confusion_pairs.csv
  confusion_heatmap.png
```

---

## 3. Experiment Matrix (What to Run)

### Unit I - Long-Tail Loss Dynamics
Run 4 configs (loss x balanced):
1. CE + Balanced
2. CE + Unbalanced
3. Focal + Balanced
4. Focal + Unbalanced

Command template:
```bash
python train.py \
  --data_root /home/23uec552/Butterfree/indolepatlas_data/data/butterflies \
  --phase 3 \
  --loss ce \
  --balanced_sampling true \
  --exp_name unit1_ce_bal
```
Repeat with `loss=focal` and `balanced_sampling=false`.

### Unit II - Feature-Fusion Impact (Ablation)
Runs:
1. Phase 1 (Baseline)
2. Phase 2 (+CA)
3. Phase 3 (+CA+MLFI)

Commands:
```bash
python train.py --phase 1 --exp_name unit2_phase1
python train.py --phase 2 --exp_name unit2_phase2
python train.py --phase 3 --exp_name unit2_phase3
```

### Unit III - Layer Freezing Strategy
Runs:
1. none
2. head_only
3. freeze_early
4. freeze_late

Commands:
```bash
python train.py --phase 3 --freeze_strategy none --exp_name unit3_freeze_none
python train.py --phase 3 --freeze_strategy head_only --exp_name unit3_head_only
python train.py --phase 3 --freeze_strategy freeze_early --exp_name unit3_freeze_early
python train.py --phase 3 --freeze_strategy freeze_late --exp_name unit3_freeze_late
```

---

## 4. Data Extraction and Plot Generation

### 4.1 Evaluation for Each Run
Update file: `evaluate.py` to save outputs.

For each run:
```bash
python evaluate.py \
  --data_root /home/23uec552/Butterfree/indolepatlas_data/data/butterflies \
  --checkpoint runs/<RUN>/best_model.pth \
  --output_dir runs/<RUN>/eval_results \
  --sparse_threshold 50
```

### 4.2 Aggregation Script (Recommended)
Add file: `analysis/collect_results.py`

This script will:
1. Read all `runs/*/eval_results/summary.json`
2. Produce `results/summary.csv`
3. Generate plots:
   - loss/accuracy curves
   - class-freq vs accuracy
   - freezing strategy comparison

---

## 5. Result Formatting (for Report)

Tables:
1. Unit I Loss Dynamics: 4-row comparison table
2. Unit II Feature Fusion: 3-row ablation table
3. Unit III Freezing: 4-row strategy comparison

Figures:
1. Train/Val loss curve (best run)
2. Confusion shift (Phase 1 vs Phase 3)
3. Class frequency vs accuracy
4. Overfitting gap vs freezing strategy

---

## 6. Report Assembly (4-6 pages)

Section format for each Unit:
1. Question
2. Experimental design
3. Results (tables + plots)
4. Mechanistic interpretation

---

## 7. Execution Checklist

1. Fix train.py bool parsing + freeze strategy
2. Update evaluate.py to save JSON/CSV/plots
3. Run Unit I matrix (4 runs)
4. Run Unit II phases (3 runs)
5. Run Unit III freezing (4 runs)
6. Run aggregation script
7. Draft report + slides

---

## 8. Expected Outputs Folder Layout

```
runs/
  unit1_ce_bal/
    metrics.csv
    eval_results/
      summary.json
      per_stratum.json
      confusion_pairs.csv
      confusion_heatmap.png
  unit2_phase3/
    ...
results/
  summary.csv
  plots/
    class_freq_vs_acc.png
    freeze_gap.png
```

---

## 9. Notes and Risks

- Balanced sampling flag must parse correctly.
- Confusion analysis requires saving confusion pairs into a CSV.
- Freezing must not break optimizer (only include params with requires_grad).


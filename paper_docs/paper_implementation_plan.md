<!--# 48-Hour Blitz Plan: IndoLepAtlas Paper Additions

**Deadline:** ~48 Hours | **Resources:** 2 People + DGX Access (Multiple GPUs)

---

## ⚡ Execution Strategy: Parallel Experiments (Hours 0–24)
Both people launch GPU jobs **immediately**. All training must be completed and results collected by the 24-hour mark to leave the final 24 hours for writing and assembly.

| Timeframe | Person A (Baselines & MLFI) | Person B (Geotemporal & Data) |
| :--- | :--- | :--- |
| **Hours 0–2 (Now)** | Write A1.1 (ResNet101) & A1.2 (ViT) scripts. Implement A2.1 (MLFI Warmup). | Smoke test Phase 5 (B1.1). Launch B1.2 (Phase 5 Training). |
| **Hours 2–12 (Night 1)** | **Launch A1.3 (ResNet) & A1.4 (ViT)** on separate GPUs. Run overnight. | **Training: Phase 5 (B1.2)** continues. Start B2.1 (iNat Analysis) in background. |
| **Hours 12–18 (Day 1)** | Evaluate A1.5. **Launch A2.2 (MLFI Warmup)**. | Evaluate B1.3. Run B1.4/B1.5 (Zone/Month Analysis). Finish B2.3 (Dataset Table). |
| **Hours 18–24 (Night 2)** | Evaluate A2.3. *Optional: Launch EfficientNet-B5 if time/GPU permits.* | Finalize all B-unit documentation. Prep tables for LaTeX. |

---

## ✍️ Writing & Assembly (Hours 24–48)
Start writing **only** after results are locked.

| Section | Owner | Key Deliverables |
| :--- | :---: | :--- |
| **Dataset & SOTA Comparison** | Person B | iNat coverage stats + Comparison Table. |
| **Architecture & Baselines** | Person A | ResNet vs ViT vs Phase 2 (CA) Table + Bar Chart. |
| **Geotemporal Results** | Person B | Phase 5 interpretation + Zone/Month Heatmaps. |
| **MLFI & Layer Analysis** | Person A | Updated Unit II/III narrative with Warmup results. |
| **Final Assembly** | Joint | Abstract, Conclusion, BibTeX, Presentation update. |

---

## 📋 Task Tracker

### Person A Tasks
- [ ] A1.1: ResNet-101 Training Script ⬜
- [ ] A1.2: ViT-B/16 Training Script ⬜
- [ ] A1.3: Train ResNet-101 (40 Epochs) ⬜
- [ ] A1.4: Train ViT-B/16 (40 Epochs) ⬜
- [ ] A2.1: Implement MLFI Warmup in `train.py` ⬜
- [ ] A2.2: Train Phase 3 + Warmup ⬜
- [ ] A1.6/A2.4: Summarize all results in `results_A.md` ⬜

### Person B Tasks
- [ ] B1.1: Smoke test Phase 5 ⬜
- [ ] B1.2: Train Phase 5 (40 Epochs) ⬜
- [ ] B2.1: iNaturalist/GBIF India species coverage script ⬜
- [ ] B2.3: Cross-dataset comparison table ⬜
- [ ] B1.4/B1.5: Zone/Month performance analysis ⬜
- [ ] B1.6/B2.4: Summarize all results in `results_B.md` ⬜

---

## 🚨 Critical Dependencies
1. **GPU Availability:** Assumes Person A can get 2 GPUs for parallel ResNet/ViT runs.
2. **iNat API/Data:** Assumes iNaturalist counts can be fetched quickly (or via GBIF metadata).
3. **Phase 5 Stability:** Assumes the existing `geotemporal.py` is bug-free for a full 40-epoch run.-->

# IndoLepAtlas Paper: Two-Person Parallel Execution Plan

## Team Roles

| | **Akshit (Person A)** | **Kriti (Person B)** |
|---|---|---|
| **Focus** | Code changes, experiment setup, debugging, GPU orchestration | Paper writing, dataset analysis, figures, literature |
| **Tools** | Codebase + AI assistant + DGX terminal | Paper docs + dataset CSVs + visualization |
| **Key rule** | Prepares ALL scripts/code before experiments launch | Runs experiments using provided commands, writes paper sections |

> [!IMPORTANT]
> **GPU Strategy:** With DGX, aim to run as many experiments simultaneously as possible. All experiments below are single-GPU jobs (~3-6 hours each at 40 epochs). If you have 4+ GPUs free, the entire experimental phase completes in **one overnight batch.**

---

## Timeline Overview

```
DAY 1 (Today/Tomorrow)
├── A: Code prep (baseline models, experiment scripts, MLFI fix)
├── B: Dataset analysis + Related Work draft
└── EVENING: Launch ALL experiments overnight (4-6 GPUs)

DAY 2
├── A: Collect results, debug any failed runs, generate result tables
├── B: Continue paper sections (Dataset, Methodology)
└── EVENING: Relaunch any failed/additional experiments

DAY 3
├── A+B CONVERGE: Assemble paper
├── A: Results section, figures, final report.tex edits
├── B: Intro, Discussion, Conclusion rewrites
└── Paper draft complete
```

---

## DAY 1 — Parallel Setup

### Person A (Akshit): Code Preparation

All code changes happen BEFORE any experiments launch. Prepare everything, test locally with 1-epoch dry runs, then hand commands to Person B or launch the batch yourself.

#### Task A1: Add Baseline Model Support `[~45 min]`

Create a new file or extend `backbone.py` to support vanilla baselines:

**File:** `indolep_model/models/baselines.py` (NEW)

```python
import timm
import torch.nn as nn

def build_baseline_model(arch: str, num_classes: int, pretrained: bool = True):
    """
    Standard off-the-shelf models for comparison table.
    Uses same timm library, same input resolution.
    """
    if arch == 'resnet101':
        model = timm.create_model('resnet101', pretrained=pretrained, num_classes=num_classes)
    elif arch == 'vit_base_patch16':
        model = timm.create_model('vit_base_patch16_224', pretrained=pretrained, num_classes=num_classes)
    elif arch == 'efficientnet_b5':
        model = timm.create_model('efficientnet_b5', pretrained=pretrained, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown baseline: {arch}")
    return model
```

**Modify `train.py`** to accept a `--baseline` flag:
- When `--baseline resnet101` is passed, use `build_baseline_model()` instead of `build_model()`
- Keep everything else identical: same dataloader, same optimizer, same balanced sampling, same metrics
- The differential LR still applies: backbone layers at 0.1x, classifier head at 1x

**Test:** `python train.py --baseline resnet101 --epochs 1 --batch_size 4` (dry run)

---

#### Task A2: Prepare Geotemporal Experiment Scripts `[~30 min]`

Phase 5 code already exists. Create run scripts:

**File:** `indolep_model/run_geo_experiments.sh` (NEW)

```bash
#!/bin/bash
# Experiment: Geotemporal Fusion (Phase 5)
# GPU 0: Normal geotemporal
CUDA_VISIBLE_DEVICES=0 python train.py \
    --data_root /path/to/data \
    --phase 5 --loss ce --balanced \
    --epochs 40 --batch_size 32 --lr 1e-4 \
    --exp_name geo_phase5_normal \
    --pretrained True &

# GPU 1: Shuffled-geo control (needs --shuffle_geo flag, see Task A3)
CUDA_VISIBLE_DEVICES=1 python train.py \
    --data_root /path/to/data \
    --phase 5 --loss ce --balanced \
    --epochs 40 --batch_size 32 --lr 1e-4 \
    --exp_name geo_phase5_shuffled \
    --shuffle_geo \
    --pretrained True &

wait
echo "Geotemporal experiments complete"
```

---

#### Task A3: Add Shuffled-Geo Control Flag `[~20 min]`

Add `--shuffle_geo` argument to `train.py`:
- At **test/validation time only**, randomly permute `zone_idx` and `month_enc` within each batch
- Training uses real geo features (model learns normally)
- At eval, shuffling destroys the geo signal — if accuracy drops, the model relied on real correlations

This is a ~10-line change in the evaluation loop of `train.py`.

---

#### Task A4: MLFI Warmup Fix `[~30 min]` *(Optional but recommended)*

Modify `train.py` optimizer setup:

```python
# Current: MLFI branches get full lr from epoch 0
# Fix: MLFI branches start at backbone_lr (0.1x), ramp to full lr after warmup

# Create separate param groups:
# Group 1: backbone params → 0.1 * lr (existing)
# Group 2: CA params → lr (existing)  
# Group 3: MLFI params → start at 0.1 * lr, ramp to lr after epoch 10
# Group 4: classifier head → lr (existing)
```

Add a scheduler callback that increases MLFI lr after warmup period.

---

#### Task A5: Create Master Launch Script `[~20 min]`

**File:** `indolep_model/run_paper_experiments.sh` (NEW)

```bash
#!/bin/bash
# MASTER SCRIPT: Launch all paper experiments in parallel
# Assumes 6 GPUs available on DGX

DATA_ROOT="/path/to/indolepatlas/data"
COMMON="--loss ce --balanced --epochs 40 --batch_size 32 --lr 1e-4 --pretrained True"

echo "=== Launching all experiments ==="

# GPU 0: ResNet-101 baseline
CUDA_VISIBLE_DEVICES=0 python train.py --data_root $DATA_ROOT \
    --baseline resnet101 $COMMON --exp_name baseline_resnet101 &

# GPU 1: ViT-B/16 baseline  
CUDA_VISIBLE_DEVICES=1 python train.py --data_root $DATA_ROOT \
    --baseline vit_base_patch16 $COMMON --exp_name baseline_vit_b16 &

# GPU 2: Geotemporal fusion (Phase 5)
CUDA_VISIBLE_DEVICES=2 python train.py --data_root $DATA_ROOT \
    --phase 5 $COMMON --exp_name geo_phase5 &

# GPU 3: Geotemporal shuffled control
CUDA_VISIBLE_DEVICES=3 python train.py --data_root $DATA_ROOT \
    --phase 5 $COMMON --shuffle_geo --exp_name geo_phase5_shuffled &

# GPU 4: MLFI with warmup fix (Phase 3)
CUDA_VISIBLE_DEVICES=4 python train.py --data_root $DATA_ROOT \
    --phase 3 $COMMON --mlfi_warmup 10 --exp_name mlfi_warmup_fix &

# GPU 5: EfficientNet-B5 baseline (optional)
CUDA_VISIBLE_DEVICES=5 python train.py --data_root $DATA_ROOT \
    --baseline efficientnet_b5 $COMMON --exp_name baseline_effnet_b5 &

wait
echo "=== All experiments complete ==="
```

#### Task A6: Dry Run Validation `[~30 min]`

Before the overnight batch:
- Run each experiment for 1 epoch with `--epochs 1` to catch import errors, shape mismatches, dataloader issues
- Verify all 6 experiments start correctly and log to their respective `runs/` folders
- Check that geotemporal features flow correctly in Phase 5

---

### Person B (Kriti): Analysis + Paper Prep

While Akshit prepares code, Person B works on content that requires **zero code changes.**

#### Task B1: iNaturalist Coverage Gap Analysis `[~2 hours]`

This is critical for the Dataset section and answers the reviewer question *"Why not use existing datasets?"*

**Method:**
1. Download the iNaturalist India Lepidoptera species list (from GBIF or iNaturalist export)
2. Cross-reference against your 966 species list
3. For each of your species, count how many research-grade iNaturalist India observations exist
4. **Key statistic to compute:** *"X% of IndoLepAtlas species have fewer than 100 India-specific iNaturalist records"*

**Expected finding:** Many India-endemic species and subspecies will be severely under-represented in global datasets. This number will be striking and directly justifies the dataset.

**Output:** A comparison table + paragraph for the paper.

---

#### Task B2: Related Work Section Draft `[~2 hours]`

Write a Related Work section covering three threads:

1. **Fine-grained visual classification (FGVC):**
   - CUB-200-2011, Stanford Dogs, iNaturalist challenge
   - FGBNet (Yuan et al. 2025), thrips paper (Amarathunga et al. 2022)
   - Evolution: CNN → attention → multi-level fusion → ViT

2. **Insect/butterfly classification systems:**
   - SLR (Amarathunga et al. 2021) — cite the open problems they identify
   - Alfatemi et al. (2024) bird paper — manual curation for similar species
   - Note: no existing work incorporates geotemporal context

3. **Geotemporal features in ecological ML:**
   - Species distribution modeling (MaxEnt, etc.)
   - iNaturalist's geo-aware species suggestions
   - Key gap: nobody has fused this into a classification pipeline end-to-end

**Output:** 1-2 page LaTeX section ready to paste into `report.tex`.

---

#### Task B3: Expanded Dataset Section Draft `[~1.5 hours]`

Expand the current 1-paragraph dataset description into a proper dataset contribution section:

Using data from `distribution_stats.md`, write about:
1. **Collection protocol:** Sources (Indian Lepidoptera atlas, iNaturalist), filtering criteria, adult-only selection
2. **Taxonomic coverage:** 966 species, 6 families (Nymphalidae 7168, Lycaenidae 6769, etc.), relation to known Indian butterfly diversity (~1500 species)
3. **Geographic coverage:** 34 states, 9 biogeographic zones, state distribution table
4. **Temporal coverage:** 12-month span, monsoon peak pattern, ecological significance
5. **Class distribution analysis:** Dense (≥50 images, ~500 classes) vs Sparse (<50 images, ~400 classes), long-tail statistics
6. **Metadata completeness:** 98.3% location, 76.1% date, 85.8% sex missing (acknowledged limitation)
7. **Quality assurance:** Filtering pipeline, early-stage removal (3,684 removed)

**Output:** 2-3 page LaTeX section with a dataset statistics table.

---

#### Task B4: Prepare Figures for Dataset Section `[~1 hour]`

Create/collect these figures (can use the audit plots from `data_root/audit/`):
1. **Geographic distribution map** of India showing observation density by state/zone
2. **Species frequency distribution** (log-scale bar chart — from audit)
3. **Monthly observation distribution** bar chart showing monsoon peak
4. **Example images** grid: selected species pairs showing inter-species similarity and intra-species variation

---

## DAY 1 EVENING — Launch Experiments

### Handoff Point: A → B (or A launches directly)

Once Akshit's code prep is done and dry-runs pass:

```bash
# SSH into DGX
ssh dgx-server

# Navigate to project
cd /path/to/IndoLepAtlas/indolep_model

# Launch master script (tmux/screen recommended)
tmux new -s experiments
bash run_paper_experiments.sh

# Monitor
watch -n 60 'for d in runs/baseline_* runs/geo_* runs/mlfi_*; do
    echo "=== $d ===";
    tail -1 "$d/metrics.csv" 2>/dev/null;
done'
```

**Expected runtime:** ~4-6 hours per experiment at 40 epochs. Overnight batch = all done by morning.

---

## DAY 2 — Results Collection + Paper Assembly

### Person A (Akshit): Results Processing

#### Task A7: Collect and Tabulate Results `[~1 hour]`

After experiments finish:
1. Run `evaluate.py` on each experiment's best checkpoint
2. Extract: Top-1, Top-5, Macro-F1, Macro-Precision, Dense acc, Sparse acc
3. Build the master comparison table

**Target tables for the paper:**

**Table 1: External Baseline Comparison**

| Model | Params | Top-1 | Top-5 | Macro-F1 | Dense | Sparse |
|---|---|---|---|---|---|---|
| ResNet-101 | 44.5M | ? | ? | ? | ? | ? |
| ViT-B/16 | 86.6M | ? | ? | ? | ? | ? |
| EfficientNet-B5 | 30.4M | ? | ? | ? | ? | ? |
| **Ours (ConvNeXt+CA)** | ~29M | **86.77** | ? | **85.75** | 86.13 | **88.05** |

**Table 2: Geotemporal Ablation**

| Configuration | Top-1 | Macro-F1 | Dense | Sparse | Δ Sparse |
|---|---|---|---|---|---|
| Vision-only (Phase 2) | 86.77 | 85.75 | 86.13 | 88.05 | — |
| + Geotemporal (Phase 5) | ? | ? | ? | ? | ? |
| + Geo-shuffled (control) | ? | ? | ? | ? | ? |

---

#### Task A8: Generate Result Figures `[~1.5 hours]`

Create publication-quality plots:
1. **Bar chart:** All models compared on Top-1 + Macro-F1
2. **Training curves:** Phase 5 vs Phase 2 convergence comparison
3. **Geotemporal ablation bar chart:** Vision-only vs +Geo vs +Geo-shuffled
4. **(If MLFI fix works):** Before/after MLFI bar chart

Can reuse/adapt `generate_report_assets.py` for these.

---

#### Task A9: Debug & Relaunch Failed Runs `[as needed]`

Check each run's `progress.log` for:
- OOM errors (reduce batch size for ViT-B/16 — it's bigger)
- NaN losses
- Dataloader errors

Relaunch on free GPUs as needed.

---

### Person B (Kriti): Paper Sections

#### Task B5: Write Methodology Section `[~2 hours]`

Restructure the existing Units I-III content into a proper Methods section:
1. **Architecture overview:** ConvNeXt-Tiny backbone → CA → MLFI pipeline (keep existing content, clean up)
2. **Geotemporal encoding:** NEW subsection
   - Biogeographic zone encoding (9 zones, learned 32-dim embedding)
   - Cyclic month encoding (sin/cos)
   - Late fusion strategy and rationale
3. **Training protocol:** Loss function, balanced sampling, optimizer, scheduler, checkpointing
4. **Evaluation protocol:** Metrics (Top-1, Top-5, Macro-F1), Dense/Sparse stratification

#### Task B6: Draft Introduction Rewrite `[~1.5 hours]`

Reframe from course project to research contribution:
1. **Opening:** India as butterfly biodiversity hotspot, ~1500 species, conservation imperative
2. **Gap:** Existing datasets (iNaturalist, GBIF) are global, dilute India-specific subspecies. No dataset has geotemporal metadata integrated into classification.
3. **Our contributions** (numbered list):
   - IndoLepAtlas: 966-species India-specific dataset with geotemporal metadata
   - Systematic baseline establishment through architectural ablation
   - First geotemporal fusion for butterfly classification
   - Domain-shift analysis showing ImageNet feature misalignment

#### Task B7: Draft Discussion + Conclusion `[~1 hour]`

- Integrate the three narrative arcs (dataset, architecture, geotemporal)
- Discuss the domain-shift finding and its implications beyond butterflies
- Honest limitations (metadata gaps, 15-epoch constraint for some runs, species with <5 images)
- Future work: prototypical heads, self-supervised pretraining, hierarchical classification

---

## DAY 3 — Convergence

### Both: Paper Assembly `[~4-6 hours together]`

#### Task AB1: Merge All Sections into report.tex

Person A provides:
- Results tables (LaTeX formatted)
- Result figures (.png/.pdf)
- Updated architecture diagram if needed

Person B provides:
- Introduction, Related Work, Dataset, Methodology, Discussion, Conclusion (LaTeX text)
- Dataset figures

Merge into single `report.tex` with this structure:

```
1. Introduction                    [B wrote, A reviews]
2. Related Work                    [B wrote]
3. IndoLepAtlas Dataset            [B wrote, uses A's stats]
4. Methodology                    [B restructured from existing + new geo section]
   4.1 Architecture
   4.2 Geotemporal Encoding
   4.3 Training Protocol
5. Experiments                     [A wrote results, B helps narrative]
   5.1 External Baselines
   5.2 Loss & Sampling Ablation (existing Unit I)
   5.3 Feature Fusion Ablation (existing Unit II)
   5.4 Transfer Learning Analysis (existing Unit III)
   5.5 Geotemporal Fusion
6. Analysis & Discussion           [Joint]
7. Conclusion                      [B wrote, A reviews]
References
```

#### Task AB2: Final Review Checklist

- [ ] All tables have consistent formatting and correct numbers
- [ ] All figures are referenced in text
- [ ] Bibliography complete (add Dosovitskiy ViT, iNaturalist dataset paper, etc.)
- [ ] Abstract rewritten to match new framing
- [ ] Dense/Sparse reporting consistent across all new experiments
- [ ] Geotemporal shuffled-control result interpreted correctly
- [ ] No leftover "Unit I/II/III" terminology (or deliberately kept as organizational labels)

---

## Complete Experiment Matrix

| # | Experiment | GPU | Person | Status | Priority |
|---|---|---|---|---|---|
| E1 | ResNet-101 baseline | 0 | A sets up, auto | ⬜ | CRITICAL |
| E2 | ViT-B/16 baseline | 1 | A sets up, auto | ⬜ | CRITICAL |
| E3 | Geotemporal Phase 5 | 2 | A sets up, auto | ⬜ | CRITICAL |
| E4 | Geo-shuffled control | 3 | A sets up, auto | ⬜ | CRITICAL |
| E5 | MLFI warmup fix | 4 | A sets up, auto | ⬜ | RECOMMENDED |
| E6 | EfficientNet-B5 baseline | 5 | A sets up, auto | ⬜ | OPTIONAL |

All 6 run simultaneously overnight. **Total wall-clock time for experiments: ~6 hours.**

---

## Complete Task Assignment Summary

### Person A (Akshit) — Day 1

| Task | Duration | Dependency |
|---|---|---|
| A1: Baseline model support | 45 min | None |
| A2: Geo experiment scripts | 30 min | None |
| A3: Shuffled-geo flag | 20 min | None |
| A4: MLFI warmup fix | 30 min | None |
| A5: Master launch script | 20 min | A1-A4 |
| A6: Dry run validation | 30 min | A5 |
| **Launch experiments** | 5 min | A6 |
| **Total Day 1:** | **~3 hours** | |

### Person A (Akshit) — Day 2

| Task | Duration | Dependency |
|---|---|---|
| A7: Collect results + tables | 1 hour | Experiments done |
| A8: Generate result figures | 1.5 hours | A7 |
| A9: Debug/relaunch failures | variable | A7 |
| **Total Day 2:** | **~3-4 hours** | |

### Person B (Kriti) — Day 1

| Task | Duration | Dependency |
|---|---|---|
| B1: iNaturalist coverage analysis | 2 hours | None |
| B2: Related Work draft | 2 hours | None |
| B3: Dataset section draft | 1.5 hours | B1 |
| B4: Dataset figures | 1 hour | B3 |
| **Total Day 1:** | **~6 hours** | |

### Person B (Kriti) — Day 2

| Task | Duration | Dependency |
|---|---|---|
| B5: Methodology section | 2 hours | None |
| B6: Introduction rewrite | 1.5 hours | B1, B2 |
| B7: Discussion + Conclusion | 1 hour | None (update after results) |
| **Total Day 2:** | **~4.5 hours** | |

### Day 3 — Joint

| Task | Duration | Who |
|---|---|---|
| AB1: Merge into report.tex | 3-4 hours | Both |
| AB2: Final review | 1-2 hours | Both |
| **Total Day 3:** | **~5-6 hours** | |

---

> [!WARNING]
> **Biggest risk:** ViT-B/16 may OOM on DGX V100 with batch_size=32. Person A should test with `--batch_size 16` for ViT during dry runs. ResNet-101 and EfficientNet-B5 should be fine at 32.

> [!TIP]
> **Quick win if short on time:** Skip E5 (MLFI fix) and E6 (EfficientNet). The minimum viable set is E1+E2+E3+E4 (4 experiments, 4 GPUs, one night). That gives you baselines + geotemporal + control = everything needed for a publishable paper.

# Model Outline: End-to-End Butterfly Classification Pipeline

This document explains what the current codebase does from raw metadata to final model evaluation, based on:

- `run_all.sh`
- `data_audit.py`
- `dataset.py`
- `train.py`
- `models/` (`backbone.py`, `coord_attention.py`, `mlfi.py`, `geotemporal.py`)

---

## 1) Big-Picture Execution Order

The intended pipeline order is:

1. **Data audit + filtering** (`data_audit.py`)  
   Cleans `metadata.csv`, keeps butterfly rows, removes early-stage images, creates analysis plots, and writes `metadata_filtered.csv`.
2. **Training** (`train.py`)  
   Builds dataloaders from filtered metadata, constructs model by phase, trains with focal/CE loss, logs metrics, and saves best checkpoint.
3. **Evaluation** (`evaluate.py`, launched by `run_all.sh`)  
   Loads best checkpoint from latest run and computes Top-1/Top-5/F1/confusions.

`run_all.sh` orchestrates this sequence automatically.

---

## 2) `run_all.sh`: The Orchestrator

`run_all.sh` does three major steps:

1. **Audit step**
   - Checks if `metadata_filtered.csv` exists.
   - If missing, runs:
     - `python data_audit.py --data_root "$DATA_ROOT"`

2. **Training step**
   - Runs `train.py` with:
     - `--data_root`
     - `--phase` (default set to `3` in script)
     - `--loss focal`
     - `--epochs`, `--batch_size`, `--lr`, `--pretrained True`

3. **Evaluation step**
   - Finds latest run folder in `runs/`
   - Uses `best_model.pth` from that run
   - Runs `evaluate.py` and writes results inside the same run directory.

So this shell script is the **single-command full lifecycle pipeline**.

---

## 3) Data Audit and Preprocessing (`data_audit.py`)

This script is "Phase 0" dataset quality control and metadata preparation.

## 3.1 Metadata loading and fallback search

- Starts from `data_root/metadata.csv`
- If missing, tries common alternatives:
  - `images/metadata.csv`
  - `../metadata.csv`
  - `butterfly_metadata.csv`

## 3.2 Filtering rules

- Attempts to identify a type/category column (`type`, `category`, etc.).
- If found, keeps only rows containing butterfly-like strings.
- Removes early-stage samples using:
  - life-stage columns (`life_stage`, `stage`, etc.), and/or
  - filename regex patterns for `earlystage` variants.

Net effect: metadata becomes **adult-butterfly-focused**.

## 3.3 Audit analyses generated

It prints statistics and saves plots under `data_root/audit/`:

- **Species distribution**
  - counts, sparse-threshold tables (`<10`, `<20`, `<30`, `<50`, `<100`)
  - bar + histogram
- **Family distribution** (if family column exists)
- **Geographic distribution**
  - maps state to biogeographic zone
  - reports unknown/unmapped states
- **Temporal distribution**
  - month extraction, invalid month handling, month histogram
- **Split health checks**
  - train/val/test counts
  - species missing from a split
- **Image file verification**
  - compares metadata rows with actual image files in `images/`

## 3.4 Output artifact

- Saves filtered training metadata to:
  - `data_root/metadata_filtered.csv`

This file is what `train.py` uses by default.

---

## 4) Dataset Construction and Data Flow (`dataset.py`)

`dataset.py` defines the runtime data pipeline used by training/evaluation.

## 4.1 Input assumptions

Expected dataset layout:

- `data_root/images/<species_folder>/<image files>`
- `data_root/metadata_filtered.csv` (or fallback to `metadata.csv`)

## 4.2 Column auto-detection

The dataset class auto-discovers schema variations:

- species column from candidates like `species`, `species_name`, etc.
- optional split/state/date/filepath columns if present.

This allows the code to tolerate slightly different metadata formats.

## 4.3 Split filtering + exclusion logic

For each split (`train`, `val`, `test`):

- filters rows by split column (if present),
- removes early-stage rows using filename regex,
- drops rows with missing/empty species labels.

## 4.4 Image path resolution strategy

For each metadata row, `_resolve_image_path()` tries multiple path-recovery routes:

1. absolute path from metadata,
2. relative path under `data_root`,
3. normalized path if metadata includes extra prefixes,
4. filename directly under `images/`,
5. filename under species-specific folder.

Invalid/unresolvable samples are removed.

## 4.5 Label mapping and unification across splits

Important detail:

- Each split initially builds its own species-to-index map.
- `create_dataloaders()` then creates a **unified mapping** across train+val+test species.
- All splits are remapped to this global class index space.

This avoids class-index mismatches across splits.

## 4.6 Image transforms

- **Train transforms**:
  - `RandomResizedCrop`
  - horizontal flip
  - random rotation
  - color jitter
  - ImageNet normalization
- **Val/Test transforms**:
  - deterministic resize + center crop
  - same normalization

So training is stochastic and regularized; validation/test are stable.

## 4.7 Geotemporal feature creation

If geotemporal is enabled:

- **State -> Zone index** using `STATE_TO_ZONE` (`NUM_ZONES = 9`, last is unknown),
- **Date -> month -> cyclic encoding**:
  - `[sin(2*pi*m/12), cos(2*pi*m/12)]`
  - invalid/missing months become `[0, 0]`.

Batch output contains:

- always: `image`, `label`
- optionally: `zone_idx`, `month_enc`

## 4.8 Class imbalance handling

`dataset.py` provides:

- inverse-frequency class weights (`get_class_weights`) for focal/CE weighting,
- weighted random sampler (`get_sampler`) for class-balanced sampling.

This is central for long-tail species distributions.

---

## 5) Model Architecture and Sequence (`models/`)

The model is a **progressive architecture** with phase-based ablations.

## 5.1 Base model in `models/backbone.py`

`ButterflyClassifier` uses:

- **Backbone**: `convnext_tiny` from timm with `features_only=True`
- extracts all 4 stage feature maps.

From there, optional modules are added depending on phase.

## 5.2 Phase definitions

`build_model()` phase config:

- **Phase 1**: ConvNeXt feature extraction + pooled final stage + classifier
- **Phase 2**: Phase 1 + Coordinate Attention (CA) after each stage
- **Phase 3**: Phase 2 + MLFI multi-stage fusion
- **Phase 5**: Phase 3 + geotemporal late fusion

(`train.py` allows phases `1,2,3,5`, default `3`.)

## 5.3 Coordinate Attention (`coord_attention.py`)

For each feature map:

- pools along width and height separately,
- learns axis-aware attention maps,
- multiplies attention back onto features.

Purpose: preserve **where** pattern evidence exists on butterfly wings, not just if it exists.

## 5.4 MLFI (`mlfi.py`)

MLFI uses 4 DIS branches (one per backbone stage):

- adaptive max pool each stage,
- FC projection per stage,
- concatenate projected vectors.

With proportions `[2,4,1,8]` and base dim `96`:

- stage outputs: `192, 384, 96, 768`
- concatenated visual feature dim = **1440**.

Purpose: keep low/mid-level fine-grained texture cues while retaining deep semantic cues.

## 5.5 Geotemporal fusion (`geotemporal.py`)

When enabled:

- zone index -> learned embedding (default 32-dim),
- month cyclic vector (2-dim, optionally projected),
- concatenate `[visual_features, zone_embedding, month_features]`,
- layer norm on fused vector.

With defaults in code:

- `1440 + 32 + 2 = 1474` feature dim into classifier head.

Purpose: disambiguate visually similar species via geography/season priors.

## 5.6 Final head

Classifier head is:

- `LayerNorm`
- `Dropout`
- `Linear -> num_classes logits`

No softmax in model; loss handles logits directly.

---

## 6) Training Pipeline (`train.py`)

## 6.1 Initialization

- Parses CLI args (data paths, phase, loss, optimization, workers, etc.).
- Auto-selects GPU with max free memory (unless `--gpu` explicitly set).
- Creates run folder in `runs/<exp_name>_<timestamp>`.
- Writes:
  - `config.json`
  - `metrics.csv`
  - `progress.log`
  - TensorBoard logs.

## 6.2 Dataloader + model + loss assembly

1. `create_dataloaders(...)`
2. `build_model(num_classes, phase, pretrained, dropout)`
3. loss:
   - focal (default) or cross-entropy via `build_loss`
   - focal can receive class weights from train dataset.

## 6.3 Optimization strategy

- Optimizer: `AdamW`
- Differential LR:
  - backbone params at `0.1 * lr`
  - new modules/head at `lr`
- Scheduler: cosine annealing with warmup epochs.
- Uses AMP (`autocast`) + gradient scaler.
- Clips grad norm to `5.0`.

## 6.4 Per-batch data flow

For each batch:

1. Fetch `image`, `label` (+optional `zone_idx`, `month_enc`)
2. Move tensors to device
3. Forward:
   - `model(images, zone_idx, month_enc)`
4. Loss on logits vs labels
5. Backprop with AMP scaler
6. Optimizer step, metrics accumulation.

## 6.5 Per-epoch logic

- Train pass + validation pass
- Compute:
  - loss, accuracy, macro-F1, macro-precision
- Append row to `metrics.csv`
- Regenerate metric plots (`metrics_curve.png`, `loss_curve.png`)
- Save:
  - best checkpoint by **validation macro-F1**
  - periodic checkpoints every 10 epochs.

So model selection criterion is explicitly long-tail-sensitive (macro-F1), not only accuracy.

---

## 7) End-to-End Data Processing Flow (Concise Graph)

Raw dataset
-> `metadata.csv` + `images/`
-> `data_audit.py` filtering/audit
-> `metadata_filtered.csv` (+ audit plots)
-> `dataset.py` split-wise sample build
-> path resolution + label mapping + transforms
-> DataLoaders (+ optional weighted sampler)
-> `train.py` forward/backward loop
-> best checkpoint (`best_model.pth`)
-> `evaluate.py` metrics + confusion analysis.

---

## 8) What the Model Is Ultimately Doing

At a system level, this model is designed for **fine-grained, long-tail butterfly species recognition** under class imbalance and subtle inter-class visual differences.

It does this by combining:

1. **Strong visual backbone** (ConvNeXt-Tiny),
2. **Spatially aware attention** (Coordinate Attention),
3. **Multi-level feature preservation** (MLFI, retaining both local pattern and global semantics),
4. **Long-tail training strategy** (class-balanced sampling + focal loss + macro-F1 checkpointing),
5. **Optional ecological priors** (geotemporal fusion in phase 5).

In short:  
the pipeline cleans metadata, builds robust split-aware datasets, extracts multi-scale wing-pattern features, optionally fuses location/season context, and optimizes for balanced species-level performance rather than dominance by common classes.

---

## 9) Key Files and Their Roles

- `run_all.sh`: full automation entrypoint (audit -> train -> evaluate)
- `data_audit.py`: metadata filtering + diagnostics + `metadata_filtered.csv` generation
- `dataset.py`: runtime dataset, transforms, geotemporal encoding, class balancing/sampler
- `train.py`: experiment lifecycle, optimization, logging, checkpointing
- `models/backbone.py`: phase-based model assembly
- `models/coord_attention.py`: positional/channel attention refinement
- `models/mlfi.py`: multi-stage feature fusion head
- `models/geotemporal.py`: optional late fusion of zone/month metadata


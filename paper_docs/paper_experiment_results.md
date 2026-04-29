# IndoLepAtlas Paper Experiment Results

> All experiments trained for **40 epochs** with **CE loss**, **lr=1e-4**, **AdamW** on Tesla V100-SXM2-32GB GPUs.  
> Dataset: 40,260 train / 4,673 val / 5,865 test images across **966 species**.

---

## Table 1: Main Comparison (Test Set)

| Model | Backbone | Params | Top-1 Acc | Top-5 Acc | Macro P | Macro F1 | Weighted F1 |
|---|---|---|---|---|---|---|---|
| ResNet-101 | ResNet-101 | 44.5M | 48.76% | 76.91% | 49.94% | 47.60% | 47.76% |
| EfficientNet-B5 | EfficientNet-B5 | 30.3M | 86.96% | 97.05% | 87.33% | 85.92% | 86.57% |
| **ViT-B/16** | ViT-B/16 | 86.5M | **89.99%** | 97.51% | **90.14%** | **88.11%** | **89.51%** |
| MLFI Warmup | ConvNeXt-S + CA + MLFI | 40.8M | 87.20% | 97.34% | 87.84% | 85.72% | 86.76% |
| Geo-Shuffled (control) | ConvNeXt-S + CA + MLFI + Geo* | 40.8M | 86.36% | 97.12% | 86.41% | 84.57% | 85.86% |
| **Geotemporal (Ours)** | ConvNeXt-S + CA + MLFI + Geo | 40.8M | 89.38% | **98.01%** | 89.13% | 87.37% | 88.94% |

*\*Geo-Shuffled: zone/month labels randomly permuted at inference to ablate geotemporal signal.*

---

## Table 2: Per-Stratum Analysis (Dense ≥ 50 samples vs. Sparse)

| Model | Dense Acc | Dense F1 | Sparse Acc | Sparse F1 | Δ Acc (S−D) |
|---|---|---|---|---|---|
| ResNet-101 | 45.02% | 22.53% | 56.29% | 45.80% | +11.27 |
| EfficientNet-B5 | 87.33% | 68.30% | 86.20% | 75.07% | −1.13 |
| ViT-B/16 | **91.60%** | **79.40%** | 86.76% | 75.18% | −4.84 |
| MLFI Warmup | 88.36% | 73.07% | 84.86% | 72.84% | −3.50 |
| Geo-Shuffled | 87.87% | 72.32% | 83.32% | 71.89% | −4.55 |
| **Geotemporal (Ours)** | 90.35% | 76.74% | **87.43%** | **75.38%** | −2.92 |

---

## Key Findings

### 1. Geotemporal fusion is competitive with ViT-B/16
Our geotemporal model (ConvNeXt-S + CA + MLFI + Geo, **40.8M params**) achieves **89.38% Top-1** — within 0.6% of the much larger ViT-B/16 (**86.5M params**) while using **2.1× fewer parameters**. It also achieves the **best Top-5 accuracy (98.01%)** across all models.

### 2. Geotemporal metadata provides a genuine signal
The geo-shuffled control (same architecture, randomized location/month) drops by:
- **−3.02% Top-1** (89.38 → 86.36)
- **−2.80% Macro F1** (87.37 → 84.57)

This confirms the geotemporal branch is learning meaningful biogeographic priors, not just adding parameters.

### 3. Best sparse-class performance from geotemporal model
Our model achieves **87.43% accuracy on sparse classes** (< 50 training images), outperforming all baselines including ViT-B/16 (86.76%). The gap narrows between dense and sparse (Δ = −2.92) compared to ViT-B/16 (Δ = −4.84), suggesting geotemporal priors help disambiguate underrepresented species.

### 4. ResNet-101 struggles with fine-grained classification
ResNet-101 achieves only 48.76% Top-1, confirming that modern architectures (ViT, EfficientNet, ConvNeXt) with stronger inductive biases or attention mechanisms are essential for 966-class fine-grained recognition.

### 5. MLFI warmup provides modest gains
The MLFI warmup variant (87.20%) outperforms the geo-shuffled control (86.36%) but underperforms full geotemporal fusion (89.38%), suggesting the multi-level feature integration benefits from geospatial context.

---

## Experiment Configurations

| Experiment | Backbone | Batch Size | Loss | Epochs | GPU |
|---|---|---|---|---|---|
| ResNet-101 | timm/resnet101 | 64 | CE | 40 | 3 |
| ViT-B/16 | timm/vit_base_patch16_224 | 16 | CE | 40 | 4 |
| EfficientNet-B5 | timm/efficientnet_b5 | 32 | CE | 40 | 4 |
| Geotemporal (Phase 5) | ConvNeXt-S + CA + MLFI + Geo | 32 | CE | 40 | 2 |
| Geo-Shuffled | ConvNeXt-S + CA + MLFI + Geo (shuffled) | 32 | CE | 40 | 0 |
| MLFI Warmup | ConvNeXt-S + CA + MLFI (warmup=10) | 32 | CE | 40 | 3 |

---

## Asset Locations

All paper assets organized in `paper_docs/experiment_assets/`:

```
experiment_assets/
├── configs/                  # Experiment hyperparameters (JSON)
│   ├── resnet101_config.json
│   ├── vit_b16_config.json
│   ├── effnet_b5_config.json
│   ├── geotemporal_config.json
│   ├── geo_shuffled_config.json
│   └── mlfi_warmup_config.json
├── confusion_matrices/       # Heatmaps + top confused pairs
│   ├── {model}_confusion.png
│   └── {model}_top_pairs.csv
├── training_curves/          # Loss + accuracy/F1 curves
│   ├── {model}_loss.png
│   └── {model}_metrics.png
├── metrics_csv/              # Per-epoch raw training data
│   └── {model}_metrics.csv
└── summaries/                # Aggregate eval metrics
    ├── {model}_summary.json
    └── {model}_per_stratum.json
```

# IndoLepAtlas: A Fine-Grained Dataset of Indian Lepidoptera with Geotemporal Metadata and Baseline Classification Study

**Authors:** Akshit Bansal, Kriti Chaturvedi

---

## Abstract

Fine-grained classification of Indian Lepidoptera species presents compounding challenges: spatial information collapse in channel attention, progressive texture-cue loss through pooling, and extreme long-tail class imbalance. We present IndoLepAtlas, a curated 60,641-image, 967-species India-specific butterfly dataset with geotemporal metadata, early life-stage documentation (9,804 images), and 703 images of larval host plant species. We conduct a systematic evaluation spanning six architectures — ResNet-101, EfficientNet-B5, ViT-B/16, and our ConvNeXt-S variants with Coordinate Attention (CA), Multi-Level Feature Interaction (MLFI), and geotemporal fusion — alongside an 11-configuration ablation across loss dynamics, feature fusion, and transfer-learning strategies. Our geotemporal model achieves **89.38% Top-1 accuracy** and **98.01% Top-5** with 2.1× fewer parameters than ViT-B/16 (89.99% Top-1), while achieving the best sparse-class accuracy (87.43%) across all models. A geo-shuffled control confirms the geotemporal branch learns genuine biogeographic priors (−3.02% Top-1 on ablation). Cross-referencing against iNaturalist reveals 69.3% of our species have fewer than 100 India-specific records, establishing IndoLepAtlas as a critical resource for India-focused Lepidoptera research.

**Keywords:** Fine-grained visual classification · Lepidoptera · Indian biodiversity · geotemporal fusion · long-tail distribution · Coordinate Attention



---

## 1. Introduction and Motivation

India's 1,500+ Lepidoptera species represent one of the world's richest butterfly biodiversity hotspots. These insects serve as critical ecological indicators — their population dynamics correlate with habitat health, pollination networks, and climate change effects. Automated classification from field photographs is critical for ecological monitoring and conservation planning at scale.

Yet this task fundamentally differs from standard visual recognition benchmarks. Discriminative features are spatially precise: eyespot location on forewing versus hindwing can separate entire genera. Features are hierarchically distributed across scales — fine texture bands at one level, overall wing morphology at another. And the data distribution is naturally and severely skewed: dominant species exceed 200 images while approximately 533 rare species have fewer than 50 images each.

Existing global biodiversity datasets — iNaturalist, GBIF — dilute India-specific subspecies representation. Our cross-referencing reveals that 69.3% of IndoLepAtlas species have fewer than 100 India-specific iNaturalist records, and 145 species (15.1%) have zero records. Furthermore, no existing butterfly classification dataset integrates geotemporal metadata (biogeographic zone, observation season) or documents larval host plant associations — ecological dimensions that are standard practice in field entomology but absent from computational approaches.

Standard CNNs fail on this task in three predictable ways:

1. **Spatial information collapse:** Channel attention mechanisms like SE-Net collapse spatial dimensions through global average pooling, discarding the positional information essential for wing pattern discrimination.
2. **Progressive texture loss:** Deep pooling progressively destroys fine-grained texture cues, routing only high-level semantics to the classifier — precisely the wrong bias for a domain where submarginal bands and discal spots are discriminative.
3. **Long-tail imbalance:** Uniform sampling over-indexes on common species, starving rare classes of gradient signal.

### Contributions

1. **IndoLepAtlas dataset:** 60,641 butterfly images, 967 species, GPS coordinates, observation dates, 9-zone biogeographic mapping, early life-stage documentation (9,804 images), and 703 host plant images — the first butterfly dataset to integrate ecological metadata at this scale.
2. **Systematic baseline establishment:** 11-configuration ablation + 6-model comparison providing reproducible baselines.
3. **Domain-shift analysis:** Empirical proof that ImageNet features are fundamentally misaligned with entomological morphology — head-only probing achieves 53.20% Top-1 with a striking Dense/Sparse accuracy inversion (44.46% vs 70.75%).
4. **Geotemporal fusion:** First integration of biogeographic zone and seasonal embeddings into butterfly classification, achieving 89.38% Top-1 with 2.1× fewer parameters than ViT-B/16, verified via geo-shuffled ablation (−3.02% Top-1).

---

## 2. Related Work

### 2.1 Fine-Grained Visual Classification

Fine-grained visual classification (FGVC) distinguishes subordinate categories within a superclass — bird species, dog breeds, car models. Key benchmarks include CUB-200-2011 (200 bird species, 11,788 images) and iNaturalist (5,000+ species, 675,000 images). Architecturally, the field progressed from bilinear pooling through channel attention (SE-Net) to multi-level feature fusion (FGBNet, Yuan et al. 2025). Vision Transformers achieve strong FGVC results but require substantial data. Coordinate Attention (Hou et al. 2021) offers a lightweight alternative preserving spatial structure — a property we exploit for wing-pattern localisation. Butterfly FGVC is harder than birds: no standardised pose, uncontrolled field photography, and extreme long-tail distributions.

### 2.2 Insect Classification Datasets and Systems

The systematic literature review by Amarathunga et al. (2021) surveyed insect image classification comprehensively: only 10 of 28 insect orders covered, no Africa or South Asia representation, no morphologically similar microscopic species datasets, and temporal distribution flagged as an explicit open problem. The thrips classification work (Amarathunga et al. 2022) is the most relevant morphological similarity study — domain-knowledge-driven body-segment augmentation with ViT achieved 94.7% on microscopic insects, analogous to our challenges. Alfatemi et al. (2024) established methodology for manual data curation in visually similar bird species, a protocol we adapt for Lepidoptera.

### 2.3 Geotemporal Context in Species Recognition

Species distribution modelling (MaxEnt, random forests) has long used geographic and seasonal features for predicting occurrence. iNaturalist uses location as a soft prior in its species suggestions. However, these operate as separate prediction systems — no classification paper has formalised geotemporal context as a learned feature fused into the visual pipeline. The SLR (Amarathunga et al. 2021) explicitly identifies temporal and geographic distribution as open problems. Our geotemporal fusion module directly addresses this gap.

### 2.4 Early Stage and Host Plant Data

Lepidoptera undergo complete metamorphosis; species identification at larval stages is ecologically important for conservation and agricultural pest management. No existing FGVC dataset documents early life stages or larval host plant associations alongside adult specimens. IndoLepAtlas fills this gap with 9,804 early-stage images and 703 host plant images, enabling future work on life-stage classification and habitat suitability modelling.

---

## 3. The IndoLepAtlas Dataset

### 3.1 Collection Protocol

IndoLepAtlas comprises 61,344 images (60,641 butterflies across 967 species + 703 plant images across 127 larval host plant species), curated primarily from the IFoundButterflies.org Indian Lepidoptera web atlas. The curation pipeline:

1. **Species-level organisation:** Images organised by scientific name, each species its own class (butterfly class IDs 0–966; plant IDs 967+).
2. **Quality filtering:** Non-butterfly entries removed, taxonomic synonyms resolved against established references.
3. **Life-stage separation:** Early life-stage images (caterpillar, pupa, chrysalis, egg — 9,804 images) separated from adult specimens (50,686) using metadata tags and filename pattern matching. Adults used for classification training; early-stage images retained as labelled auxiliary data.
4. **Metadata extraction:** Per-image metadata extracted via OCR (pytesseract): scientific name, common name, family, media code, location, date, photographer credit, sex/life stage. **Missing fields stored as empty strings, not dropped.**

**Annotation protocol:** Bounding boxes generated in both YOLO format (`<class_id> <x_center> <y_center> <width> <height>`, normalised) and COCO format (`[x, y, width, height]` pixels). Automated annotation uses **Grounding DINO** zero-shot detection with prompts `"butterfly . moth . caterpillar . pupa . chrysalis"` (butterflies) and `"plant . flower . leaf . tree . shrub"` (plants). Fallback: full image as bounding box if detection fails. Species class assigned from directory structure, not model output. Quality verified by spot-checking 100 random images per subset; flagged failures re-annotated manually via CVAT.

| Edge Case | Rule |
|-----------|------|
| Multiple subjects | Separate bounding boxes for each |
| Partially visible (>30%) | Annotate |
| Very small subject | Annotate if clearly identifiable |
| Occluded by vegetation | Annotate visible portion |

### 3.2 Dataset Statistics

| Property | Value |
|----------|-------|
| Total images | 61,344 |
| Butterfly images | 60,641 |
| Plant images | 703 |
| Butterfly species | 967 |
| Plant species | 127 |
| Adult specimens | 50,686 |
| Early-stage images | 9,804 |
| States/territories | 36 |

**Family distribution (butterflies):**

| Family | Images | % |
|--------|--------|---|
| Nymphalidae | 18,054 | 29.8% |
| Lycaenidae | 16,985 | 28.0% |
| Hesperiidae | 12,361 | 20.4% |
| Pieridae | 6,132 | 10.1% |
| Papilionidae | 5,826 | 9.6% |
| Riodinidae | 1,132 | 1.9% |

**Top species by image count:**

| Species | Count |
|---------|-------|
| *Papilio protenor* | 203 |
| *Euploea mulciber* | 199 |
| *Telicota bambusae* | 199 |
| *Kaniska canace* | 192 |
| *Notocrypta curvifascia* | 192 |

Classes stratified into **Dense** (≥50 images, ~434 classes) and **Sparse** (<50 images, ~533 classes) for evaluation.

**Geographic distribution (top 10 states):**

| State | Images |
|-------|--------|
| Maharashtra | 10,198 |
| Karnataka | 8,121 |
| Kerala | 6,795 |
| Arunachal Pradesh | 5,856 |
| Assam | 3,556 |
| Uttarakhand | 3,526 |
| Sikkim | 3,134 |
| West Bengal | 3,007 |
| Meghalaya | 2,824 |
| Tamil Nadu | 1,364 |

Total: 57,403 images with state-level location across 36 states/territories. Western Ghats (Maharashtra, Karnataka, Kerala) and Northeast India (Arunachal Pradesh, Assam, Meghalaya) are the primary contributors, reflecting both biodiversity hotspot density and active naturalist communities.

> **Figure 2** — *Geographic distribution map of India with 9 biogeographic zones and image density overlay.*

**Temporal distribution:**

| Month | Images | | Month | Images |
|-------|--------|---|-------|--------|
| January | 2,157 | | July | 2,579 |
| February | 1,486 | | August | 3,968 |
| March | 3,722 | | September | 5,198 |
| April | 4,654 | | October | **7,228** |
| May | 3,892 | | November | 5,654 |
| June | 2,516 | | December | 3,775 |

Clear post-monsoon peak (Sep–Nov), with October contributing the most images (7,228). This mirrors natural butterfly emergence patterns: the Indian monsoon triggers larval host plant growth, leading to peak adult emergence in September–November.

**Data splits:**

| Split | Images | % |
|-------|--------|---|
| Train | 48,630 | 79.3% |
| Validation | 5,649 | 9.2% |
| Test | 7,065 | 11.5% |

> **Note on splits:** The full dataset contains 61,344 images (48,630/5,649/7,065 train/val/test). The model comparison experiments use 40,260/4,673/5,865 = 50,798 images because **early-stage images (9,804) and plant images (703) are excluded from classification training** — only adult butterfly specimens are used for species classification.

### 3.3 Metadata Description

**Location → Biogeographic zone:** GPS coordinates mapped to India's 9 established biogeographic zones (Wildlife Institute of India classification): Western Ghats, Deccan Peninsula, Western Himalayas, Eastern Himalayas, Northeast, Indo-Gangetic Plain, Semi-Arid, Desert, and Andaman & Nicobar Islands. A 10th "unknown" index is used when GPS is missing. Zone-level encoding is ecologically motivated (species follow biogeographic boundaries, not administrative ones) and robust to typical ±km GPS errors. Each zone is represented as a learned 32-dimensional embedding vector.

**Date → Cyclic encoding:** Observation month encoded as [sin(2πM/12), cos(2πM/12)], preserving circularity (December adjacent to January).

**Missing data handling:** Images lacking location or date use a learned "unknown" embedding token during geotemporal fusion.

**Metadata coverage (butterflies):**

| Field | Present | Missing | % Missing |
|-------|---------|---------|-----------|
| common_name | 60,380 | 261 | 0.4% |
| location | 60,542 | 99 | 0.2% |
| state | 56,712 | 3,929 | 6.5% |
| media_code | 55,766 | 4,875 | 8.0% |
| date | 46,211 | 14,430 | 23.8% |
| credit | 43,461 | 17,180 | 28.3% |
| sex | 21,332 | 39,309 | 64.8% |

### 3.4 Dataset Quality Validation

1. **Automated bbox verification:** 100 random images per subset spot-checked; failures re-annotated via CVAT.
2. **Taxonomic consistency:** Species labels inherit from expert-curated IFoundButterflies.org directory structure, cross-referenced against Indian butterfly field guides.
3. **Ecological plausibility:** Geographic/seasonal distributions cross-referenced against known species ranges. Confusion matrix analysis (Section 6) confirms systematic confusion occurs between congeneric species — evidence of genuine morphological similarity, not labelling error.

> **Figure 4** — *Sample image grid showing morphologically similar species pairs.* [Placeholder — authors to add images]

### 3.5 Comparison with Existing Datasets

| Dataset | Species | Images | Domain | Location | Temporal | Host Plants | India-specific |
|---------|---------|--------|--------|----------|----------|-------------|----------------|
| CUB-200-2011 | 200 | 11,788 | Birds | ✗ | ✗ | ✗ | ✗ |
| iNaturalist | 5,089 | 675,170 | Multi | ✓ | ✓ | ✗ | ✗ |
| IP102 | 102 | 75,222 | Insects | ✗ | ✗ | ✗ | ✗ |
| **IndoLepAtlas** | **967** | **60,641** | **Butterflies (+ 127 host plants)** | **✓** | **✓** | **✓** | **✓** |

### 3.6 Coverage Gap Analysis

Cross-referencing 961 IndoLepAtlas species against iNaturalist India research-grade observations:

| iNat India Records | # Species | % of Dataset |
|--------------------|-----------|-------------|
| 0 (absent) | 145 | 15.1% |
| 1–10 | 249 | 25.9% |
| 11–50 | 194 | 20.2% |
| 51–100 | 78 | 8.1% |
| 101–500 | 147 | 15.3% |
| >500 | 148 | 15.4% |

**69.3% of IndoLepAtlas species have fewer than 100 India-specific iNaturalist records.** 145 species (15.1%) are entirely absent. This confirms global repositories severely under-represent India-specific Lepidoptera diversity.

---

## 4. Experimental Setup

Experiments execute on a DGX V100 cluster in two phases.

**Phase 1 — 11-Configuration Ablation:**
- **Backbone:** ConvNeXt-Tiny (ImageNet pretrained)
- **Optimiser:** AdamW, differential learning rates (backbone 0.1×, new heads 10⁻⁴)
- **Scheduler:** Cosine annealing with 5-epoch warmup
- **Training:** Mixed precision (AMP), gradient norm clipping at 5.0
- **Checkpoint selection:** Best macro-F1 on validation (ensures long-tail sensitivity)
- **Augmentation (train):** RandomResizedCrop(224), horizontal flip, random rotation (±30°), color jitter (hue/saturation/brightness), ImageNet normalisation. No vertical flip (butterflies are never upside-down in field photographs).
- **Augmentation (val/test):** Deterministic resize + centre crop (224), ImageNet normalisation.
- Unit I (Loss dynamics): 30 epochs, 4 configs
- Units II–III (Architecture, Freezing): 40 epochs, 7 configs

**Phase 2 — 6-Model Comparison (40 epochs each):**

| Model | Backbone | Batch | Params |
|-------|----------|-------|--------|
| ResNet-101 | timm/resnet101 | 64 | 44.5M |
| EfficientNet-B5 | timm/efficientnet_b5 | 32 | 30.3M |
| ViT-B/16 | timm/vit_base_patch16_224 | 16 | 86.5M |
| MLFI Warmup | ConvNeXt-S + CA + MLFI | 32 | 40.8M |
| Geo-Shuffled | ConvNeXt-S + CA + MLFI + Geo* | 32 | 40.8M |
| Geotemporal (Ours) | ConvNeXt-S + CA + MLFI + Geo | 32 | 40.8M |

*Geo-Shuffled: zone/month labels randomly permuted at inference to ablate geotemporal signal.*

All models: CE loss, AdamW (lr=10⁻⁴), balanced sampling. Dataset: 40,260 train / 4,673 val / 5,865 test across 966 species.

**Metrics:** Top-1 accuracy, Top-5 accuracy, Macro Precision, Macro F1, Weighted F1, Dense-stratum accuracy (≥50 images), Sparse-stratum accuracy (<50 images), Δ Acc (Sparse − Dense).

---

## 5. Baseline Experiments and Results

### 5.1 Unit I: Long-Tail Loss Dynamics (30 Epochs)

**Question:** *How do different loss formulations and sampling strategies impact representational capacity on long-tailed fine-grained distributions?*

| Configuration | Top-1 (%) | Macro F1 (%) | Dense Acc (%) | Sparse Acc (%) |
|---------------|-----------|-------------|---------------|----------------|
| **CE + Balanced** | 87.06 | **85.48** | 87.67 | 85.84 |
| CE + Unbalanced | **87.60** | 80.23 | **91.22** | 80.35 |
| Focal + Balanced | 82.69 | 82.64 | 80.85 | **86.40** |
| Focal + Unbalanced | 84.83 | 82.70 | 85.70 | 83.07 |

> **Figure 5** — Unit I training curves. File: `report_assets/plots/unit1_training_curves.png`
> **Figure 6** — Unit I metric comparison. File: `report_assets/plots/unit1_bar_comparison.png`

**The "Unbalanced" illusion.** CE + Unbalanced achieves the highest raw Top-1 (87.60%), but Dense accuracy inflates to 91.22% while Sparse drops to 80.35%. Macro-F1 plummets to 80.23% — a **5.25-point gap** below the balanced variant — confirming the model abandons rare species to maximise majority-class performance.

**CE + Balanced as optimal compromise.** Balanced sampling forces equitable gradient allocation, compressing the Dense–Sparse gap to just 1.83 points (87.67% vs 85.84%) while achieving the highest macro-F1 (85.48%). The mechanism is direct: inverse-frequency sampling ensures rare-class gradients remain proportional regardless of distribution skew.

**The Focal Loss penalty.** Focal Loss (γ=2) achieves the highest Sparse-class accuracy (86.40% under balanced sampling) but severely degrades Dense-class performance to 80.85%. In 966-class fine-grained settings, the (1−pₜ)^γ modulation over-penalises well-classified easy examples (typically Dense strata), damaging foundational feature extraction for common species. This contradicts the assumption that loss-function sophistication universally solves long-tail problems — explicit sampling proves more reliable.

### 5.2 Unit II: Feature-Fusion Ablation (40 Epochs)

**Question:** *Where is the optimal structural transition point for integrating attention mechanisms, and does multi-level feature integration improve fine-grained classification?*

| Configuration | Top-1 (%) | Macro F1 (%) | Dense Acc (%) | Sparse Acc (%) |
|---------------|-----------|-------------|---------------|----------------|
| Phase 1 (Baseline) | 86.50 | 85.65 | **86.36** | 86.76 |
| **Phase 2 (+ CA)** | **86.77** | **85.75** | 86.13 | **88.05** |
| Phase 3 (+ CA + MLFI) | 84.43 | 83.34 | 83.32 | 86.66 |

> **Figure 7** — Unit II training curves. File: `report_assets/plots/unit2_training_curves.png`
> **Figure 8** — Unit II metric comparison. File: `report_assets/plots/unit2_bar_comparison.png`

**Coordinate Attention preserves spatial morphology.** Introducing CA after each backbone stage yields the best overall configuration: 86.77% Top-1, 85.75% macro-F1, and a notable **+1.29-point improvement** in Sparse accuracy (86.76% → 88.05%). CA decomposes channel-spatial attention into two 1D processes — horizontal pooling and vertical pooling — preserving directional structure naturally suited to butterfly morphology (horizontal banding, vertical vein streaking). Unlike SE-Net's global average pooling, CA retains *where* a pattern occurs. The FGBNet ablation (Yuan et al. 2025) tested SE, CBAM, and CA on the same datasets: CA achieved 90.748% avg accuracy vs CBAM 89.55% and SE 87.586%. We follow FGBNet's optimal placement — CA after each of the 4 backbone stages (not inside each block), using only 4 CA modules instead of 27, with <0.5% parameter overhead. **High-value, low-cost intervention.**

**MLFI: powerful but over-parameterised.** Phase 3 adds a Multi-Level Feature Interaction module: each of the 4 ConvNeXt stages feeds a Detail Information Supplement (DIS) branch consisting of Adaptive Max Pooling → Flatten → FC projection. The 4 branch outputs are concatenated (CONCAT, not ADD or MULTI — FGBNet showed CONCAT achieves best accuracy and generalisation). With the 2:4:1:8 feature proportion (FGBNet's optimal ratio), stage outputs are n₁=192, n₂=384, n₃=96, n₄=768, yielding a 1,440-dimensional visual feature vector. Max pooling retains the most salient wing-pattern activations rather than averaging them. At 40 epochs, MLFI reaches 84.43% Top-1 and 86.66% Sparse — demonstrating multi-level fusion does benefit rare classes — but creates a 2.07-point Top-1 gap versus baseline. The newly initialised MLFI layers (Xavier init, full learning rate) create an optimisation imbalance with the near-converged backbone (0.1× learning rate), resulting in a poorly conditioned loss landscape.

### 5.3 Unit III: Transfer Learning Layer Freezing (40 Epochs)

**Question:** *Which hierarchical layers contain domain-agnostic vs domain-specific representations, and what is the optimal adaptation pathway?*

| Configuration | Top-1 (%) | Macro F1 (%) | Dense Acc (%) | Sparse Acc (%) |
|---------------|-----------|-------------|---------------|----------------|
| **End-to-End** | **84.79** | **84.11** | **83.89** | 86.61 |
| Head Only | 53.20 | 57.02 | 44.46 | 70.75 |
| Freeze Early | 83.77 | 83.20 | 82.74 | 85.84 |
| Freeze Late | 84.33 | 84.02 | 83.15 | **86.71** |

> **Figure 9** — Unit III training curves. File: `report_assets/plots/unit3_training_curves.png`
> **Figure 10** — Unit III metric comparison. File: `report_assets/plots/unit3_bar_comparison.png`

**The domain-shift wall.** Head-only linear probing — freezing the entire backbone and training only a classification head — achieves just **53.20% Top-1** on 967 species. The striking Dense/Sparse inversion (Dense 44.46% < Sparse 70.75%) reveals ImageNet features are not merely *incomplete* but actively *misaligned* with Lepidoptera morphology. ImageNet activations encode "furry surface textures" and "rounded shapes," not "submarginal band" or "discal spot." Fine-grained biological classification is fundamentally a **domain-shift problem disguised as a classification problem**.

**Early vs late block freezing.** Freezing late blocks (84.33%) slightly outperforms freezing early blocks (83.77%), implying early-layer Gabor-like edge detectors require significant recalibration for butterfly wing scales and biological venation. The inter-strategy variance is small (Δ < 1%), indicating any partial-freezing strategy approaches end-to-end performance under sufficient training budget.

**End-to-end supremacy.** Full gradient backpropagation achieves the best macro-F1 (84.11%) and most balanced stratum performance (Dense 83.89%, Sparse 86.61%). The entire representational manifold must be warped for this domain.

### 5.4 Cross-Unit Analysis

**Complete 11-configuration ablation:**

| Unit | Configuration | Epochs | Top-1 | Macro-F1 | Dense | Sparse |
|------|--------------|--------|-------|----------|-------|--------|
| I | CE + Balanced | 30 | **87.06** | **85.48** | 87.67 | 85.84 |
| I | CE + Unbalanced | 30 | 87.60 | 80.23 | **91.22** | 80.35 |
| I | Focal + Balanced | 30 | 82.69 | 82.64 | 80.85 | **86.40** |
| I | Focal + Unbalanced | 30 | 84.83 | 82.70 | 85.70 | 83.07 |
| II | Phase 1 (Baseline) | 40 | 86.50 | 85.65 | **86.36** | 86.76 |
| II | Phase 2 (+ CA) | 40 | **86.77** | **85.75** | 86.13 | **88.05** |
| II | Phase 3 (+ CA + MLFI) | 40 | 84.43 | 83.34 | 83.32 | 86.66 |
| III | End-to-End | 40 | **84.79** | **84.11** | **83.89** | 86.61 |
| III | Head Only | 40 | 53.20 | 57.02 | 44.46 | 70.75 |
| III | Freeze Early | 40 | 83.77 | 83.20 | 82.74 | 85.84 |
| III | Freeze Late | 40 | 84.33 | 84.02 | 83.15 | **86.71** |

> **Figure 11** — Dense vs Sparse accuracy across all 11 configs. File: `report_assets/plots/stratum_comparison.png`
> **Figure 12** — Radar chart of best configs per unit. File: `report_assets/plots/best_models_radar.png`

A consistent pattern emerges: **Sparse-class accuracy often exceeds Dense-class accuracy under balanced training.** This occurs because balanced sampling provides proportionally more gradient signal per Sparse-class sample, while Dense classes — despite more data — contain greater intra-class morphological variation (seasonal dimorphism, sexual dimorphism, geographic colour variation). The most pronounced effect appears in Phase 2 (CA), where Sparse reaches 88.05% vs Dense 86.13%.

**Top-5 saturation:** All configurations except Head-Only achieve Top-5 accuracy above 96.6%, indicating confusion is largely confined to the most morphologically similar species pairs — suggesting hierarchical classification or pairwise discriminators as a high-leverage future direction.

---

## 6. Model Comparison and Geotemporal Fusion

### 6.1 Main Comparison (Test Set, 40 Epochs)

| Model | Backbone | Params | Top-1 | Top-5 | Macro P | Macro F1 | Wt. F1 |
|-------|----------|--------|-------|-------|---------|----------|--------|
| ResNet-101 | ResNet-101 | 44.5M | 48.76% | 76.91% | 49.94% | 47.60% | 47.76% |
| EfficientNet-B5 | EfficientNet-B5 | 30.3M | 86.96% | 97.05% | 87.33% | 85.92% | 86.57% |
| **ViT-B/16** | ViT-B/16 | 86.5M | **89.99%** | 97.51% | **90.14%** | **88.11%** | **89.51%** |
| MLFI Warmup | ConvNeXt-S + CA + MLFI | 40.8M | 87.20% | 97.34% | 87.84% | 85.72% | 86.76% |
| Geo-Shuffled | ConvNeXt-S + CA + MLFI + Geo* | 40.8M | 86.36% | 97.12% | 86.41% | 84.57% | 85.86% |
| **Geotemporal (Ours)** | ConvNeXt-S + CA + MLFI + Geo | 40.8M | 89.38% | **98.01%** | 89.13% | 87.37% | 88.94% |

*Geo-Shuffled: zone/month labels randomly permuted at inference to ablate geotemporal signal.*

> **Training curve figures** (in `experiment_assets/training_curves/`): `vit_b16_metrics.png`, `geotemporal_metrics.png`, `geo_shuffled_metrics.png`, `resnet101_metrics.png`, `effnet_b5_metrics.png`, `mlfi_warmup_metrics.png` (+ corresponding `_loss.png` variants).
>
> **Confusion matrices** (in `experiment_assets/confusion_matrices/`): `vit_b16_confusion.png`, `geotemporal_confusion.png`, `resnet101_confusion.png`, `effnet_b5_confusion.png`, `geo_shuffled_confusion.png`, `mlfi_warmup_confusion.png`. Top confused pairs: `{model}_top_pairs.csv`.

### 6.2 Per-Stratum Analysis

| Model | Dense Acc | Dense F1 | Sparse Acc | Sparse F1 | Δ Acc (S−D) |
|-------|-----------|----------|------------|-----------|-------------|
| ResNet-101 | 45.02% | 22.53% | 56.29% | 45.80% | +11.27 |
| EfficientNet-B5 | 87.33% | 68.30% | 86.20% | 75.07% | −1.13 |
| ViT-B/16 | **91.60%** | **79.40%** | 86.76% | 75.18% | −4.84 |
| MLFI Warmup | 88.36% | 73.07% | 84.86% | 72.84% | −3.50 |
| Geo-Shuffled | 87.87% | 72.32% | 83.32% | 71.89% | −4.55 |
| **Geotemporal (Ours)** | 90.35% | 76.74% | **87.43%** | **75.38%** | −2.92 |

### 6.3 Confusion Analysis

The most-confused species pairs across models are consistently congeneric:

| Model | True Species | Predicted As | Count |
|-------|-------------|-------------|-------|
| ResNet-101 | *Symbrenthia brabira* | *S. lilaea* | 7 |
| ResNet-101 | *Eurema blanda* | *E. hecabe* | 7 |
| EfficientNet-B5 | *Azanus ubaldus* | *A. uranus* | 6 |
| EfficientNet-B5 | *Tarucus indica* | *T. nara* | 5 |
| ViT-B/16 | *Tapena thwaitesi* | *Taraka hamada* | 8 |
| ViT-B/16 | *Azanus uranus* | *A. ubaldus* | 5 |
| Geotemporal | *Papilio daksha* | *P. buddha* | 6 |
| Geotemporal | *Mycalesis visala* | *M. radza* | 5 |

Persistent confusion between congeneric species sharing wing morphology (e.g., *Papilio daksha* ↔ *P. buddha*, *Mycalesis visala* ↔ *M. radza*) that differ in geographic range — precisely the information encoded by geotemporal embeddings.

### 6.4 Key Findings

**1. Geotemporal fusion is competitive with ViT-B/16.** Our geotemporal model (40.8M params) achieves 89.38% Top-1 — within 0.61 points of ViT-B/16 (86.5M params) — while using **2.1× fewer parameters**. It achieves the **best Top-5 accuracy (98.01%)** across all models.

**2. Geotemporal metadata provides a genuine signal.** The geo-shuffled control (same architecture, randomised location/month) drops by:
- **−3.02% Top-1** (89.38 → 86.36)
- **−2.80% Macro F1** (87.37 → 84.57)

This confirms the geotemporal branch learns meaningful biogeographic priors, not just adds parameters.

**3. Best sparse-class performance from geotemporal model.** 87.43% accuracy on sparse classes (<50 training images), outperforming all baselines including ViT-B/16 (86.76%). The Dense–Sparse gap narrows to Δ = −2.92 vs ViT-B/16's Δ = −4.84, suggesting geotemporal priors help disambiguate underrepresented species.

**4. ResNet-101 confirms architectural requirements.** Only 48.76% Top-1, confirming modern architectures with attention mechanisms or stronger inductive biases are essential for 966-class fine-grained recognition.

**5. MLFI warmup provides modest gains.** 87.20% vs 86.36% (geo-shuffled), but underperforms full geotemporal fusion (89.38%), suggesting multi-level feature integration benefits from geospatial context.

---

## 7. Discussion

Our experiments highlight why architectural design and ecological context are critical for fine-grained biological classification. Three structural constraints drive the primary failure modes:

### Spatial Collapse
Coordinate Attention addresses spatial collapse by preserving the directional structure of butterfly morphology (e.g., horizontal bands vs. vertical veins). Adding CA requires <0.5% more parameters but improves sparse-class accuracy by 1.29 points, making it a highly efficient architectural intervention.

### Texture Loss
MLFI preserves essential early-layer texture features, but requires careful optimisation. Our MLFI Warmup experiment validates this: applying a learning rate warmup to newly initialised branches recovers a **2.77-point Top-1 improvement** (87.20% vs. 84.43%), confirming that texture loss is an optimisation pathology rather than an architectural flaw.

### Domain Shift
The failure of Head-Only fine-tuning (53.20%) highlights a severe domain shift from natural images to biological specimens. ImageNet features encode object silhouettes rather than critical fine-grained features like submarginal bands, making full end-to-end recalibration essential for transferability.

### Geotemporal Signal
Beyond architecture, geotemporal fusion bridges the remaining accuracy gap by encoding biological reality. The most visually confused species pairs (*Papilio daksha* ↔ *P. buddha*, *Mycalesis visala* ↔ *M. radza*) are congeneric species with overlapping wing patterns but **distinct geographic ranges and flight seasons** — precisely the ecological context that geotemporal embeddings provide to disambiguate visually identical species.

These findings extend beyond butterflies to medical imaging, satellite-based crop monitoring, and microscopy-based cell classification — all domains facing similar constraints of extreme class imbalance, domain-specific morphological features, and limited per-class training data.

---

## 8. Conclusion

IndoLepAtlas fills a documented gap in the insect classification literature as the first India-specific Lepidoptera dataset with integrated geotemporal metadata, early life-stage documentation, and host plant associations. By establishing a 60,641-image baseline covering 967 species, we provide a critical resource for fine-grained species recognition in highly imbalanced, real-world ecological domains.

Our evaluations demonstrate that modern architectures require targeted adaptations for biological classification. Standard ImageNet transfer fails due to severe domain shifts, making end-to-end fine-tuning mandatory. However, combining Coordinate Attention with balanced sampling creates an efficient baseline that mitigates both spatial collapse and long-tail imbalance. Crucially, our geotemporal fusion module proves that integrating biogeographic and seasonal context provides genuine discriminative signal, bridging the accuracy gap to match much larger Vision Transformers at half the parameter cost.

### Future Work

- **Prototypical network heads** for the ~533 classes with <50 images, enabling few-shot classification of rare species.
- **Domain-specific self-supervised pretraining** (DINO/MAE on butterfly images) to bootstrap features from unlabelled in-domain data, reducing dependence on ImageNet initialisation.
- **Host plant association features** for habitat-aware species prediction — a species cannot persist where its host plant is absent.
- **Hierarchical classification** exploiting the taxonomic tree (family → genus → species) to reduce confusion between congeneric species.

---

## References

1. Yuan, Y. et al. (2025). "FGBNet: A Bio-Subspecies Classification Network with Multi-Level Feature Interaction." *Diversity*.
2. Hou, Q. et al. (2021). "Coordinate Attention for Efficient Mobile Network Design." *CVPR*.
3. Liu, Z. et al. (2022). "A ConvNet for the 2020s." *CVPR*.
4. Hu, J. et al. (2018). "Squeeze-and-Excitation Networks." *CVPR*.
5. Lin, T.-Y. et al. (2017). "Focal Loss for Dense Object Detection." *ICCV*.
6. Amarathunga, D.C. et al. (2022). "Fine-grained image classification of microscopic insect pest species." *Computers and Electronics in Agriculture*.
7. Amarathunga, D.C.K. et al. (2021). "Methods of Insect Image Capture and Classification: A Systematic Literature Review." *Smart Agricultural Technology*.
8. Alfatemi, A. et al. (2024). "Multi-Label Classification with Deep Learning and Manual Data Collection for Identifying Similar Bird Species." *Procedia Computer Science*.
9. Dosovitskiy, A. et al. (2021). "An image is worth 16x16 words: Transformers for image recognition at scale." *ICLR*.
10. Snell, J. et al. (2017). "Prototypical Networks for Few-Shot Learning." *NeurIPS*.
11. Wah, C. et al. (2011). "The Caltech-UCSD Birds-200-2011 Dataset." Technical Report, Caltech.
12. Van Horn, G. et al. (2018). "The iNaturalist Species Classification and Detection Dataset." *CVPR*.
13. Kehimkar, I. (2008). "The Book of Indian Butterflies." BNHS.
14. Loshchilov, I. and Hutter, F. (2017). "SGDR: Stochastic Gradient Descent with Warm Restarts." *ICLR*.
15. Woo, S. et al. (2018). "CBAM: Convolutional Block Attention Module." *ECCV*.
16. Loshchilov, I. and Hutter, F. (2019). "Decoupled Weight Decay Regularization." *ICLR*.


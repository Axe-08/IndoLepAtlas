## Paper Title
**IndoLepAtlas: A Fine-Grained Dataset of Indian Lepidoptera with Geotemporal Metadata and Baseline Classification Study**
 ### title running: IndoLepAtlas: A India-Specific Lepidoptera Dataset
*(or variant: "...for Biodiversity Monitoring and Species Recognition")*

---
## Abstract
- 150-200 words
- add keywords at the end

## Section 1 — Introduction (~1 page)

**Narrative arc:** India is a butterfly biodiversity hotspot → automated classification is critical for conservation → existing datasets fail India specifically → we fix this.

**Paragraphs:**
- Open with the ecological stakes: 1,500+ Indian Lepidoptera species, role in pollination monitoring, climate change indicators (cite the bird paper's framing — Alfatemi et al. 2024 uses this exact argument, just swap birds for butterflies)
- The gap: global datasets (iNaturalist, GBIF) dilute India-specific subspecies. Datasets don't captures the geotemporal context, early life stages' or host plants' information that makes Indian butterfly distribution ecologically meaningful well
- What makes this hard: your three failure modes from the report — spatial information collapse, texture loss through pooling, long-tail imbalance. One crisp sentence each.
- Contributions bullet list (4 bullets):
  - IndoLepAtlas: 61,314 images, 967 species, 9-zone biogeographic mapping, GPS + date metadata
  - Systematic 11-configuration ablation establishing strong baselines
  - Empirical proof that ImageNet features are actively misaligned with Lepidoptera morphology
  - Geotemporal fusion results

---

## Section 2 — Related Work (~1 page)

**Three subsections, roughly one paragraph each:**

**2.1 Fine-Grained Visual Classification** — Standard FGVC problem, CUB-200-2011 as canonical benchmark, ConvNeXt/ViT as current backbones, FGBNet (Yuan et al. 2025) as most recent fine-grained subspecies work. One sentence on why butterfly FGVC is harder than birds (no standardized pose, field photography, extreme long tail).

**2.2 Insect Classification Datasets and Systems** — Cite the SLR (Amarathunga et al. 2021) heavily here. Key stats from it: only 10 of 28 insect orders covered, no Africa or South Asia coverage, no morphologically-similar microscopic species datasets, temporal distribution flagged as open problem. The thrips paper (Amarathunga et al. 2022) as the most relevant morphological similarity work. Alfatemi bird paper for manual curation methodology.

**2.3 Geotemporal Context in Species Recognition** — This is where you establish novelty. Note that the SLR explicitly identifies temporal and geographic distribution as open problems. iNaturalist uses location as a soft prior but no classification paper has formalized it as a learned feature. One sentence is enough — the point is no one has done what you're about to do.

**2.4 Early Stage and Larval Host Plant data availability** - Extra semantic information for each butterfly species to enahance classification and add more context.

---

## Section 3 — The IndoLepAtlas Dataset (~2 pages)

*This is the heart of the paper. Most of the figures live here.*

**3.1 Collection Protocol**

How images were sourced (IFoundButterflies.org), filtering criteria (species with at least N images post-filtering, quality thresholds), how metadata was validated.
use information from annotation_guide.md and annotations/classes.txt for added reference

**3.2 Dataset Statistics**

> **Figure 1 — Class distribution histogram (log scale)**
> X-axis: number of images per class, Y-axis: number of classes. Show the long tail clearly. Mark the Dense/Sparse threshold at 50 images. This single figure justifies your entire imbalance handling strategy.

> **Figure 2 — Geographic distribution map of India**
> A map of India with the 9 biogeographic zones colored/shaded, with dots or a heatmap showing image collection density per zone. Maharashtra, Karnataka, Arunachal Pradesh should be visually prominent. This is your most visually striking figure and should probably be full-width. Tools: matplotlib + cartopy or geopandas with India shapefile.

> **Figure 3 — Temporal/seasonal distribution**
> A 12-bar histogram (one per month) showing image count. Show the monsoon peak (Aug–Nov) clearly. Optionally overlay a second series showing species richness per month (number of unique species observed) — this tells a richer ecological story than just image count.

> **Table 1 — Dataset statistics summary**
> Clean table: Total images, Total species, Dense classes (≥50), Sparse classes (<50), Location metadata coverage (98.3%), Date metadata coverage (76.1%), Geographic zones, States covered. Compare a column against iNaturalist India subset.

**3.3 Metadata Description**

GPS coordinates → biogeographic zone mapping (explain the 9-zone scheme, cite the Wildlife Institute of India biogeographic classification), date → month → cyclic encoding and/or meteorological season. Explain coverage gaps (23.9% missing dates) and how you handle them.

**3.4 Dataset Quality Validation**

> **Figure 4 — Sample image grid**
> 3×4 or 4×5 grid showing representative images across species, including morphologically similar pairs side by side. Pick 2-3 pairs that look nearly identical to visually motivate why this is hard. Caption should name the species and point out what differs (e.g., "submarginal band absent in X, present in Y"). add placeholders here, ill add the images myself later

Mention inter-rater verification. Use annotation_guide.md to write on how ambiguity/empty fields etc were handled

**3.5 Comparison with Existing Datasets**

> **Table 2 — Dataset comparison table**
> Columns: Dataset, Species, Images, Domain, Location metadata, Temporal metadata, India-specific. Rows: iNaturalist (global), GBIF (global), IP102 (insects, from SLR), CUB-200-2011 (birds, fine-grained reference), **IndoLepAtlas (yours)**. This table does a lot of work — it justifies the dataset's existence in one glance.

---

## Section 4 — Experimental Setup (~0.75 page)

Keep this tight. Readers will skim it.

- Dataset splits: train/val/test percentages, stratified by Dense/Sparse
- Backbone
- Optimizer
- Scheduler
- Checkpoint selection
- Hardware: DGX V100
- Experimental units, epochs per unit 
- Evaluation metrics: Top-1 accuracy, Top-5 accuracy, Macro-F1, Dense stratum accuracy, Sparse stratum accuracy etc

No figure needed here. Maybe a compact setup table if the venue requires it.

---

## Section 5 — Baseline Experiments and Results (~2.5 pages)

*Three subsections mirroring your three units. Each has one figure and feeds into the cross-unit analysis.*

**5.1 Unit I: Loss Formulation and Sampling Strategy**

One paragraph explaining what you tested and why (CE vs Focal, balanced vs unbalanced — motivate from the long-tail problem). Then results.

> **Figure 5 — Unit I training curves** (your Figure 1 from the report)
> Keep both panels (val accuracy + val loss). The visual separation between CE+Balanced and Focal+Unbalanced is clean and convincing.

> **Figure 6 — Unit I metric comparison bar chart** (your Figure 2 from the report)
> The grouped bars across Top-1, Top-5, Macro Prec, Macro F1. The "unbalanced illusion" is immediately visible here.

Key finding to highlight in prose: the CE+Unbalanced illusion — highest raw Top-1 but 5.25-point macro-F1 gap. Name it this way in the paper, it's a memorable framing. The focal loss finding (best Sparse but hurts Dense) is a counterintuitive result worth one strong sentence.

**5.2 Unit II: Architectural Feature Fusion**

> **Figure 7 — Unit II training curves** (your Figure 3)

> **Figure 8 — Unit II metric comparison** (your Figure 4)

Key finding: CA is high-value low-cost (<0.5% parameters, +1.29 Sparse accuracy). The MLFI optimization imbalance diagnosis is important — explain Xavier initialization + full learning rate vs near-converged backbone creating a poorly conditioned loss landscape. This is sophisticated and reviewers will appreciate that you diagnosed *why* it fails, not just that it does.

**5.3 Unit III: Transfer Learning and Layer Freezing**

> **Figure 9 — Unit III training curves** (your Figure 5)

> **Figure 10 — Unit III metric comparison** (your Figure 6)

This is your most generalisable finding. The Dense-class inversion under head-only probing (44.46% Dense vs 70.75% Sparse) needs to be foregrounded strongly. Explain the mechanism: ImageNet encodes object silhouettes and fur textures, not submarginal bands and discal spots. The features are not merely incomplete — they actively interfere with distinguishing morphologically similar dominant species.

**5.4 Cross-Unit Analysis**

> **Figure 11 — Dense vs Sparse accuracy across all 11 configurations** (your Figure 7)
> This is one of your best figures. The consistent pattern — Sparse often exceeds Dense under balanced training — tells the whole imbalance story in one visual. Keep this.

> **Figure 12 — Radar chart of best configurations** (your Figure 8)
> Good for visual summary but put it here rather than in discussion. Rename the three traces more descriptively: "Best Loss Config (CE+Bal)", "Best Architecture (CA)", "Best Transfer (E2E)".

> **Table 3 — Complete 11-configuration ablation table** (your Table 1)
> This should be here, not in an appendix. Bold your best result per column. It's the paper's central empirical contribution.

---

## Section 6 — Geotemporal Fusion (~1 page)

*This section is what separates the paper from a pure dataset paper and gives it methodological novelty. Even preliminary results are fine.*

**6.1 Motivation** — Two paragraphs. Many Indian species pairs are visually near-identical but have non-overlapping geographic ranges or seasonal flight windows. At the classification decision boundary, location becomes the deciding feature. Cite the SLR's identification of temporal distribution as an open problem.

**6.2 Encoding** — Location: GPS → 9-zone biogeographic embedding (learned, d=32). Season: cyclic month encoding [sin(2πM/12), cos(2πM/12)]. Late fusion: concatenate to MLFI feature vector before classification head. Missing metadata: learned unknown-zone token.

**6.3 Results**

> **Table 4 — Geotemporal fusion ablation**
> Rows: Vision only (best from Unit I) | + Location | + Season | + Location + Season. Columns: Top-1, Macro-F1, Dense, Sparse. Even small gains on Sparse accuracy matter here.

> **Figure 13 — Per-zone accuracy heatmap** (optional but strong)
> Map of India with zones colored by classification accuracy. Shows where the model struggles geographically. Ecologically interesting and visually compelling.

If results are neutral or negative, frame it honestly: *"Geotemporal features do not provide discriminative signal beyond visual features at the current dataset scale, suggesting that zone-level granularity may be insufficient — finer habitat-type encoding or larger per-zone sample sizes may be required."* That's a real finding.

---

## Section 7 — Discussion (~0.75 page)

Three paragraphs, one per structural constraint identified:

**Spatial collapse:** CA addresses it effectively and cheaply. The directional structure of CA (independent row/column processing) is naturally suited to butterfly morphology — horizontal banding vs. vertical vein streaking. Connect to the thrips paper's finding that attention to body sub-regions drives performance.

**Texture loss:** MLFI in principle addresses this but has an optimization pathology at current epoch budgets. The fix (learning rate warmup for newly initialized branches) is a concrete future direction, not a failure.

**Domain shift:** Your most generalizable finding. The head-only result is not just a butterfly-dataset problem — it applies to any biological or medical imaging domain far from ImageNet. Connect to Amarathunga et al. (2022) thrips paper's similar observation about needing domain-specific features.

---

## Section 8 — Conclusion (~0.5 page)

Four sentences per contribution:
- IndoLepAtlas fills a documented gap in the insect classification literature
- Geotemporal info + larval stage images + host plant images
- Systematic ablation establishes that CE + Balanced sampling + Coordinate Attention is the optimal baseline configuration
- ImageNet features are actively misaligned with Lepidoptera morphology — end-to-end fine-tuning is mandatory
- Geotemporal fusion is demonstrated as a viable direction [results sentence]


---

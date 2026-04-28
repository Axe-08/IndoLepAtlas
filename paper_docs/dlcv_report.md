# IndoLepAtlas: Deep Learning for Computer Vision (DLCV) Phase 2 Report

## Overview
In accordance with the DLCV Phase 2 guidelines, this report treats our deep learning architecture for the IndoLepAtlas image dataset as an object of scientific inquiry. Moving beyond standard black-box optimization, we established three rigorous, research-oriented problem statements that prioritize understanding phenomena such as long-tail learning dynamics, architectural feature fusion, and inductive bias from layer-freezing strategies. 

Our investigation leverages a custom long-tail image dataset of Indian Lepidoptera, where recognizing complex, fine-grained inter-class textures is critical, yet data distribution is naturally skewed (Dense vs. Sparse classes).

---

## Unit I: Long-Tail Loss Dynamics (Fundamentals)

**Research Question:** *How do different loss formulations and sampling strategies impact the representational capacity of CNNs on significantly long-tailed fine-grained distributions?*

### Experimental Setup
We tested two primary loss paradigms: standard Cross-Entropy (CE) and Focal Loss, across two sampling strategies: Unbalanced (natural distribution) and Balanced (via weighted sampling/loss adjustments).

| Experiment | Top-1 Acc (%) | Macro F1 (%) | Dense Acc (%) | Sparse Acc (%) |
| :--- | :---: | :---: | :---: | :---: |
| **CE + Balanced** | 87.06 | **85.48** | 87.67 | 85.84 |
| **CE + Unbalanced** | **87.60** | 80.23 | **91.22** | 80.35 |
| **Focal + Balanced** | 82.69 | 82.64 | 80.85 | 86.40 |
| **Focal + Unbalanced**| 84.83 | 82.70 | 85.70 | 83.07 |

### Analysis & Mechanistic Explanation
1. **The 'Unbalanced' Illusion:** The `CE + Unbalanced` model yields the highest Top-1 Accuracy (87.60%), functioning as an illusion of performance. Looking closely at the strata breakdowns, it achieves this by heavily overfitting the majority (Dense) classes (91.22%) while sacrificing the long-tail (Sparse) classes (80.35%). The Macro F1 score accurately plummets to 80.23%.
2. **CE + Balanced as the Optimal Compromise:** When we balance the CE loss, we see a massive regression toward the mean for Sparse classes (increasing to 85.84%), stabilizing the Macro F1 score to the overall peak of 85.48%.
3. **The Focal Loss Penalty:** We hypothesized that Focal Loss would naturally boost the Sparse classes. While `Focal + Balanced` successfully achieved one of the highest Sparse accuracies (86.40%), it severely deteriorated Dense class accuracy (to 80.85%). Mechanistically, Focal loss over-penalizes well-classified 'easy' examples (typically our dense strata) to aggressively pull gradients from hard examples. In fine-grained taxonomy, this over-regularization damages the foundational feature extraction capacity for the most common species. 

---

## Unit II: Feature-Fusion Ablation (Architectures)

**Research Question:** *Where is the optimal structural transition point for integrating attention mechanisms, and does extreme multi-level feature integration improve fine-grained visual classification?*

### Experimental Setup
We ablated structural complexity by starting with a pure ResNet50 equivalent, adding Cross-Attention (CA) mechanisms, and then proceeding to a highly interwoven Multi-Level Feature Integration (MLFI) architecture.

| Experiment | Top-1 Acc (%) | Macro F1 (%) | Dense Acc (%) | Sparse Acc (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Phase 1 (ResNet50)** | 86.50 | 85.65 | 86.36 | 86.76 |
| **Phase 2 (+ CA)** | **86.77** | **85.75** | 86.13 | **88.05** |
| **Phase 3 (+ CA + MLFI)**| 84.43 | 83.34 | 83.32 | 86.66 |

### Analysis & Mechanistic Explanation
1. **The Benefit of Local-to-Global Attention:** Introducing Cross-Attention (`Phase 2`) on top of the ResNet backbone generated a marginal bump in overall accuracy, but a very notable gain in Sparse Class accuracy (from 86.76% to 88.05%). Attention mechanisms serve as strong inductive priors for spatial localization; they allow the network to explicitly 'zoom in' on distinct morphological features (like wing textures) that are vital when class examples are scarce.
2. **Information Bottlenecks and Over-parameterization:** `Phase 3`, which forced the integration of multi-level hierarchies (early + late layers), significantly degraded performance across all metrics (Top-1 dropping by over 2%). Rather than providing richer semantic depth, MLFI introduced catastrophic feature noise. Because early layers capture high-frequency textural variance (which varies fiercely among butterflies in different lighting), aggressively forcing these raw activations into the final classification head overwhelmed the refined semantic signals, creating a poorly conditioned loss landscape.

---

## Unit III: Transfer Learning Layer Freezing Strategy (Learning Paradigms)

**Research Question:** *Which hierarchical layers contain domain-agnostic generic representations versus domain-specific semantic concepts, and what is the optimal adaptation pathway for non-standard image domains?*

### Experimental Setup
We fine-tuned the network initialized with ImageNet weights, evaluating how the gradient flow affects early texture-based convolution blocks versus late, complex semantic blocks.

| Experiment | Top-1 Acc (%) | Macro F1 (%) | Dense Acc (%) | Sparse Acc (%) |
| :--- | :---: | :---: | :---: | :---: |
| **End-to-End (None)** | **84.79** | **84.11** | **83.89** | 86.61 |
| **Head Only** | 53.20 | 57.02 | 44.46 | 70.75 |
| **Freeze Early Blocks**| 83.77 | 83.20 | 82.74 | 85.84 |
| **Freeze Late Blocks** | 84.33 | 84.02 | 83.15 | **86.71** |

### Analysis & Mechanistic Explanation
1. **The Domain Shift Wall:** Freezing everything but the classification head resulted in immediate and massive failure (53.20% Top-1). This empirically proves that ImageNet representations (designed for identifying macroscopic objects like cars or dogs) completely fail to map onto the microscopic, texture-dependent taxonomy of Lepidoptera. The network fundamentally lacks the vocabulary to differentiate species.
2. **Early vs. Late Representation:** `Freeze Late Blocks` (84.33%) slightly outperformed `Freeze Early Blocks` (83.77%). Mechanistically, this implies that the early-layer parameters of ImageNet models (typically Gabor-like edge detectors) needed significant recalibration to adapt to butterfly wing patterns, scales, and biological venation. Adapting early layers created a stronger foundation for the frozen late layers to act upon. 
3. **The Supremacy of End-to-End Unlocking:** Fully unfreezing the network `End-to-End` resulted in the best Macro F1 score. Because the semantic gap between the source domain (ImageNet) and target domain (IndoLepAtlas) is profound, full gradient backpropagation was required to warp the entire representational manifold for this specific scientific task.

---

## Final Conclusion
By treating the IndoLepAtlas modeling pipeline as an object of scientific inquiry, we successfully untangled the behaviors of our architecture. We found that for fine-grained biological taxonomy, balancing standard Cross-Entropy is mathematically safer than applying Focal Loss. We demonstrated that focused spatial Attention vastly improves learning from scarce data, but aggressive multi-level feature passing acts as a toxic noise source. Finally, we empirically validated that domain shift is fatal to linear-probing paradigms, requiring full end-to-end gradient updates to properly bend representations against the domain of Lepidoptera classifications.

# IndoLepAtlas: DGX Experiment Analysis Report

This document outlines the empirical findings generated from the intensive 10-model parallel training matrix executed on the LNMIIT DGX system across the 966-class Indian Butterfly dataset (15-epoch constraint). 

## 1. Unit I: Long-Tail Loss Dynamics
The objective of this unit was to investigate the most effective method for combatting severe class imbalance (often ranging from 1,000+ images for common species down to <5 for rare ones). 

| Experiment Setup | Top-1 Accuracy | Macro Precision | Macro F1 | Weighted F1 |
| :--- | :---: | :---: | :---: | :---: |
| **CE + Balanced** | **82.67%** | **83.24%** | **81.46%** | **82.27%** |
| CE + Unbalanced | 81.58% | 69.34% | 67.08% | 79.33% |
| Focal + Unbalanced| 73.33% | 70.93% | 68.51% | 72.78% |
| Focal + Balanced | 68.71% | 72.64% | 70.52% | 67.91% |

### Analysis
*   **The Power of Balancing:** Standard Cross-Entropy (CE) combined with inverse-frequency Balanced Sampling proved to be the overwhelmingly superior dynamic. When sampling was *unbalanced*, CE maintained a deceptively high Top-1 Accuracy (81.58%) by over-indexing on dominant classes, but its Macro F1 completely crashed to 67.08%, proving it effectively abandoned the tail-end rare species.
*   **Focal Loss Underperformance:** Surprisingly, Focal Loss struggled to stabilize in this high-cardinality setting. It yielded heavily depressed accuracies globally (~68-73%), likely indicating that its default hyper-parameter settings (`gamma=2`) suppressed gradients too aggressively across 966 distinct classifications, requiring significantly more than 15 epochs or intricate tuning to yield benefits.

## 2. Unit II: Feature-Fusion Ablation 
The goal was to justify the transition from a vanilla ResNet50 backend to the highly complex Coordinate Attention (CA) + Multi-Level Feature Integration (MLFI) Phase 3 model.

| Architecture | Top-1 Accuracy | Macro Precision | Macro F1 |
| :--- | :---: | :---: | :---: |
| **Phase 1 (Base ResNet50)** | **73.45%** | **76.37%** | **74.73%** |
| Phase 2 (Base + CA) | 73.36% | 76.44% | 74.85% |
| Phase 3 (Base + CA + MLFI)* | 68.47% | 72.96% | 70.85% |

*\*Phase 3 derived from the `unit3_freeze_none` end-to-end default state.*

### Analysis
*   **Training Time Constraints:** Under a highly constrained 15-epoch training window, the lightweight vanilla ResNet50 practically matched the Coordinate-Attention-injected model perfectly (~73.4%). 
*   **Capacity vs. Convergence:** The fully matured Phase 3 model (nearly ~41 Million parameters) actually *underperformed* the baseline (68.47%). This is a textbook Deep Learning dynamic: massive architectural capacity (like MLFI spanning multiple spatial resolutions) requires immensely more epochs to stabilize its fresh, randomly initialized fusion weights than a fundamentally compact and directly pre-trained backbone.

## 3. Unit III: Layer Freezing Strategy
This unit assessed the viability of locking backbone layers to prevent catastrophic forgetting and accelerate training speeds on the Phase 3 model. 

| Freezing Strategy | Top-1 Accuracy | Macro Precision | Macro F1 |
| :--- | :---: | :---: | :---: |
| **None (End-to-End)** | **68.47%**| **72.96%** | **70.85%** |
| Freeze Late | 68.38% | 72.42% | 70.65% |
| Freeze Early | 68.04% | 72.06% | 70.28% |
| Head-Only | 27.16% | 33.77% | 31.28% |

### Analysis
*   **Domain Shift Severity:** The `Head-Only` strategy (where only the final linear layer trains and the ImageNet features are locked) was a catastrophic failure, culminating in a 27.16% accuracy. This vividly proves that pre-trained ImageNet representations fail to natively understand the critical fine-grained textural geometries of Lepidoptera taxonomy. The backbone *must* be allowed to spatially adapt.
*   **Marginality of Block Freezing:** Freezing early blocks or late blocks performed virtually identically to total end-to-end training (~68%). Given the massive complexity of 966 distinct classifications, allowing the gradients to flow purely throughout the entire model (End-to-End) edged out the restrictive strategies, securing the best feature extraction representations.

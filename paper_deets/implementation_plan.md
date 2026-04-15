# Indian Butterfly Classifier: Architecture & Execution Plan

This document details what has been built thus far for the Indian Butterfly Classifier, how the components interact, why those architectural choices were made, and outlines the planned updates for training execution and monitoring.

## 1. What Was Built (The Architecture)

We built a modular, state-of-the-art fine-grained visual classification model optimized for a 208-class butterfly dataset, designed specifically for a single internal DGX V100 GPU structure. The following architectural components were established:

1.  **ConvNeXt-Tiny Backbone (`models/backbone.py`)**: A highly capable, modern CNN that matches Vision Transformer (ViT) performance while maintaining efficiency.
2.  **Coordinate Attention Modules (`models/coord_attention.py`)**: Attention mechanisms capable of distinguishing features across height and width independently. 
3.  **Multi-Level Feature Interaction (`models/mlfi.py`)**: An interaction module that extracts activation maps from *all four* stages of the ConvNeXt network using Max Pooling instead of just relying on the terminal fully-connected layer. 
4.  **Geotemporal Feature Fusion (`models/geotemporal.py`)**: A late-fusion enhancement that processes Biogeographic Zone embeddings and Cyclic Month embeddings, appending them just prior to final classification.
5.  **Focal Loss Implementation (`losses.py`)**: A specialized loss function that explicitly prioritizes hard-to-classify samples and de-prioritizes commonly identified species to combat class imbalance.
6.  **Training & Evaluation Ecosystem (`train.py` & `evaluate.py`)**: Full PyTorch pipelines complete with Mixed Precision (AMP), TensorBoard metrics, dynamic learning rate annealing, and evaluation pipelines that compute accurate sub-stratum logic (sparse < 30 samples vs dense).

## 2. How the Components Interoperate

1.  Input images are passed through the **ConvNeXt-Tiny** backbone. Based on the selected "Phase Configuration" (1 through 5), the model dynamically attaches enhancements.
2.  During the forward pass, **Coordinate Attention** adjusts the focus of the receptive fields sequentially after each major spatial down-sampling block in the ConvNeXt. 
3.  Instead of destroying low-level features (which contain texture and wing banding patterns), the **MLFI Module** aggressively max-pools and flattens out responses from varying depths of the backbone (at an exact 2:4:1:8 ratio). 
4.  The network optionally consumes integer coordinates mapping to location and season, resolving the **Geotemporal Encoded** vector, to help discriminate species that look identical but do not share geographical/temporal bounds.
5.  All the logic iterates under a `FocalLoss` boundary guided by AdamW, aggressively scaling through GPU operations. 

## 3. Why These Specific Choices

-   **ConvNeXt-Tiny > ViT for Resource Constrained Contexts**: ViTs demand immense swaths of clean data to map relations (self-attention lacks inductive bias compared to CNNs). ConvNeXt has strong spatial biases innate to CNNs while mimicking transformer block layout, resulting in faster and superior convergence on an un-pre-trained domain. 
-   **Why MLFI & Coordinate Attention?**: Butterflies share macro-morphology perfectly. The actual determinable properties (like a specific submarginal wing spot) exist at lower texture levels that standard networks typically wash away deep in the network. Co-ordinate Attention and MLFI forcibly preserve the *where* and *what* throughout processing. 
-   **Why 208 Classes (Filtered)?**: The decision to exclude early-stage forms (like caterpillars and eggs) enforces model purity so that the CNN receptive fields aren't confused by violently dissimilar morphologies associated with the same taxonomical label. 

---

## User Review Required

The following adjustments have been proposed to finalize the execution framework based on your request. Please review the planned implementation below.

### Planned Changes (Pending Approval)

#### 1. Implement Execution Driver (`run_all.sh`)
**Goal:** Create a single master terminal executable. 
-   **Function:** Run `data_audit.py` to assert data integrity and formulate metadata -> Initiate `train.py` with the baseline configuration (Phase 1) progressing through enhancements -> Finalize execution through `evaluate.py`.
-   **Toggles:** Will allow parameter overrides via bash flags to dictate precisely which model architecture "Phase" should sequentially run.

#### 2. Progress Bar & Real-Time Monitoring Integration 
**Goal:** Guarantee complete observability inside terminal systems via `tqdm` and local persistent textual logs.
-   **Modify `train.py`:** Wrap all DataLoaders in `tqdm()` to grant live GPU progression pacing.
-   **Modify `evaluate.py`:** Wrap validation DataLoaders equally in `tqdm()`. 
-   **Create Logging Hooks (`progress.log`)**: Implement custom Python rotational I/O logging extending into both training and evaluation. It will append:
    *   **Epoch / Batch Status**
    *   **Current Iteration Loss & Exponential Moving Average (EMA) Total Loss**
    *   **Batch evaluated species count vs Total species footprint evaluated**

## Open Questions
- Do you want `run_all.sh` to conditionally invoke *all* phases in a singular run sequentially (i.e. train Phase 1 Baseline to completion -> Train Phase 2 -> Train Phase 3 etc.) or would you prefer it takes a flag to execute the entire ecosystem for a *singular* desired architecture configuration at a time?

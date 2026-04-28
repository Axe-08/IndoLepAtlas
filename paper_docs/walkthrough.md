# Evaluation & Pipeline Monitoring Infrastructure

I've finalized your training pipeline by tying everything together into a streamlined and highly observable package so that you can trace what the model is learning precisely and automatically. 

## 1. Unified Execution System (`run_all.sh`)
I've produced a single bash script that acts as the deployment commander for the entire pipeline.
*   **Optimal Strategy:** Because the dataset is small and heavily scattered, the script is intrinsically targeted to run **Phase 3 (ConvNeXt + CA + MLFI)** utilizing **Focal Loss**, without the added geo metadata. This acts as the strongest visual-focused ceiling possible against overfitting.
*   **Sequence:** Automatically runs the `metadata_filtered.csv` audit -> Executes training to completion under the optimized target -> Extracts the best weights dynamically and automatically passes them to `evaluate.py`. 

## 2. Advanced Performance Logging (`progress.log`) 
Instead of relying on the console buffer, training logic is now exported per-iteration directly via Python's `logging` module so that terminal disconnects do not destroy your evaluation bounds:
*   Displays the active Epoch, the `tqdm` processing rate, and precisely which sub-batch is executing.
*   Shows raw `curr_loss` against an Exponential Moving Average (`EMA_loss`) to give a smoother trend signal over how training is actually behaving outside of the jitter.
*   Per-batch, displays precisely **how many distinct species have been fully evaluated** vs the total evaluated across all loops. 

## 3. Metrics Generation & Graphing (`metrics.csv` & `matplotlib`)
*   **Evaluation Upgrades:** Integrated calculating and returning `Precision` into `evaluate.py` alongside the existing Top-1/Top-5 and F1 metrics.
*   **History Ledger:** `train.py` continuously drafts a `metrics.csv` containing the state of the Train Loss, Validation Loss, Accuracy, F1, Precision, and current Learning Rate. 
*   **Plotting Output:** At the close of each evaluated validation epoch, `matplotlib` natively renders `.png` files visualising overlapping curves:
    *   `metrics_curve.png`: Graphing Train Accuracy against Validation Accuracy alongside Validation Precision and Macro-F1.
    *   `loss_curve.png`: Graphical curve comparing raw Train Loss against Validation Loss.

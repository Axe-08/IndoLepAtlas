#!/usr/bin/env python3
"""
IndoLepAtlas — Report Asset Generator
=====================================
Reads all experiment run data and generates publication-quality plots
organized into a clean report_assets/ directory.
"""

import os
import json
import csv
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from pathlib import Path
import shutil

# ─── Configuration ───────────────────────────────────────────────────────────
RUNS_DIR = Path("indolep_model/runs")
OUTPUT_DIR = Path("report_assets")

# The 11 canonical runs from the 30-40 epoch convergence session
LATEST_RUNS = {
    # Unit I: Long-Tail Loss Dynamics
    "unit1_ce_bal":       "unit1_ce_bal_20260426_154859",
    "unit1_ce_unbal":     "unit1_ce_unbal_20260426_154904",
    "unit1_focal_bal":    "unit1_focal_bal_20260426_154909",
    "unit1_focal_unbal":  "unit1_focal_unbal_20260426_154915",
    # Unit II: Feature-Fusion Ablation
    "unit2_phase1":       "unit2_phase1_20260426_154859",
    "unit2_phase2":       "unit2_phase2_20260426_154904",
    "unit2_phase3":       "unit2_phase3_20260426_154909",
    # Unit III: Layer Freezing Strategy
    "unit3_freeze_none":  "unit3_freeze_none_20260426_154859",
    "unit3_head_only":    "unit3_head_only_20260426_154905",
    "unit3_freeze_early": "unit3_freeze_early_20260426_154909",
    "unit3_freeze_late":  "unit3_freeze_late_20260426_154859",
}

# Display names for plots
DISPLAY_NAMES = {
    "unit1_ce_bal":       "CE + Balanced",
    "unit1_ce_unbal":     "CE + Unbalanced",
    "unit1_focal_bal":    "Focal + Balanced",
    "unit1_focal_unbal":  "Focal + Unbalanced",
    "unit2_phase1":       "Phase 1\n(ResNet50)",
    "unit2_phase2":       "Phase 2\n(+ CA)",
    "unit2_phase3":       "Phase 3\n(+ CA + MLFI)",
    "unit3_freeze_none":  "End-to-End\n(None)",
    "unit3_head_only":    "Head Only",
    "unit3_freeze_early": "Freeze Early\nBlocks",
    "unit3_freeze_late":  "Freeze Late\nBlocks",
}

# Color palettes
UNIT1_COLORS = ["#2ecc71", "#27ae60", "#e74c3c", "#c0392b"]
UNIT2_COLORS = ["#3498db", "#9b59b6", "#e67e22"]
UNIT3_COLORS = ["#1abc9c", "#e74c3c", "#f39c12", "#9b59b6"]

# ─── Styling ─────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'axes.labelsize': 12,
    'figure.facecolor': 'white',
    'axes.facecolor': '#fafafa',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'legend.fontsize': 9,
    'legend.framealpha': 0.9,
})


def load_metrics(run_key):
    """Load per-epoch metrics.csv for a run."""
    path = RUNS_DIR / LATEST_RUNS[run_key] / "metrics.csv"
    return pd.read_csv(path)


def load_eval_summary(run_key):
    """Load final test evaluation summary.json for a run."""
    path = RUNS_DIR / LATEST_RUNS[run_key] / "eval_results" / "summary.json"
    with open(path) as f:
        return json.load(f)


def load_stratum(run_key):
    """Load per-stratum breakdown."""
    path = RUNS_DIR / LATEST_RUNS[run_key] / "eval_results" / "per_stratum.json"
    with open(path) as f:
        return json.load(f)


def load_confusion_pairs(run_key):
    """Load top confused species pairs."""
    path = RUNS_DIR / LATEST_RUNS[run_key] / "eval_results" / "confusion_pairs.csv"
    return pd.read_csv(path)


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT GENERATORS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_unit1_bar_comparison(out_dir):
    """Unit I: Grouped bar chart comparing all 4 loss configurations."""
    keys = ["unit1_ce_bal", "unit1_ce_unbal", "unit1_focal_bal", "unit1_focal_unbal"]
    labels = [DISPLAY_NAMES[k] for k in keys]
    
    metrics = {}
    for k in keys:
        s = load_eval_summary(k)
        metrics[k] = s

    metric_names = ["top1_accuracy", "top5_accuracy", "macro_precision", "macro_f1"]
    metric_labels = ["Top-1 Acc", "Top-5 Acc", "Macro Prec", "Macro F1"]
    
    x = np.arange(len(metric_labels))
    width = 0.18
    
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (k, color) in enumerate(zip(keys, UNIT1_COLORS)):
        vals = [metrics[k][m] * 100 for m in metric_names]
        bars = ax.bar(x + i * width, vals, width, label=DISPLAY_NAMES[k], color=color, edgecolor='white', linewidth=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    f'{val:.1f}', ha='center', va='bottom', fontsize=7.5, fontweight='bold')
    
    ax.set_ylabel('Score (%)')
    ax.set_title('Unit I: Long-Tail Loss Dynamics — Test Set Comparison')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 105)
    ax.legend(loc='lower right')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    
    plt.tight_layout()
    plt.savefig(out_dir / "unit1_bar_comparison.png", dpi=200, bbox_inches='tight')
    plt.close()
    print("  ✓ unit1_bar_comparison.png")


def plot_unit1_training_curves(out_dir):
    """Unit I: Validation accuracy and loss curves over epochs."""
    keys = ["unit1_ce_bal", "unit1_ce_unbal", "unit1_focal_bal", "unit1_focal_unbal"]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for k, color in zip(keys, UNIT1_COLORS):
        df = load_metrics(k)
        axes[0].plot(df['epoch'], df['val_acc'] * 100, label=DISPLAY_NAMES[k], color=color, linewidth=1.8)
        axes[1].plot(df['epoch'], df['val_loss'], label=DISPLAY_NAMES[k], color=color, linewidth=1.8)
    
    axes[0].set_title('Unit I: Validation Accuracy Over Epochs')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Val Accuracy (%)')
    axes[0].yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    axes[0].legend()
    
    axes[1].set_title('Unit I: Validation Loss Over Epochs')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Val Loss')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(out_dir / "unit1_training_curves.png", dpi=200, bbox_inches='tight')
    plt.close()
    print("  ✓ unit1_training_curves.png")


def plot_unit2_bar_comparison(out_dir):
    """Unit II: Feature fusion ablation bar chart."""
    keys = ["unit2_phase1", "unit2_phase2", "unit2_phase3"]
    labels = [DISPLAY_NAMES[k] for k in keys]
    
    metric_names = ["top1_accuracy", "top5_accuracy", "macro_precision", "macro_f1"]
    metric_labels = ["Top-1 Acc", "Top-5 Acc", "Macro Prec", "Macro F1"]
    
    x = np.arange(len(metric_labels))
    width = 0.22
    
    fig, ax = plt.subplots(figsize=(9, 6))
    for i, (k, color) in enumerate(zip(keys, UNIT2_COLORS)):
        s = load_eval_summary(k)
        vals = [s[m] * 100 for m in metric_names]
        bars = ax.bar(x + i * width, vals, width, label=DISPLAY_NAMES[k].replace('\n', ' '), 
                      color=color, edgecolor='white', linewidth=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax.set_ylabel('Score (%)')
    ax.set_title('Unit II: Feature-Fusion Ablation — Test Set Comparison')
    ax.set_xticks(x + width)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 105)
    ax.legend(loc='lower right')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    
    plt.tight_layout()
    plt.savefig(out_dir / "unit2_bar_comparison.png", dpi=200, bbox_inches='tight')
    plt.close()
    print("  ✓ unit2_bar_comparison.png")


def plot_unit2_training_curves(out_dir):
    """Unit II: Training curves for all 3 phases."""
    keys = ["unit2_phase1", "unit2_phase2", "unit2_phase3"]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for k, color in zip(keys, UNIT2_COLORS):
        df = load_metrics(k)
        axes[0].plot(df['epoch'], df['val_acc'] * 100, label=DISPLAY_NAMES[k].replace('\n', ' '), 
                     color=color, linewidth=1.8)
        axes[1].plot(df['epoch'], df['val_loss'], label=DISPLAY_NAMES[k].replace('\n', ' '), 
                     color=color, linewidth=1.8)
    
    axes[0].set_title('Unit II: Validation Accuracy Over Epochs')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Val Accuracy (%)')
    axes[0].yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    axes[0].legend()
    
    axes[1].set_title('Unit II: Validation Loss Over Epochs')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Val Loss')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(out_dir / "unit2_training_curves.png", dpi=200, bbox_inches='tight')
    plt.close()
    print("  ✓ unit2_training_curves.png")


def plot_unit3_bar_comparison(out_dir):
    """Unit III: Freezing strategy bar chart."""
    keys = ["unit3_freeze_none", "unit3_head_only", "unit3_freeze_early", "unit3_freeze_late"]
    
    metric_names = ["top1_accuracy", "top5_accuracy", "macro_precision", "macro_f1"]
    metric_labels = ["Top-1 Acc", "Top-5 Acc", "Macro Prec", "Macro F1"]
    
    x = np.arange(len(metric_labels))
    width = 0.18
    
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (k, color) in enumerate(zip(keys, UNIT3_COLORS)):
        s = load_eval_summary(k)
        vals = [s[m] * 100 for m in metric_names]
        bars = ax.bar(x + i * width, vals, width, label=DISPLAY_NAMES[k].replace('\n', ' '), 
                      color=color, edgecolor='white', linewidth=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=7.5, fontweight='bold')
    
    ax.set_ylabel('Score (%)')
    ax.set_title('Unit III: Layer Freezing Strategy — Test Set Comparison')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 105)
    ax.legend(loc='lower right')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    
    plt.tight_layout()
    plt.savefig(out_dir / "unit3_bar_comparison.png", dpi=200, bbox_inches='tight')
    plt.close()
    print("  ✓ unit3_bar_comparison.png")


def plot_unit3_training_curves(out_dir):
    """Unit III: Training curves for all freezing strategies."""
    keys = ["unit3_freeze_none", "unit3_head_only", "unit3_freeze_early", "unit3_freeze_late"]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for k, color in zip(keys, UNIT3_COLORS):
        df = load_metrics(k)
        axes[0].plot(df['epoch'], df['val_acc'] * 100, label=DISPLAY_NAMES[k].replace('\n', ' '), 
                     color=color, linewidth=1.8)
        axes[1].plot(df['epoch'], df['val_loss'], label=DISPLAY_NAMES[k].replace('\n', ' '), 
                     color=color, linewidth=1.8)
    
    axes[0].set_title('Unit III: Validation Accuracy Over Epochs')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Val Accuracy (%)')
    axes[0].yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    axes[0].legend()
    
    axes[1].set_title('Unit III: Validation Loss Over Epochs')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Val Loss')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(out_dir / "unit3_training_curves.png", dpi=200, bbox_inches='tight')
    plt.close()
    print("  ✓ unit3_training_curves.png")


def plot_stratum_comparison(out_dir):
    """Cross-unit: Dense vs Sparse accuracy for all 11 experiments."""
    all_keys = list(LATEST_RUNS.keys())
    
    dense_accs = []
    sparse_accs = []
    labels = []
    
    for k in all_keys:
        s = load_stratum(k)
        dense_accs.append(s['dense']['accuracy'] * 100)
        sparse_accs.append(s['sparse']['accuracy'] * 100)
        labels.append(DISPLAY_NAMES[k].replace('\n', ' '))
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(14, 6))
    bars1 = ax.bar(x - width/2, dense_accs, width, label='Dense Classes (≥50 imgs)', color='#3498db', edgecolor='white')
    bars2 = ax.bar(x + width/2, sparse_accs, width, label='Sparse Classes (<50 imgs)', color='#e74c3c', edgecolor='white')
    
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Dense vs Sparse Class Accuracy Across All Experiments')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=8)
    ax.legend()
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    
    plt.tight_layout()
    plt.savefig(out_dir / "stratum_comparison.png", dpi=200, bbox_inches='tight')
    plt.close()
    print("  ✓ stratum_comparison.png")


def plot_overall_radar(out_dir):
    """Radar chart comparing the best model from each unit."""
    best_keys = {
        "Unit I Best\n(CE+Bal)": "unit1_ce_bal",
        "Unit II Best\n(Phase 1)": "unit2_phase1",
        "Unit III Best\n(End-to-End)": "unit3_freeze_none",
    }
    
    metric_names = ["top1_accuracy", "top5_accuracy", "macro_precision", "macro_f1", "weighted_f1"]
    metric_labels = ["Top-1 Acc", "Top-5 Acc", "Macro Prec", "Macro F1", "Weighted F1"]
    
    angles = np.linspace(0, 2 * np.pi, len(metric_labels), endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors = ["#2ecc71", "#3498db", "#e67e22"]
    
    for (label, k), color in zip(best_keys.items(), colors):
        s = load_eval_summary(k)
        vals = [s[m] * 100 for m in metric_names]
        vals += vals[:1]
        ax.plot(angles, vals, 'o-', linewidth=2, label=label.replace('\n', ' '), color=color)
        ax.fill(angles, vals, alpha=0.1, color=color)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_title('Best Model Per Unit — Performance Radar', pad=20)
    ax.legend(loc='lower right', bbox_to_anchor=(1.3, 0))
    
    plt.tight_layout()
    plt.savefig(out_dir / "best_models_radar.png", dpi=200, bbox_inches='tight')
    plt.close()
    print("  ✓ best_models_radar.png")


def generate_master_table(out_dir):
    """Generate a clean CSV with only the 11 latest runs and human-readable labels."""
    rows = []
    for key in LATEST_RUNS:
        s = load_eval_summary(key)
        st = load_stratum(key)
        unit = key.split("_")[0]  # unit1, unit2, unit3
        rows.append({
            "Unit": unit.replace("unit", "Unit "),
            "Experiment": DISPLAY_NAMES[key].replace('\n', ' '),
            "Run": LATEST_RUNS[key],
            "Epochs": len(load_metrics(key)),
            "Top-1 Acc (%)": round(s["top1_accuracy"] * 100, 2),
            "Top-5 Acc (%)": round(s["top5_accuracy"] * 100, 2),
            "Macro Precision (%)": round(s["macro_precision"] * 100, 2),
            "Macro F1 (%)": round(s["macro_f1"] * 100, 2),
            "Weighted F1 (%)": round(s["weighted_f1"] * 100, 2),
            "Dense Acc (%)": round(st["dense"]["accuracy"] * 100, 2),
            "Sparse Acc (%)": round(st["sparse"]["accuracy"] * 100, 2),
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "master_results.csv", index=False)
    print("  ✓ master_results.csv")
    return df


def copy_confusion_heatmaps(out_dir):
    """Copy confusion heatmaps from each run into the output directory."""
    heatmap_dir = out_dir / "confusion_heatmaps"
    heatmap_dir.mkdir(exist_ok=True)
    
    for key, run_dir in LATEST_RUNS.items():
        src = RUNS_DIR / run_dir / "eval_results" / "confusion_heatmap.png"
        if src.exists():
            shutil.copy2(src, heatmap_dir / f"{key}_confusion.png")
    
    print(f"  ✓ Copied {len(LATEST_RUNS)} confusion heatmaps")


def copy_training_curves(out_dir):
    """Copy the auto-generated loss/metrics curves from each run."""
    curves_dir = out_dir / "per_run_curves"
    curves_dir.mkdir(exist_ok=True)
    
    for key, run_dir in LATEST_RUNS.items():
        for fname in ["loss_curve.png", "metrics_curve.png"]:
            src = RUNS_DIR / run_dir / fname
            if src.exists():
                shutil.copy2(src, curves_dir / f"{key}_{fname}")
    
    print("  ✓ Copied per-run training curves")


def copy_metrics_csvs(out_dir):
    """Copy per-epoch metrics CSVs for reference."""
    data_dir = out_dir / "raw_metrics"
    data_dir.mkdir(exist_ok=True)
    
    for key, run_dir in LATEST_RUNS.items():
        src = RUNS_DIR / run_dir / "metrics.csv"
        if src.exists():
            shutil.copy2(src, data_dir / f"{key}_metrics.csv")
    
    print("  ✓ Copied per-run metrics CSVs")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  IndoLepAtlas Report Asset Generator")
    print("=" * 60)
    
    # Create output structure
    plots_dir = OUTPUT_DIR / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Generate publication plots
    print("\n📊 Generating plots...")
    plot_unit1_bar_comparison(plots_dir)
    plot_unit1_training_curves(plots_dir)
    plot_unit2_bar_comparison(plots_dir)
    plot_unit2_training_curves(plots_dir)
    plot_unit3_bar_comparison(plots_dir)
    plot_unit3_training_curves(plots_dir)
    plot_stratum_comparison(plots_dir)
    plot_overall_radar(plots_dir)
    
    # 2. Generate master results table
    print("\n📋 Generating master results table...")
    df = generate_master_table(OUTPUT_DIR)
    
    # 3. Copy raw data
    print("\n📁 Organizing raw data...")
    copy_confusion_heatmaps(OUTPUT_DIR)
    copy_training_curves(OUTPUT_DIR)
    copy_metrics_csvs(OUTPUT_DIR)
    
    # 4. Print summary
    print("\n" + "=" * 60)
    print("  FINAL RESULTS SUMMARY")
    print("=" * 60)
    print(df.to_string(index=False))
    
    print(f"\n✅ All assets saved to: {OUTPUT_DIR.resolve()}")
    print(f"   📊 plots/              — 8 publication-quality charts")
    print(f"   📁 confusion_heatmaps/ — 11 confusion matrices")
    print(f"   📁 per_run_curves/     — Per-run loss & accuracy plots")
    print(f"   📁 raw_metrics/        — Per-epoch CSV data")
    print(f"   📋 master_results.csv  — Clean results table")


if __name__ == "__main__":
    main()

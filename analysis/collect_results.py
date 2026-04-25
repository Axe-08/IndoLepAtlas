import os
import glob
import json
import csv
import pandas as pd
import matplotlib.pyplot as plt

def collect_summaries(runs_dir, results_dir):
    """Aggregates all summary.json files into a single CSV."""
    summary_files = glob.glob(os.path.join(runs_dir, '*/eval_results/summary.json'))
    data = []
    
    for file_path in summary_files:
        run_name = file_path.split('/')[-3]
        with open(file_path, 'r') as f:
            try:
                summary = json.load(f)
                summary['run_name'] = run_name
                data.append(summary)
            except json.JSONDecodeError:
                print(f"Skipping {file_path} - Invalid JSON")

    if not data:
        print("No summary.json files found.")
        return None

    df = pd.DataFrame(data)
    
    # Ensure results directory exists
    os.makedirs(results_dir, exist_ok=True)
    
    csv_path = os.path.join(results_dir, 'summary.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved aggregated results to {csv_path}")
    return df

def plot_class_freq_vs_acc(runs_dir, results_dir):
    """Plots Class Frequency vs Accuracy from Unit I runs."""
    plots_dir = os.path.join(results_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    # Unit 1 runs to compare
    target_runs = ['unit1_ce_bal', 'unit1_ce_unbal', 'unit1_focal_bal', 'unit1_focal_unbal']
    
    plt.figure(figsize=(10, 6))
    
    for run in target_runs:
        per_stratum_file = os.path.join(runs_dir, run, 'eval_results', 'per_stratum.json')
        if not os.path.exists(per_stratum_file):
            continue
            
        with open(per_stratum_file, 'r') as f:
            data = json.load(f)
            
        categories = ['sparse', 'dense']
        accuracies = [data.get(cat, {}).get('accuracy', 0) for cat in categories]
        
        plt.plot(categories, accuracies, marker='o', label=run)
        
    plt.title('Class Frequency (Stratum) vs Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Stratum')
    plt.legend()
    plt.grid(True)
    
    out_path = os.path.join(plots_dir, 'class_freq_vs_acc.png')
    plt.savefig(out_path)
    plt.close()
    print(f"Saved plot: {out_path}")

def plot_freeze_gap(df, results_dir):
    """Plots Overfitting Gap vs Freezing Strategy from Unit III runs."""
    plots_dir = os.path.join(results_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    unit3_runs = df[df['run_name'].str.startswith('unit3')]
    
    if unit3_runs.empty:
        print("No Unit III runs found to plot freeze gap.")
        return

    # Assuming we have train_acc and val_acc in the summary or can infer overfitting gap.
    # For now, we plot val accuracy and macro-f1 against strategies
    
    plt.figure(figsize=(10, 6))
    strategies = unit3_runs['run_name'].str.replace('unit3_', '')
    val_acc = unit3_runs['top1_accuracy']
    
    plt.bar(strategies, val_acc, color='skyblue')
    plt.title('Validation Accuracy by Freezing Strategy')
    plt.ylabel('Top-1 Accuracy')
    plt.xlabel('Strategy')
    plt.grid(axis='y')
    
    out_path = os.path.join(plots_dir, 'freeze_acc.png')
    plt.savefig(out_path)
    plt.close()
    print(f"Saved plot: {out_path}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    runs_directory = os.path.join(base_dir, 'indolep_model', 'runs')
    results_directory = os.path.join(base_dir, 'results')

    print("Collecting evaluation results...")
    df_summary = collect_summaries(runs_directory, results_directory)
    
    if df_summary is not None:
        print("Generating plots...")
        plot_class_freq_vs_acc(runs_directory, results_directory)
        plot_freeze_gap(df_summary, results_directory)
        print("Analysis complete.")
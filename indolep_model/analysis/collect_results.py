import os
import json
import csv
import pandas as pd
import matplotlib.pyplot as plt

def main():
    runs_dir = '../runs'
    results_dir = '../../results'
    
    # We are inside indolep_model/analysis/
    # So runs_dir is ../runs
    if not os.path.exists(runs_dir):
        runs_dir = '../indolep_model/runs' # fallback if run from root
        
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'plots'), exist_ok=True)
    
    summary_data = []
    
    if os.path.exists(runs_dir):
        for run_name in os.listdir(runs_dir):
            run_path = os.path.join(runs_dir, run_name)
            if not os.path.isdir(run_path):
                continue
                
            summary_path = os.path.join(run_path, 'eval_results', 'summary.json')
            if os.path.exists(summary_path):
                with open(summary_path, 'r') as f:
                    data = json.load(f)
                    data['exp_name'] = run_name
                    summary_data.append(data)
    
    if summary_data:
        df = pd.DataFrame(summary_data)
        df.to_csv(os.path.join(results_dir, 'summary.csv'), index=False)
        print(f"Saved summary.csv with {len(summary_data)} runs.")
        
        # Simple placeholder plots until real runs are available
        plt.figure(figsize=(10,6))
        if 'exp_name' in df.columns and 'top1_accuracy' in df.columns:
            plt.bar(df['exp_name'], df['top1_accuracy'])
            plt.xticks(rotation=45, ha='right')
            plt.ylabel('Top-1 Accuracy')
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, 'plots', 'accuracy_comparison.png'))
            print("Saved accuracy_comparison.png")
    else:
        print("No summary.json files found yet. Run experiments and evaluate first.")

if __name__ == '__main__':
    main()

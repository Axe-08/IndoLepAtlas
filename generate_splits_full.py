#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import random
from tqdm import tqdm

def generate_splits(base_dir=".", seed=42):
    random.seed(seed)
    np.random.seed(seed)
    
    all_files = []
    
    for ds in ["butterflies", "plants"]:
        csv_path = os.path.join(base_dir, "data", ds, "metadata.csv")
        if not os.path.exists(csv_path):
            print(f"Metadata not found for {ds}")
            continue
            
        df = pd.read_csv(csv_path)
        # We assume 'filename' and 'species' columns exist
        for _, row in df.iterrows():
            all_files.append({
                "dataset": ds,
                "filename": row["filename"],
                "species": row["species"]
            })
            
    print(f"Total files collected: {len(all_files)}")
    
    # Stratified split by species
    files_by_species = {}
    for f in all_files:
        files_by_species.setdefault(f["species"], []).append(f)
        
    train_list, val_list, test_list = [], [], []
    
    for species, files in tqdm(files_by_species.items(), desc="Splitting species"):
        random.shuffle(files)
        n = len(files)
        n_train = max(1, int(0.8 * n))
        n_val = max(0, int(0.1 * n))
        
        train_list.extend(files[:n_train])
        val_list.extend(files[n_train:n_train+n_val])
        test_list.extend(files[n_train+n_val:])
        
    # Write to splits/
    splits_dir = os.path.join(base_dir, "splits")
    os.makedirs(splits_dir, exist_ok=True)
    
    for name, flist in [("train", train_list), ("val", val_list), ("test", test_list)]:
        with open(os.path.join(splits_dir, f"{name}.txt"), "w") as f:
            for item in sorted(flist, key=lambda x: x["filename"]):
                f.write(f"{item['filename']}\n")
        print(f"Split {name}: {len(flist)} files")

if __name__ == "__main__":
    generate_splits()

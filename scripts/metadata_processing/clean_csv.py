import pandas as pd
import os

def clean_metadata(dataset):
    csv_path = f"data/{dataset}/metadata.csv"
    if not os.path.exists(csv_path):
        print(f"CSV not found: {csv_path}")
        return
        
    print(f"Cleaning {dataset} metadata...")
    df = pd.read_csv(csv_path)
    initial_count = len(df)
    
    # Standardize filenames
    # Deduplicate based on raw_filename (source of truth)
    # Keep the last entry (likely the most recent EasyOCR one)
    df = df.drop_duplicates(subset=["raw_filename"], keep="last")
    
    # Reset image IDs to be sequential
    df["image_id"] = range(1, len(df) + 1)
    
    final_count = len(df)
    print(f"Dropped {initial_count - final_count} duplicates. Final count: {final_count}")
    
    df.to_csv(csv_path, index=False)

if __name__ == "__main__":
    clean_metadata("butterflies")
    clean_metadata("plants")

import os
import json
import csv
import logging
import torch
import easyocr
import pandas as pd
from PIL import Image, ImageFile
from tqdm import tqdm
import numpy as np

ImageFile.LOAD_TRUNCATED_IMAGES = True
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def refine_metadata():
    for dataset in ["butterflies", "plants"]:
        csv_path = f"data/{dataset}/metadata.csv"
        if not os.path.exists(csv_path):
            logging.error(f"Metadata CSV not found: {csv_path}")
            continue

        logging.info(f"Initializing EasyOCR reader for {dataset} (GPU enabled)...")
        reader = easyocr.Reader(["en"], gpu=True)
        
        df = pd.read_csv(csv_path)
        
        refined_count = 0
        
        for i, row in tqdm(df.iterrows(), total=len(df), desc=f"Refining {dataset} Metadata"):
            raw_rel_path = row.get("raw_filename")
            if not raw_rel_path:
                continue
            
            raw_path = os.path.join("data", raw_rel_path)
            if not os.path.exists(raw_path):
                if os.path.exists(raw_rel_path):
                    raw_path = raw_rel_path
                else:
                    continue
                
            try:
                img = Image.open(raw_path)
                w, h = img.size
                # Bottom 12% crop
                bot = img.crop((0, int(h * 0.88), w, h))
                
                bot_np = np.array(bot)
                results = reader.readtext(bot_np, detail=0)
                
                if results:
                    raw_text = " ".join(results)
                    df.at[i, "location_refined"] = raw_text
                
                refined_count += 1
                
                if refined_count % 200 == 0:
                    df.to_csv(csv_path, index=False)
                    
            except Exception as e:
                logging.error(f"Error processing {raw_path}: {e}")
                continue

        df.to_csv(csv_path, index=False)
        logging.info(f"Metadata refinement for {dataset} complete. Refined {refined_count} rows.")

if __name__ == "__main__":
    refine_metadata()

import os
import sys
import json
import logging
import easyocr
import pandas as pd
from PIL import Image, ImageFile
from tqdm import tqdm
import numpy as np
import multiprocessing as mp
import time

ImageFile.LOAD_TRUNCATED_IMAGES = True
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def process_chunk(chunk_id, gpu_id, dataset):
    logging.info(f"Worker {chunk_id} starting on GPU {gpu_id} for {dataset}")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    try:
        # EasyOCR init
        reader = easyocr.Reader(["en"], gpu=True)
    except Exception as e:
        logging.error(f"Worker {chunk_id} failed to init EasyOCR on GPU {gpu_id}: {e}")
        return
        
    chunk_file = f"data/chunks/chunk_{dataset}_{chunk_id}.csv"
    if not os.path.exists(chunk_file):
        logging.error(f"Chunk file not found: {chunk_file}")
        return
        
    df = pd.read_csv(chunk_file)
    # Ensure column exists and is object type
    if "location_refined" not in df.columns:
        df["location_refined"] = ""
    df["location_refined"] = df["location_refined"].astype(object)
    
    for i, row in tqdm(df.iterrows(), total=len(df), desc=f"GPU {gpu_id} {dataset}"):
        raw_rel_path = row.get("raw_filename")
        if not raw_rel_path: continue
        
        raw_path = os.path.join("data", raw_rel_path)
        if not os.path.exists(raw_path):
            if os.path.exists(raw_rel_path): raw_path = raw_rel_path
            else: continue
            
        try:
            img = Image.open(raw_path)
            w, h = img.size
            bot = img.crop((0, int(h * 0.88), w, h))
            bot_np = np.array(bot)
            results = reader.readtext(bot_np, detail=0)
            if results:
                df.at[i, "location_refined"] = " ".join(results)
        except Exception as e:
            continue
            
    df.to_csv(f"data/chunks/chunk_{dataset}_{chunk_id}_refined.csv", index=False)
    logging.info(f"Worker {chunk_id} finished {dataset}.")

def main():
    datasets = ["butterflies", "plants"]
    # Selecting the best GPUs based on nvidia-smi check
    AVAILABLE_GPUS = [0, 1, 3, 4, 5, 7]
    num_workers = len(AVAILABLE_GPUS)
    
    os.makedirs("data/chunks", exist_ok=True)
    
    for dataset in datasets:
        csv_path = f"data/{dataset}/metadata.csv"
        if not os.path.exists(csv_path): continue
        
        logging.info(f"Splitting {dataset} into {num_workers} chunks...")
        df = pd.read_csv(csv_path)
        
        rows_per_chunk = len(df) // num_workers + 1
        processes = []
        for i, gpu_id in enumerate(AVAILABLE_GPUS):
            start = i * rows_per_chunk
            end = min((i + 1) * rows_per_chunk, len(df))
            if start >= len(df): break
            
            chunk = df.iloc[start:end]
            chunk.to_csv(f"data/chunks/chunk_{dataset}_{i}.csv", index=False)
            
            p = mp.Process(target=process_chunk, args=(i, gpu_id, dataset))
            p.start()
            processes.append(p)
            
        for p in processes:
            p.join()
            
        logging.info(f"Merging {dataset} chunks...")
        all_refined = []
        for i in range(num_workers):
            f = f"data/chunks/chunk_{dataset}_{i}_refined.csv"
            if os.path.exists(f):
                all_refined.append(pd.read_csv(f))
        
        if all_refined:
            df_final = pd.concat(all_refined)
            df_final.to_csv(csv_path, index=False)
            logging.info(f"Refinement for {dataset} finished and merged.")

if __name__ == "__main__":
    main()

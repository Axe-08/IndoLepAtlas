import os
import re
import csv
import json
import logging
import easyocr
import pandas as pd
from PIL import Image, ImageFile
from tqdm import tqdm
import numpy as np
import multiprocessing as mp
import time
from pathlib import Path

ImageFile.LOAD_TRUNCATED_IMAGES = True
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ── CSV Column Definitions ───────────────────────────────────────────────────

BUTTERFLY_CSV_COLUMNS = [
    "image_id", "filename", "raw_filename", "species", "common_name",
    "family", "subfamily", "genus", "order", "life_stage", "sex",
    "media_code", "location", "state", "date", "credit",
    "source_url", "source", "split",
]

PLANT_CSV_COLUMNS = [
    "image_id", "filename", "raw_filename", "species", "family",
    "genus", "image_type", "media_code", "location", "state",
    "date", "credit", "butterfly_hosts", "source_url", "source", "split",
]

# ── Parsing Helpers ──────────────────────────────────────────────────────────

def extract_state(location):
    if not location: return None
    indian_states = [
        "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar",
        "Chhattisgarh", "Goa", "Gujarat", "Haryana", "Himachal Pradesh",
        "Jharkhand", "Karnataka", "Kerala", "Madhya Pradesh", "Maharashtra",
        "Manipur", "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab",
        "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura",
        "Uttar Pradesh", "Uttarakhand", "West Bengal", "Delhi",
        "Andaman and Nicobar", "Chandigarh", "Dadra", "Daman",
        "Jammu and Kashmir", "Ladakh", "Lakshadweep", "Puducherry",
    ]
    for state in indian_states:
        if state.lower() in location.lower():
            return state
    return None

def parse_top(text_list, dataset_type):
    result = {"scientific_name": None, "common_or_family": None, "media_code": None}
    if not text_list: return result
    
    full_text = " ".join(text_list)
    result["scientific_name"] = text_list[0] if text_list else None
    
    media_match = re.search(r"[Mm]edia\s*code[:\s]+(\S+)", full_text)
    if media_match:
        result["media_code"] = media_match.group(1)
        
    if len(text_list) >= 2:
        candidate = text_list[1]
        if "media code" not in candidate.lower() and candidate != result["scientific_name"]:
            result["common_or_family"] = candidate
            
    return result

def parse_bottom(text_list):
    result = {"sex": None, "location": None, "date": None, "credit": None}
    if not text_list: return result
    
    full_text = " ".join(text_list)
    
    for t in text_list:
        if "©" in t or "@" in t:
            result["credit"] = t.replace("©", "").replace("@", "").strip()
            break
            
    date_match = re.search(r"(\d{4}/\d{2}/\d{2})", full_text)
    if date_match:
        result["date"] = date_match.group(1)
        
    sex_match = re.search(r"(Male|Female|Unknown)", full_text, re.IGNORECASE)
    if sex_match:
        result["sex"] = sex_match.group(1).capitalize()
        
    loc_parts = []
    for t in text_list:
        if t == result["date"] or t == result["credit"] or (result["credit"] and result["credit"] in t):
            continue
        if result["sex"] and result["sex"].lower() in t.lower():
            t = re.sub(r"(Male|Female|Unknown)\b\.?\s*", "", t, flags=re.IGNORECASE)
        if t.strip():
            loc_parts.append(t.strip())
            
    if loc_parts:
        result["location"] = " ".join(loc_parts)
        
    return result

# ── Data Loaders ─────────────────────────────────────────────────────────────

def load_existing_butterfly_metadata(raw_dir):
    existing = {}
    if not os.path.exists(raw_dir): return existing
    for slug in os.listdir(raw_dir):
        meta_path = os.path.join(raw_dir, slug, "metadata.jsonl")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        existing[record.get("file_name", "")] = record
                    except: continue
    return existing

def load_existing_plant_metadata(raw_dir):
    existing = {}
    if not os.path.exists(raw_dir): return existing
    registry_path = os.path.join(raw_dir, "registry.json")
    registry = {}
    if os.path.exists(registry_path):
        with open(registry_path, "r") as f:
            registry = json.load(f)
    meta_path = os.path.join(raw_dir, "metadata.jsonl")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            for line in f:
                try:
                    record = json.loads(line)
                    pk = record.get("plant_key", "")
                    if pk in registry: record["_registry"] = registry[pk]
                    existing[record.get("file_name", "")] = record
                except: continue
    return existing

# ── Worker Logic ─────────────────────────────────────────────────────────────

def process_chunk(chunk_id, gpu_id, dataset, items_to_process, raw_dir):
    logger.info(f"Worker {chunk_id} starting on GPU {gpu_id}")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    try:
        reader = easyocr.Reader(["en"], gpu=True)
    except Exception as e:
        logger.error(f"Failed to init EasyOCR on GPU {gpu_id}: {e}")
        return

    results = []
    for slug, filename, rel_path in tqdm(items_to_process, desc=f"GPU {gpu_id}"):
        img_path = os.path.join(raw_dir, slug, filename)
        try:
            img = Image.open(img_path)
            w, h = img.size
            top_img = np.array(img.crop((0, 0, w, int(h * 0.12))))
            top_texts = reader.readtext(top_img, detail=0)
            top_data = parse_top(top_texts, dataset)
            
            bot_img = np.array(img.crop((0, int(h * 0.88), w, h)))
            bot_texts = reader.readtext(bot_img, detail=0)
            bot_data = parse_bottom(bot_texts)
            
            # Simple state extraction
            bot_data["state"] = extract_state(bot_data.get("location"))
            
            results.append({
                "rel_path": rel_path,
                "slug": slug,
                "filename": filename,
                **top_data,
                **bot_data
            })
        except Exception as e:
            logger.error(f"Error processing {img_path}: {e}")
            continue
            
    df = pd.DataFrame(results)
    df.to_csv(f"data/chunks/enrich_{dataset}_{chunk_id}.csv", index=False)

def enrich_dataset(dataset, base_dir, gpus):
    raw_dir = os.path.join(base_dir, "data", dataset, "raw")
    csv_path = os.path.join(base_dir, "data", dataset, "metadata.csv")
    manifest_path = os.path.join(base_dir, "data", dataset, ".processed_easyocr")
    
    if not os.path.exists(raw_dir):
        logger.error(f"Raw dir not found: {raw_dir}")
        return

    processed = set()
    if os.path.exists(manifest_path):
        with open(manifest_path, "r") as f:
            processed = {line.strip() for line in f if line.strip()}
            
    all_images = []
    for slug in sorted(os.listdir(raw_dir)):
        sd = os.path.join(raw_dir, slug)
        if not os.path.isdir(sd): continue
        for val in os.listdir(sd):
            if val.lower().endswith((".jpg", ".jpeg", ".png")):
                rp = f"{slug}/{val}"
                if rp not in processed:
                    all_images.append((slug, val, rp))
                    
    if not all_images:
        logger.info(f"No new images to enrich for {dataset}")
        return
        
    logger.info(f"Processing {len(all_images)} images for {dataset} using {len(gpus)} GPUs")
    
    num_workers = len(gpus)
    chunk_size = len(all_images) // num_workers + 1
    processes = []
    for i, gpu_id in enumerate(gpus):
        start = i * chunk_size
        end = min((i+1) * chunk_size, len(all_images))
        if start >= len(all_images): break
        chunk_items = all_images[start:end]
        p = mp.Process(target=process_chunk, args=(i, gpu_id, dataset, chunk_items, raw_dir))
        p.start()
        processes.append(p)
        
    for p in processes: p.join()
    
    # Merging and master CSV update
    logger.info(f"Merging results for {dataset}...")
    existing_meta = load_existing_butterfly_metadata(raw_dir) if dataset == "butterflies" else load_existing_plant_metadata(raw_dir)
    columns = BUTTERFLY_CSV_COLUMNS if dataset == "butterflies" else PLANT_CSV_COLUMNS
    
    # Determine start ID
    next_id = 1
    if os.path.exists(csv_path):
        try:
            mdf = pd.read_csv(csv_path)
            if not mdf.empty: next_id = mdf["image_id"].max() + 1
        except: pass

    write_header = not os.path.exists(csv_path)
    csv_file = open(csv_path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=columns)
    if write_header: writer.writeheader()
    
    manifest_file = open(manifest_path, "a")
    
    added_count = 0
    for i in range(num_workers):
        f_part = f"data/chunks/enrich_{dataset}_{i}.csv"
        if not os.path.exists(f_part): continue
        pdf = pd.read_csv(f_part)
        for _, ocr_row in pdf.iterrows():
            filename = ocr_row["filename"]
            slug = ocr_row["slug"]
            rel_path = ocr_row["rel_path"]
            
            existing = None
            if dataset == "butterflies":
                keys = [f"data/{slug.replace('_', '-')}/{filename}", filename]
                for k in keys:
                    if k in existing_meta:
                        existing = existing_meta[k]
                        break
                
                row = {
                    "image_id": next_id,
                    "filename": f"butterflies/images/{slug}/{filename}",
                    "raw_filename": f"butterflies/raw/{slug}/{filename}",
                    "species": f"{existing.get('Genus', '')} {existing.get('Species', '')}" if existing else ocr_row["scientific_name"],
                    "common_name": ocr_row["common_or_family"],
                    "family": existing.get("Family") if existing else None,
                    "subfamily": existing.get("Subfamily") if existing else None,
                    "genus": existing.get("Genus") if existing else None,
                    "order": existing.get("Order", "Lepidoptera") if existing else "Lepidoptera",
                    "life_stage": existing.get("life_stage") if existing else None,
                    "sex": ocr_row["sex"],
                    "media_code": ocr_row["media_code"],
                    "location": ocr_row["location"],
                    "state": ocr_row["state"],
                    "date": ocr_row["date"],
                    "credit": ocr_row["credit"],
                    "source_url": existing.get("source_url", "") if existing else "",
                    "source": "ifoundbutterflies.org",
                    "split": "",
                }
            else: # plants
                keys = [f"host_plants/{slug}/{filename}", f"host_plants/{slug}/{slug}_{filename}"]
                for k in keys:
                    if k in existing_meta:
                        existing = existing_meta[k]
                        break
                img_type = "hero" if "hero" in filename.lower() else ("gallery" if "gallery" in filename.lower() else "other")
                row = {
                    "image_id": next_id,
                    "filename": f"plants/images/{slug}/{filename}",
                    "raw_filename": f"plants/raw/{slug}/{filename}",
                    "species": existing.get("plant_scientific") if existing else ocr_row["scientific_name"],
                    "family": existing.get("plant_family") if existing else ocr_row["common_or_family"],
                    "genus": existing.get("plant_genus") if existing else None,
                    "image_type": img_type,
                    "media_code": ocr_row["media_code"],
                    "location": ocr_row["location"],
                    "state": ocr_row["state"],
                    "date": ocr_row["date"],
                    "credit": ocr_row["credit"],
                    "butterfly_hosts": ", ".join(h for h in existing.get("butterfly_hosts", []) if not h.startswith("http")) if existing else "",
                    "source_url": existing.get("source_url", "") if existing else "",
                    "source": "ifoundbutterflies.org",
                    "split": "",
                }
            
            writer.writerow(row)
            manifest_file.write(rel_path + "\n")
            next_id += 1
            added_count += 1
            
    csv_file.close()
    manifest_file.close()
    logger.info(f"Updated {csv_path} with {added_count} new records.")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["butterflies", "plants", "all"], default="all")
    args = parser.parse_args()
    
    AVAILABLE_GPUS = [0, 1, 3, 4, 5, 7]
    os.makedirs("data/chunks", exist_ok=True)
    
    datasets = ["butterflies", "plants"] if args.dataset == "all" else [args.dataset]
    for ds in datasets:
        enrich_dataset(ds, ".", AVAILABLE_GPUS)

if __name__ == "__main__":
    main()

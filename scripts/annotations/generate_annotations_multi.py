#!/usr/bin/env python3
"""
generate_annotations_multi.py — Scalable multi-GPU bounding box generation.

Distributes Grounding DINO zero-shot detection across all available GPUs
to handle 60k+ images in parallel.

Outputs:
  annotations/<dataset>/<slug>/<filename>.txt (YOLO)
  annotations/annotations.json (Master COCO)
"""

import os
import json
import argparse
import logging
import torch
import torch.multiprocessing as mp
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import copy

# Force multiprocessing to use spawn for CUDA
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Grounding DINO Loading ───────────────────────────────────────────────────

def load_grounding_dino(device):
    """Load model on specific device."""
    try:
        from groundingdino.util.inference import load_model, predict
        import groundingdino.datasets.transforms as T

        config_path = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        weights_path = "groundingdino_swint_ogc.pth"

        model = load_model(config_path, weights_path, device=device)
        return model
    except Exception as e:
        logger.error(f"Failed to load Grounding DINO on {device}: {e}")
        return None

def detect_subject(model, image_path, prompt, box_threshold=0.35, text_threshold=0.25):
    """Detect subject using Grounding DINO."""
    from groundingdino.util.inference import load_image, predict
    try:
        image_source, image = load_image(image_path)
        boxes, logits, phrases = predict(
            model=model,
            image=image,
            caption=prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
        # boxes are cxcywh, normalized [0,1]
        return boxes.cpu().numpy().tolist() if len(boxes) > 0 else [[0.5, 0.5, 1.0, 1.0]]
    except Exception:
        return [[0.5, 0.5, 1.0, 1.0]]

# ── Worker Process ────────────────────────────────────────────────────────────

def process_chunk(gpu_id, chunk_images, class_mapping, base_dir):
    """Worker function for a single GPU."""
    device = f"cuda:{gpu_id}"
    logger.info(f"Worker on GPU {gpu_id} starting for {len(chunk_images)} images.")
    
    model = load_grounding_dino(device)
    if not model:
        return []

    results = []
    
    for img_info in tqdm(chunk_images, desc=f"GPU {gpu_id}"):
        image_path = img_info["abs_path"]
        slug = img_info["slug"]
        filename = img_info["filename"]
        dataset_type = img_info["dataset"]
        class_id = class_mapping.get(slug, -1)
        
        prompt = "butterfly . moth . caterpillar" if dataset_type == "butterflies" else "plant . flower . leaf"
        
        # Detect
        boxes = detect_subject(model, image_path, prompt)
        
        # Write YOLO
        slug_anno_dir = os.path.join(base_dir, "annotations", dataset_type, slug)
        os.makedirs(slug_anno_dir, exist_ok=True)
        yolo_path = os.path.join(slug_anno_dir, os.path.splitext(filename)[0] + ".txt")
        
        with open(yolo_path, "w") as f:
            for box in boxes:
                cx, cy, w, h = box
                f.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
        
        # Prep COCO entries
        try:
            with Image.open(image_path) as img:
                width, height = img.size
        except:
            width, height = 700, 600

        res_entry = {
            "image": {
                "id": img_info["global_id"],
                "file_name": f"data/{dataset_type}/images/{slug}/{filename}",
                "width": width,
                "height": height
            },
            "annotations": []
        }
        
        for i, box in enumerate(boxes):
            cx, cy, bw, bh = box
            # COCO: [x, y, width, height]
            px = (cx - bw/2) * width
            py = (cy - bh/2) * height
            pw = bw * width
            ph = bh * height
            
            res_entry["annotations"].append({
                "category_id": class_id,
                "bbox": [round(px, 1), round(py, 1), round(pw, 1), round(ph, 1)],
                "area": round(pw * ph, 1),
                "iscrowd": 0
            })
            
        results.append(res_entry)
        
    return results

# ── Main Orchestration ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=int, nargs="+", default=[0])
    parser.add_argument("--base-dir", type=str, default=".")
    args = parser.parse_args()

    # 1. Build Class Mapping
    mapping = {}
    cid = 0
    for ds in ["butterflies", "plants"]:
        raw_dir = os.path.join(args.base_dir, "data", ds, "images")
        if not os.path.isdir(raw_dir): continue
        for slug in sorted(os.listdir(raw_dir)):
            if os.path.isdir(os.path.join(raw_dir, slug)):
                mapping[slug] = cid
                cid += 1
    
    # 2. Collect Images
    all_images = []
    global_id = 1
    for ds in ["butterflies", "plants"]:
        img_dir = os.path.join(args.base_dir, "data", ds, "images")
        if not os.path.isdir(img_dir): continue
        for slug in sorted(os.listdir(img_dir)):
            slug_path = os.path.join(img_dir, slug)
            for f in sorted(os.listdir(slug_path)):
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    all_images.append({
                        "abs_path": os.path.join(slug_path, f),
                        "dataset": ds,
                        "slug": slug,
                        "filename": f,
                        "global_id": global_id
                    })
                    global_id += 1

    logger.info(f"Collected {len(all_images)} images to process.")

    # 3. Parallel Execution
    gpu_list = args.gpus
    num_workers = len(gpu_list)
    chunks = [all_images[i::num_workers] for i in range(num_workers)]
    
    final_coco_images = []
    final_coco_annos = []
    anno_global_id = 1

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for i, gpu_id in enumerate(gpu_list):
            futures.append(executor.submit(
                process_chunk, gpu_id, chunks[i], mapping, args.base_dir
            ))

        for future in futures:
            chunk_results = future.result()
            for res in chunk_results:
                final_coco_images.append(res["image"])
                for a in res["annotations"]:
                    a["id"] = anno_global_id
                    a["image_id"] = res["image"]["id"]
                    final_coco_annos.append(a)
                    anno_global_id += 1

    # 4. Write COCO Output
    coco = {
        "info": {"description": "IndoLepAtlas Full Dataset"},
        "categories": [{"id": v, "name": k, "supercategory": "species"} for k, v in sorted(mapping.items(), key=lambda x: x[1])],
        "images": final_coco_images,
        "annotations": final_coco_annos
    }
    
    out_path = os.path.join(args.base_dir, "annotations", "annotations.json")
    with open(out_path, "w") as f:
        json.dump(coco, f)
    
    logger.info(f"Finalized annotations.json with {len(final_coco_images)} images.")

if __name__ == "__main__":
    main()

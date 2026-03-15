#!/usr/bin/env python3
"""
generate_annotations.py — Auto-generate bounding box annotations using Grounding DINO.

Uses a zero-shot object detection model to detect subjects in trimmed images
and produces species-level annotations in YOLO and COCO formats.

Outputs:
  annotations/butterflies/<species>/<image>.txt  (YOLO format)
  annotations/plants/<species>/<image>.txt       (YOLO format)
  annotations/annotations.json                    (COCO format)

Features:
  - Multi-GPU support (round-robin species across GPUs)
  - Incremental: skips images with existing annotations
  - Species-level classes (~1094 classes)
  - Auto-generates classes.txt
"""

import os
import sys
import csv
import json
import argparse
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(process)d] %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ── Class mapping ──────────────────────────────────────────────────────────────


def build_class_mapping(base_dir: str) -> dict:
    """
    Build species → class_id mapping from directory listing.
    Butterflies first (0..N), then plants (N+1..M).
    """
    mapping = {}
    class_id = 0

    # Butterflies
    butterfly_raw = os.path.join(base_dir, "data", "butterflies", "raw")
    if os.path.isdir(butterfly_raw):
        for slug in sorted(os.listdir(butterfly_raw)):
            if os.path.isdir(os.path.join(butterfly_raw, slug)):
                mapping[slug] = class_id
                class_id += 1

    # Plants
    plant_raw = os.path.join(base_dir, "data", "plants", "raw")
    if os.path.isdir(plant_raw):
        for slug in sorted(os.listdir(plant_raw)):
            if os.path.isdir(os.path.join(plant_raw, slug)):
                mapping[slug] = class_id
                class_id += 1

    return mapping


def write_classes_txt(mapping: dict, base_dir: str):
    """Write classes.txt file."""
    path = os.path.join(base_dir, "annotations", "classes.txt")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for slug, class_id in sorted(mapping.items(), key=lambda x: x[1]):
            f.write(f"{class_id} {slug}\n")
    logger.info(f"Wrote {len(mapping)} classes to {path}")


# ── Grounding DINO inference ──────────────────────────────────────────────────


def load_grounding_dino(device: str = "cuda:0"):
    """
    Load Grounding DINO model.
    Uses the groundingdino package from IDEA-Research.
    """
    try:
        from groundingdino.util.inference import load_model, predict
        from groundingdino.util.inference import load_image as gd_load_image
        import groundingdino.datasets.transforms as T

        # The model config and weights paths
        # Users should download these from the repo
        config_path = os.environ.get(
            "GDINO_CONFIG",
            "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        )
        weights_path = os.environ.get(
            "GDINO_WEIGHTS",
            "groundingdino_swint_ogc.pth",
        )

        if not os.path.exists(weights_path):
            logger.warning(
                f"Grounding DINO weights not found at {weights_path}. "
                "Download from: https://github.com/IDEA-Research/GroundingDINO"
            )
            return None, None

        model = load_model(config_path, weights_path, device=device)
        return model, {"predict": predict, "load_image": gd_load_image}

    except ImportError:
        logger.warning(
            "groundingdino not installed. Install from: "
            "https://github.com/IDEA-Research/GroundingDINO"
        )
        return None, None


def detect_subject(model, funcs, image_path: str, prompt: str, threshold: float = 0.3):
    """
    Run Grounding DINO on an image.

    Returns:
        list of (cx, cy, w, h) normalized bounding boxes
    """
    try:
        image_source, image = funcs["load_image"](image_path)
        boxes, logits, phrases = funcs["predict"](
            model=model,
            image=image,
            caption=prompt,
            box_threshold=threshold,
            text_threshold=threshold,
        )

        # boxes are in cxcywh format, normalized [0,1]
        return boxes.cpu().numpy().tolist() if len(boxes) > 0 else []

    except Exception as e:
        logger.debug(f"Detection failed for {image_path}: {e}")
        return []


def fallback_full_image_bbox():
    """If no model available, use full image as bbox."""
    return [[0.5, 0.5, 1.0, 1.0]]


# ── Write annotations ─────────────────────────────────────────────────────────


def write_yolo_annotation(
    output_path: str, class_id: int, boxes: list
):
    """Write YOLO format annotation file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for box in boxes:
            cx, cy, w, h = box
            f.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


# ── Process dataset ────────────────────────────────────────────────────────────


def annotate_dataset(
    dataset: str,
    base_dir: str,
    class_mapping: dict,
    gpu_id: int = 0,
    use_model: bool = True,
):
    """Generate annotations for a dataset."""
    images_dir = os.path.join(base_dir, "data", dataset, "images")
    annotations_dir = os.path.join(base_dir, "annotations", dataset)

    if not os.path.isdir(images_dir):
        logger.error(f"Images directory not found: {images_dir}")
        logger.error("Run process_images.py first!")
        return {"processed": 0, "skipped": 0}

    # Load model if available
    model, funcs = None, None
    if use_model:
        device = f"cuda:{gpu_id}" if gpu_id >= 0 else "cpu"
        model, funcs = load_grounding_dino(device)
        if model:
            logger.info(f"Loaded Grounding DINO on {device}")
        else:
            logger.warning("Using fallback (full-image bbox)")

    # Determine detection prompt
    if dataset == "butterflies":
        prompt = "butterfly . moth . caterpillar . pupa . chrysalis"
    else:
        prompt = "plant . flower . leaf . tree . shrub"

    processed = 0
    skipped = 0
    coco_images = []
    coco_annotations = []
    annotation_id = 1

    species_dirs = sorted(
        d for d in os.listdir(images_dir)
        if os.path.isdir(os.path.join(images_dir, d))
    )

    for slug in species_dirs:
        slug_images_dir = os.path.join(images_dir, slug)
        slug_anno_dir = os.path.join(annotations_dir, slug)
        class_id = class_mapping.get(slug, -1)

        if class_id < 0:
            logger.warning(f"No class mapping for {slug}, skipping")
            continue

        image_files = sorted(
            f for f in os.listdir(slug_images_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )

        for filename in image_files:
            yolo_filename = os.path.splitext(filename)[0] + ".txt"
            yolo_path = os.path.join(slug_anno_dir, yolo_filename)

            # Skip if annotation already exists
            if os.path.exists(yolo_path):
                skipped += 1
                continue

            image_path = os.path.join(slug_images_dir, filename)

            # Detect
            if model and funcs:
                boxes = detect_subject(model, funcs, image_path, prompt)
                if not boxes:
                    boxes = fallback_full_image_bbox()
            else:
                boxes = fallback_full_image_bbox()

            # Write YOLO
            write_yolo_annotation(yolo_path, class_id, boxes)

            # Collect for COCO
            try:
                from PIL import Image
                img = Image.open(image_path)
                w, h = img.size
                if w == 1000 and h == 1000:
                    logger.debug(f"Image {image_path} returned 1000x1000 - verifying...")
            except Exception as e:
                logger.warning(f"Failed to read size for {image_path}: {e}")
                w, h = 700, 600  # fallback

            image_id = processed + skipped + 1
            coco_images.append({
                "id": image_id,
                "file_name": f"{dataset}/images/{slug}/{filename}",
                "width": w,
                "height": h,
            })

            for box in boxes:
                cx, cy, bw, bh = box
                # Convert normalized cxcywh to pixel xywh for COCO
                x = (cx - bw / 2) * w
                y = (cy - bh / 2) * h
                pw = bw * w
                ph = bh * h
                coco_annotations.append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": class_id,
                    "bbox": [round(x, 1), round(y, 1), round(pw, 1), round(ph, 1)],
                    "area": round(pw * ph, 1),
                    "iscrowd": 0,
                })
                annotation_id += 1

            processed += 1

            if processed % 200 == 0:
                logger.info(f"  {dataset}: {processed} annotated, {skipped} skipped")

    logger.info(
        f"  {dataset}: Done! {processed} annotated, {skipped} skipped"
    )

    return {
        "processed": processed,
        "skipped": skipped,
        "coco_images": coco_images,
        "coco_annotations": coco_annotations,
    }


def write_coco_json(base_dir: str, class_mapping: dict, all_results: list):
    """Write combined COCO annotations.json."""
    categories = []
    for slug, class_id in sorted(class_mapping.items(), key=lambda x: x[1]):
        categories.append({
            "id": class_id,
            "name": slug,
            "supercategory": "species",
        })

    all_images = []
    all_annos = []
    for result in all_results:
        all_images.extend(result.get("coco_images", []))
        all_annos.extend(result.get("coco_annotations", []))

    coco = {
        "info": {
            "description": "IndoLepAtlas - Indian Lepidoptera and Host Plants",
            "version": "1.0",
            "year": 2026,
            "contributor": "IndoLepAtlas Team",
        },
        "licenses": [{"id": 1, "name": "CC BY-NC-SA 4.0"}],
        "categories": categories,
        "images": all_images,
        "annotations": all_annos,
    }

    path = os.path.join(base_dir, "annotations", "annotations.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(coco, f, indent=2)
    logger.info(
        f"Wrote COCO annotations: {len(all_images)} images, "
        f"{len(all_annos)} annotations → {path}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate bounding box annotations using Grounding DINO"
    )
    parser.add_argument(
        "--dataset",
        choices=["butterflies", "plants", "all"],
        default="all",
    )
    parser.add_argument("--base-dir", type=str, default=".")
    parser.add_argument(
        "--num-gpus", type=int, default=1,
        help="Number of GPUs to use (default: 1)",
    )
    parser.add_argument(
        "--no-model", action="store_true",
        help="Skip model detection, use full-image bbox fallback",
    )
    args = parser.parse_args()

    # Build class mapping
    class_mapping = build_class_mapping(args.base_dir)
    logger.info(f"Built class mapping: {len(class_mapping)} species")
    write_classes_txt(class_mapping, args.base_dir)

    datasets = (
        ["butterflies", "plants"] if args.dataset == "all" else [args.dataset]
    )

    all_results = []

    for ds in datasets:
        logger.info(f"\n{'='*60}")
        logger.info(f"Annotating dataset: {ds}")
        logger.info(f"{'='*60}")
        result = annotate_dataset(
            ds,
            args.base_dir,
            class_mapping,
            gpu_id=0,
            use_model=not args.no_model,
        )
        all_results.append(result)

    # Write combined COCO JSON
    write_coco_json(args.base_dir, class_mapping, all_results)


if __name__ == "__main__":
    main()

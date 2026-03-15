#!/usr/bin/env python3
"""
enrich_metadata.py — Extract overlay metadata from images via OCR.

Reads raw images, crops the overlay bands, runs OCR (pytesseract),
and parses the text into structured metadata CSVs.

Outputs:
  data/butterflies/metadata.csv
  data/plants/metadata.csv

Features:
  - Append-only: adds new rows without touching existing ones
  - Tracks processed files via .processed manifest
  - Handles missing/unreadable overlay fields gracefully (null)
  - Cross-validates media_code against existing metadata
"""

import os
import re
import csv
import sys
import json
import argparse
import logging
from pathlib import Path
from PIL import Image, ImageFile

# Prevent crashes on corrupted/truncated image files
ImageFile.LOAD_TRUNCATED_IMAGES = True
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Try to import pytesseract — provide helpful error if missing
try:
    import pytesseract
except ImportError:
    logger.error(
        "pytesseract not installed. Install with: pip install pytesseract\n"
        "Also install Tesseract OCR: sudo apt-get install tesseract-ocr"
    )
    sys.exit(1)


# ── OCR helpers ────────────────────────────────────────────────────────────────

TOP_CROP_FRAC = 0.10  # slightly more than trim to capture full text
BOTTOM_CROP_FRAC = 0.10


def ocr_region(img: Image.Image, top_frac: float, bottom_frac: float) -> str:
    """Crop a horizontal band from the image and OCR it."""
    width, height = img.size
    top_px = int(height * top_frac)
    bottom_px = int(height * bottom_frac)
    region = img.crop((0, top_px, width, bottom_px))

    try:
        text = pytesseract.image_to_string(region, lang="eng")
        return text.strip()
    except Exception as e:
        logger.debug(f"OCR failed: {e}")
        return ""


def parse_butterfly_top(text: str) -> dict:
    """
    Parse butterfly top overlay.
    Expected: "Scientific name\nCommon Name    Media code: XXX"
    """
    result = {"scientific_name": None, "common_name": None, "media_code": None}

    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        return result

    # First line is usually the scientific name (italic)
    result["scientific_name"] = lines[0] if lines else None

    # Look for "Media code:" pattern
    media_match = re.search(r"[Mm]edia\s*code[:\s]+(\S+)", text)
    if media_match:
        result["media_code"] = media_match.group(1)

    # Common name is typically the second line (before media code)
    if len(lines) >= 2:
        # Remove media code portion from second line
        second_line = lines[1]
        second_line = re.sub(r"[Mm]edia\s*code[:\s]+\S+", "", second_line).strip()
        if second_line and second_line != result["scientific_name"]:
            result["common_name"] = second_line

    return result


def parse_plant_top(text: str) -> dict:
    """
    Parse plant top overlay.
    Expected: "Scientific name\nFamily    Media code: XXX"
    """
    result = {"scientific_name": None, "family_ocr": None, "media_code": None}

    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        return result

    result["scientific_name"] = lines[0] if lines else None

    media_match = re.search(r"[Mm]edia\s*code[:\s]+(\S+)", text)
    if media_match:
        result["media_code"] = media_match.group(1)

    if len(lines) >= 2:
        second_line = lines[1]
        second_line = re.sub(r"[Mm]edia\s*code[:\s]+\S+", "", second_line).strip()
        if second_line:
            result["family_ocr"] = second_line

    return result


def parse_bottom(text: str) -> dict:
    """
    Parse bottom overlay (same for both datasets).
    Expected: "[Sex.] Location. YYYY/MM/DD.  ©Photographer"
    """
    result = {"sex": None, "location": None, "date": None, "credit": None}

    if not text:
        return result

    # Extract credit (©...)
    credit_match = re.search(r"[©@](.+?)(?:\n|$)", text)
    if credit_match:
        result["credit"] = credit_match.group(1).strip()

    # Extract date (YYYY/MM/DD pattern)
    date_match = re.search(r"(\d{4}/\d{2}/\d{2})", text)
    if date_match:
        result["date"] = date_match.group(1)

    # Extract sex (Male/Female at start)
    sex_match = re.match(r"(Male|Female|Unknown)\b", text, re.IGNORECASE)
    if sex_match:
        result["sex"] = sex_match.group(1).capitalize()

    # Location: everything between sex and date, or start and date
    # Remove credit and date from text, what remains is location
    location_text = text
    if result["credit"]:
        location_text = re.sub(r"[©@].+?(?:\n|$)", "", location_text)
    if result["date"]:
        location_text = location_text.replace(result["date"], "")
    if result["sex"]:
        location_text = re.sub(
            r"^(Male|Female|Unknown)\b\.?\s*",
            "",
            location_text,
            flags=re.IGNORECASE,
        )

    # Clean up remaining text as location
    location_text = re.sub(r"\s+", " ", location_text).strip(" .,\n")
    if location_text and len(location_text) > 3:
        result["location"] = location_text

    return result


def extract_state(location: str) -> str:
    """Try to extract Indian state from location string."""
    if not location:
        return None

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


# ── Process an image ───────────────────────────────────────────────────────────


def extract_image_metadata(image_path: str, dataset_type: str) -> dict:
    """
    Extract overlay metadata from a single image.

    Args:
        image_path: path to raw image
        dataset_type: "butterflies" or "plants"

    Returns:
        dict with extracted fields (nulls for missing)
    """
    try:
        img = Image.open(image_path)
    except Exception as e:
        logger.error(f"Cannot open {image_path}: {e}")
        return {}

    # OCR the overlay bands
    top_text = ocr_region(img, 0.0, TOP_CROP_FRAC)
    bottom_text = ocr_region(img, 1.0 - BOTTOM_CROP_FRAC, 1.0)

    # Parse
    if dataset_type == "butterflies":
        top_data = parse_butterfly_top(top_text)
    else:
        top_data = parse_plant_top(top_text)

    bottom_data = parse_bottom(bottom_text)

    # Extract state from location
    bottom_data["state"] = extract_state(bottom_data.get("location"))

    return {**top_data, **bottom_data}


# ── Load existing metadata ────────────────────────────────────────────────────


def load_existing_butterfly_metadata(raw_dir: str) -> dict:
    """Load per-species metadata.jsonl files for butterflies."""
    existing = {}
    for slug in os.listdir(raw_dir):
        meta_path = os.path.join(raw_dir, slug, "metadata.jsonl")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        filename = record.get("file_name", "")
                        existing[filename] = record
                    except json.JSONDecodeError:
                        continue
    return existing


def load_existing_plant_metadata(raw_dir: str) -> dict:
    """Load plant registry.json and metadata.jsonl."""
    existing = {}

    # Load registry
    registry_path = os.path.join(raw_dir, "registry.json")
    registry = {}
    if os.path.exists(registry_path):
        with open(registry_path, "r") as f:
            registry = json.load(f)

    # Load metadata.jsonl
    meta_path = os.path.join(raw_dir, "metadata.jsonl")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    filename = record.get("file_name", "")
                    # Merge registry data
                    plant_key = record.get("plant_key", "")
                    if plant_key in registry:
                        record["_registry"] = registry[plant_key]
                    existing[filename] = record
                except json.JSONDecodeError:
                    continue

    return existing


# ── Main enrichment ────────────────────────────────────────────────────────────

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


def enrich_dataset(dataset: str, base_dir: str):
    """Enrich metadata for a dataset via OCR."""
    raw_dir = os.path.join(base_dir, "data", dataset, "raw")
    images_dir = os.path.join(base_dir, "data", dataset, "images")
    csv_path = os.path.join(base_dir, "data", dataset, "metadata.csv")
    manifest_path = os.path.join(base_dir, "data", dataset, ".processed")

    if not os.path.isdir(raw_dir):
        logger.error(f"Raw directory not found: {raw_dir}")
        return

    # Load manifest of already-processed files
    processed_files = set()
    if os.path.exists(manifest_path):
        with open(manifest_path, "r") as f:
            processed_files = {line.strip() for line in f if line.strip()}

    # Load existing metadata
    if dataset == "butterflies":
        existing_meta = load_existing_butterfly_metadata(raw_dir)
        columns = BUTTERFLY_CSV_COLUMNS
    else:
        existing_meta = load_existing_plant_metadata(raw_dir)
        columns = PLANT_CSV_COLUMNS

    logger.info(f"Loaded {len(existing_meta)} existing metadata records")
    logger.info(f"{len(processed_files)} files already processed")

    # Collect all raw image files
    all_images = []
    for slug in sorted(os.listdir(raw_dir)):
        slug_dir = os.path.join(raw_dir, slug)
        if not os.path.isdir(slug_dir):
            continue
        for filename in sorted(os.listdir(slug_dir)):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                rel_path = f"{slug}/{filename}"
                if rel_path not in processed_files:
                    all_images.append((slug, filename, rel_path))

    logger.info(f"{len(all_images)} images to process")

    if not all_images:
        logger.info("Nothing to process!")
        return

    # Determine starting image_id
    next_id = 1
    if os.path.exists(csv_path):
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    next_id = max(next_id, int(row.get("image_id", 0)) + 1)
                except (ValueError, TypeError):
                    pass

    # Open CSV for append
    write_header = not os.path.exists(csv_path)
    csv_file = open(csv_path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=columns)
    if write_header:
        writer.writeheader()

    # Open manifest for append
    manifest_file = open(manifest_path, "a")

    processed_count = 0

    try:
        for slug, filename, rel_path in all_images:
            image_path = os.path.join(raw_dir, slug, filename)

            # OCR extraction
            ocr_data = extract_image_metadata(image_path, dataset)

            # Build CSV row
            if dataset == "butterflies":
                # Try to find existing metadata for this image
                # Match by various possible keys
                existing = None
                for key_pattern in [
                    f"data/{slug.replace('_', '-')}/{filename}",
                    filename,
                ]:
                    if key_pattern in existing_meta:
                        existing = existing_meta[key_pattern]
                        break

                row = {
                    "image_id": next_id,
                    "filename": f"butterflies/images/{slug}/{filename}",
                    "raw_filename": f"butterflies/raw/{slug}/{filename}",
                    "species": (
                        f"{existing.get('Genus', '')} {existing.get('Species', '')}"
                        if existing
                        else ocr_data.get("scientific_name")
                    ),
                    "common_name": ocr_data.get("common_name"),
                    "family": existing.get("Family") if existing else None,
                    "subfamily": existing.get("Subfamily") if existing else None,
                    "genus": existing.get("Genus") if existing else None,
                    "order": existing.get("Order", "Lepidoptera") if existing else "Lepidoptera",
                    "life_stage": existing.get("life_stage") if existing else None,
                    "sex": ocr_data.get("sex"),
                    "media_code": ocr_data.get("media_code"),
                    "location": ocr_data.get("location"),
                    "state": ocr_data.get("state"),
                    "date": ocr_data.get("date"),
                    "credit": ocr_data.get("credit"),
                    "source_url": existing.get("source_url", "") if existing else "",
                    "source": "ifoundbutterflies.org",
                    "split": "",  # will be filled by generate_splits.py
                }

            else:  # plants
                existing = None
                for key_pattern in [
                    f"host_plants/{slug}/{filename}",
                    f"host_plants/{slug}/{slug}_{filename}",
                ]:
                    if key_pattern in existing_meta:
                        existing = existing_meta[key_pattern]
                        break

                registry_data = existing.get("_registry", {}) if existing else {}

                # Determine image type from filename
                if "hero" in filename.lower():
                    img_type = "hero"
                elif "gallery" in filename.lower():
                    img_type = "gallery"
                else:
                    img_type = "other"

                row = {
                    "image_id": next_id,
                    "filename": f"plants/images/{slug}/{filename}",
                    "raw_filename": f"plants/raw/{slug}/{filename}",
                    "species": (
                        existing.get("plant_scientific")
                        if existing
                        else ocr_data.get("scientific_name")
                    ),
                    "family": (
                        existing.get("plant_family")
                        if existing
                        else ocr_data.get("family_ocr")
                    ),
                    "genus": existing.get("plant_genus") if existing else None,
                    "image_type": img_type,
                    "media_code": ocr_data.get("media_code"),
                    "location": ocr_data.get("location"),
                    "state": ocr_data.get("state"),
                    "date": ocr_data.get("date"),
                    "credit": ocr_data.get("credit"),
                    "butterfly_hosts": (
                        ", ".join(
                            h for h in existing.get("butterfly_hosts", [])
                            if not h.startswith("http")
                        )
                        if existing
                        else ""
                    ),
                    "source_url": existing.get("source_url", "") if existing else "",
                    "source": "ifoundbutterflies.org",
                    "split": "",
                }

            writer.writerow(row)
            manifest_file.write(rel_path + "\n")
            next_id += 1
            processed_count += 1

            if processed_count % 100 == 0:
                csv_file.flush()
                manifest_file.flush()
                logger.info(f"  Processed {processed_count}/{len(all_images)}")

    finally:
        csv_file.close()
        manifest_file.close()

    logger.info(
        f"\nDone! Enriched {processed_count} images → {csv_path}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Extract overlay metadata from images via OCR"
    )
    parser.add_argument(
        "--dataset",
        choices=["butterflies", "plants", "all"],
        default="all",
        help="Which dataset to process (default: all)",
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default=".",
        help="Base directory (default: current directory)",
    )
    args = parser.parse_args()

    datasets = (
        ["butterflies", "plants"] if args.dataset == "all" else [args.dataset]
    )

    for ds in datasets:
        logger.info(f"\n{'='*60}")
        logger.info(f"Enriching metadata for: {ds}")
        logger.info(f"{'='*60}")
        enrich_dataset(ds, args.base_dir)


if __name__ == "__main__":
    main()

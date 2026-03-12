#!/usr/bin/env python3
"""
process_images.py — Trim overlay bands from raw images.

Removes the top ~8% and bottom ~8% overlay bands (containing metadata text)
from raw images, producing clean versions for model training.

Reads from:  data/{butterflies,plants}/raw/<species>/
Writes to:   data/{butterflies,plants}/images/<species>/

Features:
  - Incremental: skips species that already have trimmed images
  - Configurable crop percentages
  - Preserves directory structure
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Default crop percentages (fraction of image height)
# Tested on both butterfly and plant images — 12%/10% cleanly removes all overlay text
DEFAULT_TOP_CROP = 0.12
DEFAULT_BOTTOM_CROP = 0.10


def trim_image(
    input_path: str,
    output_path: str,
    top_crop: float = DEFAULT_TOP_CROP,
    bottom_crop: float = DEFAULT_BOTTOM_CROP,
) -> bool:
    """
    Trim top and bottom overlay bands from an image.

    Args:
        input_path: path to raw image
        output_path: path to save trimmed image
        top_crop: fraction of height to crop from top
        bottom_crop: fraction of height to crop from bottom

    Returns:
        True if successful
    """
    try:
        img = Image.open(input_path)
        width, height = img.size

        top_px = int(height * top_crop)
        bottom_px = int(height * (1 - bottom_crop))

        # Ensure we don't crop too much
        if bottom_px <= top_px:
            logger.warning(
                f"Crop would remove entire image: {input_path}, skipping"
            )
            return False

        trimmed = img.crop((0, top_px, width, bottom_px))

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        trimmed.save(output_path, quality=95)
        return True
    except Exception as e:
        logger.error(f"Failed to trim {input_path}: {e}")
        return False


def process_species(
    raw_dir: str,
    images_dir: str,
    species_slug: str,
    top_crop: float,
    bottom_crop: float,
) -> dict:
    """
    Process all images for a single species.

    Returns:
        dict with counts: {"processed": N, "skipped": N, "failed": N}
    """
    raw_species = os.path.join(raw_dir, species_slug)
    out_species = os.path.join(images_dir, species_slug)

    result = {"species": species_slug, "processed": 0, "skipped": 0, "failed": 0}

    if not os.path.isdir(raw_species):
        return result

    # Get list of image files
    image_files = [
        f
        for f in os.listdir(raw_species)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    for filename in image_files:
        input_path = os.path.join(raw_species, filename)
        output_path = os.path.join(out_species, filename)

        # Skip if already processed
        if os.path.exists(output_path):
            result["skipped"] += 1
            continue

        if trim_image(input_path, output_path, top_crop, bottom_crop):
            result["processed"] += 1
        else:
            result["failed"] += 1

    # Write manifest if we processed anything
    if result["processed"] > 0 or result["skipped"] > 0:
        manifest_path = os.path.join(out_species, ".trimmed")
        os.makedirs(out_species, exist_ok=True)
        with open(manifest_path, "w") as f:
            f.write(
                f"processed={result['processed']} "
                f"skipped={result['skipped']} "
                f"failed={result['failed']}\n"
            )

    return result


def process_dataset(
    dataset: str,
    base_dir: str,
    top_crop: float,
    bottom_crop: float,
    workers: int,
):
    """Process all species in a dataset."""
    raw_dir = os.path.join(base_dir, "data", dataset, "raw")
    images_dir = os.path.join(base_dir, "data", dataset, "images")

    if not os.path.isdir(raw_dir):
        logger.error(f"Raw directory not found: {raw_dir}")
        logger.error("Run pull_hf_data.py first!")
        return

    # Get species list
    species_list = sorted(
        d
        for d in os.listdir(raw_dir)
        if os.path.isdir(os.path.join(raw_dir, d))
    )

    logger.info(f"Found {len(species_list)} species in {raw_dir}")

    # Filter already-fully-processed species
    pending = []
    skipped_species = 0
    for slug in species_list:
        manifest = os.path.join(images_dir, slug, ".trimmed")
        if os.path.exists(manifest):
            # Check if there are new raw images not yet trimmed
            raw_count = len([
                f for f in os.listdir(os.path.join(raw_dir, slug))
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ])
            trimmed_count = len([
                f for f in os.listdir(os.path.join(images_dir, slug))
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]) if os.path.isdir(os.path.join(images_dir, slug)) else 0

            if trimmed_count >= raw_count:
                skipped_species += 1
                continue

        pending.append(slug)

    logger.info(
        f"{skipped_species} species fully processed, "
        f"{len(pending)} species to process"
    )

    if not pending:
        logger.info("Nothing to process!")
        return

    total_processed = 0
    total_failed = 0

    # Process species (use multiprocessing for CPU-bound image cropping)
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    process_species,
                    raw_dir,
                    images_dir,
                    slug,
                    top_crop,
                    bottom_crop,
                ): slug
                for slug in pending
            }

            for future in as_completed(futures):
                result = future.result()
                total_processed += result["processed"]
                total_failed += result["failed"]
                if result["processed"] > 0:
                    logger.info(
                        f"  ✓ {result['species']}: "
                        f"{result['processed']} trimmed, "
                        f"{result['skipped']} skipped"
                    )
    else:
        for slug in pending:
            result = process_species(
                raw_dir, images_dir, slug, top_crop, bottom_crop
            )
            total_processed += result["processed"]
            total_failed += result["failed"]
            if result["processed"] > 0:
                logger.info(
                    f"  ✓ {result['species']}: "
                    f"{result['processed']} trimmed, "
                    f"{result['skipped']} skipped"
                )

    logger.info(
        f"\nDone! Trimmed {total_processed} images, "
        f"{total_failed} failed."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Trim overlay bands from raw images"
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
    parser.add_argument(
        "--top-crop",
        type=float,
        default=DEFAULT_TOP_CROP,
        help=f"Fraction of height to crop from top (default: {DEFAULT_TOP_CROP})",
    )
    parser.add_argument(
        "--bottom-crop",
        type=float,
        default=DEFAULT_BOTTOM_CROP,
        help=f"Fraction of height to crop from bottom (default: {DEFAULT_BOTTOM_CROP})",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel worker processes (default: 4)",
    )
    args = parser.parse_args()

    datasets = (
        ["butterflies", "plants"] if args.dataset == "all" else [args.dataset]
    )

    for ds in datasets:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing dataset: {ds}")
        logger.info(f"{'='*60}")
        process_dataset(
            ds, args.base_dir, args.top_crop, args.bottom_crop, args.workers
        )


if __name__ == "__main__":
    main()

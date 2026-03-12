#!/usr/bin/env python3
"""
pull_hf_data.py — Download butterfly and plant datasets from Hugging Face.

Pulls from DihelseeWee/IndoLepAtlas into the clean directory structure:
  data/butterflies/raw/<species_slug>/
  data/plants/raw/<species_slug>/

Features:
  - Batch processing (--batch-size species at a time)
  - Resumable (skips already-downloaded species)
  - Supports --dataset butterflies|plants|all
"""

import os
import sys
import json
import time
import argparse
import logging
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.environ.get("HF_TOKEN")
REPO_ID = "DihelseeWee/IndoLepAtlas"
API_BASE = f"https://huggingface.co/api/datasets/{REPO_ID}"
RAW_BASE = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ── Helpers ────────────────────────────────────────────────────────────────────


def hf_headers():
    return {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}


def list_hf_dir(path: str) -> list:
    """List contents of a directory on HF repo."""
    url = f"{API_BASE}/tree/main/{path}" if path else f"{API_BASE}/tree/main"
    resp = requests.get(url, headers=hf_headers(), timeout=60)
    resp.raise_for_status()
    return resp.json()


def download_file(remote_path: str, local_path: str, retries: int = 3):
    """Download a single file from HF with retry."""
    url = f"{RAW_BASE}/{remote_path}"
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=hf_headers(), stream=True, timeout=60)
            resp.raise_for_status()
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            with open(local_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
        except Exception as e:
            if attempt < retries - 1:
                logger.warning(f"Retry {attempt+1}/{retries} for {remote_path}: {e}")
                time.sleep(2 * (attempt + 1))
            else:
                logger.error(f"Failed to download {remote_path}: {e}")
                return False


# ── Species download ───────────────────────────────────────────────────────────


def download_species(
    hf_dir: str, local_dir: str, species_name: str, download_threads: int = 4
):
    """
    Download all files for a single species from HF to local.

    Args:
        hf_dir: HF path, e.g. "data/Abisara-attenuata"
        local_dir: local path, e.g. "data/butterflies/raw/Abisara_attenuata"
        species_name: for logging
        download_threads: parallel file downloads within species
    """
    try:
        items = list_hf_dir(hf_dir)
    except Exception as e:
        logger.error(f"Failed to list {hf_dir}: {e}")
        return 0

    files = [item for item in items if item.get("type") == "file"]
    if not files:
        logger.warning(f"No files found in {hf_dir}")
        return 0

    downloaded = 0

    def _download_one(item):
        remote_path = item["path"]
        filename = os.path.basename(remote_path)
        local_path = os.path.join(local_dir, filename)

        # Skip if already downloaded and same size
        if os.path.exists(local_path):
            local_size = os.path.getsize(local_path)
            remote_size = item.get("size", -1)
            if remote_size > 0 and local_size == remote_size:
                return True  # already have it

        return download_file(remote_path, local_path)

    with ThreadPoolExecutor(max_workers=download_threads) as executor:
        futures = {executor.submit(_download_one, f): f for f in files}
        for future in as_completed(futures):
            if future.result():
                downloaded += 1

    return downloaded


# ── Main pipeline ──────────────────────────────────────────────────────────────


def get_species_dirs(hf_parent: str) -> list:
    """Get list of species directories from HF."""
    items = list_hf_dir(hf_parent)
    return [
        item for item in items
        if item.get("type") == "directory"
    ]


def normalize_slug(slug: str) -> str:
    """Normalize species slug: 'Abisara-attenuata' → 'Abisara_attenuata'."""
    return slug.replace("-", "_")


def pull_dataset(
    dataset: str,
    base_dir: str,
    batch_size: int,
    download_threads: int,
):
    """
    Pull a dataset (butterflies or plants) from HF.

    Args:
        dataset: "butterflies" or "plants"
        base_dir: base output directory (default: current dir)
        batch_size: number of species per batch
        download_threads: parallel downloads per species
    """
    if dataset == "butterflies":
        hf_parent = "data/butterflies/raw"
        local_parent = os.path.join(base_dir, "data", "butterflies", "raw")
    elif dataset == "plants":
        hf_parent = "data/plants/raw"
        local_parent = os.path.join(base_dir, "data", "plants", "raw")
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    logger.info(f"Pulling {dataset} from HF:{hf_parent} → {local_parent}")

    # Get species list from HF
    species_dirs = get_species_dirs(hf_parent)
    logger.info(f"Found {len(species_dirs)} species directories on HF")

    # Filter out already-downloaded species
    pending = []
    skipped = 0
    for sp_item in species_dirs:
        slug = os.path.basename(sp_item["path"])
        local_slug = normalize_slug(slug)
        local_dir = os.path.join(local_parent, local_slug)
        manifest = os.path.join(local_dir, ".downloaded")

        if os.path.exists(manifest):
            skipped += 1
            continue

        pending.append((sp_item, slug, local_slug, local_dir))

    logger.info(
        f"{skipped} species already downloaded, {len(pending)} pending"
    )

    if not pending:
        logger.info("Nothing to download!")
        return

    # Process in batches
    total_downloaded = 0
    total_files = 0

    for batch_start in range(0, len(pending), batch_size):
        batch = pending[batch_start : batch_start + batch_size]
        batch_num = batch_start // batch_size + 1
        total_batches = (len(pending) + batch_size - 1) // batch_size

        logger.info(
            f"Batch {batch_num}/{total_batches}: "
            f"processing {len(batch)} species"
        )

        for sp_item, slug, local_slug, local_dir in batch:
            hf_path = sp_item["path"]
            count = download_species(
                hf_path, local_dir, local_slug, download_threads
            )

            if count > 0:
                # Write manifest to mark as downloaded
                os.makedirs(local_dir, exist_ok=True)
                with open(os.path.join(local_dir, ".downloaded"), "w") as f:
                    f.write(f"{count} files downloaded\n")

                total_downloaded += 1
                total_files += count
                logger.info(
                    f"  ✓ {local_slug}: {count} files "
                    f"({total_downloaded}/{len(pending)})"
                )
            else:
                logger.warning(f"  ✗ {local_slug}: no files downloaded")

        # Small pause between batches to be nice to HF API
        if batch_start + batch_size < len(pending):
            logger.info("Pausing 2s between batches...")
            time.sleep(2)

    logger.info(
        f"\nDone! Downloaded {total_files} files "
        f"across {total_downloaded} species."
    )


# ── Also download root-level metadata files ────────────────────────────────────


def pull_root_metadata(dataset: str, base_dir: str):
    """Download root-level metadata files (registry.json, metadata.jsonl)."""
    if dataset == "plants":
        root_files = ["data/plants/raw/registry.json", "data/plants/raw/metadata.jsonl"]
        local_parent = os.path.join(base_dir, "data", "plants", "raw")
    elif dataset == "butterflies":
        # Butterfly metadata is per-species (inside each dir), no root files
        return
    else:
        return

    for remote_path in root_files:
        filename = os.path.basename(remote_path)
        local_path = os.path.join(local_parent, filename)
        if os.path.exists(local_path):
            logger.info(f"  Root metadata {filename} already exists, skipping")
            continue
        logger.info(f"  Downloading root metadata: {filename}")
        download_file(remote_path, local_path)


# ── CLI ────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Pull IndoLepAtlas datasets from Hugging Face"
    )
    parser.add_argument(
        "--dataset",
        choices=["butterflies", "plants", "all"],
        default="all",
        help="Which dataset to pull (default: all)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Number of species to process per batch (default: 50)",
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default=".",
        help="Base output directory (default: current directory)",
    )
    parser.add_argument(
        "--download-threads",
        type=int,
        default=4,
        help="Parallel file downloads per species (default: 4)",
    )
    args = parser.parse_args()

    if not HF_TOKEN:
        logger.error("HF_TOKEN not set. Add it to your .env file.")
        sys.exit(1)

    datasets = (
        ["butterflies", "plants"] if args.dataset == "all" else [args.dataset]
    )

    for ds in datasets:
        logger.info(f"\n{'='*60}")
        logger.info(f"Pulling dataset: {ds}")
        logger.info(f"{'='*60}")
        pull_root_metadata(ds, args.base_dir)
        pull_dataset(ds, args.base_dir, args.batch_size, args.download_threads)


if __name__ == "__main__":
    main()

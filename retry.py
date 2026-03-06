#!/usr/bin/env python3
"""
retry.py - Retry failed species from a previous crawler run.

Reads failed_species.log, filters out any already completed species,
and re-attempts scraping + uploading them.

Usage:
    python retry.py                    # Retry all failed species
    python retry.py --max 50           # Retry at most 50 species
    python retry.py --delay 5          # 5 second delay between species
"""
import os
import sys
import time
import json
import base64
import logging
import argparse
import requests
import shutil
from tqdm import tqdm
from dotenv import load_dotenv

from scraper_prototype import scrape_species_page

# Load environment variables
load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")
REPO_ID = "AXE8/IndoLepAtlas"

COMPLETED_LOG = "completed_species.log"
FAILED_LOG = "failed_species.log"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_completed_species():
    completed = set()
    if os.path.exists(COMPLETED_LOG):
        with open(COMPLETED_LOG, "r") as f:
            for line in f:
                completed.add(line.strip())
    return completed


def load_failed_urls():
    if not os.path.exists(FAILED_LOG):
        logging.error(f"No {FAILED_LOG} found. Nothing to retry.")
        return []
    with open(FAILED_LOG, "r") as f:
        urls = [line.strip() for line in f if line.strip()]
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for url in urls:
        if url not in seen:
            seen.add(url)
            unique.append(url)
    return unique


def upload_batch(batch_dir, slug):
    """Upload all files in a batch folder to HF via REST API."""
    files_list = [f for f in os.listdir(batch_dir) if not f.startswith('.')]
    if not files_list:
        shutil.rmtree(batch_dir)
        return 0

    commit_url = f"https://huggingface.co/api/datasets/{REPO_ID}/commit/main"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}

    operations = []
    for filename in files_list:
        filepath = os.path.join(batch_dir, filename)
        with open(filepath, "rb") as f:
            content = f.read()
        encoded = base64.b64encode(content).decode("ascii")
        operations.append({
            "op": "addOrUpdate",
            "path": f"data/{slug}/{filename}",
            "encoding": "base64",
            "content": encoded,
        })

    payload = {
        "summary": f"Retry: Add {slug} ({len(files_list)} files)",
        "operations": operations,
    }

    resp = requests.post(commit_url, headers=headers, json=payload, timeout=120)
    resp.raise_for_status()

    shutil.rmtree(batch_dir)
    return len(files_list)


def main():
    parser = argparse.ArgumentParser(description="Retry failed species scrapes")
    parser.add_argument("--max", type=int, default=None, help="Max species to retry")
    parser.add_argument("--delay", type=float, default=3.0, help="Delay between species (seconds)")
    args = parser.parse_args()

    # Load completed and failed lists
    completed = load_completed_species()
    failed_urls = load_failed_urls()

    if not failed_urls:
        return

    # Filter out any that have since been completed
    to_retry = [url for url in failed_urls if url.rstrip('/').split('/')[-1] not in completed]
    already_done = len(failed_urls) - len(to_retry)

    if already_done:
        logging.info(f"{already_done} previously-failed species are now completed. Skipping them.")

    if args.max:
        to_retry = to_retry[:args.max]

    logging.info(f"Retrying {len(to_retry)} species...")

    # Verify HF
    hf_ready = False
    if HF_TOKEN:
        headers = {"Authorization": f"Bearer {HF_TOKEN}"}
        resp = requests.get(f"https://huggingface.co/api/datasets/{REPO_ID}", headers=headers, timeout=10)
        hf_ready = resp.status_code == 200
        if hf_ready:
            logging.info(f"HF repo {REPO_ID} verified.")
        else:
            logging.warning(f"HF repo check returned {resp.status_code}. Uploads disabled.")

    master_dir = "dataset_test_run"
    os.makedirs(master_dir, exist_ok=True)

    success_count = 0
    fail_count = 0
    images_synced = 0
    still_failing = []

    pbar = tqdm(to_retry, desc="Retrying failed species", unit="species")
    for url in pbar:
        slug = url.rstrip('/').split('/')[-1]
        batch_dir = os.path.join(master_dir, f"batch_{slug}")

        try:
            success, dl_count = scrape_species_page(url, output_dir=batch_dir)

            if not success:
                raise Exception("Scraper returned failure.")

            if hf_ready and os.path.exists(batch_dir):
                upload_batch(batch_dir, slug)

            success_count += 1
            images_synced += dl_count

            # Log as completed
            with open(COMPLETED_LOG, "a") as f:
                f.write(f"{slug}\n")

        except Exception as e:
            fail_count += 1
            still_failing.append(url)
            logging.debug(f"Retry failed for {slug}: {e}")

        pbar.set_postfix({"OK": success_count, "Fail": fail_count, "Imgs": images_synced})
        time.sleep(args.delay)

    # Rewrite failed_species.log with only the still-failing URLs
    with open(FAILED_LOG, "w") as f:
        for url in still_failing:
            f.write(f"{url}\n")

    logging.info(f"Retry complete! Recovered: {success_count}, Still failing: {fail_count}, Images: {images_synced}")
    if still_failing:
        logging.info(f"{FAILED_LOG} updated with {len(still_failing)} remaining failures.")


if __name__ == "__main__":
    main()

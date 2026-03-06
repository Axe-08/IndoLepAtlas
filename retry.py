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
REPO_ID = "DihelseeWee/IndoLepAtlas"

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


def pull_logs_from_hf():
    """Download existing logs from HF and merge them locally."""
    logging.info("Pulling sync logs from Hugging Face...")
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    for log_file in [COMPLETED_LOG, FAILED_LOG]:
        url = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/{log_file}"
        try:
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code == 200:
                existing = set()
                if os.path.exists(log_file):
                    with open(log_file, "r") as f:
                        existing.update(line.strip() for line in f if line.strip())
                        
                remote_lines = resp.text.splitlines()
                existing.update(line.strip() for line in remote_lines if line.strip())
                
                with open(log_file, "w") as f:
                    for line in sorted(existing):
                        f.write(line + "\n")
                logging.info(f"Merged remote {log_file} ({len(remote_lines)} lines). Total now: {len(existing)}")
        except Exception as e:
            logging.debug(f"Could not pull {log_file}: {e}")


def push_logs_to_hf():
    """Upload the current local logs to Hugging Face."""
    logging.info("Pushing sync logs to Hugging Face...")
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    commit_url = f"https://huggingface.co/api/datasets/{REPO_ID}/commit/main"
    
    lines = [json.dumps({"key": "header", "value": {"summary": "Sync progress logs (Retry Run)"}})]
    
    for log_file in [COMPLETED_LOG, FAILED_LOG]:
        if os.path.exists(log_file):
            with open(log_file, "rb") as f:
                content = f.read()
            b64 = base64.b64encode(content).decode('ascii')
            lines.append(json.dumps({
                "key": "file",
                "value": {"path": log_file, "encoding": "base64", "content": b64}
            }))
            
    if len(lines) > 1:
        ndjson_body = "\n".join(lines)
        try:
            resp = requests.post(
                commit_url, 
                headers={**headers, "Content-Type": "application/x-ndjson"},
                data=ndjson_body.encode('utf-8'),
                timeout=30
            )
            resp.raise_for_status()
            logging.info("Sync logs pushed successfully.")
        except Exception as e:
            logging.error(f"Failed to push sync logs: {e}")


def upload_batch(batch_dir, slug):
    """Upload all files in a batch folder to HF via REST API, cleanly handling Git LFS."""
    import hashlib
    files_list = [f for f in os.listdir(batch_dir) if not f.startswith('.')]
    if not files_list:
        shutil.rmtree(batch_dir)
        return 0

    commit_url = f"https://huggingface.co/api/datasets/{REPO_ID}/commit/main"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}

    # 1. Prepare files
    lfs_objects = []
    file_metadata = {}

    for filename in files_list:
        filepath = os.path.join(batch_dir, filename)
        with open(filepath, "rb") as f:
            content = f.read()
        size = len(content)
        sha256 = hashlib.sha256(content).hexdigest()
        is_lfs = filename.lower().endswith(('.jpg', '.jpeg', '.png'))

        file_metadata[filename] = {
            'content': content,
            'size': size,
            'sha256': sha256,
            'is_lfs': is_lfs
        }
        if is_lfs:
            lfs_objects.append({'oid': sha256, 'size': size})

    # 2. Upload LFS blobs to S3
    if lfs_objects:
        lfs_url = f"https://huggingface.co/datasets/{REPO_ID}.git/info/lfs/objects/batch"
        lfs_headers = {**headers, "Content-Type": "application/vnd.git-lfs+json", "Accept": "application/vnd.git-lfs+json"}
        lfs_payload = {
            "operation": "upload",
            "transfers": ["basic"],
            "objects": lfs_objects
        }
        resp = requests.post(lfs_url, headers=lfs_headers, json=lfs_payload, timeout=60)
        resp.raise_for_status()

        for obj in resp.json().get('objects', []):
            if 'actions' in obj and 'upload' in obj['actions']:
                up_action = obj['actions']['upload']
                content_to_upload = next(m['content'] for m in file_metadata.values() if m['sha256'] == obj['oid'])
                up_resp = requests.put(up_action['href'], headers=up_action.get('header', {}), data=content_to_upload, timeout=120)
                up_resp.raise_for_status()

    # 3. Commit via NDJSON
    lines = [json.dumps({"key": "header", "value": {"summary": f"Retry: Add {slug} ({len(files_list)} files)"}})]

    for filename, meta in file_metadata.items():
        path_in_repo = f"data/{slug}/{filename}"
        if meta['is_lfs']:
            lines.append(json.dumps({
                "key": "lfsFile",
                "value": {"path": path_in_repo, "algo": "sha256", "oid": meta['sha256'], "size": meta['size']}
            }))
        else:
            b64 = base64.b64encode(meta['content']).decode('ascii')
            lines.append(json.dumps({
                "key": "file",
                "value": {"path": path_in_repo, "encoding": "base64", "content": b64}
            }))

    ndjson_body = "\n".join(lines)
    commit_resp = requests.post(
        commit_url,
        headers={**headers, "Content-Type": "application/x-ndjson"},
        data=ndjson_body.encode('utf-8'),
        timeout=120
    )
    commit_resp.raise_for_status()

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
            pull_logs_from_hf()
        else:
            logging.warning(f"HF repo check returned {resp.status_code}. Uploads disabled.")

    master_dir = "dataset_test_run"
    os.makedirs(master_dir, exist_ok=True)

    success_count = 0
    fail_count = 0
    images_synced = 0
    still_failing = []

    try:
        pbar = tqdm(to_retry, desc="Retrying failed species", unit="species")
        for url in pbar:
            slug = url.rstrip('/').split('/')[-1]
            batch_dir = os.path.join(master_dir, f"batch_{slug}")

            try:
                success, dl_count = scrape_species_page(url, output_dir=batch_dir, pbar=pbar)

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
    finally:
        if hf_ready:
            push_logs_to_hf()


if __name__ == "__main__":
    main()

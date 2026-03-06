import os
import json
import time
import math
import base64
import logging
import argparse
import requests
import shutil
from tqdm import tqdm
from bs4 import BeautifulSoup
from urllib.parse import urljoin
# NOTE: We use the HF REST API directly via requests (huggingface_hub v0.4.0 is too old)
from dotenv import load_dotenv

from scraper_prototype import scrape_species_page

# Load environment variables
load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")
REPO_ID = "AXE8/IndoLepAtlas"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

BASE_URL = "https://www.ifoundbutterflies.org"
SPECIES_LIST_URL = "https://www.ifoundbutterflies.org/species-list"

def get_species_links(limit=None):
    """
    Scrape the main species list page to get links to individual species pages.
    Limits to `limit` number of species if specified, else all.
    """
    logging.info(f"Fetching species list from {SPECIES_LIST_URL}")
    
    max_retries = 3
    response = None
    for attempt in range(max_retries):
        try:
            # Increased timeout to 30s as the site can occasionally be very slow
            response = requests.get(SPECIES_LIST_URL, timeout=30)
            response.raise_for_status()
            break
        except requests.exceptions.RequestException as e:
            logging.warning(f"Attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                logging.error("Max retries reached. Aborting.")
                raise
            time.sleep(5)
    
    soup = BeautifulSoup(response.text, 'html.parser')

    species_links = []
    
    main_content = soup.find('div', class_='region-content') or soup
    
    for a in main_content.find_all('a', href=True):
        href = a['href']
        # Species links usually look like /Genus-species, eg: /Papilio-machaon
        if href.startswith('/') and len(href) > 2 and href[1].isupper() and '-' in href:
            if not any(ext in href for ext in ['.css', '.js', '.png', '.jpg', '?', 'user', 'about', 'contact']):
                full_url = urljoin(BASE_URL, href)
                # Avoid duplicates
                if full_url not in species_links:
                    species_links.append(full_url)
                    
        if limit and len(species_links) >= limit:
            break
            
    return species_links

def verify_hf_repo():
    """Verify the HF repo exists using the REST API. Returns True if ready."""
    if not HF_TOKEN:
        logging.warning("No HF_TOKEN found. Will scrape locally without cloud sync.")
        return False

    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    
    # Check if the repo exists
    check_url = f"https://huggingface.co/api/datasets/{REPO_ID}"
    resp = requests.get(check_url, headers=headers, timeout=10)
    
    if resp.status_code == 200:
        logging.info(f"HF repo {REPO_ID} exists and is accessible.")
        return True
    elif resp.status_code == 404:
        # Try to create it
        create_url = "https://huggingface.co/api/repos/create"
        payload = {"type": "dataset", "name": REPO_ID.split('/')[-1], "organization": REPO_ID.split('/')[0]}
        create_resp = requests.post(create_url, headers=headers, json=payload, timeout=10)
        if create_resp.status_code in (200, 201):
            logging.info(f"Created HF repo {REPO_ID}.")
            return True
        else:
            logging.error(f"Failed to create repo: {create_resp.status_code} {create_resp.text}")
            return False
    else:
        logging.error(f"Failed to verify repo: {resp.status_code} {resp.text}")
        return False


def upload_batch(batch_dir, slug):
    """
    Upload all files in a batch folder to HF via REST API, properly handling Git LFS.
    Deletes the local folder after success. Returns file count.
    """
    import hashlib
    files_list = [f for f in os.listdir(batch_dir) if not f.startswith('.')]
    if not files_list:
        shutil.rmtree(batch_dir)
        return 0
    
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
                # Find matching file content
                content_to_upload = next(m['content'] for m in file_metadata.values() if m['sha256'] == obj['oid'])
                up_resp = requests.put(up_action['href'], headers=up_action.get('header', {}), data=content_to_upload, timeout=120)
                up_resp.raise_for_status()
                
    # 3. Commit via NDJSON
    commit_url = f"https://huggingface.co/api/datasets/{REPO_ID}/commit/main"
    lines = [json.dumps({"key": "header", "value": {"summary": f"Add {slug} ({len(files_list)} files)"}})]
    
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


def reupload_existing_batches(master_dir):
    """Scan for leftover batch folders from a previous run and upload them."""
    if not os.path.exists(master_dir):
        return 0, 0
    
    batch_dirs = [d for d in os.listdir(master_dir)
                  if d.startswith("batch_") and os.path.isdir(os.path.join(master_dir, d))]
    
    if not batch_dirs:
        return 0, 0
    
    logging.info(f"Found {len(batch_dirs)} leftover batch folders. Re-uploading...")
    uploaded = 0
    files_uploaded = 0
    
    pbar = tqdm(batch_dirs, desc="Re-uploading leftovers", unit="batch")
    for batch_name in pbar:
        batch_path = os.path.join(master_dir, batch_name)
        slug = batch_name.replace("batch_", "", 1)
        try:
            count = upload_batch(batch_path, slug)
            uploaded += 1
            files_uploaded += count
            pbar.set_postfix({"Uploaded": uploaded, "Files": files_uploaded})
        except Exception as e:
            logging.error(f"Failed to re-upload {batch_name}: {e}")
    
    logging.info(f"Re-upload complete. {uploaded}/{len(batch_dirs)} batches, {files_uploaded} files.")
    return uploaded, files_uploaded


COMPLETED_LOG = "completed_species.log"
FAILED_LOG = "failed_species.log"
CIRCUIT_BREAKER_THRESHOLD = 10  # consecutive fails before pause
CIRCUIT_BREAKER_PAUSE = 1800   # 30 minutes in seconds
CIRCUIT_BREAKER_MAX_PAUSES = 3 # stop after this many pauses


def load_completed_species():
    """Load the set of already-completed species slugs."""
    completed = set()
    if os.path.exists(COMPLETED_LOG):
        with open(COMPLETED_LOG, "r") as f:
            for line in f:
                completed.add(line.strip())
    return completed


def pull_logs_from_hf():
    """Download existing logs from HF and merge them locally."""
    logging.info("Pulling sync logs from Hugging Face...")
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    for log_file in [COMPLETED_LOG, FAILED_LOG]:
        url = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/{log_file}"
        try:
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code == 200:
                # Merge remote with local
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
    
    lines = [json.dumps({"key": "header", "value": {"summary": "Sync progress logs"}})]
    
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


def run_crawler(chunk=1, total_chunks=1):
    logging.info(f"Starting production crawler (Chunk {chunk}/{total_chunks})...")
    
    master_dir = "dataset_test_run"
    os.makedirs(master_dir, exist_ok=True)
    
    # --- Phase 0: Verify HF repo ---
    hf_ready = verify_hf_repo()
    
    # --- Phase 1: Pull HF Logs & Re-upload leftover batches ---
    reupload_count = 0
    reupload_files = 0
    if hf_ready:
        pull_logs_from_hf()
        reupload_count, reupload_files = reupload_existing_batches(master_dir)
    
    # --- Phase 2: Load completed species to skip ---
    completed = load_completed_species()
    if completed:
        logging.info(f"Loaded {len(completed)} already-completed species. Will skip them.")
    
    # --- Phase 3: Scrape new species ---
    links = get_species_links(limit=None)
    
    if not links:
        logging.error("Found no species links.")
        return
        
    # Apply chunking
    if total_chunks > 1:
        chunk_size = math.ceil(len(links) / total_chunks)
        start_idx = (chunk - 1) * chunk_size
        end_idx = start_idx + chunk_size
        links = links[start_idx:end_idx]
        logging.info(f"Assigned chunk {chunk}/{total_chunks}: {len(links)} species (indices {start_idx} to {min(end_idx, len(links))-1})")
    
    # Filter out already-completed species
    remaining = [url for url in links if url.rstrip('/').split('/')[-1] not in completed]
    skipped = len(links) - len(remaining)
    
    logging.info(f"Found {len(links)} total species. Skipping {skipped} already done. {len(remaining)} remaining.")

    success_count = reupload_count + len(completed)
    fail_count = 0
    images_synced = reupload_files
    consecutive_fails = 0
    pause_count = 0
    
    try:
        pbar = tqdm(remaining, desc="Scraping Species", unit="species")
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
                consecutive_fails = 0  # Reset on success
                
                # Log completed species
                with open(COMPLETED_LOG, "a") as f:
                    f.write(f"{slug}\n")
                
            except Exception as e:
                fail_count += 1
                consecutive_fails += 1
                with open(FAILED_LOG, "a") as f:
                    f.write(f"{url}\n")
                logging.debug(f"Error processing {slug}: {e}")
                
                # --- Circuit Breaker ---
                if consecutive_fails >= CIRCUIT_BREAKER_THRESHOLD:
                    pause_count += 1
                    if pause_count >= CIRCUIT_BREAKER_MAX_PAUSES:
                        logging.error(f"Circuit breaker tripped {CIRCUIT_BREAKER_MAX_PAUSES} times. Server appears down. Stopping.")
                        break
                    logging.warning(f"Circuit breaker: {consecutive_fails} consecutive fails. Pausing {CIRCUIT_BREAKER_PAUSE//60} min (pause {pause_count}/{CIRCUIT_BREAKER_MAX_PAUSES})...")
                    time.sleep(CIRCUIT_BREAKER_PAUSE)
                    consecutive_fails = 0  # Reset after pause
                
            pbar.set_postfix({"OK": success_count, "Fail": fail_count, "Imgs": images_synced})
            time.sleep(3)
                
        logging.info(f"Crawler complete! Success: {success_count}, Fails: {fail_count}, Images: {images_synced}")
    finally:
        if hf_ready:
            push_logs_to_hf()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="IndoLepAtlas Distributed Crawler")
    parser.add_argument("--chunk", type=int, default=1, help="Which chunk of the workload to run (1-indexed)")
    parser.add_argument("--total-chunks", type=int, default=1, help="Total number of chunks to divide the workload into")
    args = parser.parse_args()
    
    if args.chunk < 1 or args.chunk > args.total_chunks:
        logging.error("--chunk must be between 1 and --total-chunks")
        exit(1)
        
    run_crawler(chunk=args.chunk, total_chunks=args.total_chunks)

import os
import json
import time
import base64
import logging
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
    Upload all files in a batch folder to HF in a single commit via the REST API.
    Uses JSON payload with base64-encoded file content.
    Deletes the local folder after success. Returns file count.
    """
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
        "summary": f"Add {slug} ({len(files_list)} files)",
        "operations": operations,
    }
    
    resp = requests.post(commit_url, headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    
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


def run_crawler():
    logging.info("Starting production crawler...")
    
    master_dir = "dataset_test_run"
    os.makedirs(master_dir, exist_ok=True)
    
    # --- Phase 0: Verify HF repo ---
    hf_ready = verify_hf_repo()
    
    # --- Phase 1: Re-upload any leftover batches from previous runs ---
    reupload_count = 0
    reupload_files = 0
    if hf_ready:
        reupload_count, reupload_files = reupload_existing_batches(master_dir)
    
    # --- Phase 2: Scrape new species ---
    links = get_species_links(limit=None)
    
    if not links:
        logging.error("Found no species links.")
        return
        
    logging.info(f"Found {len(links)} species. Starting scrape loop...")

    success_count = reupload_count
    fail_count = 0
    images_synced = reupload_files
    
    pbar = tqdm(links, desc="Scraping Species", unit="species")
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
            
        except Exception as e:
            fail_count += 1
            with open("failed_species.log", "a") as f:
                f.write(f"{url}\n")
            logging.debug(f"Error processing {slug}: {e}")
            
        pbar.set_postfix({"OK": success_count, "Fail": fail_count, "Imgs": images_synced})
        time.sleep(3)
            
    logging.info(f"Crawler complete! Success: {success_count}, Fails: {fail_count}, Images: {images_synced}")

if __name__ == '__main__':
    run_crawler()

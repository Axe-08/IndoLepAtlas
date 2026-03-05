import os
import json
import time
import logging
import requests
import shutil
from tqdm import tqdm
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from huggingface_hub import HfApi, create_repo
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

def setup_hf_api():
    """Set up and return a Hugging Face API client, or None if unavailable."""
    if not HF_TOKEN:
        logging.warning("No HF_TOKEN found. Will scrape locally without cloud sync.")
        return None

    try:
        from huggingface_hub.hf_api import HfFolder
        HfFolder.save_token(HF_TOKEN)
    except ImportError:
        pass

    try:
        api = HfApi(token=HF_TOKEN)
    except TypeError:
        api = HfApi()

    try:
        create_repo(repo_id=REPO_ID, repo_type="dataset", exist_ok=True, token=HF_TOKEN)
        logging.info(f"Targeting dataset repo: {REPO_ID}")
    except TypeError:
        try:
            if '/' in REPO_ID:
                org, name = REPO_ID.split('/')
                create_repo(name=name, organization=org, repo_type="dataset", exist_ok=True, token=HF_TOKEN)
            else:
                create_repo(name=REPO_ID, repo_type="dataset", exist_ok=True, token=HF_TOKEN)
            logging.info(f"Targeting dataset repo: {REPO_ID}")
        except Exception as e:
            logging.error(f"Failed to verify/create repo {REPO_ID} (legacy): {e}")
            return None
    except Exception as e:
        logging.error(f"Failed to verify/create repo {REPO_ID}: {e}")
        return None

    return api


def upload_batch(api, batch_dir, slug):
    """Upload a single batch folder to Hugging Face and delete it locally. Returns file count."""
    file_count = len([f for f in os.listdir(batch_dir) if not f.startswith('.')])
    
    if hasattr(api, "upload_folder"):
        api.upload_folder(
            repo_id=REPO_ID,
            repo_type="dataset",
            folder_path=batch_dir,
            path_in_repo=f"data/{slug}"
        )
    else:
        for filename in os.listdir(batch_dir):
            filepath = os.path.join(batch_dir, filename)
            path_in_repo = f"data/{slug}/{filename}"
            try:
                api.upload_file(
                    path_or_fileobj=filepath,
                    path_in_repo=path_in_repo,
                    repo_id=REPO_ID,
                    repo_type="dataset",
                    token=HF_TOKEN
                )
            except TypeError:
                api.upload_file(
                    path_or_fileobj=filepath,
                    path_in_repo=path_in_repo,
                    repo_id=REPO_ID,
                    repo_type="dataset"
                )
    
    shutil.rmtree(batch_dir)
    return file_count


def reupload_existing_batches(api, master_dir):
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
            count = upload_batch(api, batch_path, slug)
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
    
    # --- Phase 0: Set up HF API ---
    api = setup_hf_api()
    
    # --- Phase 1: Re-upload any leftover batches from previous runs ---
    reupload_count = 0
    reupload_files = 0
    if api:
        reupload_count, reupload_files = reupload_existing_batches(api, master_dir)
    
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
        
        # Skip if this species was already uploaded (batch folder gone)
        # but still exists as a directory (needs upload)
        
        try:
            success, dl_count = scrape_species_page(url, output_dir=batch_dir)
            
            if not success:
                raise Exception("Scraper returned failure.")
            
            if api and os.path.exists(batch_dir):
                upload_batch(api, batch_dir, slug)
            
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

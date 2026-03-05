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

def run_crawler_test():
    logging.info("Starting production crawler...")
    links = get_species_links(limit=None)
    
    if not links:
        logging.error("Found no species links to test.")
        return
        
    logging.info(f"Found {len(links)} species. Starting scrape loop...")
    
    # Set up a master output directory for the crawler test
    master_dir = "dataset_test_run"
    os.makedirs(master_dir, exist_ok=True)
    
    # Set up Hugging Face API
    api = None
    if HF_TOKEN:
        try:
            # Login globally so old versions pick it up automatically
            from huggingface_hub.hf_api import HfFolder
            HfFolder.save_token(HF_TOKEN)
        except ImportError:
            pass

        try:
            # New huggingface_hub
            api = HfApi(token=HF_TOKEN)
        except TypeError:
            # Legacy huggingface_hub (e.g. v0.4.0 on Python 3.6)
            api = HfApi()

        try:
            create_repo(repo_id=REPO_ID, repo_type="dataset", exist_ok=True, token=HF_TOKEN)
            logging.info(f"Targeting dataset repo: {REPO_ID}")
        except TypeError:
            try:
                # Legacy version (v0.4.0) uses 'name' and 'organization' instead of 'repo_id'
                if '/' in REPO_ID:
                    org, name = REPO_ID.split('/')
                    create_repo(name=name, organization=org, repo_type="dataset", exist_ok=True, token=HF_TOKEN)
                else:
                    create_repo(name=REPO_ID, repo_type="dataset", exist_ok=True, token=HF_TOKEN)
                logging.info(f"Targeting dataset repo: {REPO_ID}")
            except Exception as e:
                logging.error(f"Failed to verify/create repo {REPO_ID} (legacy): {e}")
                api = None
        except Exception as e:
            logging.error(f"Failed to verify/create repo {REPO_ID}: {e}")
            api = None
    else:
        logging.warning("No HF_TOKEN found. Will scrape locally without cloud sync.")

    success_count = 0
    fail_count = 0
    images_synced = 0
    
    # Progress tracking loop
    pbar = tqdm(links, desc="Scraping Species", unit="species")
    for url in pbar:
        # Create a unique batch folder for each species based on its URL slug
        slug = url.rstrip('/').split('/')[-1]
        batch_dir = os.path.join(master_dir, f"batch_{slug}")
        
        try:
            # We reuse the robust logic from the prototype
            success, dl_count = scrape_species_page(url, output_dir=batch_dir)
            
            if not success:
                raise Exception("Scraper prototype returned failure.")
            
            # --- Cloud Sync & Local Clean --
            if api and os.path.exists(batch_dir):
                logging.info(f"Syncing {batch_dir} to Hugging Face...")
                if hasattr(api, "upload_folder"):
                    # Modern huggingface_hub method
                    api.upload_folder(
                        repo_id=REPO_ID,
                        repo_type="dataset",
                        folder_path=batch_dir,
                        path_in_repo=f"data/{slug}" # Keep species separate in HF
                    )
                else:
                    # Legacy fallback for old huggingface_hub versions (like v0.4.0)
                    for filename in os.listdir(batch_dir):
                        filepath = os.path.join(batch_dir, filename)
                        path_in_repo = f"data/{slug}/{filename}"
                        try:
                            # Try modern arg
                            api.upload_file(
                                path_or_fileobj=filepath,
                                path_in_repo=path_in_repo,
                                repo_id=REPO_ID,
                                repo_type="dataset",
                                token=HF_TOKEN
                            )
                        except TypeError:
                            # Try old arg syntax
                            api.upload_file(
                                path_or_fileobj=filepath,
                                path_in_repo=path_in_repo,
                                repo_id=REPO_ID,
                                repo_type="dataset"
                            )
                logging.debug(f"Upload successful. Deleting local batch folder {batch_dir}")
                shutil.rmtree(batch_dir)
            
            success_count += 1
            images_synced += dl_count
            
        except Exception as e:
            fail_count += 1
            with open("failed_species.log", "a") as f:
                f.write(f"{url}\n")
            logging.debug(f"Error scraping or syncing {url}: {e}")
            
        # Update progress bar metrics
        pbar.set_postfix({"Success": success_count, "Fails": fail_count, "Images Synced": images_synced})
            
        # Respectful rate limiting: wait 3 seconds before hitting the next species page
        time.sleep(3)
            
    logging.info(f"Crawler completion. Total success: {success_count}, Total fails: {fail_count}, Total images synced: {images_synced}")

if __name__ == '__main__':
    run_crawler_test()

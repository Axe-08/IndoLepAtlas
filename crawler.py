import os
import json
import time
import logging
import requests
import shutil
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

def get_species_links(limit=2):
    """
    Scrape the main species list page to get links to individual species pages.
    Limits to `limit` number of species for testing.
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
    
    # Let's find links inside the main views container, ignoring header links
    # Often, species links have italic text `<em>` or are nested in specific divs
    # Let's just find links that have a capitalized first letter, lowercase second word (Binomial)
    # Or simply grab links from the main content region
    
    main_content = soup.find('div', class_='region-content') or soup
    
    for a in main_content.find_all('a', href=True):
        href = a['href']
        # Species links usually look like /Genus-species, eg: /Papilio-machaon
        # We can check if it starts with a capital letter after the slash
        if href.startswith('/') and len(href) > 2 and href[1].isupper() and '-' in href:
            if not any(ext in href for ext in ['.css', '.js', '.png', '.jpg', '?', 'user', 'about', 'contact']):
                full_url = urljoin(BASE_URL, href)
                # Avoid duplicates
                if full_url not in species_links:
                    species_links.append(full_url)
                    
        if len(species_links) >= limit:
            break
            
    return species_links

def run_crawler_test():
    logging.info("Starting crawler test...")
    links = get_species_links(limit=2)
    
    if not links:
        logging.error("Found no species links to test.")
        return
        
    logging.info(f"Found {len(links)} links to test: {links}")
    
    # Set up a master output directory for the crawler test
    master_dir = "dataset_test_run"
    os.makedirs(master_dir, exist_ok=True)
    
    # Set up Hugging Face API
    api = None
    if HF_TOKEN:
        api = HfApi(token=HF_TOKEN)
        try:
            create_repo(repo_id=REPO_ID, repo_type="dataset", exist_ok=True, token=HF_TOKEN)
            logging.info(f"Targeting dataset repo: {REPO_ID}")
        except Exception as e:
            logging.error(f"Failed to verify/create repo {REPO_ID}: {e}")
            api = None
    else:
        logging.warning("No HF_TOKEN found. Will scrape locally without cloud sync.")

    for url in links:
        # Create a unique batch folder for each species based on its URL slug
        slug = url.rstrip('/').split('/')[-1]
        batch_dir = os.path.join(master_dir, f"batch_{slug}")
        
        logging.info(f"--- Scraping Species: {slug} ---")
        try:
            # We reuse the robust logic from the prototype
            scrape_species_page(url, output_dir=batch_dir)
            
            # --- Cloud Sync & Local Clean --
            if api and os.path.exists(batch_dir):
                logging.info(f"Syncing {batch_dir} to Hugging Face...")
                # Upload the folder content into a path inside the HF dataset repo
                api.upload_folder(
                    repo_id=REPO_ID,
                    repo_type="dataset",
                    folder_path=batch_dir,
                    path_in_repo=f"data/{slug}" # Keep species separate in HF
                )
                logging.info(f"Upload successful. Deleting local batch folder {batch_dir}")
                shutil.rmtree(batch_dir)
                
            # Respectful rate limiting: wait 3 seconds before hitting the next species page
            time.sleep(3)
                
        except Exception as e:
            logging.error(f"Error scraping or syncing {url}: {e}")
            
    logging.info("Crawler test completion.")

if __name__ == '__main__':
    run_crawler_test()

import os
import json
import time
import logging
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_taxonomy(soup):
    """Extract full taxonomy from breadcrumbs, preferring scientific names (first matched)."""
    taxonomy = {}
    for li in soup.select('ul#system-breadcrumb-listing li'):
        for a in li.find_all('a'):
            span = a.find('span')
            if span:
                classes = span.get('class', [''])
                if classes:
                    rank = classes[0].replace('style-', '')
                    name = clean_text(span.text)
                    # Use the first encountered value for each rank (scientific name)
                    if rank and rank not in taxonomy:
                        taxonomy[rank] = name
    return taxonomy

def clean_text(text):
    return " ".join(text.split()).strip()

def extract_metadata_tabs(soup):
    """Extract structured metadata from tabs."""
    data = {}
    
    # Early Stages
    early_div = soup.find('div', id='early')
    if early_div:
        paragraphs = [clean_text(p.get_text()) for p in early_div.find_all('p')]
        early_texts = [p for p in paragraphs if len(p.replace(',', '').strip()) > 0]
        data['Early Stages Text'] = early_texts
        
    # Distribution
    dist_div = soup.find('div', id='dist')
    if dist_div:
        data['Distribution'] = clean_text(dist_div.get_text(separator=' ', strip=True))
        
    # Status, Habitat and Habits
    stat_div = soup.find('div', id='stat')
    if stat_div:
        stat_data = {}
        paragraphs = stat_div.find_all('p', recursive=False)
        if paragraphs:
            stat_data['Description'] = [clean_text(p.get_text()) for p in paragraphs if clean_text(p.get_text())]
            
        table = stat_div.find('table')
        if table:
            headers = [th.get_text(strip=True) for th in table.find_all('th')]
            rows = []
            for tr in table.find_all('tr')[1:]:
                cells = [td.get_text(strip=True) for td in tr.find_all(['td', 'th'])]
                if cells:
                    row_data = {}
                    for i, cell in enumerate(cells):
                        if i < len(headers):
                            val = cell
                            if val.isdigit(): val = int(val)
                            row_data[headers[i]] = val
                    if row_data:
                        rows.append(row_data)
            stat_data['Sighting Table'] = rows
            
        data['Status, Habitats, and Habits'] = stat_data
            
    # Larval Host Plants
    larval_div = soup.find('div', id='laraval')
    if larval_div:
        data['Larval Host Plants'] = clean_text(larval_div.get_text(separator=' ', strip=True))
        
    return data

def extract_image_id(a_tag):
    inner_img = a_tag.find('img')
    alt_text = ""
    if inner_img:
        alt_text = inner_img.get('alt', '')
    elif a_tag.get('data-cbox-img-attrs'):
        try:
            attrs = json.loads(a_tag.get('data-cbox-img-attrs'))
            alt_text = attrs.get('alt', '')
        except:
            pass
    alt_text = clean_text(alt_text)
    return alt_text if alt_text else "Unknown"

def get_images_from_tab(soup, tab_id, life_stage):
    images = []
    for a in soup.select(f'div#{tab_id} a.colorbox'):
        img_url = a.get('href')
        if img_url and img_url.startswith('/'):
            img_url = "https://www.ifoundbutterflies.org" + img_url

        if img_url:
            photographer_or_id = extract_image_id(a)
            images.append({
                'url': img_url,
                'photographer_or_id': photographer_or_id,
                'life_stage': life_stage
            })
    return images

def scrape_species_page(url_or_filepath, output_dir="dataset_batch", pbar=None):
    """
    Scrape a single species page, downloading its images and metadata.
    Returns: (success: bool, downloaded_count: int)
    """
    if pbar:
        slug = url_or_filepath.rstrip('/').split('/')[-1]
        pbar.set_description(f"Fetching HTML: {slug}")
        
    logging.debug(f"Scraping: {url_or_filepath}")
    
    try:
        from tqdm import tqdm
        if url_or_filepath.startswith("http"):
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = requests.get(url_or_filepath, timeout=30)
                    response.raise_for_status()
                    html_content = response.text
                    break
                except requests.exceptions.RequestException as e:
                    tqdm.write(f"Attempt {attempt + 1}/{max_retries} failed to fetch HTML for {url_or_filepath}. Retrying in 5s...")
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(5)
        else:
            with open(url_or_filepath, "r", encoding="utf-8") as f:
                html_content = f.read()
    except Exception as e:
        tqdm.write(f"ERROR: Failed to fetch {url_or_filepath} after retries: {e}")
        return False, 0

    soup = BeautifulSoup(html_content, 'html.parser')

    title = soup.title.text.strip() if soup.title else 'Unknown Title'
    
    # Taxonomy & Metadata
    taxonomy = extract_taxonomy(soup)
    tab_metadata = extract_metadata_tabs(soup)

    scientific_name = f"{taxonomy.get('Genus', 'Unknown')}_{taxonomy.get('Species', 'Unknown')}".replace(' ', '_')

    source_url = url_or_filepath if url_or_filepath.startswith("http") else "file://" + url_or_filepath

    # Extract images from both #home and #early
    images_to_download = get_images_from_tab(soup, 'home', 'Adult/Unknown')
    early_images = get_images_from_tab(soup, 'early', 'Early Stage')
    
    # Download all available images
    sample_images = images_to_download + early_images

    os.makedirs(output_dir, exist_ok=True)
    metadata_records = []
    downloaded_count = 0

    logging.debug(f"Found {len(images_to_download)} adult & {len(early_images)} early images. Downloading all {len(sample_images)} images...")

    from tqdm import tqdm
    for i, img_data in enumerate(tqdm(sample_images, desc=f"Downloading {scientific_name}", leave=False, unit="img")):
        img_url = img_data['url']
        
        try:
            img_resp = None
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    img_resp = requests.get(img_url, stream=True, timeout=30)
                    img_resp.raise_for_status()
                    break
                except requests.exceptions.RequestException as e:
                    logging.warning(f"Attempt {attempt + 1} for image {img_url} failed: {e}")
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(2)
            
            safe_stage = img_data['life_stage'].replace('/', '-').replace(' ', '')
            ext = os.path.splitext(urlparse(img_url).path)[1]
            if not ext: ext = '.jpg'
            
            filename = f"{scientific_name}_{safe_stage}_{i+1:03d}{ext}"
            local_filepath = os.path.join(output_dir, filename)
            
            with open(local_filepath, 'wb') as f:
                for chunk in img_resp.iter_content(1024):
                    f.write(chunk)
            
            logging.debug(f"Downloaded: {filename}")
            downloaded_count += 1
            
            record = {
                "file_name": filename,
                "dataset_source": "ifoundbutterflies.org",
                "source_url": source_url,
                "image_id": img_data['photographer_or_id'],
                "life_stage": img_data['life_stage'],
                "page_title": title,
                **taxonomy,      
                **tab_metadata   
            }
            metadata_records.append(record)
            
            # Respectful rate limiting: wait 1.5 seconds between image downloads
            time.sleep(1)
            
        except Exception as e:
            logging.error(f"Failed to download image {img_url}: {e}")

    # Write metadata.jsonl
    metadata_file = os.path.join(output_dir, "metadata.jsonl")
    with open(metadata_file, "w", encoding="utf-8") as f:
        for record in metadata_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            
    logging.debug(f"Wrote {len(metadata_records)} records to {metadata_file}")
    logging.debug(f"Prototype scrape complete. Files ready in '{output_dir}'.")
    return True, downloaded_count

if __name__ == '__main__':
    test_file = "/home/akshit/Projects/IndoLepAtlas/butterflies.html"
    scrape_species_page(test_file, output_dir="dataset_batch")

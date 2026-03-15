"""
plant_scraper.py

Scrapes a single plant species page from ifoundbutterflies.org/plant-species-list.

Each plant page (e.g. /axonopus-compressus) contains:
  - Taxonomy in breadcrumbs  (Family, Genus, Species, common name)
  - A hero plant image       (ihgwraapper div)
  - A gallery of plant imgs  (a.colorbox inside species_gallery view)
  - A list of butterfly spp. this plant hosts (a.spe-a links)

Output per plant:
  host_plants/{PlantKey}/
      {PlantKey}_hero.jpg             <- main plant image
      {PlantKey}_gallery_{NNN}.jpg    <- gallery images
  host_plants/registry.json           <- append-only shared registry
  host_plants/metadata.jsonl          <- append-only shared metadata

Returns: (success: bool, image_count: int)
"""

import os
import re
import json
import time
import logging
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin

BASE_URL = "https://www.ifoundbutterflies.org"

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


# ── Helpers ────────────────────────────────────────────────────────────────────

def clean_text(text):
    return " ".join(text.split()).strip()


def make_plant_key(name):
    """Turn a plant name into a safe folder/key string."""
    key = re.sub(r'\s+', '_', name.strip())
    key = re.sub(r'[^A-Za-z0-9_\-]', '', key)
    return key or "Unknown_Plant"


def fetch_html(url, max_retries=3):
    """Fetch a URL with retries. Returns HTML string or raises."""
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            return resp.text
        except requests.exceptions.RequestException as e:
            logging.warning(f"Attempt {attempt + 1}/{max_retries} failed for {url}: {e}")
            if attempt == max_retries - 1:
                raise
            time.sleep(5)


def download_image(url, filepath, max_retries=3):
    """Download a single image to filepath. Skips if already exists."""
    if os.path.exists(filepath):
        return True
    for attempt in range(max_retries):
        try:
            r = requests.get(url, stream=True, timeout=30)
            r.raise_for_status()
            with open(filepath, 'wb') as f:
                for chunk in r.iter_content(1024):
                    f.write(chunk)
            time.sleep(1)
            return True
        except requests.exceptions.RequestException as e:
            logging.warning(f"Image attempt {attempt + 1} failed for {url}: {e}")
            if attempt == max_retries - 1:
                logging.error(f"Failed to download {url}")
                return False
            time.sleep(2)
    return False


# ── Plant page parsers ─────────────────────────────────────────────────────────

def extract_plant_taxonomy(soup):
    """
    Extract taxonomy from the breadcrumb.
    Returns dict with keys like 'Superfamily', 'Family', 'Genus', 'Species'
    and optional common names.
    """
    taxonomy = {}
    common_names = {}
    for li in soup.select('ul#system-breadcrumb-listing li'):
        for a in li.find_all('a'):
            spans = a.find_all('span')
            if len(spans) >= 1:
                cls = spans[0].get('class', [''])[0].replace('style-', '')
                sci = clean_text(spans[0].text)
                if cls and sci and cls not in taxonomy:
                    taxonomy[cls] = sci
            if len(spans) >= 2:
                common_names[cls] = clean_text(spans[1].text)
    return taxonomy, common_names


def extract_butterfly_hosts(soup):
    """
    Extract the list of butterfly species this plant hosts.
    Returns list of dicts: {'scientific_name': str, 'common_name': str, 'slug': str}
    """
    butterflies = []
    # Butterfly links are in the first .textContent div as <a class="spe-a">
    text_divs = soup.find_all('div', class_='textContent')
    for div in text_divs:
        h4 = div.find('h4')
        if h4 and 'Larval Host' or 'Nectar Plant' in h4.text:
            for a in div.find_all('a', class_='spe-a'):
                href  = a.get('href', '')
                slug  = href.rstrip('/').split('/')[-1]
                em    = a.find('em')
                sci   = clean_text(em.text) if em else clean_text(a.text.split('–')[0])
                rest  = clean_text(a.text)
                common = rest.split('–')[-1].strip() if '–' in rest else ''
                butterflies.append({
                    'scientific_name': sci,
                    'common_name':     common,
                    'slug':            slug,
                    'url':             urljoin(BASE_URL, href),
                })
    return butterflies


def extract_plant_images(soup):
    """
    Extract all plant images from the page.
    Returns list of {'url': str, 'type': 'hero'|'gallery', 'alt': str}
    """
    images = []

    # Hero image (ihgwraapper div)
    hero_div = soup.find('div', class_='ihgwraapper')
    if hero_div:
        img = hero_div.find('img')
        if img and img.get('src'):
            src = img['src']
            if src.startswith('/'):
                src = BASE_URL + src
            images.append({'url': src, 'type': 'hero', 'alt': img.get('alt', '')})

    # Gallery images (colorbox links inside the species_gallery view)
    for a in soup.select('div.view-species_gallery a.colorbox, a.colorbox'):
        href = a.get('href', '')
        if href.startswith('/'):
            href = BASE_URL + href
        if href and href not in [i['url'] for i in images]:
            # Get alt from inner img or data-cbox-img-attrs
            alt = ''
            inner = a.find('img')
            if inner:
                alt = inner.get('alt', '')
            elif a.get('data-cbox-img-attrs'):
                try:
                    alt = json.loads(a['data-cbox-img-attrs']).get('alt', '')
                except Exception:
                    pass
            images.append({'url': href, 'type': 'gallery', 'alt': clean_text(alt)})

    return images


# ── Registry helpers ───────────────────────────────────────────────────────────

def load_registry(registry_path):
    if os.path.exists(registry_path):
        with open(registry_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_registry(registry, registry_path):
    os.makedirs(os.path.dirname(os.path.abspath(registry_path)), exist_ok=True)
    with open(registry_path, 'w', encoding='utf-8') as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)


# ── Main scrape function ───────────────────────────────────────────────────────

def scrape_plant_page(url, host_plants_dir="host_plants", pbar=None):
    """
    Scrape a single plant species page.

    Args:
        url:             Full URL, e.g. https://www.ifoundbutterflies.org/axonopus-compressus
        host_plants_dir: Root output folder for all host plant data.
        pbar:            Optional tqdm bar for status updates.

    Returns: (success: bool, image_count: int)
    """
    slug = url.rstrip('/').split('/')[-1]
    if pbar:
        pbar.set_description(f"[Plant] {slug}")

    # ── Fetch page ─────────────────────────────────────────────────────────
    try:
        html = fetch_html(url)
    except Exception as e:
        logging.error(f"Failed to fetch plant page {url}: {e}")
        return False, 0

    soup = BeautifulSoup(html, 'html.parser')

    # ── Parse ──────────────────────────────────────────────────────────────
    taxonomy, common_names = extract_plant_taxonomy(soup)
    butterflies            = extract_butterfly_hosts(soup)
    images                 = extract_plant_images(soup)

    genus   = taxonomy.get('Genus', 'Unknown')
    species = taxonomy.get('Species', 'Unknown')
    family  = taxonomy.get('Family', 'Unknown')
    sci_name = f"{genus} {species}".strip()
    plant_key = make_plant_key(sci_name) if sci_name != "Unknown Unknown" else make_plant_key(slug)
    common_name = common_names.get('Species', common_names.get('Genus', ''))

    if not images:
        logging.debug(f"No images found for plant {slug}.")
        return True, 0

    # ── Create folder ──────────────────────────────────────────────────────
    plant_dir = os.path.join(host_plants_dir, plant_key)
    os.makedirs(plant_dir, exist_ok=True)

    # ── Download images ────────────────────────────────────────────────────
    registry_path = os.path.join(host_plants_dir, "registry.json")
    registry      = load_registry(registry_path)
    metadata_records = []
    downloaded = 0

    for i, img in enumerate(images):
        img_url = img['url']
        ext     = os.path.splitext(urlparse(img_url).path)[1] or '.jpg'

        if img['type'] == 'hero':
            filename = f"{plant_key}_hero{ext}"
        else:
            filename = f"{plant_key}_gallery_{i:03d}{ext}"

        filepath = os.path.join(plant_dir, filename)
        hf_path  = f"host_plants/{plant_key}/{filename}"

        ok = download_image(img_url, filepath)
        if not ok:
            continue
        downloaded += 1

        metadata_records.append({
            "file_name":         hf_path,
            "image_type":        img['type'],
            "alt":               img['alt'],
            "plant_key":         plant_key,
            "plant_scientific":  sci_name,
            "plant_common":      common_name,
            "plant_family":      family,
            "plant_genus":       genus,
            "plant_species":     species,
            "taxonomy":          taxonomy,
            "source_url":        url,
            "dataset_source":    "ifoundbutterflies.org",
            "butterfly_hosts":   [b['scientific_name'] for b in butterflies],
        })

    # ── Update registry ────────────────────────────────────────────────────
    if plant_key not in registry:
        registry[plant_key] = {
            "plant_scientific":  sci_name,
            "plant_common":      common_name,
            "plant_family":      family,
            "taxonomy":          taxonomy,
            "common_names":      common_names,
            "source_url":        url,
            "butterfly_species": [],
            "images":            [],
        }

    for b in butterflies:
        entry = b['scientific_name']
        if entry not in registry[plant_key]["butterfly_species"]:
            registry[plant_key]["butterfly_species"].append(entry)

    for rec in metadata_records:
        if rec["file_name"] not in registry[plant_key]["images"]:
            registry[plant_key]["images"].append(rec["file_name"])

    save_registry(registry, registry_path)

    # ── Append to shared metadata.jsonl ───────────────────────────────────
    if metadata_records:
        meta_path = os.path.join(host_plants_dir, "metadata.jsonl")
        with open(meta_path, 'a', encoding='utf-8') as f:
            for rec in metadata_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    logging.info(f"Plant {sci_name}: {downloaded} image(s), "
                 f"{len(butterflies)} butterfly host(s) recorded.")
    return True, downloaded


# ── Standalone test ────────────────────────────────────────────────────────────

if __name__ == '__main__':
    test_url = "https://www.ifoundbutterflies.org/axonopus-compressus"
    ok, count = scrape_plant_page(test_url, host_plants_dir="host_plants_test")
    print(f"Success={ok}, images={count}")
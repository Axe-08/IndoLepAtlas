import os
import re
import json
import time
import logging
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def extract_taxonomy(soup):
    """Extract full taxonomy from breadcrumbs, preferring scientific names (first matched)."""
    taxonomy = {}
    for li in soup.select("ul#system-breadcrumb-listing li"):
        for a in li.find_all("a"):
            span = a.find("span")
            if span:
                classes = span.get("class", [""])
                if classes:
                    rank = classes[0].replace("style-", "")
                    name = clean_text(span.text)
                    if rank and rank not in taxonomy:
                        taxonomy[rank] = name
    return taxonomy


def clean_text(text):
    return " ".join(text.split()).strip()


def extract_metadata_tabs(soup):
    """Extract structured metadata from tabs."""
    data = {}

    early_div = soup.find("div", id="early")
    if early_div:
        paragraphs = [clean_text(p.get_text()) for p in early_div.find_all("p")]
        early_texts = [p for p in paragraphs if len(p.replace(",", "").strip()) > 0]
        data["Early Stages Text"] = early_texts

    dist_div = soup.find("div", id="dist")
    if dist_div:
        data["Distribution"] = clean_text(dist_div.get_text(separator=" ", strip=True))

    stat_div = soup.find("div", id="stat")
    if stat_div:
        stat_data = {}
        paragraphs = stat_div.find_all("p", recursive=False)
        if paragraphs:
            stat_data["Description"] = [
                clean_text(p.get_text()) for p in paragraphs if clean_text(p.get_text())
            ]
        table = stat_div.find("table")
        if table:
            headers = [th.get_text(strip=True) for th in table.find_all("th")]
            rows = []
            for tr in table.find_all("tr")[1:]:
                cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
                if cells:
                    row_data = {}
                    for i, cell in enumerate(cells):
                        if i < len(headers):
                            val = cell
                            if val.isdigit():
                                val = int(val)
                            row_data[headers[i]] = val
                    if row_data:
                        rows.append(row_data)
            stat_data["Sighting Table"] = rows
        data["Status, Habitats, and Habits"] = stat_data

    larval_div = soup.find("div", id="laraval")
    if larval_div:
        data["Larval Host Plants"] = clean_text(
            larval_div.get_text(separator=" ", strip=True)
        )

    return data


def extract_image_id(a_tag):
    inner_img = a_tag.find("img")
    alt_text = ""
    if inner_img:
        alt_text = inner_img.get("alt", "")
    elif a_tag.get("data-cbox-img-attrs"):
        try:
            attrs = json.loads(a_tag.get("data-cbox-img-attrs"))
            alt_text = attrs.get("alt", "")
        except:
            pass
    alt_text = clean_text(alt_text)
    return alt_text if alt_text else "Unknown"


def get_images_from_tab(soup, tab_id, life_stage):
    images = []
    for a in soup.select(f"div#{tab_id} a.colorbox"):
        img_url = a.get("href")
        if img_url and img_url.startswith("/"):
            img_url = "https://www.ifoundbutterflies.org" + img_url
        if img_url:
            photographer_or_id = extract_image_id(a)
            images.append(
                {
                    "url": img_url,
                    "photographer_or_id": photographer_or_id,
                    "life_stage": life_stage,
                }
            )
    return images


# ──────────────────────────────────────────────
# Host plant helpers
# ──────────────────────────────────────────────


def get_host_plant_images(soup):
    """
    Extract image URLs from the Larval Host Plants tab (id='laraval').
    Returns list of {'url': ..., 'photographer_or_id': ...}
    """
    images = []
    larval_div = soup.find("div", id="laraval")
    if not larval_div:
        return images
    for a in larval_div.select("a.colorbox"):
        img_url = a.get("href")
        if img_url and img_url.startswith("/"):
            img_url = "https://www.ifoundbutterflies.org" + img_url
        if img_url:
            images.append(
                {
                    "url": img_url,
                    "photographer_or_id": extract_image_id(a),
                }
            )
    return images


def extract_host_plant_names(soup):
    """
    Parse individual plant names from the Larval Host Plants tab text.
    Returns a list of cleaned plant name strings.
    """
    larval_div = soup.find("div", id="laraval")
    if not larval_div:
        return []
    text = larval_div.get_text(separator=" ", strip=True)
    raw = re.split(r"[,;]", text)
    plants = []
    for r in raw:
        name = re.sub(r"\(.*?\)", "", r).strip()
        name = re.sub(r"\s+", " ", name).strip()
        if 2 < len(name) < 80:
            plants.append(name)
    return plants


def load_host_plant_registry(registry_path):
    """Load shared host plant registry from JSON, or return empty dict."""
    if os.path.exists(registry_path):
        with open(registry_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_host_plant_registry(registry, registry_path):
    """Persist the host plant registry to disk."""
    os.makedirs(os.path.dirname(os.path.abspath(registry_path)), exist_ok=True)
    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)


def download_host_plant_images(
    soup,
    scientific_name,
    genus,
    species,
    host_plants_dir,
    registry_path,
    common_meta=None,
):
    """
    Download all images from the Larval Host Plants tab (id='laraval').

    Folder structure:
        host_plants/
            {PlantKey}/
                {PlantKey}_{ScientificName}_{NNN}.jpg
            registry.json       <- shared, append-only across all species
            metadata.jsonl      <- appended across all species

    Returns:
        list of metadata record dicts (one per downloaded image).
        Returns empty list immediately if the tab has no images at all —
        no network requests are made in that case.
    """
    from tqdm import tqdm

    images = get_host_plant_images(soup)
    plant_names = extract_host_plant_names(soup)

    # ── Early-exit: no host plant images on this page ──────────────────────
    if not images:
        logging.debug(f"{scientific_name}: no host plant images found on page.")
        return []

    registry = load_host_plant_registry(registry_path)
    records = []

    for i, img_data in enumerate(
        tqdm(images, desc=f"Host plants {scientific_name}", leave=False, unit="img")
    ):
        img_url = img_data["url"]
        plant_name = plant_names[i] if i < len(plant_names) else "Unknown_Host_Plant"
        plant_key = re.sub(r"\s+", "_", plant_name.strip())
        plant_key = re.sub(r"[^A-Za-z0-9_\-]", "", plant_key)

        plant_dir = os.path.join(host_plants_dir, plant_key)
        os.makedirs(plant_dir, exist_ok=True)

        ext = os.path.splitext(urlparse(img_url).path)[1] or ".jpg"
        filename = f"{plant_key}_{scientific_name}_{i+1:03d}{ext}"
        filepath = os.path.join(plant_dir, filename)
        hf_path = f"host_plants/{plant_key}/{filename}"

        # Download (skip if already on disk)
        if not os.path.exists(filepath):
            try:
                for attempt in range(3):
                    try:
                        r = requests.get(img_url, stream=True, timeout=30)
                        r.raise_for_status()
                        with open(filepath, "wb") as f:
                            for chunk in r.iter_content(1024):
                                f.write(chunk)
                        break
                    except requests.exceptions.RequestException as e:
                        logging.warning(
                            f"Attempt {attempt+1} failed for {img_url}: {e}"
                        )
                        if attempt == 2:
                            raise
                        time.sleep(2)
            except Exception as e:
                logging.error(f"Failed to download host plant image {img_url}: {e}")
                continue
            time.sleep(1)

        # Update shared registry (append-only per plant)
        butterfly_entry = f"{genus} {species}"
        if plant_key not in registry:
            registry[plant_key] = {
                "plant_name": plant_name,
                "butterfly_species": [],
                "images": [],
            }
        if butterfly_entry not in registry[plant_key]["butterfly_species"]:
            registry[plant_key]["butterfly_species"].append(butterfly_entry)
        if hf_path not in registry[plant_key]["images"]:
            registry[plant_key]["images"].append(hf_path)

        record = {
            "file_name": hf_path,
            "life_stage": "Host Plant",
            "plant_name": plant_name,
            "plant_key": plant_key,
            "image_id": img_data["photographer_or_id"],
            "original_url": img_url,
        }
        if common_meta:
            record.update(common_meta)
        records.append(record)

    save_host_plant_registry(registry, registry_path)
    logging.info(f"{scientific_name}: {len(records)} host plant image(s) downloaded")
    return records


# ──────────────────────────────────────────────
# Main scrape function
# ──────────────────────────────────────────────


def scrape_species_page(
    url_or_filepath,
    output_dir="dataset_batch",
    host_plants_dir="host_plants",
    pbar=None,
    host_plants_only=False,
):
    """
    Scrape a single species page.

    Args:
        url_or_filepath:  URL or local HTML file path
        output_dir:       Where to write adult + early stage images/metadata.
                          Ignored (and no files written) when host_plants_only=True.
        host_plants_dir:  Top-level folder for host plant images + registry.
        pbar:             Optional tqdm progress bar for description updates.
        host_plants_only: When True, skip fetching adult/early images entirely.
                          Only host plant data is downloaded. The page HTML is
                          still fetched once. Returns (success, hp_image_count).

    Returns: (success: bool, downloaded_count: int)
    """
    if pbar:
        slug = url_or_filepath.rstrip("/").split("/")[-1]
        pbar.set_description(f"{'[HP]' if host_plants_only else ''} Fetching: {slug}")

    logging.debug(
        f"Scraping ({'host plants only' if host_plants_only else 'full'}): {url_or_filepath}"
    )

    # ── Fetch page HTML ──────────────────────────────────────────────────────
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
                    tqdm.write(
                        f"Attempt {attempt + 1}/{max_retries} failed for {url_or_filepath}. Retrying in 5s..."
                    )
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(5)
        else:
            with open(url_or_filepath, "r", encoding="utf-8") as f:
                html_content = f.read()
    except Exception as e:
        from tqdm import tqdm

        tqdm.write(f"ERROR: Failed to fetch {url_or_filepath}: {e}")
        return False, 0

    soup = BeautifulSoup(html_content, "html.parser")
    title = soup.title.text.strip() if soup.title else "Unknown Title"
    taxonomy = extract_taxonomy(soup)
    tab_metadata = extract_metadata_tabs(soup)

    genus = taxonomy.get("Genus", "Unknown")
    species = taxonomy.get("Species", "Unknown")
    scientific_name = f"{genus}_{species}".replace(" ", "_")

    source_url = (
        url_or_filepath
        if url_or_filepath.startswith("http")
        else "file://" + url_or_filepath
    )

    downloaded_count = 0

    # ── Adult + early stage images (skipped when host_plants_only=True) ─────
    if not host_plants_only:
        images_to_download = get_images_from_tab(soup, "home", "Adult/Unknown")
        early_images = get_images_from_tab(soup, "early", "Early Stage")
        sample_images = images_to_download + early_images

        os.makedirs(output_dir, exist_ok=True)
        metadata_records = []

        from tqdm import tqdm

        for i, img_data in enumerate(
            tqdm(
                sample_images,
                desc=f"Downloading {scientific_name}",
                leave=False,
                unit="img",
            )
        ):
            img_url = img_data["url"]
            try:
                img_resp = None
                for attempt in range(3):
                    try:
                        img_resp = requests.get(img_url, stream=True, timeout=30)
                        img_resp.raise_for_status()
                        break
                    except requests.exceptions.RequestException as e:
                        logging.warning(
                            f"Attempt {attempt + 1} for image {img_url} failed: {e}"
                        )
                        if attempt == 2:
                            raise
                        time.sleep(2)

                safe_stage = img_data["life_stage"].replace("/", "-").replace(" ", "")
                ext = os.path.splitext(urlparse(img_url).path)[1] or ".jpg"
                filename = f"{scientific_name}_{safe_stage}_{i+1:03d}{ext}"
                local_filepath = os.path.join(output_dir, filename)

                with open(local_filepath, "wb") as f:
                    for chunk in img_resp.iter_content(1024):
                        f.write(chunk)

                downloaded_count += 1
                record = {
                    "file_name": filename,
                    "dataset_source": "ifoundbutterflies.org",
                    "source_url": source_url,
                    "image_id": img_data["photographer_or_id"],
                    "life_stage": img_data["life_stage"],
                    "page_title": title,
                    **taxonomy,
                    **tab_metadata,
                }
                metadata_records.append(record)
                time.sleep(1)

            except Exception as e:
                logging.error(f"Failed to download image {img_url}: {e}")

        # Write species metadata.jsonl (adult + early stage only)
        metadata_file = os.path.join(output_dir, "metadata.jsonl")
        with open(metadata_file, "w", encoding="utf-8") as f:
            for record in metadata_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ── Host plant images (always runs unless page has none) ─────────────────
    common_meta = {
        "dataset_source": "ifoundbutterflies.org",
        "source_url": source_url,
        "page_title": title,
        **taxonomy,
        **tab_metadata,
    }
    registry_path = os.path.join(host_plants_dir, "registry.json")
    host_plant_records = download_host_plant_images(
        soup,
        scientific_name,
        genus,
        species,
        host_plants_dir,
        registry_path,
        common_meta=common_meta,
    )
    downloaded_count += len(host_plant_records)

    if host_plant_records:
        hp_meta_file = os.path.join(host_plants_dir, "metadata.jsonl")
        with open(hp_meta_file, "a", encoding="utf-8") as f:
            for record in host_plant_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logging.debug(
        f"Scrape complete. output_dir='{output_dir}', host_plants='{host_plants_dir}'"
    )
    return True, downloaded_count


if __name__ == "__main__":
    test_url = "https://www.ifoundbutterflies.org/Papilio-paris"
    scrape_species_page(
        test_url, output_dir="dataset_batch", host_plants_dir="host_plants"
    )

"""
plant_crawler.py

Crawls all plant species pages from:
  https://www.ifoundbutterflies.org/plant-species-list

Each plant page lives at e.g. /axonopus-compressus and has:
  - Clean taxonomy (breadcrumbs)
  - A list of butterfly species the plant hosts
  - Plant images (hero + gallery)

Run standalone:
    python plant_crawler.py

Run distributed (e.g. 4 machines):
    python plant_crawler.py --chunk 1 --total-chunks 4
    python plant_crawler.py --chunk 2 --total-chunks 4
    ...
"""

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
from dotenv import load_dotenv

from plant_scraper import scrape_plant_page

load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")
REPO_ID  = "DihelseeWee/IndoLepAtlas"

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

BASE_URL            = "https://www.ifoundbutterflies.org"
PLANT_LIST_URL = "https://www.ifoundbutterflies.org/plant-species-list"

PLANT_LIST_LOG      = "plant_list.log"        # cached plant URLs
PLANT_COMPLETED_LOG = "plant_completed.log"   # slugs fully scraped
PLANT_FAILED_LOG    = "plant_failed.log"      # slugs that errored

LOCK_TTL_SECONDS    = 600   # 10 min stale lock TTL

CIRCUIT_BREAKER_THRESHOLD  = 10
CIRCUIT_BREAKER_PAUSE      = 1800
CIRCUIT_BREAKER_MAX_PAUSES = 3


# ── Distributed lock ───────────────────────────────────────────────────────────

def try_acquire_hf_lock(slug):
    headers   = {"Authorization": f"Bearer {HF_TOKEN}"}
    lock_path = f"locks/plant_{slug}.lock"
    check_url = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/{lock_path}"
    try:
        resp = requests.get(check_url, headers=headers, timeout=10)
        if resp.status_code == 200:
            age = time.time() - resp.json().get("timestamp", 0)
            if age < LOCK_TTL_SECONDS:
                return False
    except Exception:
        pass

    payload  = json.dumps({"slug": slug, "timestamp": time.time(), "pid": os.getpid()})
    b64      = base64.b64encode(payload.encode()).decode('ascii')
    body = "\n".join([
        json.dumps({"key": "header", "value": {"summary": f"Lock plant {slug}"}}),
        json.dumps({"key": "file",   "value": {"path": lock_path,
                                                "encoding": "base64", "content": b64}}),
    ])
    try:
        r = requests.post(
            f"https://huggingface.co/api/datasets/{REPO_ID}/commit/main/host_plants",
            headers={**headers, "Content-Type": "application/x-ndjson"},
            data=body.encode('utf-8'), timeout=30,
        )
        return r.status_code in (200, 201)
    except Exception as e:
        logging.debug(f"Could not acquire lock for plant {slug}: {e}")
        return False


def release_hf_lock(slug):
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    body = "\n".join([
        json.dumps({"key": "header",      "value": {"summary": f"Unlock plant {slug}"}}),
        json.dumps({"key": "deletedFile", "value": {"path": f"locks/plant_{slug}.lock"}}),
    ])
    try:
        requests.post(
            f"https://huggingface.co/api/datasets/{REPO_ID}/commit/main/host_plants",
            headers={**headers, "Content-Type": "application/x-ndjson"},
            data=body.encode('utf-8'), timeout=30,
        )
    except Exception as e:
        logging.debug(f"Could not release lock for plant {slug}: {e}")


# ── HF upload ─────────────────────────────────────────────────────────────────

def upload_host_plants_dir(host_plants_dir, slug=None):
    """
    Upload a tree of host plant data to HuggingFace.

    ``host_plants_dir`` may point to the top-level ``host_plants`` folder or
    to any of its subdirectories (for example when uploading a single
    species batch).  Images are pushed via Git LFS; JSON/JSONL files are
    committed normally.  The local filesystem is not modified by this
    helper; callers should remove directories after a successful upload if
    desired.

    When ``slug`` is supplied the commit summary will mention the plant
    ("added {slug} plant").  If a subdirectory matching the slug exists we
    restrict the upload to that folder so the file count in the message
    reflects only the files actually added.  This mirrors the behaviour in
    ``crawler_logged.upload_batch`` and keeps the HF history readable.
    """
    import hashlib

    # if caller provided a slug and there is a corresponding subdirectory,
    # only upload that subfolder rather than the entire tree.  upload_root
    # will be used for walking below.
    upload_root = host_plants_dir
    if slug:
        candidate = os.path.join(host_plants_dir, slug)
        if os.path.isdir(candidate):
            upload_root = candidate

    if not os.path.exists(upload_root):
        return 0

    headers       = {"Authorization": f"Bearer {HF_TOKEN}"}
    lfs_objects   = []
    file_metadata = {}

    for root, dirs, files in os.walk(upload_root):
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        for filename in files:
            if filename.startswith('.'):
                continue
            filepath  = os.path.join(root, filename)
            rel_path  = os.path.relpath(filepath, os.path.dirname(host_plants_dir))
            repo_path = rel_path.replace(os.sep, '/')
            with open(filepath, 'rb') as f:
                content = f.read()
            size   = len(content)
            sha256 = hashlib.sha256(content).hexdigest()
            is_lfs = filename.lower().endswith(('.jpg', '.jpeg', '.png'))
            file_metadata[repo_path] = {'content': content, 'size': size,
                                        'sha256': sha256, 'is_lfs': is_lfs}
            if is_lfs:
                lfs_objects.append({'oid': sha256, 'size': size})

    if not file_metadata:
        return 0

    if lfs_objects:
        lfs_url = f"https://huggingface.co/datasets/{REPO_ID}.git/info/lfs/objects/batch"
        resp = requests.post(
            lfs_url,
            headers={**headers, "Content-Type": "application/vnd.git-lfs+json",
                     "Accept": "application/vnd.git-lfs+json"},
            json={"operation": "upload", "transfers": ["basic"], "objects": lfs_objects},
            timeout=60,
        )
        resp.raise_for_status()
        for obj in resp.json().get('objects', []):
            if 'actions' in obj and 'upload' in obj['actions']:
                up   = obj['actions']['upload']
                data = next(m['content'] for m in file_metadata.values()
                            if m['sha256'] == obj['oid'])
                requests.put(up['href'], headers=up.get('header', {}),
                             data=data, timeout=120).raise_for_status()

    commit_url = f"https://huggingface.co/api/datasets/{REPO_ID}/commit/main/host_plants"
    # customise the summary when a slug is known
    if slug:
        summary = f"added {slug} plant ({len(file_metadata)} files)"
    else:
        summary = f"Sync host_plants ({len(file_metadata)} files)"
    lines = [json.dumps({"key": "header",
                         "value": {"summary": summary}})]
    for repo_path, meta in file_metadata.items():
        if meta['is_lfs']:
            lines.append(json.dumps({"key": "lfsFile",
                                     "value": {"path": repo_path, "algo": "sha256",
                                               "oid": meta['sha256'], "size": meta['size']}}))
        else:
            b64 = base64.b64encode(meta['content']).decode('ascii')
            lines.append(json.dumps({"key": "file",
                                     "value": {"path": repo_path,
                                               "encoding": "base64", "content": b64}}))

    requests.post(commit_url,
                  headers={**headers, "Content-Type": "application/x-ndjson"},
                  data="\n".join(lines).encode('utf-8'),
                  timeout=120).raise_for_status()

    logging.info(f"Uploaded {len(file_metadata)} host plant file(s) to HF.")
    return len(file_metadata)


# ── Log sync ──────────────────────────────────────────────────────────────────

def verify_hf_repo():
    if not HF_TOKEN:
        logging.warning("No HF_TOKEN. Running locally without cloud sync.")
        return False
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    resp    = requests.get(f"https://huggingface.co/api/datasets/{REPO_ID}",
                           headers=headers, timeout=10)
    if resp.status_code == 200:
        return True
    elif resp.status_code == 404:
        cr = requests.post("https://huggingface.co/api/repos/create",
                           headers=headers,
                           json={"type": "dataset", "name": REPO_ID.split('/')[-1],
                                 "organization": REPO_ID.split('/')[0]},
                           timeout=10)
        return cr.status_code in (200, 201)
    return False


def pull_logs_from_hf():
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    for log_file in [PLANT_LIST_LOG, PLANT_COMPLETED_LOG, PLANT_FAILED_LOG]:
        url = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/{log_file}"
        try:
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code != 200:
                continue
            existing     = set()
            remote_lines = resp.text.splitlines()
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    existing.update(line.strip() for line in f if line.strip())
            if log_file == PLANT_LIST_LOG:
                merged = list(dict.fromkeys(remote_lines + list(existing)))
                with open(log_file, 'w') as f:
                    for line in merged:
                        if line.strip():
                            f.write(line.strip() + "\n")
            else:
                existing.update(line.strip() for line in remote_lines if line.strip())
                with open(log_file, 'w') as f:
                    for line in sorted(existing):
                        f.write(line + "\n")
            logging.info(f"Merged {log_file} from HF.")
        except Exception as e:
            logging.debug(f"Could not pull {log_file}: {e}")


def push_logs_to_hf():
    """Push all three log files to HF.

    Before writing we *pull* the remote copy and merge it with our local
    mirror.  This makes the operation additive and prevents a second crawler
    from clobbering the first one's progress when both are running
    concurrently.  (The original implementation simply uploaded whatever
    happened to be in the working directory at push time, so a later push
    could erase entries added by another worker.)
    """
    # merge any changes that might have landed since we started crawling
    try:
        pull_logs_from_hf()
    except Exception:
        # if pulling fails we'll still attempt to push; errors will be logged
        pass

    headers    = {"Authorization": f"Bearer {HF_TOKEN}"}
    commit_url = f"https://huggingface.co/api/datasets/{REPO_ID}/commit/main/host_plants"
    lines      = [json.dumps({"key": "header", "value": {"summary": "Sync plant logs"}})]
    for log_file in [PLANT_LIST_LOG, PLANT_COMPLETED_LOG, PLANT_FAILED_LOG]:
        if os.path.exists(log_file):
            with open(log_file, 'rb') as f:
                content = f.read()
            b64 = base64.b64encode(content).decode('ascii')
            lines.append(json.dumps({"key": "file",
                                     "value": {"path": log_file,
                                               "encoding": "base64", "content": b64}}))
    if len(lines) > 1:
        try:
            requests.post(commit_url,
                          headers={**headers, "Content-Type": "application/x-ndjson"},
                          data="\n".join(lines).encode('utf-8'),
                          timeout=30).raise_for_status()
            logging.info("Plant logs pushed to HF.")
        except Exception as e:
            logging.error(f"Failed to push plant logs: {e}")


def reupload_existing_batches(master_dir):
    """Scan for leftover plant subdirectories and re-upload them.

    The original crawler created folders named after the plant slug (e.g.
    ``host_plants/Abrus_precatorius``).  If a previous run was interrupted
    before pushing to HF we may be left with these directories locally.  This
    helper finds every non-hidden subdirectory of ``master_dir`` and attempts
    to upload it; on success the folder is removed so it won't be retried.

    Returns a tuple ``(folders_uploaded, files_uploaded)`` matching the
    earlier implementation.
    """
    if not os.path.exists(master_dir):
        return 0, 0
    plant_dirs = [
        d
        for d in os.listdir(master_dir)
        if not d.startswith('.') and os.path.isdir(os.path.join(master_dir, d))
    ]
    if not plant_dirs:
        return 0, 0

    logging.info(f"Found {len(plant_dirs)} leftover plant folders. Re-uploading...")
    uploaded = files_uploaded = 0
    pbar = tqdm(plant_dirs, desc="Re-uploading leftovers", unit="folder")
    for name in pbar:
        slug = name
        try:
            count = upload_host_plants_dir(os.path.join(master_dir, name), slug)
            uploaded += 1
            files_uploaded += count
            pbar.set_postfix({"Uploaded": uploaded, "Files": files_uploaded})
            # clear out folder once done
            try:
                shutil.rmtree(os.path.join(master_dir, name))
            except Exception:
                logging.debug(f"Could not delete leftover folder {name}")
        except Exception as e:
            logging.error(f"Failed to re-upload {name}: {e}")

    logging.info(
        f"Re-upload complete. {uploaded}/{len(plant_dirs)} folders, {files_uploaded} files."
    )
    return uploaded, files_uploaded


# ── Plant list scraper ────────────────────────────────────────────────────────

def get_plant_links(limit=None):
    """
    Return plant page URLs from local cache or by scraping plant-species-list.
    Plant links follow the pattern: href starts with '/' and the page title
    is a plant page (class path-node page-node-type-plant-species-form).
    We detect them by the <a class="spe-a"> anchor pattern on the list page.
    """
    if os.path.exists(PLANT_LIST_LOG):
        with open(PLANT_LIST_LOG, 'r') as f:
            links = [line.strip() for line in f if line.strip()]
        if links:
            logging.info(f"Loaded {len(links)} plant URLs from cache.")
            return links[:limit] if limit else links

    logging.info(f"Scraping plant list from {PLANT_LIST_URL} ...")
    html  = None
    for attempt in range(3):
        try:
            resp = requests.get(PLANT_LIST_URL, timeout=30)
            resp.raise_for_status()
            html = resp.text
            break
        except requests.exceptions.RequestException as e:
            logging.warning(f"Attempt {attempt+1} failed: {e}")
            if attempt == 2:
                raise
            time.sleep(5)

    soup  = BeautifulSoup(html, 'html.parser')
    links = []

    # Plant links are <a href="/slug-name"> inside the main content,
    # matching a simple pattern: starts with '/', lowercase, contains a hyphen or letter run
    main = soup.find('div', class_='region-content') or soup
    for a in main.find_all('a', href=True):
        href = a['href']
        # Plant slugs should be single‑segment paths such as
        # "/axonopus-compressus".  Historically we filtered out a few
        # unwanted URLs, but the old exclusion list mistakenly contained
        # '/' which matches *every* href, so nothing ever passed the
        # check.  That is why you were seeing "0 plant urls found" even
        # though the site clearly has many.
        #
        # The updated logic keeps the original intent:
        #  * must start with '/'
        #  * only one slash (no sub‑paths)
        #  * lowercase first character (butterfly pages start uppercase)
        #  * contains a hyphen (plant slugs do)
        #  * does not contain any of the known junk tokens
        if (href.startswith('/')
                and len(href) > 2
                and href.count('/') == 1
                and not href[1].isupper()          # NOT a butterfly page (those start uppercase)
                and '-' in href
                and not any(x in href for x in ['.css', '.js', '?', '#',
                                                 'user', 'about', 'contact', 'search',
                                                  'species-list',
                                                 'larval-host', 'hostplant'])):
            full_url = urljoin(BASE_URL, href)
            if full_url not in links:
                links.append(full_url)

    with open(PLANT_LIST_LOG, 'w') as f:
        for url in links:
            f.write(url + "\n")
    logging.info(f"Found and cached {len(links)} plant URLs.")
    return links[:limit] if limit else links


# ── Main crawler ──────────────────────────────────────────────────────────────

def run_plant_crawler(chunk=1, total_chunks=1):
    logging.info(f"Starting plant crawler (Chunk {chunk}/{total_chunks})...")

    host_plants_dir = "host_plants"
    os.makedirs(host_plants_dir, exist_ok=True)

    hf_ready = verify_hf_repo()

    if hf_ready:
        pull_logs_from_hf()
        reupload_count, reupload_files = reupload_existing_batches(host_plants_dir)

    # Load completed
    completed = set()
    if os.path.exists(PLANT_COMPLETED_LOG):
        with open(PLANT_COMPLETED_LOG, 'r') as f:
            completed = {line.strip() for line in f if line.strip()}
    logging.info(f"{len(completed)} plants already completed.")

    # Get full plant link list
    all_links = get_plant_links(limit=None)
    if not all_links:
        logging.error("No plant links found.")
        return

    # Apply chunking
    links = all_links
    if total_chunks > 1:
        chunk_size = math.ceil(len(links) / total_chunks)
        start_idx  = (chunk - 1) * chunk_size
        end_idx    = start_idx + chunk_size
        links      = links[start_idx:end_idx]
        logging.info(f"Chunk {chunk}/{total_chunks}: {len(links)} plants "
                     f"(indices {start_idx}–{min(end_idx, len(all_links))-1})")

    remaining = [u for u in links if u.rstrip('/').split('/')[-1] not in completed]
    logging.info(f"{len(links)} in chunk, {len(links)-len(remaining)} done, "
                 f"{len(remaining)} remaining.")

    success_count = fail_count = images_synced = 0
    consecutive_fails = pause_count = 0

    try:
        pbar = tqdm(remaining, desc="Scraping plants", unit="plant")
        for url in pbar:
            slug = url.rstrip('/').split('/')[-1]

            # Distributed lock
            if hf_ready:
                if not try_acquire_hf_lock(slug):
                    logging.debug(f"Skipping {slug} — locked.")
                    continue

            try:
                success, img_count = scrape_plant_page(
                    url,
                    host_plants_dir=host_plants_dir,
                    pbar=pbar,
                )
                if not success:
                    raise Exception("Scraper returned failure.")

                if hf_ready and img_count > 0:
                    # upload only the folder for the current plant; the helper
                    # itself will also restrict to the slug if necessary but
                    # this makes the intent explicit.
                    upload_host_plants_dir(os.path.join(host_plants_dir, slug), slug=slug)

                success_count     += 1
                images_synced     += img_count
                consecutive_fails  = 0

                with open(PLANT_COMPLETED_LOG, 'a') as f:
                    f.write(f"{slug}\n")

            except Exception as e:
                fail_count        += 1
                consecutive_fails += 1
                with open(PLANT_FAILED_LOG, 'a') as f:
                    f.write(f"{url}\n")
                logging.debug(f"Error on plant {slug}: {e}")

                if consecutive_fails >= CIRCUIT_BREAKER_THRESHOLD:
                    pause_count += 1
                    if pause_count >= CIRCUIT_BREAKER_MAX_PAUSES:
                        logging.error("Circuit breaker max pauses reached. Stopping.")
                        break
                    logging.warning(f"Circuit breaker: pausing {CIRCUIT_BREAKER_PAUSE//60} min...")
                    time.sleep(CIRCUIT_BREAKER_PAUSE)
                    consecutive_fails = 0

            finally:
                if hf_ready:
                    release_hf_lock(slug)

            pbar.set_postfix({"OK": success_count, "Fail": fail_count, "Imgs": images_synced})
            time.sleep(3)

        logging.info(f"Plant crawler done. OK: {success_count}, "
                     f"Fail: {fail_count}, Images: {images_synced}")

    finally:
        if hf_ready:
            push_logs_to_hf()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="IndoLepAtlas Plant Crawler")
    parser.add_argument("--chunk",        type=int, default=1)
    parser.add_argument("--total-chunks", type=int, default=1)
    args = parser.parse_args()

    if args.chunk < 1 or args.chunk > args.total_chunks:
        logging.error("--chunk must be between 1 and --total-chunks")
        exit(1)

    run_plant_crawler(chunk=args.chunk, total_chunks=args.total_chunks)

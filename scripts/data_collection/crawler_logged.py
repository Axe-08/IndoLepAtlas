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
REPO_ID = "DihelseeWee/IndoLepAtlas"

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

BASE_URL = "https://www.ifoundbutterflies.org"
SPECIES_LIST_URL = "https://www.ifoundbutterflies.org/species-list"

COMPLETED_LOG = "completed_species.log"
FAILED_LOG = "failed_species.log"
SPECIES_LIST_LOG = "species_list.log"
# Tracks every slug we have attempted a host-plant backfill on.
# A slug is written here after every attempt (success OR "no plants on page")

CIRCUIT_BREAKER_THRESHOLD = 10
CIRCUIT_BREAKER_PAUSE = 1800  # 30 minutes
CIRCUIT_BREAKER_MAX_PAUSES = 3

LOCK_TTL_SECONDS = 600  # 10 minutes


# ── Distributed lock helpers ───────────────────────────────────────────────────


def try_acquire_hf_lock(slug):
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    lock_path = f"locks/butterfly_{slug}.lock"
    check_url = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/{lock_path}"
    try:
        resp = requests.get(check_url, headers=headers, timeout=10)
        if resp.status_code == 200:
            age = time.time() - resp.json().get("timestamp", 0)
            if age < LOCK_TTL_SECONDS:
                return False  # fresh lock — another crawler owns this slug
    except Exception:
        pass  # no lock / unreadable — attempt to claim

    lock_payload = json.dumps(
        {"slug": slug, "timestamp": time.time(), "pid": os.getpid()}
    )
    b64 = base64.b64encode(lock_payload.encode()).decode("ascii")
    commit_url = f"https://huggingface.co/api/datasets/{REPO_ID}/commit/main"
    body = "\n".join(
        [
            json.dumps({"key": "header", "value": {"summary": f"Lock {slug}"}}),
            json.dumps(
                {
                    "key": "file",
                    "value": {
                        "path": f"locks/{slug}.lock",
                        "encoding": "base64",
                        "content": b64,
                    },
                }
            ),
        ]
    )
    try:
        r = requests.post(
            commit_url,
            headers={**headers, "Content-Type": "application/x-ndjson"},
            data=body.encode("utf-8"),
            timeout=30,
        )
        return r.status_code in (200, 201)
    except Exception as e:
        logging.debug(f"Could not acquire HF lock for {slug}: {e}")
        return False


def release_hf_lock(slug):
    """Delete the HF lock file for slug. Best-effort — never raises."""
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    commit_url = f"https://huggingface.co/api/datasets/{REPO_ID}/commit/main"
    body = "\n".join(
        [
            json.dumps({"key": "header", "value": {"summary": f"Unlock {slug}"}}),
            json.dumps({"key": "deletedFile", "value": {"path": f"locks/butterfly_{slug}.lock"}}),
        ]
    )
    try:
        requests.post(
            commit_url,
            headers={**headers, "Content-Type": "application/x-ndjson"},
            data=body.encode("utf-8"),
            timeout=30,
        )
    except Exception as e:
        logging.debug(f"Could not release HF lock for {slug}: {e}")


# ── HF upload helpers ──────────────────────────────────────────────────────────


def upload_batch(batch_dir, slug):
    """
    Upload all files in batch_dir to HF under data/{slug}/, with Git LFS for
    images. Deletes the local folder on success. Returns file count.
    """
    import hashlib

    files_list = [f for f in os.listdir(batch_dir) if not f.startswith(".")]
    if not files_list:
        shutil.rmtree(batch_dir)
        return 0

    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    lfs_objects = []
    file_metadata = {}

    for filename in files_list:
        with open(os.path.join(batch_dir, filename), "rb") as f:
            content = f.read()
        size = len(content)
        sha256 = hashlib.sha256(content).hexdigest()
        is_lfs = filename.lower().endswith((".jpg", ".jpeg", ".png"))
        file_metadata[filename] = {
            "content": content,
            "size": size,
            "sha256": sha256,
            "is_lfs": is_lfs,
        }
        if is_lfs:
            lfs_objects.append({"oid": sha256, "size": size})

    if lfs_objects:
        lfs_url = (
            f"https://huggingface.co/datasets/{REPO_ID}.git/info/lfs/objects/batch"
        )
        resp = requests.post(
            lfs_url,
            headers={
                **headers,
                "Content-Type": "application/vnd.git-lfs+json",
                "Accept": "application/vnd.git-lfs+json",
            },
            json={
                "operation": "upload",
                "transfers": ["basic"],
                "objects": lfs_objects,
            },
            timeout=60,
        )
        resp.raise_for_status()
        for obj in resp.json().get("objects", []):
            if "actions" in obj and "upload" in obj["actions"]:
                up = obj["actions"]["upload"]
                data = next(
                    m["content"]
                    for m in file_metadata.values()
                    if m["sha256"] == obj["oid"]
                )
                requests.put(
                    up["href"], headers=up.get("header", {}), data=data, timeout=120
                ).raise_for_status()

    commit_url = f"https://huggingface.co/api/datasets/{REPO_ID}/commit/main"
    lines = [
        json.dumps(
            {
                "key": "header",
                "value": {"summary": f"Add {slug} ({len(files_list)} files)"},
            }
        )
    ]
    for filename, meta in file_metadata.items():
        path_in_repo = f"data/butterflies/raw/{slug}/{filename}"
        if meta["is_lfs"]:
            lines.append(
                json.dumps(
                    {
                        "key": "lfsFile",
                        "value": {
                            "path": path_in_repo,
                            "algo": "sha256",
                            "oid": meta["sha256"],
                            "size": meta["size"],
                        },
                    }
                )
            )
        else:
            b64 = base64.b64encode(meta["content"]).decode("ascii")
            lines.append(
                json.dumps(
                    {
                        "key": "file",
                        "value": {
                            "path": path_in_repo,
                            "encoding": "base64",
                            "content": b64,
                        },
                    }
                )
            )

    requests.post(
        commit_url,
        headers={**headers, "Content-Type": "application/x-ndjson"},
        data="\n".join(lines).encode("utf-8"),
        timeout=120,
    ).raise_for_status()
    shutil.rmtree(batch_dir)
    return len(files_list)



def reupload_existing_batches(master_dir):
    """Scan for leftover batch folders from a previous run and re-upload them."""
    if not os.path.exists(master_dir):
        return 0, 0
    batch_dirs = [
        d
        for d in os.listdir(master_dir)
        if d.startswith("batch_") and os.path.isdir(os.path.join(master_dir, d))
    ]
    if not batch_dirs:
        return 0, 0

    logging.info(f"Found {len(batch_dirs)} leftover batch folders. Re-uploading...")
    uploaded = files_uploaded = 0
    pbar = tqdm(batch_dirs, desc="Re-uploading leftovers", unit="batch")
    for batch_name in pbar:
        slug = batch_name.replace("batch_", "", 1)
        try:
            count = upload_batch(os.path.join(master_dir, batch_name), slug)
            uploaded += 1
            files_uploaded += count
            pbar.set_postfix({"Uploaded": uploaded, "Files": files_uploaded})
        except Exception as e:
            logging.error(f"Failed to re-upload {batch_name}: {e}")

    logging.info(
        f"Re-upload complete. {uploaded}/{len(batch_dirs)} batches, {files_uploaded} files."
    )
    return uploaded, files_uploaded


# ── Species link helpers ───────────────────────────────────────────────────────


def get_species_links(limit=None):
    """Return species links from local cache, or scrape and cache them."""
    if os.path.exists(SPECIES_LIST_LOG):
        with open(SPECIES_LIST_LOG, "r") as f:
            links = [line.strip() for line in f if line.strip()]
        if links:
            logging.info(f"Loaded {len(links)} species URLs from local cache.")
            return links[:limit] if limit else links

    logging.info(f"No local species cache found. Scraping {SPECIES_LIST_URL}...")
    for attempt in range(3):
        try:
            response = requests.get(SPECIES_LIST_URL, timeout=30)
            response.raise_for_status()
            break
        except requests.exceptions.RequestException as e:
            logging.warning(f"Attempt {attempt + 1} failed: {e}")
            if attempt == 2:
                logging.error("Max retries reached. Aborting.")
                raise
            time.sleep(5)

    soup = BeautifulSoup(response.text, "html.parser")
    species_links = []
    main_content = soup.find("div", class_="region-content") or soup
    for a in main_content.find_all("a", href=True):
        href = a["href"]
        if href.startswith("/") and len(href) > 2 and href[1].isupper() and "-" in href:
            if not any(
                x in href
                for x in [
                    ".css",
                    ".js",
                    ".png",
                    ".jpg",
                    "?",
                    "user",
                    "about",
                    "contact",
                ]
            ):
                full_url = urljoin(BASE_URL, href)
                if full_url not in species_links:
                    species_links.append(full_url)

    with open(SPECIES_LIST_LOG, "w") as f:
        for url in species_links:
            f.write(url + "\n")
    logging.info(f"Scraped and cached {len(species_links)} species URLs.")
    return species_links[:limit] if limit else species_links


# ── HF repo / log sync ────────────────────────────────────────────────────────


def verify_hf_repo():
    if not HF_TOKEN:
        logging.warning("No HF_TOKEN found. Will scrape locally without cloud sync.")
        return False
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    resp = requests.get(
        f"https://huggingface.co/api/datasets/{REPO_ID}", headers=headers, timeout=10
    )
    if resp.status_code == 200:
        logging.info(f"HF repo {REPO_ID} exists and is accessible.")
        return True
    elif resp.status_code == 404:
        create_resp = requests.post(
            "https://huggingface.co/api/repos/create",
            headers=headers,
            json={
                "type": "dataset",
                "name": REPO_ID.split("/")[-1],
                "organization": REPO_ID.split("/")[0],
            },
            timeout=10,
        )
        if create_resp.status_code in (200, 201):
            logging.info(f"Created HF repo {REPO_ID}.")
            return True
        logging.error(
            f"Failed to create repo: {create_resp.status_code} {create_resp.text}"
        )
        return False
    logging.error(f"Failed to verify repo: {resp.status_code} {resp.text}")
    return False


def load_completed_species():
    if not os.path.exists(COMPLETED_LOG):
        return set()
    with open(COMPLETED_LOG, "r") as f:
        return {line.strip() for line in f if line.strip()}


def pull_logs_from_hf():
    logging.info("Pulling sync logs from Hugging Face...")
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    for log_file in [COMPLETED_LOG, FAILED_LOG, SPECIES_LIST_LOG]:
        url = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/{log_file}"
        try:
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code != 200:
                continue
            existing = set()
            remote_lines = resp.text.splitlines()
            if os.path.exists(log_file):
                with open(log_file, "r") as f:
                    existing.update(line.strip() for line in f if line.strip())
            if log_file == SPECIES_LIST_LOG:
                merged = list(dict.fromkeys(remote_lines + list(existing)))
                with open(log_file, "w") as f:
                    for line in merged:
                        if line.strip():
                            f.write(line.strip() + "\n")
                logging.info(f"Merged {log_file}: {len(merged)} lines.")
            else:
                existing.update(line.strip() for line in remote_lines if line.strip())
                with open(log_file, "w") as f:
                    for line in sorted(existing):
                        f.write(line + "\n")
                logging.info(f"Merged {log_file}: {len(existing)} lines.")
        except Exception as e:
            logging.debug(f"Could not pull {log_file}: {e}")


def push_logs_to_hf():
    logging.info("Pushing sync logs to Hugging Face...")
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    commit_url = f"https://huggingface.co/api/datasets/{REPO_ID}/commit/main"
    lines = [json.dumps({"key": "header", "value": {"summary": "Sync progress logs"}})]
    for log_file in [COMPLETED_LOG, FAILED_LOG, SPECIES_LIST_LOG]:
        if os.path.exists(log_file):
            with open(log_file, "rb") as f:
                content = f.read()
            b64 = base64.b64encode(content).decode("ascii")
            lines.append(
                json.dumps(
                    {
                        "key": "file",
                        "value": {
                            "path": log_file,
                            "encoding": "base64",
                            "content": b64,
                        },
                    }
                )
            )
    if len(lines) > 1:
        try:
            requests.post(
                commit_url,
                headers={**headers, "Content-Type": "application/x-ndjson"},
                data="\n".join(lines).encode("utf-8"),
                timeout=30,
            ).raise_for_status()
            logging.info("Sync logs pushed successfully.")
        except Exception as e:
            logging.error(f"Failed to push sync logs: {e}")


# ── Main crawler ───────────────────────────────────────────────────────────────


def run_crawler(chunk=1, total_chunks=1):
    logging.info(f"Starting production crawler (Chunk {chunk}/{total_chunks})...")

    master_dir = "dataset_test_run"
    os.makedirs(master_dir, exist_ok=True)

    # Phase 0: Verify HF repo
    hf_ready = verify_hf_repo()

    # Phase 1: Pull remote logs + re-upload leftover batches
    reupload_count = reupload_files = 0
    if hf_ready:
        pull_logs_from_hf()
        reupload_count, reupload_files = reupload_existing_batches(master_dir)

    # Phase 2: Load state
    completed = load_completed_species()
    logging.info(f"Loaded {len(completed)} already-completed species.")

    # Phase 3: Get full species link list + build slug->url map
    all_links = get_species_links(limit=None)
    if not all_links:
        logging.error("Found no species links.")
        return
    slug_to_url = {u.rstrip("/").split("/")[-1]: u for u in all_links}

    # Apply chunking
    links = all_links
    if total_chunks > 1:
        chunk_size = math.ceil(len(links) / total_chunks)
        start_idx = (chunk - 1) * chunk_size
        end_idx = start_idx + chunk_size
        links = links[start_idx:end_idx]
        logging.info(
            f"Chunk {chunk}/{total_chunks}: {len(links)} species "
            f"(indices {start_idx}–{min(end_idx, len(all_links)) - 1})"
        )

    remaining = [u for u in links if u.rstrip("/").split("/")[-1] not in completed]
    logging.info(
        f"{len(links)} in chunk, {len(links)-len(remaining)} already done, "
        f"{len(remaining)} to scrape."
    )

    success_count = reupload_count + len(completed)
    fail_count = 0
    images_synced = reupload_files
    consecutive_fails = pause_count = 0

    try:
        # ════════════════════════════════════════════════════════════════════
        # PASS 1 — full scrape of new species (adult + early + host plants)
        # ════════════════════════════════════════════════════════════════════
        pbar = tqdm(remaining, desc="Scraping new species", unit="species")
        for url in pbar:
            slug = url.rstrip("/").split("/")[-1]
            batch_dir = os.path.join(master_dir, f"batch_{slug}")

            if hf_ready:
                if not try_acquire_hf_lock(slug):
                    logging.debug(f"Skipping {slug} — locked by another crawler.")
                    continue

            try:
                success, dl_count = scrape_species_page(
                    url,
                    output_dir=batch_dir,
                    pbar=pbar,
                )
                if not success:
                    raise Exception("Scraper returned failure.")

                if hf_ready:
                    if os.path.exists(batch_dir):
                        upload_batch(batch_dir, slug)

                success_count += 1
                images_synced += dl_count
                consecutive_fails = 0

                with open(COMPLETED_LOG, "a") as f:
                    f.write(f"{slug}\n")
                # Mark host plants as attempted so Pass 2 skips this slug


            except Exception as e:
                fail_count += 1
                consecutive_fails += 1
                with open(FAILED_LOG, "a") as f:
                    f.write(f"{url}\n")
                logging.debug(f"Error processing {slug}: {e}")

                if consecutive_fails >= CIRCUIT_BREAKER_THRESHOLD:
                    pause_count += 1
                    if pause_count >= CIRCUIT_BREAKER_MAX_PAUSES:
                        logging.error("Circuit breaker max pauses reached. Stopping.")
                        break
                    logging.warning(
                        f"Circuit breaker: pausing {CIRCUIT_BREAKER_PAUSE // 60} min "
                        f"({pause_count}/{CIRCUIT_BREAKER_MAX_PAUSES})..."
                    )
                    time.sleep(CIRCUIT_BREAKER_PAUSE)
                    consecutive_fails = 0

            finally:
                if hf_ready:
                    release_hf_lock(slug)

            pbar.set_postfix(
                {"OK": success_count, "Fail": fail_count, "Imgs": images_synced}
            )
            time.sleep(3)

        logging.info(
            f"Crawler complete! "
            f"New: {success_count}  Fails: {fail_count}  Images: {images_synced}"
        )

    finally:
        if hf_ready:
            push_logs_to_hf()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IndoLepAtlas Distributed Crawler")
    parser.add_argument("--chunk", type=int, default=1)
    parser.add_argument("--total-chunks", type=int, default=1)
    args = parser.parse_args()

    if args.chunk < 1 or args.chunk > args.total_chunks:
        logging.error("--chunk must be between 1 and --total-chunks")
        exit(1)

    run_crawler(chunk=args.chunk, total_chunks=args.total_chunks)

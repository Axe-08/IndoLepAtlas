"""
migrate_hf_plants.py

One-time migration script to move plant directories that were
incorrectly uploaded to the root of the HF dataset into
the correct `host_plants/` directory.

Usage:
    python migrate_hf_plants.py                # dry-run (default)
    python migrate_hf_plants.py --execute      # actually perform the migration
    python migrate_hf_plants.py --batch-size 10  # change batch size (default 20)
"""

import os
import json
import base64
import logging
import argparse
import requests
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")
REPO_ID  = "DihelseeWee/IndoLepAtlas"

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

# Directories that legitimately live at the root of the HF repo
KNOWN_ROOT_DIRS = {
    "host_plants", "data", "locks", ".gitattributes",
}

# Log files that live at the root
KNOWN_ROOT_FILES = {
    "README.md", ".gitattributes",
    "completed_species.log", "failed_species.log",
    "species_list.log",
    "plant_list.log", "plant_completed.log", "plant_failed.log",
}


def list_repo_files():
    """Return list of all file paths in the HF dataset repo."""
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    url = f"https://huggingface.co/api/datasets/{REPO_ID}/tree/main"
    all_files = []
    params = {"recursive": "true"}

    # The tree API is paginated; keep fetching until done.
    while True:
        resp = requests.get(url, headers=headers, params=params, timeout=30)
        resp.raise_for_status()
        items = resp.json()
        if not items:
            break
        for item in items:
            if item.get("type") == "file":
                all_files.append(item)
        # HF uses Link header for pagination
        link = resp.headers.get("Link", "")
        if 'rel="next"' in link:
            # Extract next URL from Link header
            next_url = link.split(";")[0].strip().strip("<>")
            url = next_url
            params = {}  # params are in the URL now
        else:
            break

    return all_files


def identify_misplaced_plants(all_files):
    """
    Find files at root level that look like plant directories
    (e.g. Abrus_precatorius/metadata.json).

    Returns a dict: {plant_slug: [file_info, ...]}
    """
    misplaced = defaultdict(list)

    for f in all_files:
        path = f["path"]
        parts = path.split("/")

        # Skip anything already under known root dirs or single root files
        if parts[0] in KNOWN_ROOT_DIRS or path in KNOWN_ROOT_FILES:
            continue

        # Skip log files
        if path.endswith(".log"):
            continue

        # If it's a multi-part path like "Slug_name/file.jpg",
        # the top-level dir is a misplaced plant
        if len(parts) >= 2:
            slug = parts[0]
            misplaced[slug].append(f)

    return dict(misplaced)


def move_batch(slug_files_pairs, dry_run=True):
    """
    Move a batch of plant directories from root to host_plants/.

    Each commit contains deletedFile + file/lfsFile operations.
    For LFS files, we reference by oid so no re-upload is needed.
    """
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    commit_url = f"https://huggingface.co/api/datasets/{REPO_ID}/commit/main"

    slugs = [s for s, _ in slug_files_pairs]
    summary = f"Migrate {len(slugs)} plant dirs to host_plants/: {', '.join(slugs[:5])}"
    if len(slugs) > 5:
        summary += f"... (+{len(slugs)-5} more)"

    lines = [json.dumps({"key": "header", "value": {"summary": summary}})]

    move_count = 0
    for slug, files in slug_files_pairs:
        for f in files:
            old_path = f["path"]
            new_path = f"host_plants/{old_path}"
            lfs = f.get("lfs")

            # Delete old path
            lines.append(json.dumps({"key": "deletedFile",
                                     "value": {"path": old_path}}))

            if lfs:
                # LFS file: just point to existing oid — no re-upload!
                lines.append(json.dumps({"key": "lfsFile",
                                         "value": {"path": new_path,
                                                   "algo": "sha256",
                                                   "oid": lfs["oid"],
                                                   "size": lfs["size"]}}))
            else:
                # Regular file: download content and re-add at new path
                dl_url = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/{old_path}"
                resp = requests.get(dl_url, headers=headers, timeout=60)
                resp.raise_for_status()
                b64 = base64.b64encode(resp.content).decode("ascii")
                lines.append(json.dumps({"key": "file",
                                         "value": {"path": new_path,
                                                   "encoding": "base64",
                                                   "content": b64}}))
            move_count += 1

    if dry_run:
        logging.info(f"[DRY RUN] Would move {move_count} files for slugs: {slugs}")
        return True

    ndjson = "\n".join(lines)
    resp = requests.post(
        commit_url,
        headers={**headers, "Content-Type": "application/x-ndjson"},
        data=ndjson.encode("utf-8"),
        timeout=180,
    )
    if resp.status_code in (200, 201):
        logging.info(f"Moved {move_count} files for {len(slugs)} plants.")
        return True
    else:
        logging.error(f"Commit failed ({resp.status_code}): {resp.text[:500]}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Migrate misplaced plant dirs on HF")
    parser.add_argument("--execute", action="store_true",
                        help="Actually perform the migration (default is dry-run)")
    parser.add_argument("--batch-size", type=int, default=20,
                        help="Number of plant dirs per commit (default 20)")
    args = parser.parse_args()

    dry_run = not args.execute

    if not HF_TOKEN:
        logging.error("HF_TOKEN not set. Add it to .env")
        return

    logging.info("Listing all files in HF repo...")
    all_files = list_repo_files()
    logging.info(f"Found {len(all_files)} total files in repo.")

    misplaced = identify_misplaced_plants(all_files)
    if not misplaced:
        logging.info("No misplaced plant directories found. Nothing to do!")
        return

    logging.info(f"Found {len(misplaced)} misplaced plant directories at root level:")
    for slug, files in sorted(misplaced.items()):
        logging.info(f"  {slug}/ ({len(files)} files)")

    total_files = sum(len(f) for f in misplaced.values())
    logging.info(f"Total files to move: {total_files}")

    if dry_run:
        logging.info("[DRY RUN] Pass --execute to actually migrate.")

    # Batch by slug groups
    slug_list = sorted(misplaced.items())
    for i in range(0, len(slug_list), args.batch_size):
        batch = slug_list[i:i + args.batch_size]
        batch_num = i // args.batch_size + 1
        total_batches = (len(slug_list) + args.batch_size - 1) // args.batch_size
        logging.info(f"Processing batch {batch_num}/{total_batches}...")

        success = move_batch(batch, dry_run=dry_run)
        if not success:
            logging.error(f"Batch {batch_num} failed. Stopping.")
            return

    mode = "DRY RUN" if dry_run else "MIGRATION"
    logging.info(f"{mode} complete! {len(misplaced)} plant dirs, {total_files} files.")


if __name__ == "__main__":
    main()

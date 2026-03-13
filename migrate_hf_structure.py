"""
migrate_hf_structure.py

One-time migration script to move existing HF dataset files
into the new clean directory structure.

Moves:
1. Butterflies: `data/{slug}/` -> `data/butterflies/raw/{slug}/`
   Note: we only move things directly under data/ (so depth=2).
2. Plants: `host_plants/{slug}/` -> `data/plants/raw/{slug}/`
3. Old misplaced plants at root: `{slug}/` -> `data/plants/raw/{slug}/`
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Directories/files that legitimately live at the root of the HF repo and shouldn't be touched as plants
KNOWN_ROOT_DIRS = {"host_plants", "data", "locks", ".gitattributes", "annotations", "splits", "docs"}
KNOWN_ROOT_FILES = {
    "README.md", ".gitattributes", "completed_species.log", "failed_species.log", "species_list.log",
    "plant_list.log", "plant_completed.log", "plant_failed.log", "registry.json", "metadata.jsonl"
}

def list_repo_files():
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    url = f"https://huggingface.co/api/datasets/{REPO_ID}/tree/main"
    all_files = []
    params = {"recursive": "true"}

    while True:
        resp = requests.get(url, headers=headers, params=params, timeout=30)
        resp.raise_for_status()
        items = resp.json()
        if not items: break
        for item in items:
            if item.get("type") == "file":
                all_files.append(item)
        link = resp.headers.get("Link", "")
        if 'rel="next"' in link:
            next_url = link.split(";")[0].strip().strip("<>")
            url = next_url
            params = {}
        else:
            break
    return all_files

def identify_migrations(all_files):
    moves = defaultdict(list)
    for f in all_files:
        path = f["path"]
        parts = path.split("/")

        # 1. Old butterflies: data/{slug}/...
        if len(parts) >= 3 and parts[0] == "data" and parts[1] not in ("butterflies", "plants"):
            slug = parts[1]
            rest = "/".join(parts[2:])
            new_path = f"data/butterflies/raw/{slug}/{rest}"
            moves[path] = {"new_path": new_path, "lfs": f.get("lfs")}
            continue

        # 2. Old plant dir: host_plants/{slug}/...
        if len(parts) >= 3 and parts[0] == "host_plants":
            slug = parts[1]
            rest = "/".join(parts[2:])
            new_path = f"data/plants/raw/{slug}/{rest}"
            moves[path] = {"new_path": new_path, "lfs": f.get("lfs")}
            continue

        # 3. Misplaced plants: {slug}/... at root
        if len(parts) >= 2 and parts[0] not in KNOWN_ROOT_DIRS and path not in KNOWN_ROOT_FILES and not path.endswith(".log"):
            slug = parts[0]
            rest = "/".join(parts[1:])
            new_path = f"data/plants/raw/{slug}/{rest}"
            moves[path] = {"new_path": new_path, "lfs": f.get("lfs")}
            continue

    return dict(moves)

def move_batch(moves_batch, dry_run=True):
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    commit_url = f"https://huggingface.co/api/datasets/{REPO_ID}/commit/main"

    summary = f"Structure migration ({len(moves_batch)} files)"
    lines = [json.dumps({"key": "header", "value": {"summary": summary}})]

    for old_path, info in moves_batch.items():
        new_path = info["new_path"]
        lfs = info["lfs"]
        lines.append(json.dumps({"key": "deletedFile", "value": {"path": old_path}}))
        if lfs:
            lines.append(json.dumps({
                "key": "lfsFile",
                "value": {"path": new_path, "algo": "sha256", "oid": lfs["oid"], "size": lfs["size"]}
            }))
        else:
            dl_url = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/{old_path}"
            resp = requests.get(dl_url, headers=headers, timeout=60)
            resp.raise_for_status()
            b64 = base64.b64encode(resp.content).decode("ascii")
            lines.append(json.dumps({
                "key": "file",
                "value": {"path": new_path, "encoding": "base64", "content": b64}
            }))

    if dry_run:
        logging.info(f"[DRY RUN] Would move {len(moves_batch)} files")
        return True

    ndjson = "\n".join(lines)
    resp = requests.post(commit_url, headers={**headers, "Content-Type": "application/x-ndjson"}, data=ndjson.encode("utf-8"), timeout=180)
    if resp.status_code in (200, 201):
        logging.info(f"Successfully moved {len(moves_batch)} files.")
        return True
    else:
        logging.error(f"Commit failed ({resp.status_code}): {resp.text[:500]}")
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="Actually perform the migration")
    parser.add_argument("--batch-size", type=int, default=100)
    args = parser.parse_args()

    if not HF_TOKEN:
        logging.error("HF_TOKEN missing")
        return

    logging.info("Listing files...")
    all_files = list_repo_files()
    moves = identify_migrations(all_files)
    
    if not moves:
        logging.info("No files need to be migrated!")
        return
        
    logging.info(f"Found {len(moves)} files to migrate.")
    if not args.execute:
        logging.info("[DRY RUN] Showing up to 10 sample moves:")
        for k, v in list(moves.items())[:10]:
            logging.info(f"{k} -> {v['new_path']}")
        logging.info("[DRY RUN] Use --execute to run.")
        return

    items = list(moves.items())
    for i in range(0, len(items), args.batch_size):
        batch = dict(items[i:i + args.batch_size])
        batch_num = i // args.batch_size + 1
        total_batches = (len(items) + args.batch_size - 1) // args.batch_size
        logging.info(f"Moving batch {batch_num}/{total_batches}...")
        if not move_batch(batch, dry_run=False):
            logging.error("Batch failed. Stopping.")
            return

    logging.info("Migration complete!")

if __name__ == "__main__":
    main()

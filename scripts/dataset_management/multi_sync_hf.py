import os
import logging
import time
from huggingface_hub import HfApi, CommitOperationAdd
from tqdm import tqdm
from dotenv import load_dotenv

# Path to .env on DGX
ENV_PATH = "/home/23uec552/DLCV/IndoLepAtlas/.env"
load_dotenv(ENV_PATH)

token = os.environ.get("HF_TOKEN")
repo_id = "DihelseeWee/IndoLepAtlas"

# Configure logging to flush and show in console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("sync_to_hf_global.log")
    ]
)
# Force logging to be unbuffered
import sys
for handler in logging.root.handlers:
    handler.flush = sys.stdout.flush

api = HfApi(token=token)

BATCH_SIZE = 1000  # Number of files per commit

def get_local_files(directory):
    files_list = []
    for root, _, files in os.walk(directory):
        for f in files:
            full_path = os.path.join(root, f)
            # Normalized path for HF
            rel_path = os.path.relpath(full_path, start=os.getcwd())
            files_list.append((full_path, rel_path))
    return files_list

def sync():
    logging.info("Auditing remote files on HF...")
    try:
        remote_files = set(api.list_repo_files(repo_id=repo_id, repo_type="dataset"))
    except Exception as e:
        logging.error(f"Failed to list remote files: {e}")
        return

    # 1. Collect all local images and root metadata
    local_dirs = ["data/butterflies/images", "data/plants/images"]
    root_files = [
        "data/butterflies/metadata.csv",
        "data/plants/metadata.csv",
        "data/butterflies/.processed_easyocr",
        "data/plants/.processed_easyocr",
        "data/butterflies/.processed",
        "data/plants/.processed"
    ]
    
    all_local_files = []
    for d in local_dirs:
        if os.path.exists(d):
            all_local_files.extend(get_local_files(d))
            
    for f in root_files:
        if os.path.exists(f):
            all_local_files.append((f, f))

    # 2. Identify missing files
    missing_files = []
    for full_path, rel_path in all_local_files:
        if rel_path not in remote_files:
            missing_files.append((full_path, rel_path))
    
    logging.info(f"Total local files: {len(all_local_files)}")
    logging.info(f"Already uploaded: {len(all_local_files) - len(missing_files)}")
    logging.info(f"Remaining to upload: {len(missing_files)}")

    if not missing_files:
        logging.info("Everything is already synced!")
        return

    # 3. Upload in batches
    num_batches = (len(missing_files) + BATCH_SIZE - 1) // BATCH_SIZE
    logging.info(f"Starting upload in {num_batches} batches...")

    for i in range(num_batches):
        batch = missing_files[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
        operations = [
            CommitOperationAdd(path_in_repo=rel_path, path_or_fileobj=full_path)
            for full_path, rel_path in batch
        ]
        
        logging.info(f"Committing batch {i+1}/{num_batches} ({len(operations)} files)...")
        try:
            api.create_commit(
                repo_id=repo_id,
                operations=operations,
                commit_message=f"Sync images batch {i+1}/{num_batches}",
                repo_type="dataset"
            )
            logging.info(f"Batch {i+1} committed successfully.")
            # Small cooldown to avoid hitting limit again
            time.sleep(2) 
        except Exception as e:
            logging.error(f"Failed to commit batch {i+1}: {e}")
            # If rate limited, maybe wait longer?
            if "rate limit" in str(e).lower():
                logging.info("Rate limit hit, waiting 60s...")
                time.sleep(60)

    logging.info("Bulk image synchronization complete.")

if __name__ == "__main__":
    sync()

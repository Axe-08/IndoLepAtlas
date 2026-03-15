import os
import time
import logging
from huggingface_hub import HfApi, CommitOperationAdd
from dotenv import load_dotenv

# Path to .env on DGX
ENV_PATH = "/home/23uec552/DLCV/IndoLepAtlas/.env"
load_dotenv(ENV_PATH)

api = HfApi(token=os.environ.get("HF_TOKEN"))
repo_id = "DihelseeWee/IndoLepAtlas"

BATCH_SIZE = 5000
MAX_RETRIES = 5

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def get_local_files(directory):
    files_list = []
    for root, _, files in os.walk(directory):
        for f in files:
            full_path = os.path.join(root, f)
            rel_path = os.path.relpath(full_path, start=os.getcwd())
            files_list.append((full_path, rel_path))
    return files_list

def sync():
    logging.info("Auditing remote files to skip already-uploaded assets...")
    try:
        remote_files = set(api.list_repo_files(repo_id=repo_id, repo_type="dataset"))
    except Exception as e:
        logging.error(f"Failed to list remote files: {e}")
        remote_files = set()

    target_dirs = ["annotations", "splits"]
    target_files = ["data/butterflies/metadata.csv", "data/plants/metadata.csv"]

    all_local_files = []
    for d in target_dirs:
        if os.path.exists(d):
            all_local_files.extend(get_local_files(d))
    
    for f in target_files:
        if os.path.exists(f):
            all_local_files.append((f, f))

    # Identify missing or updated files (metadata always updated)
    missing_ops = []
    for full, rel in all_local_files:
        if rel not in remote_files or "metadata.csv" in rel:
            missing_ops.append(CommitOperationAdd(path_in_repo=rel, path_or_fileobj=full))

    logging.info(f"Total assets: {len(all_local_files)}")
    logging.info(f"Remaining to upload: {len(missing_ops)}")

    if not missing_ops:
        logging.info("All final assets are already on Hugging Face!")
        return

    num_batches = (len(missing_ops) + BATCH_SIZE - 1) // BATCH_SIZE
    for i in range(num_batches):
        batch = missing_ops[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
        logging.info(f"Committing batch {i+1}/{num_batches} ({len(batch)} files)...")
        
        for attempt in range(MAX_RETRIES):
            try:
                api.create_commit(
                    repo_id=repo_id,
                    operations=batch,
                    commit_message=f"Final Golden Sync: Batch {i+1}/{num_batches} (Attempt {attempt+1})",
                    repo_type="dataset"
                )
                logging.info(f"Batch {i+1} successful.")
                time.sleep(10) # Safe buffer between commits
                break
            except Exception as e:
                logging.warning(f"Batch {i+1} failed (Attempt {attempt+1}): {e}")
                if "Conflict" in str(e) or "409" in str(e):
                    wait_time = 30 * (attempt + 1)
                    logging.info(f"Conflict detected. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                elif attempt < MAX_RETRIES - 1:
                    time.sleep(15)
                else:
                    logging.error(f"Batch {i+1} failed after {MAX_RETRIES} attempts.")

if __name__ == "__main__":
    sync()

import os
import time
from huggingface_hub import HfApi, CommitOperationAdd
from dotenv import load_dotenv

# Path to .env on DGX
ENV_PATH = "/home/23uec552/DLCV/IndoLepAtlas/.env"
load_dotenv(ENV_PATH)

api = HfApi(token=os.environ.get("HF_TOKEN"))
repo_id = "DihelseeWee/IndoLepAtlas"

BATCH_SIZE = 5000

def get_local_files(directory):
    files_list = []
    for root, _, files in os.walk(directory):
        for f in files:
            full_path = os.path.join(root, f)
            rel_path = os.path.relpath(full_path, start=os.getcwd())
            files_list.append((full_path, rel_path))
    return files_list

# Define key files and directories to ensure they are pushed
target_dirs = ["annotations", "splits"]
target_files = ["data/butterflies/metadata.csv", "data/plants/metadata.csv"]

all_ops = []
for d in target_dirs:
    if os.path.exists(d):
        for full, rel in get_local_files(d):
            all_ops.append(CommitOperationAdd(path_in_repo=rel, path_or_fileobj=full))

for f in target_files:
    if os.path.exists(f):
        all_ops.append(CommitOperationAdd(path_in_repo=f, path_or_fileobj=f))

print(f"Total operations to perform: {len(all_ops)}")

num_batches = (len(all_ops) + BATCH_SIZE - 1) // BATCH_SIZE
for i in range(num_batches):
    batch = all_ops[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
    print(f"Committing batch {i+1}/{num_batches} ({len(batch)} files)... ")
    try:
        api.create_commit(
            repo_id=repo_id,
            operations=batch,
            commit_message=f"Final Golden Sync: Batch {i+1}/{num_batches}",
            repo_type="dataset"
        )
        print(f"Batch {i+1} done.")
        time.sleep(2)
    except Exception as e:
        print(f"Batch {i+1} failed: {e}")
        time.sleep(10)

import os
from huggingface_hub import HfApi
from tqdm import tqdm

repo_id = "DihelseeWee/IndoLepAtlas"
api = HfApi()

def verify():
    print("Listing remote files...")
    remote_files = set(api.list_repo_files(repo_id=repo_id, repo_type="dataset"))
    
    local_dirs = ["data/butterflies/images", "data/plants/images"]
    local_count = 0
    missing = []
    
    for d in local_dirs:
        if not os.path.exists(d):
            continue
        for root, _, files in os.walk(d):
            for f in files:
                full_path = os.path.join(root, f)
                rel_path = os.path.relpath(full_path, start=os.getcwd())
                local_count += 1
                if rel_path not in remote_files:
                    missing.append(rel_path)
    
    print(f"Total local files: {local_count}")
    print(f"Total remote files: {len(remote_files)}")
    print(f"Missing from remote: {len(missing)}")
    
    if len(missing) > 0:
        print("First 10 missing files:")
        for m in missing[:10]:
            print(f"  - {m}")
    else:
        print("VERIFIED: All local files are present on Hugging Face.")

    print("\nRecent Commits:")
    commits = list(api.list_repo_commits(repo_id=repo_id, repo_type="dataset"))
    for c in commits[:10]:
        print(f"Commit: {c.commit_id[:8]} | Date: {c.created_at} | Message: '{c.message}'")

if __name__ == "__main__":
    verify()

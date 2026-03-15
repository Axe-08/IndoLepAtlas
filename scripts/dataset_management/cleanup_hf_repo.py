import os
from huggingface_hub import HfApi
from dotenv import load_dotenv

# Load HF Token
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
REPO_ID = "DihelseeWee/IndoLepAtlas"

api = HfApi(token=HF_TOKEN)

def cleanup_repo():
    # List of legacy directories to remove
    # WARNING: These were successfully moved/copied to data/butterflies/* and data/plants/*
    
    # 1. Root 'host_plants' directory
    print(f"Deleting legacy directory: host_plants/")
    try:
        api.delete_folder(
            repo_id=REPO_ID,
            path_in_repo="host_plants",
            repo_type="dataset"
        )
        print("Successfully deleted host_plants/")
    except Exception as e:
        print(f"Error deleting host_plants/: {e}")

    # 2. Root 'plants' directory
    print(f"Deleting legacy directory: plants/")
    try:
        api.delete_folder(
            repo_id=REPO_ID,
            path_in_repo="plants",
            repo_type="dataset"
        )
        print("Successfully deleted plants/")
    except Exception as e:
        print(f"Error deleting plants/: {e}")

    # 3. Individual species directories in 'data/' that are NOT 'butterflies' or 'plants'
    print("Auditing root data/ for legacy species-specific folders...")
    files = api.list_repo_tree(repo_id=REPO_ID, path_in_repo="data", repo_type="dataset")
    dirs_to_delete = []
    for f in files:
        if f.path.startswith("data/") and "/" not in f.path[5:]:
             # This is a direct subfolder of data/
             folder_name = f.path.split("/")[-1]
             if folder_name not in ["butterflies", "plants"]:
                 dirs_to_delete.append(f.path)
    
    print(f"Found {len(dirs_to_delete)} legacy species folders in data/ to remove.")
    for folder_path in dirs_to_delete:
        try:
            print(f"Deleting {folder_path}...")
            api.delete_folder(
                repo_id=REPO_ID,
                path_in_repo=folder_path,
                repo_type="dataset"
            )
        except Exception as e:
            print(f"Error deleting {folder_path}: {e}")

    print("Cleanup complete.")

if __name__ == "__main__":
    cleanup_repo()

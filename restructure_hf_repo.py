import os
import logging
from huggingface_hub import HfApi
from dotenv import load_dotenv

load_dotenv()
token = os.environ.get("HF_TOKEN")
repo_id = "DihelseeWee/IndoLepAtlas"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
api = HfApi(token=token)

def migrate():
    files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    
    # 1. Move root plants/images/ to data/plants/images/
    plant_trimmed_files = [f for f in files if f.startswith("plants/images/")]
    if plant_trimmed_files:
        logging.info(f"Moving {len(plant_trimmed_files)} trimmed plant images to data/plants/images/...")
        for f in plant_trimmed_files:
            new_path = f.replace("plants/images/", "data/plants/images/")
            try:
                api.move_file(from_path=f, to_path=new_path, repo_id=repo_id, repo_type="dataset")
            except Exception as e:
                logging.warning(f"Failed to move {f}: {e}")
                
    # 2. Upload missing butterfly trimmed images
    logging.info("Uploading trimmed butterfly images to data/butterflies/images/...")
    try:
        api.upload_folder(
            folder_path="data/butterflies/images",
            path_in_repo="data/butterflies/images",
            repo_id=repo_id,
            repo_type="dataset"
        )
    except Exception as e:
        logging.error(f"Butterfly upload failed: {e}")

    # 3. Clean up root-level legacy directories
    legacy_dirs = ["host_plants", "plants", "locks", "butterflies"]
    for d in legacy_dirs:
        try:
            # Check if it has files
            d_files = [f for f in files if f.startswith(f"{d}/")]
            if d_files:
                logging.info(f"Deleting legacy directory: {d}")
                api.delete_folder(path_in_repo=d, repo_id=repo_id, repo_type="dataset")
        except Exception as e:
            logging.warning(f"Could not delete {d}: {e}")

    logging.info("Hugging Face restructuring complete.")

if __name__ == "__main__":
    migrate()

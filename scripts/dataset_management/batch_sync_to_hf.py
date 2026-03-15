import os
import logging
import pandas as pd
from huggingface_hub import HfApi
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()
token = os.environ.get("HF_TOKEN")
repo_id = "DihelseeWee/IndoLepAtlas"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
api = HfApi(token=token)

def sync():
    # 1. Sync Metadata Files
    metadata_files = [
        ("data/butterflies/metadata.csv", "data/butterflies/metadata.csv"),
        ("data/plants/metadata.csv", "data/plants/metadata.csv")
    ]
    
    for local_path, repo_path in metadata_files:
        if os.path.exists(local_path):
            logging.info(f"Syncing {local_path} to {repo_path}...")
            try:
                api.upload_file(
                    path_or_fileobj=local_path,
                    path_in_repo=repo_path,
                    repo_id=repo_id,
                    repo_type="dataset"
                )
            except Exception as e:
                logging.error(f"Failed to upload {local_path}: {e}")

    # 2. Batch Upload Butterfly Images (Species by Species)
    base_dir = "data/butterflies/images"
    if os.path.exists(base_dir):
        species_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        logging.info(f"Starting batch upload for {len(species_dirs)} butterfly species...")
        
        for species in tqdm(species_dirs, desc="Uploading Butterfly Species"):
            local_species_dir = os.path.join(base_dir, species)
            repo_species_dir = f"data/butterflies/images/{species}"
            
            try:
                # Check if already exists to skip if re-running (optional but good)
                api.upload_folder(
                    folder_path=local_species_dir,
                    path_in_repo=repo_species_dir,
                    repo_id=repo_id,
                    repo_type="dataset",
                    delete_patterns=None # Don't delete anything
                )
            except Exception as e:
                logging.error(f"Batch upload failed for {species}: {e}")

    # 3. Batch Upload Plant Images
    base_dir_plants = "data/plants/images"
    if os.path.exists(base_dir_plants):
        plant_dirs = [d for d in os.listdir(base_dir_plants) if os.path.isdir(os.path.join(base_dir_plants, d))]
        logging.info(f"Starting batch upload for {len(plant_dirs)} plant species...")
        
        for species in tqdm(plant_dirs, desc="Uploading Plant Species"):
            local_species_dir = os.path.join(base_dir_plants, species)
            repo_species_dir = f"data/plants/images/{species}"
            
            try:
                api.upload_folder(
                    folder_path=local_species_dir,
                    path_in_repo=repo_species_dir,
                    repo_id=repo_id,
                    repo_type="dataset"
                )
            except Exception as e:
                logging.error(f"Batch upload failed for {species}: {e}")

    logging.info("Batch synchronization complete.")

if __name__ == "__main__":
    sync()

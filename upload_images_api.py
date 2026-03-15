import os
import logging
from huggingface_hub import HfApi
from dotenv import load_dotenv

load_dotenv()
token = os.environ.get("HF_TOKEN")
repo_id = "DihelseeWee/IndoLepAtlas"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
api = HfApi(token=token)

def upload():
    logging.info("Starting upload of butterflies/images...")
    try:
        api.upload_folder(
            folder_path="data/butterflies/images",
            path_in_repo="butterflies/images",
            repo_id=repo_id,
            repo_type="dataset"
        )
    except Exception as e:
        logging.error(f"Error uploading butterflies: {e}")
        
    logging.info("Starting upload of plants/images...")
    try:
        api.upload_folder(
            folder_path="data/plants/images",
            path_in_repo="plants/images",
            repo_id=repo_id,
            repo_type="dataset"
        )
    except Exception as e:
        logging.error(f"Error uploading plants: {e}")
        
    logging.info("Image upload process complete.")

if __name__ == "__main__":
    upload()

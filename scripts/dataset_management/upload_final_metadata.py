import os
from huggingface_hub import HfApi, CommitOperationAdd
from dotenv import load_dotenv

# Load HF Token
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
REPO_ID = "DihelseeWee/IndoLepAtlas"

api = HfApi(token=HF_TOKEN)

def upload_metadata():
    operations = [
        CommitOperationAdd(
            path_in_repo="data/butterflies/metadata.csv",
            path_or_fileobj="data/butterflies/metadata.csv"
        ),
        CommitOperationAdd(
            path_in_repo="data/plants/metadata.csv",
            path_or_fileobj="data/plants/metadata.csv"
        )
    ]
    
    print(f"Uploading finalized metadata for butterflies and plants to {REPO_ID}...")
    api.create_commit(
        repo_id=REPO_ID,
        operations=operations,
        commit_message="Sync finalized, cleaned metadata for butterflies and plants",
        repo_type="dataset"
    )
    print("Metadata upload complete.")

if __name__ == "__main__":
    upload_metadata()

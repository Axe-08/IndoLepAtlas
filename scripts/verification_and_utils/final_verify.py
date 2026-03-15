import os
from huggingface_hub import HfApi
from dotenv import load_dotenv
import pandas as pd

load_dotenv("/home/23uec552/DLCV/IndoLepAtlas/.env")
api = HfApi(token=os.environ.get("HF_TOKEN"))
repo_id = "DihelseeWee/IndoLepAtlas"

files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
key_files = [
    "annotations/annotations.json",
    "annotations/classes.txt",
    "splits/train.txt",
    "splits/val.txt",
    "splits/test.txt",
    "data/butterflies/metadata.csv",
    "data/plants/metadata.csv"
]

print("--- Final Verification ---")
for k in key_files:
    if k in files:
        print(f"[PASS] {k} exists on HF.")
    else:
        print(f"[FAIL] {k} MISSING on HF!")

df = pd.read_csv("data/butterflies/metadata.csv", low_memory=False)
if "split" in df.columns:
    print(f"[PASS] Butterfly metadata has split column. Unique splits: {df['split'].unique()}")
else:
    print("[FAIL] Butterfly metadata missing split column!")

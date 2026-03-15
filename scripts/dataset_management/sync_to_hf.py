import os
import json
import base64
import logging
import requests
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")
REPO_ID = "DihelseeWee/IndoLepAtlas"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

MAX_NDJSON_SIZE = 15 * 1024 * 1024  # 15MB limit for NDJSON body

def upload_files(files_to_upload, summary_msg="Sync pipeline artifacts"):
    import hashlib
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    
    current_commit_lines = [json.dumps({"key": "header", "value": {"summary": summary_msg}})]
    current_commit_size = len(current_commit_lines[0])
    commit_batches = []
    lfs_objects_all = []
    file_metadata_all = {}
    
    for local_path, hf_path, is_lfs in files_to_upload:
        if not os.path.exists(local_path):
            continue
            
        with open(local_path, "rb") as f:
            content = f.read()
        
        size = len(content)
        sha256 = hashlib.sha256(content).hexdigest()
        
        actual_is_lfs = is_lfs or size > 5 * 1024 * 1024
        
        file_metadata_all[hf_path] = {
            "content": content, "size": size, "sha256": sha256, "is_lfs": actual_is_lfs
        }
        
        if actual_is_lfs:
            lfs_objects_all.append({"oid": sha256, "size": size})
            line = json.dumps({"key": "lfsFile", "value": {"path": hf_path, "algo": "sha256", "oid": sha256, "size": size}})
        else:
            b64 = base64.b64encode(content).decode("ascii")
            line = json.dumps({"key": "file", "value": {"path": hf_path, "encoding": "base64", "content": b64}})
            
        line_size = len(line.encode("utf-8"))
        if current_commit_size + line_size > MAX_NDJSON_SIZE and len(current_commit_lines) > 1:
            commit_batches.append(current_commit_lines)
            current_commit_lines = [json.dumps({"key": "header", "value": {"summary": summary_msg + " (part)"}})]
            current_commit_size = len(current_commit_lines[0])
            
        current_commit_lines.append(line)
        current_commit_size += line_size
        
    if len(current_commit_lines) > 1:
        commit_batches.append(current_commit_lines)
            
    if not file_metadata_all: return True
        
    if lfs_objects_all:
        logging.info(f"Uploading {len(lfs_objects_all)} LFS objects...")
        lfs_url = f"https://huggingface.co/datasets/{REPO_ID}.git/info/lfs/objects/batch"
        lfs_headers = {**headers, "Content-Type": "application/vnd.git-lfs+json", "Accept": "application/vnd.git-lfs+json"}
        resp = requests.post(lfs_url, headers=lfs_headers, json={"operation": "upload", "transfers": ["basic"], "objects": lfs_objects_all}, timeout=60)
        resp.raise_for_status()
        
        for obj in resp.json().get("objects", []):
            if "actions" in obj and "upload" in obj["actions"]:
                up_action = obj["actions"]["upload"]
                content_to_upload = next(m["content"] for m in file_metadata_all.values() if m["sha256"] == obj["oid"])
                up_resp = requests.put(up_action["href"], headers=up_action.get("header", {}), data=content_to_upload, timeout=120)
                up_resp.raise_for_status()
                
    commit_url = f"https://huggingface.co/api/datasets/{REPO_ID}/commit/main"
    for i, batch in enumerate(commit_batches):
        logging.info(f"Committing batch {i+1}/{len(commit_batches)} ({len(batch)-1} files)...")
        commit_resp = requests.post(commit_url, headers={**headers, "Content-Type": "application/x-ndjson"}, data="\n".join(batch).encode("utf-8"), timeout=180)
        if commit_resp.status_code not in (200, 201):
            logging.error(f"Commit failed ({commit_resp.status_code}): {commit_resp.text}")
            return False
            
    logging.info("All commits successful.")
    return True

def sync_artifacts():
    if not HF_TOKEN: return
    logging.info("Starting artifact sync...")
    upload_list = [
        ("data/butterflies/metadata.csv", "data/butterflies/metadata.csv", False),
        ("data/plants/metadata.csv", "data/plants/metadata.csv", False),
        ("annotations/annotations.json", "annotations/annotations.json", False),
        ("annotations/classes.txt", "annotations/classes.txt", False),
        ("splits/train.txt", "splits/train.txt", False),
        ("splits/val.txt", "splits/val.txt", False),
        ("splits/test.txt", "splits/test.txt", False),
    ]
    for doc in os.listdir("docs"):
        if doc.endswith(".md") or doc.endswith(".json"):
            upload_list.append((f"docs/{doc}", f"docs/{doc}", False))
            
    upload_files(upload_list, summary_msg="Sync pipeline artifacts (metadata, annotations, splits, docs)")

if __name__ == "__main__":
    sync_artifacts()

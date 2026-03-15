import os
from huggingface_hub import HfApi

repo_id = "DihelseeWee/IndoLepAtlas"
api = HfApi()

def audit():
    print("Auditing Hugging Face repository...")
    all_files = api.list_repo_tree(repo_id=repo_id, repo_type="dataset")
    
    # 1. Structure Audit
    b_raw_species = set()
    b_img_species = set()
    p_raw_species = set()
    p_img_species = set()
    
    for f in all_files:
        parts = f.path.split("/")
        # data/butterflies/raw/<species>/<file>
        if len(parts) >= 4 and parts[0] == "data" and parts[1] == "butterflies":
            if parts[2] == "raw":
                b_raw_species.add(parts[3])
            elif parts[2] == "images":
                b_img_species.add(parts[3])
        # data/plants/raw/<species>/<file>
        elif len(parts) >= 4 and parts[0] == "data" and parts[1] == "plants":
            if parts[2] == "raw":
                p_raw_species.add(parts[3])
            elif parts[2] == "images":
                p_img_species.add(parts[3])

    print("\n--- Hugging Face Counts ---")
    print(f"Butterflies: Raw={len(b_raw_species)}, Images={len(b_img_species)}")
    print(f"Plants:      Raw={len(p_raw_species)}, Images={len(p_img_species)}")
    
    # 2. Local Audit (on DGX)
    local_b_raw = set(os.listdir("data/butterflies/raw")) if os.path.exists("data/butterflies/raw") else set()
    local_b_img = set(os.listdir("data/butterflies/images")) if os.path.exists("data/butterflies/images") else set()
    local_p_raw = set(os.listdir("data/plants/raw")) if os.path.exists("data/plants/raw") else set()
    local_p_img = set(os.listdir("data/plants/images")) if os.path.exists("data/plants/images") else set()

    print("\n--- DGX Local Counts ---")
    print(f"Butterflies: Raw={len(local_b_raw)}, Images={len(local_b_img)}")
    print(f"Plants:      Raw={len(local_p_raw)}, Images={len(local_p_img)}")

    # 3. Discrepancy Identification
    if b_raw_species != b_img_species:
        print("\nButterfly Discrepancy (HF):")
        only_raw = b_raw_species - b_img_species
        only_img = b_img_species - b_raw_species
        if only_raw: print(f"  In Raw but not Images: {len(only_raw)} species (e.g., {list(only_raw)[:5]})")
        if only_img: print(f"  In Images but not Raw: {len(only_img)} species (e.g., {list(only_img)[:5]})")
    else:
        print("\nButterfly counts match on HF.")

    if p_raw_species != p_img_species:
        print("\nPlant Discrepancy (HF):")
        only_raw = p_raw_species - p_img_species
        only_img = p_img_species - p_raw_species
        if only_raw: print(f"  In Raw but not Images: {len(only_raw)} species (e.g., {list(only_raw)[:5]})")
        if only_img: print(f"  In Images but not Raw: {len(only_img)} species (e.g., {list(only_img)[:5]})")
    else:
        print("\nPlant counts match on HF.")

if __name__ == "__main__":
    audit()

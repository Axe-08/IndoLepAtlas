import json
import pandas as pd
import os

def get_sample():
    # Load COCO annotations
    with open("annotations/annotations.json", "r") as f:
        coco = json.load(f)
    
    # Load Metadata
    df = pd.read_csv("data/butterflies/metadata.csv")
    
    # Pick a sample that has metadata (non-null common name for example)
    # Let\s look at Acraea_terpsicore since we know it has data
    sample_df = df[df["filename"].str.contains("Acraea_terpsicore")].iloc[0]
    filename = sample_df["filename"]
    
    # Find image in COCO
    img_info = next((img for img in coco["images"] if img["file_name"] == filename), None)
    if not img_info:
        print(f"Image {filename} not found in COCO")
        return

    img_id = img_info["id"]
    annotations = [ann for ann in coco["annotations"] if ann["image_id"] == img_id]
    category = next((cat for cat in coco["categories"] if cat["id"] == annotations[0]["category_id"]), {})

    sample_output = {
        "image_metadata": sample_df.to_dict(),
        "detection_metadata": {
            "image_id": img_id,
            "width": img_info["width"],
            "height": img_info["height"],
            "annotations": [
                {
                    "category_name": category.get("name"),
                    "category_id": ann["category_id"],
                    "bbox": ann["bbox"],
                    "area": ann["area"]
                } for ann in annotations
            ]
        }
    }
    
    with open("sample_output.json", "w") as f:
        json.dump(sample_output, f, indent=4)
    print(f"Sample generated for {filename}")

if __name__ == "__main__":
    get_sample()

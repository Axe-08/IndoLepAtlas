import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

def visualize_multi_bboxes(json_path, image_dir, output_dir, num_samples=5):
    with open(json_path, "r") as f:
        coco = json.load(f)

    os.makedirs(output_dir, exist_ok=True)
    images = coco["images"]
    annotations = coco["annotations"]
    
    # Map class IDs to species names
    class_map = {cat["id"]: cat["name"] for cat in coco["categories"]}
    
    # Map image ID to its annotations
    ann_map = {}
    for ann in annotations:
        img_id = ann["image_id"]
        if img_id not in ann_map:
            ann_map[img_id] = []
        ann_map[img_id].append(ann)
        
    # Find images with multiple annotations vs single
    multi_images = [img for img in images if len(ann_map.get(img["id"], [])) > 1]
    single_images = [img for img in images if len(ann_map.get(img["id"], [])) == 1]
    
    # Mix for visual variety
    samples = multi_images[:3] + single_images[:2]

    for img_info in samples:
        img_id = img_info["id"]
        parts = img_info["file_name"].split("/")
        slug = parts[-2]
        fname = parts[-1]
        img_path = os.path.join(image_dir, slug, fname)
        
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            continue

        img = Image.open(img_path)
        fig, ax = plt.subplots(1)
        ax.imshow(img)

        if img_id in ann_map:
            for ann in ann_map[img_id]:
                bbox = ann["bbox"]
                cat_name = class_map[ann["category_id"]]
                if len(bbox) == 4:
                    x, y, w, h = bbox
                    rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor="r", facecolor="none")
                    ax.add_patch(rect)
                    ax.text(x, max(y - 5, 0), cat_name, color="red", fontsize=10, 
                            bbox=dict(facecolor="white", alpha=0.5))

        plt.axis("off")
        out_path = os.path.join(output_dir, f"verify_multi_{fname}")
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        print(f"Saved visualization to {out_path}")

if __name__ == "__main__":
    visualize_multi_bboxes(
        json_path="annotations/annotations.json",
        image_dir="data/butterflies/images",
        output_dir="verification_viz"
    )

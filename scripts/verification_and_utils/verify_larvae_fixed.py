import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

def visualize_larvae_robust(json_path, image_dir, output_dir, num_samples=10):
    with open(json_path, "r") as f:
        coco = json.load(f)

    os.makedirs(output_dir, exist_ok=True)
    
    class_map = {cat["id"]: cat["name"] for cat in coco["categories"]}
    ann_map = {}
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        if img_id not in ann_map:
            ann_map[img_id] = []
        ann_map[img_id].append(ann)

    count = 0
    for img_info in coco["images"]:
        if "EarlyStage" not in img_info["file_name"]:
            continue
            
        img_id = img_info["id"]
        # Correct path mapping
        # img_info["file_name"] is usually "butterflies/images/<slug>/<fname>"
        rel_path = img_info["file_name"].replace("butterflies/images/", "")
        img_path = os.path.join(image_dir, rel_path)
        
        if not os.path.exists(img_path):
            continue

        try:
            img = Image.open(img_path)
            actual_w, actual_h = img.size
            coco_w, coco_h = img_info["width"], img_info["height"]
            
            print(f"Image: {img_info['file_name']}")
            print(f"  Actual Size: {actual_w}x{actual_h}")
            print(f"  COCO Size: {coco_w}x{coco_h}")

            fig, ax = plt.subplots(1)
            ax.imshow(img)

            if img_id in ann_map:
                for ann in ann_map[img_id]:
                    bbox = ann["bbox"]
                    cat_name = class_map[ann["category_id"]]
                    if len(bbox) == 4:
                        x, y, w, h = bbox
                        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor="yellow", facecolor="none")
                        ax.add_patch(rect)
                        ax.text(x, max(y - 5, 0), f"{cat_name}", color="yellow", fontsize=8, 
                                bbox=dict(facecolor="black", alpha=0.5))

            plt.axis("off")
            fname = os.path.basename(img_path)
            out_path = os.path.join(output_dir, f"robust_larva_{fname}")
            plt.savefig(out_path, bbox_inches="tight", dpi=150)
            plt.close()
            print(f"  Saved to {out_path}")
            count += 1
            if count >= num_samples:
                break
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

if __name__ == "__main__":
    visualize_larvae_robust(
        json_path="annotations/annotations.json",
        image_dir="data/butterflies/images",
        output_dir="verification_viz_new",
        num_samples=10
    )

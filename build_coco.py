import json
import os
import glob

ann_dir = "annotations/butterflies"

coco = {
    "images": [],
    "annotations": [],
    "categories": []
}

classes_file = "annotations/classes.txt"
with open(classes_file, "r") as f:
    classes = [line.strip() for line in f.readlines()]
for idx, cls in enumerate(classes):
    coco["categories"].append({"id": idx, "name": cls})
    
img_id = 0
ann_id = 0

files = glob.glob(os.path.join(ann_dir, "*", "*.txt"))
for f in files:
    with open(f, "r") as lines:
        parts = f.split("/")
        slug = parts[-2]
        img_name = parts[-1].replace(".txt", ".jpg")
        
        coco["images"].append({
            "id": img_id,
            "file_name": f"{slug}/{img_name}",
            "width": 1000,
            "height": 1000
        })
        
        for line in lines:
            class_id, cx, cy, w, h = map(float, line.strip().split())
            class_id = int(class_id)
            
            # Convert cxcywh (normalized) back to pseudo pixel coordinates (assuming 1000x1000 for visualization aspect ratio parsing)
            px_w = w * 1000
            px_h = h * 1000
            px_x = (cx * 1000) - (px_w / 2)
            px_y = (cy * 1000) - (px_h / 2)
            
            coco["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": class_id,
                "bbox": [px_x, px_y, px_w, px_h],
                "area": px_w * px_h,
                "iscrowd": 0
            })
            ann_id += 1
            
        img_id += 1

with open("annotations/annotations.json", "w") as out:
    json.dump(coco, out)

print(f"Built COCO with {img_id} images and {ann_id} annotations")

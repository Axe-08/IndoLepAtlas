import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import os

def visualize():
    with open("sample_output.json", "r") as f:
        data = json.load(f)
    
    img_rel_path = data["image_metadata"]["filename"].replace("butterflies/images/", "")
    img_path = os.path.join("data/butterflies/images", img_rel_path)
    
    img = Image.open(img_path)
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    
    for ann in data["detection_metadata"]["annotations"]:
        x, y, w, h = ann["bbox"]
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor="yellow", facecolor="none")
        ax.add_patch(rect)
        ax.text(x, max(y - 5, 0), ann["category_name"], color="yellow", fontsize=10, 
                bbox=dict(facecolor="black", alpha=0.5))
    
    plt.axis("off")
    plt.savefig("sample_viz.jpg", bbox_inches="tight", dpi=150)
    plt.close()
    print("Visualization saved to sample_viz.jpg")

if __name__ == "__main__":
    visualize()

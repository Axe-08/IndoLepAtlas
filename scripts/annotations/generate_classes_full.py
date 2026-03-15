#!/usr/bin/env python3
import os

def generate_classes(base_dir="."):
    mapping = []
    
    # Butterflies
    b_dir = os.path.join(base_dir, "data", "butterflies", "images")
    if os.path.exists(b_dir):
        for slug in sorted(os.listdir(b_dir)):
            if os.path.isdir(os.path.join(b_dir, slug)):
                mapping.append(slug)
                
    # Plants
    p_dir = os.path.join(base_dir, "data", "plants", "images")
    if os.path.exists(p_dir):
        for slug in sorted(os.listdir(p_dir)):
            if os.path.isdir(os.path.join(p_dir, slug)):
                mapping.append(slug)
                
    out_path = os.path.join(base_dir, "annotations", "classes.txt")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    with open(out_path, "w") as f:
        for i, slug in enumerate(mapping):
            f.write(f"{i} {slug}\n")
            
    print(f"Wrote {len(mapping)} classes to {out_path}")

if __name__ == "__main__":
    generate_classes()

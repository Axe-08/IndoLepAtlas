import json
import os

meta_file = "data/butterflies/metadata.jsonl"
count = 0
if os.path.exists(meta_file):
    with open(meta_file, "r") as f:
        for line in f:
            if not line.strip(): continue
            rec = json.loads(line)
            if "xmin" in rec and rec["xmin"] != "":
                count += 1
print(f"Processed {count} images so far")

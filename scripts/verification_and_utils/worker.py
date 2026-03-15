import os, sys, easyocr, pandas as pd, numpy as np
from PIL import Image, ImageFile
from tqdm import tqdm

dataset = sys.argv[1]
chunk_id = sys.argv[2]
ImageFile.LOAD_TRUNCATED_IMAGES = True
reader = easyocr.Reader(["en"], gpu=True)

csv_path = f"data/chunks/chunk_{dataset}_{chunk_id}.csv"
df = pd.read_csv(csv_path)
df["location_refined"] = ""
df["location_refined"] = df["location_refined"].astype(object)

for i, row in tqdm(df.iterrows(), total=len(df)):
    raw_path = os.path.join("data", row["raw_filename"])
    if not os.path.exists(raw_path): continue
    try:
        img = Image.open(raw_path)
        w, h = img.size
        # Bottom 12%
        bot = img.crop((0, int(h * 0.88), w, h))
        res = reader.readtext(np.array(bot), detail=0)
        if res: df.at[i, "location_refined"] = " ".join(res)
    except: continue

df.to_csv(f"data/chunks/chunk_{dataset}_{chunk_id}_refined.csv", index=False)

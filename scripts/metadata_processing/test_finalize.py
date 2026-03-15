import pandas as pd
import os
import re

COMMON_OCR_NOISE = {"ee", "eee", "bo", "me", "cm", "gemma", "mma", "cma", "ey", "se", "mer", "conga", "pees", "aman", "te", "iam", "nae", "mgs", "gal", "en"}

def clean_refined(text):
    if not isinstance(text, str): return ""
    text = re.sub(r"^[^a-zA-Z0-9]+", "", text)
    text = re.sub(r"[^a-zA-Z0-9]+$", "", text)
    words = text.split()
    cleaned_words = []
    for w in words:
        wl = w.lower().strip(".,()/-")
        if wl in COMMON_OCR_NOISE: continue
        if re.match(r"^([a-z]\.)+[a-z]?$", wl): continue
        cleaned_words.append(w)
    text = " ".join(cleaned_words)
    text = text.replace("@", "").replace("{", "(").replace("}", ")")
    text = re.sub(r"\s+", " ", text).strip(" ,.-;")
    return text

def test_on_processed():
    csv_path = "data/butterflies/metadata.csv"
    df = pd.read_csv(csv_path)
    if "location_refined" in df.columns:
        sample = df[df["location_refined"].notna()].head(5)
        for i, row in sample.iterrows():
            orig_refined = row["location_refined"]
            cleaned = clean_refined(orig_refined)
            print(f"Original Refined: {orig_refined}")
            print(f"Cleaned:          {cleaned}")
            print("-" * 30)

if __name__ == "__main__":
    test_on_processed()

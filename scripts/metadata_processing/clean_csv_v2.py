import os, csv, json, re

# Known OCR noise patterns
NOISE_WORDS = {
    "mmm", "mma", "emma", "amas", "commas", "maa", "eee", "rre", "rrr", 
    "ttt", "sss", "ccc", "vvv", "bbb", "nnn", "eee", "aaa", "uuu", "iii",
    "peeee", "perre", "teper", "mcedaet", "met", "tea", "ieee", "peere", "peer"
}

def clean_noise(text):
    if not text: return ""
    
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    
    # 1. Identify gender separately to preserve it
    sex = ""
    for g in ["Male", "Female"]:
        if re.search(r"\b" + g + r"\b", text, re.IGNORECASE):
            sex = g
            break
            
    # 2. Aggressive regex cleaning
    # Remove strings of 3+ repeating letters
    text = re.sub(r"([a-zA-Z])\1{2,}", r"\1", text)
    # Only keep letters, numbers, and basic punctuation
    text = re.sub(r"[^a-zA-Z0-9\s.,()\/\-©@]", " ", text)
    
    # 3. Word-based filtering
    words = text.split()
    cleaned_words = []
    for w in words:
        wl = w.lower().strip(".,()/-")
        # Skip if it matches a noise word
        if wl in NOISE_WORDS:
            continue
        # Skip if it is too short and nonsensical (unless it is a known state abbreviation)
        if len(wl) <= 2 and not wl.isalpha():
            continue
        # Skip long strings of consonants or highly repetitive characters
        if len(wl) > 5 and not any(v in wl for v in "aeiouy"):
            continue
            
        cleaned_words.append(w)
        
    cleaned_text = " ".join(cleaned_words)
    
    # Clean up the punctuation leftover
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip(" ,.-;")
    
    # If the resulting text looks like it is just residual noise (all short tokens or no vowels)
    tokens = cleaned_text.split()
    if len(tokens) > 0:
        meaningful = [t for t in tokens if len(t) > 2 or any(c.isupper() for c in t)]
        if not meaningful and len(tokens) > 5:
            return "" # Entirely noise
            
    return cleaned_text

def process_csv(path, species_map):
    if not os.path.exists(path): return
    rows = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        for row in reader:
            # 1. Map common name (Guaranteed clean from JSONL)
            sp = row.get("species", "").strip()
            if sp in species_map:
                row["common_name"] = species_map[sp]
            
            # 2. Clean Location
            row["location"] = clean_noise(row.get("location", ""))
            
            # 3. Clean Credit
            row["credit"] = clean_noise(row.get("credit", ""))
            
            # 4. Clean Media Code
            mc = row.get("media_code", "")
            if mc:
                mc = re.sub(r"[^a-zA-Z0-9\-]", "", mc)
                if mc.lower().startswith("mediacode"): mc = mc[9:]
                row["media_code"] = mc
                
            rows.append(row)
            
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

def main():
    # Load clean species map
    species_map = {}
    for source in ["butterflies", "plants"]:
        raw_dir = f"data/{source}/raw"
        if os.path.exists(raw_dir):
            for slug in os.listdir(raw_dir):
                jsonl = os.path.join(raw_dir, slug, "metadata.jsonl")
                if os.path.exists(jsonl):
                    with open(jsonl, "r") as f:
                        for line in f:
                            try:
                                d = json.loads(line)
                                title = d.get("page_title", "")
                                if " - " in title:
                                    common = title.split(" - ")[-1].split(" | ")[0].strip()
                                    sp = (d.get("Genus", "") + " " + d.get("Species", "")).strip()
                                    species_map[sp] = common
                            except: pass
                            
    process_csv("data/butterflies/metadata.csv", species_map)
    process_csv("data/plants/metadata.csv", species_map)
    print("Cleaned metadata successfully!")

if __name__ == "__main__":
    main()

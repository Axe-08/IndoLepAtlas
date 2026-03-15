import json, os

raw_dir = "data/butterflies/raw"
for slug in os.listdir(raw_dir):
    jsonl_path = os.path.join(raw_dir, slug, "metadata.jsonl")
    if os.path.exists(jsonl_path):
        with open(jsonl_path, "r") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if "page_title" in data:
                        title = data["page_title"]
                        parts = title.split(" - ")
                        if len(parts) > 1:
                            common = parts[-1].split(" | ")[0].strip()
                            species_key = f"{data.get(Genus, )} {data.get(Species, )}".strip()
                            print(f"SUCCESS: {species_key} -> {common}")
                            exit()
                except Exception as e:
                    print(f"ERROR: {e}")
                    exit()

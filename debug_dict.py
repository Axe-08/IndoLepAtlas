import json, os

species_to_common = {}
raw_dir = "data/butterflies/raw"
slugs = os.listdir(raw_dir)
print(f"Checking {len(slugs)} slugs...")
for slug in slugs:
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
                            g = data.get("Genus", "")
                            s = data.get("Species", "")
                            species_key = (g + " " + s).strip()
                            species_to_common[species_key] = common
                            if "Athyma cama" == species_key:
                                print(f"FOUND Athyma cama: {common}")
                except Exception as e:
                    pass
print(f"Total mapped: {len(species_to_common)}")

"""
iNaturalist Coverage Analysis for IndoLepAtlas
Queries iNat API for observation counts of each butterfly species in India.
Outputs CSV: species, inat_india_count, inat_global_count, indolepatlas_count
"""
import csv
import json
import time
import urllib.request
import urllib.parse
import urllib.error
import sys
import os

# ─── Config ───────────────────────────────────────────────────────
SPECIES_FILE = os.path.join(os.path.dirname(__file__), "all_butterflies.txt")
OUTPUT_CSV = os.path.join(os.path.dirname(__file__), "inat_species_coverage.csv")
PROGRESS_FILE = os.path.join(os.path.dirname(__file__), "inat_progress.json")

# iNaturalist API (public, no auth needed)
INAT_API = "https://api.inaturalist.org/v1"

# India place_id on iNaturalist
INDIA_PLACE_ID = 6681

# Rate limit: iNat allows ~1 req/sec for unauthenticated
DELAY_SECONDS = 1.2

# Only butterfly entries (indices 0-960 in the file)
MAX_BUTTERFLY_INDEX = 960


def load_species(filepath):
    """Parse the all_butterflies.txt file → list of (index, species_name)"""
    species = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)  # split on first whitespace
            if len(parts) < 2:
                continue
            idx = int(parts[0])
            if idx > MAX_BUTTERFLY_INDEX:
                break  # stop at plants
            name = parts[1].replace('_', ' ')  # "Abisara_attenuata" → "Abisara attenuata"
            species.append((idx, name))
    return species


def load_progress(filepath):
    """Load previously queried results to allow resuming."""
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return {}


def save_progress(filepath, data):
    with open(filepath, 'w') as f:
        json.dump(data, f)


def query_inat_observations(species_name, place_id=None):
    """
    Query iNat API for observation count of a species.
    Returns (taxon_id, observation_count) or (None, 0) if not found.
    """
    params = {
        'q': species_name,
        'rank': 'species',
        'per_page': 1,
    }
    url = f"{INAT_API}/taxa?" + urllib.parse.urlencode(params)

    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'IndoLepAtlas-Research/1.0'})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
        print(f"  [WARN] Taxa lookup failed for '{species_name}': {e}")
        return None, 0, 0

    if not data.get('results'):
        return None, 0, 0

    # Find the best matching taxon
    taxon = None
    for result in data['results']:
        if result.get('name', '').lower() == species_name.lower():
            taxon = result
            break
    if taxon is None:
        taxon = data['results'][0]  # best guess

    taxon_id = taxon['id']
    global_count = taxon.get('observations_count', 0)

    # Now query India-specific count
    if place_id:
        time.sleep(DELAY_SECONDS)
        obs_params = {
            'taxon_id': taxon_id,
            'place_id': place_id,
            'quality_grade': 'research',
            'per_page': 0,
        }
        obs_url = f"{INAT_API}/observations?" + urllib.parse.urlencode(obs_params)
        try:
            req = urllib.request.Request(obs_url, headers={'User-Agent': 'IndoLepAtlas-Research/1.0'})
            with urllib.request.urlopen(req, timeout=15) as resp:
                obs_data = json.loads(resp.read().decode())
            india_count = obs_data.get('total_results', 0)
        except Exception as e:
            print(f"  [WARN] India obs query failed for '{species_name}': {e}")
            india_count = 0
    else:
        india_count = 0

    return taxon_id, global_count, india_count


def main():
    species_list = load_species(SPECIES_FILE)
    print(f"Loaded {len(species_list)} butterfly species from {SPECIES_FILE}")

    progress = load_progress(PROGRESS_FILE)
    print(f"Resuming from {len(progress)} previously queried species")

    results = []
    queried = 0
    skipped = 0

    for idx, species_name in species_list:
        if species_name in progress:
            # Already queried — use cached result
            cached = progress[species_name]
            results.append({
                'index': idx,
                'species': species_name,
                'inat_taxon_id': cached.get('taxon_id'),
                'inat_global_count': cached.get('global_count', 0),
                'inat_india_count': cached.get('india_count', 0),
            })
            skipped += 1
            continue

        queried += 1
        print(f"[{idx+1}/{len(species_list)}] Querying: {species_name} ...", end=' ', flush=True)

        taxon_id, global_count, india_count = query_inat_observations(
            species_name, place_id=INDIA_PLACE_ID
        )

        print(f"Global={global_count}, India={india_count}")

        results.append({
            'index': idx,
            'species': species_name,
            'inat_taxon_id': taxon_id,
            'inat_global_count': global_count,
            'inat_india_count': india_count,
        })

        # Save progress after each query
        progress[species_name] = {
            'taxon_id': taxon_id,
            'global_count': global_count,
            'india_count': india_count,
        }
        save_progress(PROGRESS_FILE, progress)

        time.sleep(DELAY_SECONDS)

        # Print periodic stats
        if queried % 50 == 0:
            india_counts = [r['inat_india_count'] for r in results]
            below_100 = sum(1 for c in india_counts if c < 100)
            print(f"\n  --- Progress: {queried} queried, {skipped} cached ---")
            print(f"  --- {below_100}/{len(india_counts)} species with <100 India records ---\n")

    # Sort by index and write CSV
    results.sort(key=lambda x: x['index'])

    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'index', 'species', 'inat_taxon_id',
            'inat_global_count', 'inat_india_count'
        ])
        writer.writeheader()
        writer.writerows(results)

    # Print summary statistics
    india_counts = [r['inat_india_count'] for r in results]
    total = len(india_counts)

    print(f"\n{'='*60}")
    print(f"SUMMARY: {total} species queried")
    print(f"{'='*60}")

    thresholds = [0, 5, 10, 20, 50, 100, 200, 500]
    for t in thresholds:
        count = sum(1 for c in india_counts if c < t)
        pct = count / total * 100
        print(f"  Species with < {t:>4} India iNat records: {count:>4} ({pct:.1f}%)")

    zero_count = sum(1 for c in india_counts if c == 0)
    print(f"\n  Species with ZERO India iNat records:  {zero_count} ({zero_count/total*100:.1f}%)")
    print(f"\n  Median India iNat count: {sorted(india_counts)[total//2]}")
    print(f"  Mean India iNat count:   {sum(india_counts)/total:.1f}")
    print(f"\nResults saved to: {OUTPUT_CSV}")


if __name__ == '__main__':
    main()

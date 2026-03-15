#!/usr/bin/env python3
"""
generate_stats.py — Generate distribution statistics for the dataset.

Reads metadata CSVs and produces distribution_stats.md with:
  - Images per species (top N + long tail)
  - Images per family
  - Geographic distribution (by state)
  - Temporal distribution (by month)
  - Life stage distribution
  - Missing field coverage

Output:
  docs/distribution_stats.md
"""

import os
import csv
import argparse
import logging
from collections import defaultdict, Counter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_metadata(base_dir: str) -> dict:
    """Load all metadata CSVs."""
    data = {"butterflies": [], "plants": []}

    for dataset in data:
        csv_path = os.path.join(base_dir, "data", dataset, "metadata.csv")
        if os.path.exists(csv_path):
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                data[dataset] = list(reader)
            logger.info(f"Loaded {len(data[dataset])} rows from {dataset}")

    return data


def count_distribution(rows: list, key: str) -> Counter:
    """Count distribution of a specific field."""
    return Counter(
        row.get(key, "").strip()
        for row in rows
        if row.get(key, "").strip()
    )


def missing_field_stats(rows: list, fields: list) -> dict:
    """Calculate % of rows with missing/empty values per field."""
    total = len(rows) if rows else 1
    stats = {}
    for field in fields:
        missing = sum(1 for row in rows if not row.get(field, "").strip())
        stats[field] = {
            "missing": missing,
            "present": total - missing,
            "pct_missing": round(missing / total * 100, 1),
        }
    return stats


def format_top_n(counter: Counter, n: int = 30, label: str = "Item") -> str:
    """Format top N items as a markdown table."""
    lines = []
    lines.append(f"| {label} | Count |")
    lines.append("|---|---|")
    for item, count in counter.most_common(n):
        display = item if item else "(empty)"
        lines.append(f"| {display} | {count} |")

    total = sum(counter.values())
    unique = len(counter)
    lines.append(f"\n**Total: {total} images across {unique} unique values**")

    if unique > n:
        remaining = sum(c for _, c in counter.most_common()[n:])
        lines.append(f"*(showing top {n}, {unique - n} more with {remaining} images)*")

    return "\n".join(lines)


def generate_stats(base_dir: str):
    """Generate distribution statistics markdown."""
    data = load_metadata(base_dir)

    docs_dir = os.path.join(base_dir, "docs")
    os.makedirs(docs_dir, exist_ok=True)

    all_rows = data["butterflies"] + data["plants"]

    lines = []
    lines.append("# Dataset Distribution Statistics\n")
    lines.append(f"**Total images:** {len(all_rows)}")
    lines.append(f"- Butterflies: {len(data['butterflies'])}")
    lines.append(f"- Plants: {len(data['plants'])}")
    lines.append("")

    # ── Species distribution ──
    lines.append("## 1. Images per Species\n")
    lines.append("### Butterflies\n")
    b_species = count_distribution(data["butterflies"], "species")
    lines.append(format_top_n(b_species, 30, "Species"))
    lines.append("")

    lines.append("### Plants\n")
    p_species = count_distribution(data["plants"], "species")
    lines.append(format_top_n(p_species, 30, "Species"))
    lines.append("")

    # ── Family distribution ──
    lines.append("## 2. Images per Family\n")
    lines.append("### Butterflies\n")
    b_family = count_distribution(data["butterflies"], "family")
    lines.append(format_top_n(b_family, 20, "Family"))
    lines.append("")

    lines.append("### Plants\n")
    p_family = count_distribution(data["plants"], "family")
    lines.append(format_top_n(p_family, 20, "Family"))
    lines.append("")

    # ── Geographic distribution ──
    lines.append("## 3. Geographic Distribution (by State)\n")
    state_dist = count_distribution(all_rows, "state")
    lines.append(format_top_n(state_dist, 36, "State"))
    lines.append("")

    # ── Temporal distribution ──
    lines.append("## 4. Temporal Distribution (by Month)\n")
    month_counter = Counter()
    for row in all_rows:
        date_str = row.get("date", "").strip()
        if date_str and "/" in date_str:
            parts = date_str.split("/")
            if len(parts) >= 2:
                month_counter[parts[1]] += 1

    month_names = {
        "01": "January", "02": "February", "03": "March",
        "04": "April", "05": "May", "06": "June",
        "07": "July", "08": "August", "09": "September",
        "10": "October", "11": "November", "12": "December",
    }
    lines.append("| Month | Count |")
    lines.append("|---|---|")
    for m in sorted(month_counter):
        name = month_names.get(m, m)
        lines.append(f"| {name} | {month_counter[m]} |")
    lines.append("")

    # ── Life stage distribution (butterflies only) ──
    lines.append("## 5. Life Stage Distribution (Butterflies)\n")
    stage_dist = count_distribution(data["butterflies"], "life_stage")
    lines.append(format_top_n(stage_dist, 10, "Life Stage"))
    lines.append("")

    # ── Missing field coverage ──
    lines.append("## 6. Missing Field Coverage\n")

    lines.append("### Butterflies\n")
    b_fields = ["common_name", "sex", "media_code", "location", "state", "date", "credit"]
    b_missing = missing_field_stats(data["butterflies"], b_fields)
    lines.append("| Field | Present | Missing | % Missing |")
    lines.append("|---|---|---|---|")
    for field, stats in b_missing.items():
        lines.append(
            f"| {field} | {stats['present']} | {stats['missing']} | {stats['pct_missing']}% |"
        )
    lines.append("")

    lines.append("### Plants\n")
    p_fields = ["media_code", "location", "state", "date", "credit"]
    p_missing = missing_field_stats(data["plants"], p_fields)
    lines.append("| Field | Present | Missing | % Missing |")
    lines.append("|---|---|---|---|")
    for field, stats in p_missing.items():
        lines.append(
            f"| {field} | {stats['present']} | {stats['missing']} | {stats['pct_missing']}% |"
        )
    lines.append("")

    # ── Split distribution ──
    lines.append("## 7. Split Distribution\n")
    split_dist = count_distribution(all_rows, "split")
    if split_dist:
        lines.append("| Split | Count | % |")
        lines.append("|---|---|---|")
        total = sum(split_dist.values())
        for split_name in ["train", "val", "test"]:
            count = split_dist.get(split_name, 0)
            pct = round(count / total * 100, 1) if total > 0 else 0
            lines.append(f"| {split_name} | {count} | {pct}% |")
    else:
        lines.append("*Splits not yet generated. Run `generate_splits.py` first.*")
    lines.append("")

    # Write
    output_path = os.path.join(docs_dir, "distribution_stats.md")
    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    logger.info(f"Wrote distribution stats to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate dataset distribution statistics"
    )
    parser.add_argument("--base-dir", type=str, default=".")
    args = parser.parse_args()

    generate_stats(args.base_dir)


if __name__ == "__main__":
    main()

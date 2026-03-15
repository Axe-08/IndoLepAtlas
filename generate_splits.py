#!/usr/bin/env python3
"""
generate_splits.py — Create stratified train/val/test splits.

Reads metadata CSVs, assigns splits (70/15/15) stratified by species,
and writes split files. Preserves existing assignments for stability.

Outputs:
  splits/train.txt
  splits/val.txt
  splits/test.txt

Also updates the 'split' column in metadata CSVs.
"""

import os
import csv
import random
import argparse
import logging
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


def load_existing_splits(splits_dir: str) -> dict:
    """Load existing split assignments for stability."""
    existing = {}
    for split_name in ["train", "val", "test"]:
        path = os.path.join(splits_dir, f"{split_name}.txt")
        if os.path.exists(path):
            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        existing[line] = split_name
    return existing


def generate_splits(base_dir: str, seed: int = 42):
    """Generate stratified splits from metadata CSVs."""
    splits_dir = os.path.join(base_dir, "splits")
    os.makedirs(splits_dir, exist_ok=True)

    # Load existing splits for stability
    existing_splits = load_existing_splits(splits_dir)
    logger.info(f"Loaded {len(existing_splits)} existing split assignments")

    # Collect all files grouped by species
    species_files = defaultdict(list)
    all_csv_data = {}  # dataset -> list of rows

    for dataset in ["butterflies", "plants"]:
        csv_path = os.path.join(base_dir, "data", dataset, "metadata.csv")
        if not os.path.exists(csv_path):
            logger.warning(f"No metadata CSV found: {csv_path}")
            continue

        rows = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
                filename = row.get("filename", "")
                species = row.get("species", "unknown")

                # Use existing split if available
                if filename in existing_splits:
                    row["split"] = existing_splits[filename]
                else:
                    species_files[species].append(filename)

        all_csv_data[dataset] = rows

    # Assign splits to new files using stratified sampling
    random.seed(seed)
    new_assignments = {}

    for species, files in species_files.items():
        random.shuffle(files)
        n = len(files)
        n_train = max(1, int(n * SPLIT_RATIOS["train"]))
        n_val = max(0, int(n * SPLIT_RATIOS["val"]))
        # rest goes to test

        for i, filename in enumerate(files):
            if i < n_train:
                split = "train"
            elif i < n_train + n_val:
                split = "val"
            else:
                split = "test"
            new_assignments[filename] = split

    # Merge with existing
    all_assignments = {**existing_splits, **new_assignments}

    # Update CSVs with split column
    for dataset, rows in all_csv_data.items():
        csv_path = os.path.join(base_dir, "data", dataset, "metadata.csv")
        for row in rows:
            filename = row.get("filename", "")
            row["split"] = all_assignments.get(filename, "train")

        # Rewrite CSV with updated splits
        if rows:
            fieldnames = list(rows[0].keys())
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            logger.info(f"Updated splits in {csv_path}")

    # Write split files
    split_counts = defaultdict(int)
    split_files = {name: [] for name in ["train", "val", "test"]}

    for filename, split in sorted(all_assignments.items()):
        split_files[split].append(filename)
        split_counts[split] += 1

    for split_name, filenames in split_files.items():
        path = os.path.join(splits_dir, f"{split_name}.txt")
        with open(path, "w") as f:
            for fn in sorted(filenames):
                f.write(fn + "\n")

    logger.info(
        f"\nSplit summary: "
        f"train={split_counts['train']} "
        f"val={split_counts['val']} "
        f"test={split_counts['test']} "
        f"total={sum(split_counts.values())}"
    )
    logger.info(f"New assignments: {len(new_assignments)}")
    logger.info(f"Preserved: {len(existing_splits)}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate stratified train/val/test splits"
    )
    parser.add_argument("--base-dir", type=str, default=".")
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    generate_splits(args.base_dir, args.seed)


if __name__ == "__main__":
    main()

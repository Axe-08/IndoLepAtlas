"""
Phase 0: Data Audit Script for Indian Butterfly Classification
=============================================================
Run on DGX: python data_audit.py --data_root /data/butterflies

Outputs:
  - Printed summary statistics
  - Plots saved to {data_root}/audit/
  - Filtered metadata CSV saved for training
"""

import argparse
import os
import sys
from pathlib import Path
from collections import Counter

import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for DGX
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# ─── Biogeographic Zone Mapping ──────────────────────────────────────────────
STATE_TO_ZONE = {
    # Western Ghats
    "Kerala": "Western Ghats",
    "Goa": "Western Ghats",
    # Deccan Plateau
    "Maharashtra": "Deccan Plateau",
    "Dadar": "Deccan Plateau",
    "Karnataka": "Deccan Plateau",
    "Telangana": "Deccan Plateau",
    "Andhra Pradesh": "Deccan Plateau",
    "Tamil Nadu": "Deccan Plateau",
    "Chhattisgarh": "Deccan Plateau",
    "Bihar": "Deccan Plateau",
    "Madhya Pradesh": "Deccan Plateau",
    "Odisha": "Deccan Plateau",
    "Jharkhand": "Deccan Plateau",
    # Northeast India
    "Arunachal Pradesh": "Northeast India",
    "Assam": "Northeast India",
    "Meghalaya": "Northeast India",
    "Nagaland": "Northeast India",
    "Manipur": "Northeast India",
    "Mizoram": "Northeast India",
    "Tripura": "Northeast India",
    "Sikkim": "Northeast India",
    # Indo-Gangetic Plain
    "Uttar Pradesh": "Indo-Gangetic Plain",
    "Delhi": "Indo-Gangetic Plain",
    "Haryana": "Indo-Gangetic Plain",
    "Punjab": "Indo-Gangetic Plain",
    "Chandigarh": "Indo-Gangetic Plain",
    "West Bengal": "Indo-Gangetic Plain",
    # Western Himalaya
    "Uttarakhand": "Western Himalaya",
    "Himachal Pradesh": "Western Himalaya",
    "Jammu and Kashmir": "Western Himalaya",
    "Ladakh": "Western Himalaya",
    # Semi-Arid
    "Rajasthan": "Semi-Arid",
    "Gujarat": "Semi-Arid",
    # Coasts
    "Puducherry": "Coasts",
    "Daman": "Coasts",
    # Andaman & Nicobar
    "Andaman and Nicobar": "Andaman & Nicobar",
}

MONTH_NAMES = {
    1: 'January', 2: 'February', 3: 'March', 4: 'April',
    5: 'May', 6: 'June', 7: 'July', 8: 'August',
    9: 'September', 10: 'October', 11: 'November', 12: 'December'
}


def load_and_filter(data_root: str) -> pd.DataFrame:
    """Load metadata.csv, filter to butterflies, exclude early stage."""
    csv_path = os.path.join(data_root, 'metadata.csv')
    if not os.path.exists(csv_path):
        # Try common alternative locations
        for alt in ['images/metadata.csv', '../metadata.csv', 'butterfly_metadata.csv']:
            alt_path = os.path.join(data_root, alt)
            if os.path.exists(alt_path):
                csv_path = alt_path
                break
        else:
            print(f"ERROR: Cannot find metadata.csv in {data_root}")
            print("Listing directory contents:")
            for f in os.listdir(data_root):
                print(f"  {f}")
            sys.exit(1)

    print(f"Loading metadata from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"  Total rows: {len(df)}")
    print(f"  Columns: {list(df.columns)}")

    # Show first few rows to understand structure
    print(f"\n  First 3 rows:")
    print(df.head(3).to_string())
    print()

    # --- Filter to butterflies only ---
    # Try common column names for category/type
    type_col = None
    for col_name in ['type', 'category', 'organism_type', 'kind']:
        if col_name in df.columns:
            type_col = col_name
            break

    if type_col:
        print(f"  Unique types in '{type_col}': {df[type_col].unique()}")
        # Keep only butterflies
        butterfly_mask = df[type_col].str.lower().str.contains('butterfl', na=False)
        df = df[butterfly_mask].copy()
        print(f"  After filtering to butterflies: {len(df)} rows")
    else:
        print("  WARNING: No 'type' column found. Assuming all rows are butterflies.")

    # --- Exclude early stage ---
    # Check for life_stage column
    life_stage_col = None
    for col_name in ['life_stage', 'lifestage', 'stage', 'life stage']:
        if col_name in df.columns:
            life_stage_col = col_name
            break

    # Also check filenames for _earlystage_
    if 'filename' in df.columns or 'image' in df.columns or 'filepath' in df.columns:
        file_col = next(c for c in ['filename', 'image', 'filepath', 'file', 'path', 'image_path']
                        if c in df.columns)
        early_by_filename = df[file_col].str.contains('_[Ee]arly[Ss]tage_|_[Ee]arly[-_][Ss]tage', 
                                                        regex=True, na=False)
        n_early_fname = early_by_filename.sum()
        print(f"  Early stage images (by filename pattern): {n_early_fname}")
    else:
        file_col = None
        early_by_filename = pd.Series(False, index=df.index)
        n_early_fname = 0

    if life_stage_col:
        print(f"  Life stages in '{life_stage_col}': {df[life_stage_col].value_counts().to_dict()}")
        early_by_col = df[life_stage_col].str.lower().str.contains('early', na=False)
        early_mask = early_by_col | early_by_filename
    else:
        early_mask = early_by_filename

    n_excluded = early_mask.sum()
    df = df[~early_mask].copy()
    print(f"  Excluded {n_excluded} early stage images. Remaining: {len(df)} rows")

    return df


def find_species_column(df: pd.DataFrame) -> str:
    """Find the species column name."""
    for col in ['species', 'species_name', 'scientific_name', 'label', 'class', 'taxon']:
        if col in df.columns:
            return col
    # Fallback: ask user
    print(f"  Available columns: {list(df.columns)}")
    print("  ERROR: Cannot identify species column. Please specify.")
    sys.exit(1)


def audit_species_distribution(df: pd.DataFrame, species_col: str, out_dir: str):
    """Analyze and plot per-species image counts."""
    counts = df[species_col].value_counts().sort_values(ascending=False)
    n_species = len(counts)

    print(f"\n{'='*60}")
    print(f"SPECIES DISTRIBUTION")
    print(f"{'='*60}")
    print(f"  Total species: {n_species}")
    print(f"  Total images:  {len(df)}")
    print(f"  Mean images/species: {counts.mean():.1f}")
    print(f"  Median images/species: {counts.median():.1f}")
    print(f"  Min: {counts.min()} ({counts.idxmin()})")
    print(f"  Max: {counts.max()} ({counts.idxmax()})")

    # Sparse class analysis at different thresholds
    thresholds = [10, 20, 30, 50, 100]
    print(f"\n  Sparse Class Analysis:")
    print(f"  {'Threshold':<12} {'Species <':<12} {'Images in sparse':<18} {'% of total':<10}")
    print(f"  {'-'*52}")
    for t in thresholds:
        sparse = counts[counts < t]
        n_sp = len(sparse)
        n_img = sparse.sum()
        pct = 100 * n_img / len(df)
        print(f"  <{t:<11} {n_sp:<12} {n_img:<18} {pct:.1f}%")

    # Dense class analysis
    for t in thresholds:
        dense = counts[counts >= t]
        print(f"  >={t}: {len(dense)} species, {dense.sum()} images")

    # Plot 1: Bar chart of all species counts (sorted)
    fig, ax = plt.subplots(figsize=(20, 6))
    ax.bar(range(n_species), counts.values, color='steelblue', width=1.0)
    ax.set_xlabel('Species (sorted by count)', fontsize=12)
    ax.set_ylabel('Image Count', fontsize=12)
    ax.set_title(f'Per-Species Image Distribution ({n_species} species, {len(df)} images)', fontsize=14)
    ax.axhline(y=50, color='red', linestyle='--', alpha=0.7, label='Sparse threshold (50)')
    ax.axhline(y=30, color='orange', linestyle='--', alpha=0.7, label='Sparse threshold (30)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'species_distribution_bar.png'), dpi=150)
    plt.close()

    # Plot 2: Log-scale histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(counts.values, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
    ax.set_xlabel('Images per Species', fontsize=12)
    ax.set_ylabel('Number of Species', fontsize=12)
    ax.set_title('Distribution of Images per Species', fontsize=14)
    ax.axvline(x=50, color='red', linestyle='--', alpha=0.7, label='Sparse threshold (50)')
    ax.axvline(x=30, color='orange', linestyle='--', alpha=0.7, label='Sparse threshold (30)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'species_distribution_hist.png'), dpi=150)
    plt.close()

    # Top and bottom 10
    print(f"\n  Top 10 species:")
    for sp, cnt in counts.head(10).items():
        print(f"    {sp}: {cnt}")
    print(f"\n  Bottom 10 species:")
    for sp, cnt in counts.tail(10).items():
        print(f"    {sp}: {cnt}")

    return counts


def audit_family_distribution(df: pd.DataFrame, out_dir: str):
    """Analyze family-level distribution."""
    family_col = None
    for col in ['family', 'Family', 'family_name']:
        if col in df.columns:
            family_col = col
            break
    if not family_col:
        print("\n  WARNING: No 'family' column found. Skipping family analysis.")
        return

    counts = df[family_col].value_counts()
    print(f"\n{'='*60}")
    print(f"FAMILY DISTRIBUTION")
    print(f"{'='*60}")
    for fam, cnt in counts.items():
        print(f"  {fam}: {cnt} ({100*cnt/len(df):.1f}%)")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = sns.color_palette('Set2', len(counts))
    ax.barh(counts.index[::-1], counts.values[::-1], color=colors)
    ax.set_xlabel('Image Count', fontsize=12)
    ax.set_title('Images per Butterfly Family', fontsize=14)
    for i, (fam, cnt) in enumerate(zip(counts.index[::-1], counts.values[::-1])):
        ax.text(cnt + 50, i, str(cnt), va='center', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'family_distribution.png'), dpi=150)
    plt.close()


def audit_geographic(df: pd.DataFrame, out_dir: str):
    """Analyze geographic distribution and biogeographic zone mapping."""
    state_col = None
    for col in ['state', 'State', 'state_name']:
        if col in df.columns:
            state_col = col
            break
    if not state_col:
        print("\n  WARNING: No 'state' column found. Skipping geographic analysis.")
        return

    # Missing state analysis
    n_missing = df[state_col].isna().sum() + (df[state_col] == '').sum()
    print(f"\n{'='*60}")
    print(f"GEOGRAPHIC DISTRIBUTION")
    print(f"{'='*60}")
    print(f"  Missing state: {n_missing} ({100*n_missing/len(df):.1f}%)")

    # Map to biogeographic zones
    df['biogeographic_zone'] = df[state_col].map(STATE_TO_ZONE).fillna('Unknown')
    zone_counts = df['biogeographic_zone'].value_counts()
    print(f"\n  Biogeographic Zones:")
    for zone, cnt in zone_counts.items():
        print(f"    {zone}: {cnt} ({100*cnt/len(df):.1f}%)")

    # Check unmapped states
    unmapped = df[df['biogeographic_zone'] == 'Unknown'][state_col].dropna().unique()
    if len(unmapped) > 0:
        print(f"\n  WARNING: Unmapped states: {list(unmapped)}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = sns.color_palette('viridis', len(zone_counts))
    ax.barh(zone_counts.index[::-1], zone_counts.values[::-1], color=colors)
    ax.set_xlabel('Image Count', fontsize=12)
    ax.set_title('Images per Biogeographic Zone', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'zone_distribution.png'), dpi=150)
    plt.close()


def audit_temporal(df: pd.DataFrame, out_dir: str):
    """Analyze temporal/seasonal distribution."""
    date_col = None
    for col in ['date', 'Date', 'observation_date', 'obs_date']:
        if col in df.columns:
            date_col = col
            break

    month_col = None
    for col in ['month', 'Month']:
        if col in df.columns:
            month_col = col
            break

    if not date_col and not month_col:
        print("\n  WARNING: No date/month column found. Skipping temporal analysis.")
        return

    print(f"\n{'='*60}")
    print(f"TEMPORAL DISTRIBUTION")
    print(f"{'='*60}")

    if date_col:
        n_missing = df[date_col].isna().sum()
        print(f"  Missing date: {n_missing} ({100*n_missing/len(df):.1f}%)")
        # Extract month from date
        df['month_num'] = pd.to_datetime(df[date_col], errors='coerce').dt.month
    elif month_col:
        df['month_num'] = pd.to_numeric(df[month_col], errors='coerce')

    # Clean invalid months
    valid_months = df['month_num'].between(1, 12)
    n_invalid = (~valid_months & df['month_num'].notna()).sum()
    if n_invalid > 0:
        print(f"  Invalid months (e.g., month=77): {n_invalid} — will be treated as missing")
        df.loc[~valid_months, 'month_num'] = np.nan

    month_counts = df['month_num'].dropna().astype(int).value_counts().sort_index()
    print(f"\n  Monthly distribution:")
    for m, cnt in month_counts.items():
        name = MONTH_NAMES.get(m, f"Month {m}")
        print(f"    {name}: {cnt}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    months = list(range(1, 13))
    values = [month_counts.get(m, 0) for m in months]
    names = [MONTH_NAMES[m][:3] for m in months]
    ax.bar(names, values, color='coral', edgecolor='white')
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Image Count', fontsize=12)
    ax.set_title('Seasonal Distribution of Butterfly Images', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'temporal_distribution.png'), dpi=150)
    plt.close()


def audit_splits(df: pd.DataFrame, species_col: str, out_dir: str):
    """Validate train/val/test splits."""
    split_col = None
    for col in ['split', 'Split', 'subset', 'set']:
        if col in df.columns:
            split_col = col
            break
    if not split_col:
        print("\n  WARNING: No 'split' column found. Skipping split analysis.")
        return

    print(f"\n{'='*60}")
    print(f"SPLIT DISTRIBUTION")
    print(f"{'='*60}")
    split_counts = df[split_col].value_counts()
    for split, cnt in split_counts.items():
        print(f"  {split}: {cnt} ({100*cnt/len(df):.1f}%)")

    # Check all species are in all splits
    splits = df[split_col].unique()
    species_per_split = {}
    for s in splits:
        species_in_split = set(df[df[split_col] == s][species_col].unique())
        species_per_split[s] = species_in_split
        print(f"  Species in {s}: {len(species_in_split)}")

    # Find species missing from any split
    all_species = set(df[species_col].unique())
    for s in splits:
        missing = all_species - species_per_split[s]
        if missing:
            print(f"  WARNING: {len(missing)} species missing from {s} split: {list(missing)[:5]}...")


def audit_image_files(df: pd.DataFrame, data_root: str):
    """Verify image files actually exist on disk."""
    images_dir = os.path.join(data_root, 'images')
    if not os.path.isdir(images_dir):
        print(f"\n  WARNING: Images directory not found at {images_dir}")
        return

    # Count images in folder structure
    n_folders = 0
    n_files = 0
    for species_dir in sorted(os.listdir(images_dir)):
        sp_path = os.path.join(images_dir, species_dir)
        if os.path.isdir(sp_path):
            n_folders += 1
            n_files += len([f for f in os.listdir(sp_path)
                           if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))])

    print(f"\n{'='*60}")
    print(f"IMAGE FILE VERIFICATION")
    print(f"{'='*60}")
    print(f"  Species folders found: {n_folders}")
    print(f"  Image files found:     {n_files}")
    print(f"  Metadata rows:         {len(df)}")
    if n_files != len(df):
        print(f"  MISMATCH: {abs(n_files - len(df))} difference")


def save_filtered_csv(df: pd.DataFrame, data_root: str, species_col: str):
    """Save the filtered (adult-only, butterfly-only) metadata for training."""
    out_path = os.path.join(data_root, 'metadata_filtered.csv')
    df.to_csv(out_path, index=False)
    print(f"\n  Saved filtered metadata to: {out_path}")
    print(f"  Total rows: {len(df)}, Species: {df[species_col].nunique()}")


def main():
    parser = argparse.ArgumentParser(description='Butterfly Dataset Audit')
    parser.add_argument('--data_root', type=str, default='/data/butterflies',
                        help='Root directory of the butterfly dataset')
    args = parser.parse_args()

    # Create output directory
    out_dir = os.path.join(args.data_root, 'audit')
    os.makedirs(out_dir, exist_ok=True)

    print(f"{'#'*60}")
    print(f"  INDIAN BUTTERFLY DATASET AUDIT")
    print(f"  Data root: {args.data_root}")
    print(f"  Output: {out_dir}")
    print(f"{'#'*60}\n")

    # Load and filter
    df = load_and_filter(args.data_root)

    # Find species column
    species_col = find_species_column(df)
    print(f"  Using species column: '{species_col}'")

    # Run audits
    species_counts = audit_species_distribution(df, species_col, out_dir)
    audit_family_distribution(df, out_dir)
    audit_geographic(df, out_dir)
    audit_temporal(df, out_dir)
    audit_splits(df, species_col, out_dir)
    audit_image_files(df, args.data_root)

    # Save filtered CSV
    save_filtered_csv(df, args.data_root, species_col)

    # Summary recommendation
    n_sparse_50 = (species_counts < 50).sum()
    n_sparse_30 = (species_counts < 30).sum()
    print(f"\n{'='*60}")
    print(f"RECOMMENDATION")
    print(f"{'='*60}")
    if n_sparse_50 > 30:
        print(f"  {n_sparse_50} species with <50 images → DUAL HEAD recommended")
        print(f"  Use prototypical head for sparse classes, softmax+focal for dense")
    elif n_sparse_30 > 15:
        print(f"  {n_sparse_30} species with <30 images → Consider DUAL HEAD")
        print(f"  Moderate imbalance — focal loss alone may suffice, but prototypical head worth testing")
    else:
        print(f"  Only {n_sparse_30} species with <30 images → SINGLE HEAD with Focal Loss")
        print(f"  Class imbalance is manageable with focal loss + class-balanced sampling")

    print(f"\n  Plots saved to: {out_dir}")
    print(f"  Done!")


if __name__ == '__main__':
    main()

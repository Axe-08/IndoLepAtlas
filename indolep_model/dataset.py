"""
Butterfly Dataset Loader
========================
Loads butterfly images from folder-per-species structure with metadata CSV.
Supports optional geotemporal features and proper augmentation pipeline.

Dataset structure expected:
  /data/butterflies/
    images/
      Anthene_lycaenina/
        Anthene_lycaenina_Adult-Unknown_110.jpg
        ...
      Athyma_selenophora/
        ...
    metadata.csv (or metadata_filtered.csv after audit)
"""

import os
import re
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image


# ─── Biogeographic Zone Mapping ──────────────────────────────────────────────
STATE_TO_ZONE = {
    'Kerala': 0, 'Goa': 0,                                          # Western Ghats
    'Maharashtra': 1, 'Karnataka': 1, 'Telangana': 1, 'Dadar': 1,               # Deccan Plateau
    'Andhra Pradesh': 1, 'Tamil Nadu': 1, 'Chhattisgarh': 1,
    'Madhya Pradesh': 1, 'Odisha': 1, 'Jharkhand': 1, 'Bihar': 1,
    'Arunachal Pradesh': 2, 'Assam': 2, 'Meghalaya': 2,             # Northeast India
    'Nagaland': 2, 'Manipur': 2, 'Mizoram': 2, 'Tripura': 2, 'Sikkim': 2,
    'Uttar Pradesh': 3, 'Delhi': 3, 'Haryana': 3,                   # Indo-Gangetic Plain
    'Punjab': 3, 'Chandigarh': 3, 'West Bengal': 3,
    'Uttarakhand': 4, 'Himachal Pradesh': 4,                         # Western Himalaya
    'Jammu and Kashmir': 4, 'Ladakh': 4,
    'Rajasthan': 5, 'Gujarat': 5,                                    # Semi-Arid
    'Puducherry': 6, 'Daman': 6,                                     # Coasts
    'Andaman and Nicobar': 7,                                         # Andaman & Nicobar
}
NUM_ZONES = 9  # 0-7 + 8 for unknown/missing
ZONE_NAMES = [
    'Western Ghats', 'Deccan Plateau', 'Northeast India',
    'Indo-Gangetic Plain', 'Western Himalaya', 'Semi-Arid',
    'Coasts', 'Andaman & Nicobar', 'Unknown'
]


def get_train_transforms(img_size: int = 224) -> transforms.Compose:
    """Training augmentations following architecture doc recommendations.
    - Horizontal flip (butterflies are bilaterally symmetric)
    - NO vertical flip (butterflies are never upside down)
    - Random rotation ±30°
    - Color jitter (wing colors vary with lighting)
    - Random crop and resize
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0), ratio=(0.8, 1.2)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def get_val_transforms(img_size: int = 224) -> transforms.Compose:
    """Validation/test transforms — deterministic resize + center crop."""
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def encode_month_cyclic(month: float) -> np.ndarray:
    """Cyclic encoding of month: [sin(2πM/12), cos(2πM/12)].
    Correctly places December adjacent to January.
    Returns [0, 0] for missing/invalid months.
    """
    if np.isnan(month) or month < 1 or month > 12:
        return np.array([0.0, 0.0], dtype=np.float32)
    return np.array([
        np.sin(2 * np.pi * month / 12),
        np.cos(2 * np.pi * month / 12)
    ], dtype=np.float32)


class ButterflyDataset(Dataset):
    """Indian Butterfly Fine-Grained Classification Dataset.

    Args:
        data_root: Path to dataset root (e.g., /data/butterflies)
        split: One of 'train', 'val', 'test'
        img_size: Image size for transforms (default 224)
        use_geotemporal: Whether to return geotemporal features
        metadata_file: Name of metadata CSV file
    """

    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        img_size: int = 224,
        use_geotemporal: bool = False,
        metadata_file: str = 'metadata_filtered.csv',
    ):
        self.data_root = data_root
        self.images_dir = os.path.join(data_root, 'images')
        self.split = split
        self.use_geotemporal = use_geotemporal

        # Set transforms
        if split == 'train':
            self.transform = get_train_transforms(img_size)
        else:
            self.transform = get_val_transforms(img_size)

        # Load metadata
        csv_path = os.path.join(data_root, metadata_file)
        if not os.path.exists(csv_path):
            # Fallback to original metadata
            csv_path = os.path.join(data_root, 'metadata.csv')

        df = pd.read_csv(csv_path)

        # Auto-detect column names
        self.species_col = self._find_col(df, ['species', 'species_name', 'scientific_name', 'label'])
        self.split_col = self._find_col(df, ['split', 'Split', 'subset', 'set'], required=False)
        self.state_col = self._find_col(df, ['state', 'State'], required=False)
        self.date_col = self._find_col(df, ['date', 'Date'], required=False)
        self.file_col = self._find_col(df, ['filename', 'image', 'filepath', 'file', 'path', 'image_path'],
                                        required=False)

        # Filter by split
        if self.split_col and split:
            df = df[df[self.split_col] == split].copy()

        # Exclude early stage by filename
        if self.file_col:
            early_mask = df[self.file_col].str.contains(
                '_[Ee]arly[Ss]tage_|_[Ee]arly[-_][Ss]tage|Early.?Stage',
                regex=True, na=False
            )
            df = df[~early_mask].copy()

        # Filter out rows with missing species
        df = df[df[self.species_col].notna() & (df[self.species_col] != '')].copy()

        # Build label mapping
        all_species = sorted(df[self.species_col].unique())
        self.species_to_idx = {sp: i for i, sp in enumerate(all_species)}
        self.idx_to_species = {i: sp for sp, i in self.species_to_idx.items()}
        self.num_classes = len(all_species)

        # Build sample list
        self.samples = []
        for _, row in df.iterrows():
            species = row[self.species_col]
            # print(species)
            label = self.species_to_idx[species]

            # Resolve image path
            if self.file_col and pd.notna(row[self.file_col]):
                img_path = self._resolve_image_path(row[self.file_col], species)
            else:
                img_path = None  # Will need to be resolved differently

            # Geotemporal features
            zone_idx = NUM_ZONES - 1  # Unknown by default
            month = float('nan')
            if self.use_geotemporal:
                if self.state_col and pd.notna(row.get(self.state_col)):
                    state = str(row[self.state_col]).strip()
                    zone_idx = STATE_TO_ZONE.get(state, NUM_ZONES - 1)
                if self.date_col and pd.notna(row.get(self.date_col)):
                    try:
                        month = float(pd.to_datetime(row[self.date_col]).month)
                    except Exception:
                        month = float('nan')

            self.samples.append({
                'img_path': img_path,
                'label': label,
                'species': species,
                'zone_idx': zone_idx,
                'month': month,
            })

        # Filter out samples with missing image paths
        self.samples = [s for s in self.samples if s['img_path'] is not None]

        # Rebuild mappings based on available samples
        all_species = sorted(set(s['species'] for s in self.samples))
        self.species_to_idx = {sp: i for i, sp in enumerate(all_species)}
        self.idx_to_species = {i: sp for sp, i in self.species_to_idx.items()}
        self.num_classes = len(all_species)

        print(f"  [{split.upper()}] Loaded {len(self.samples)} images, "
              f"{self.num_classes} species")

    def _find_col(self, df, candidates, required=True):
        for col in candidates:
            if col in df.columns:
                return col
        if required:
            raise ValueError(f"Cannot find column from {candidates}. "
                             f"Available: {list(df.columns)}")
        return None

    def _resolve_image_path(self, filename: str, species: str) -> Optional[str]:
        """Resolve image path from filename or construct from species folder."""
        if filename is None:
            return None

        filename = filename.strip()
        if not filename:
            return None

        # Absolute paths should be accepted if valid
        if os.path.isabs(filename):
            return filename if os.path.exists(filename) else None

        normalized = os.path.normpath(filename)

        # Strip known prefixes like 'butterflies/' or 'butterfly/'
        if normalized.startswith('butterflies/'):
            normalized = normalized[len('butterflies/'):]
        elif normalized.startswith('butterfly/'):
            normalized = normalized[len('butterfly/'):]

        # Handle 'raw/' by converting to 'images/' for sample datasets
        if normalized.startswith('raw/'):
            normalized = 'images' + normalized[3:]

        # Try relative to the dataset root
        candidate = os.path.join(self.data_root, normalized)
        if os.path.exists(candidate):
            return candidate

        # Metadata may include an extra prefix like 'butterflies/images/...'
        parts = normalized.split(os.sep)
        if 'images' in parts:
            image_index = parts.index('images')
            candidate = os.path.join(self.images_dir, *parts[image_index + 1:])
            if os.path.exists(candidate):
                return candidate

        # Try direct image filename under the images root
        candidate = os.path.join(self.images_dir, os.path.basename(normalized))
        if os.path.exists(candidate):
            return candidate

        # Fall back to the species folder if it exists
        species_folder = species.replace(' ', '_')
        species_dir = os.path.join(self.images_dir, species_folder)
        if os.path.isdir(species_dir):
            candidate = os.path.join(species_dir, os.path.basename(normalized))
            if os.path.exists(candidate):
                return candidate

        return None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        # Load image
        img_path = sample.get('img_path')
        if img_path is None or not os.path.exists(img_path):
            print(f"  WARNING: Cannot load image path {img_path!r}")
            img = Image.new('RGB', (224, 224), (0, 0, 0))
        else:
            try:
                img = Image.open(img_path).convert('RGB')
            except Exception as e:
                print(f"  WARNING: Cannot load {img_path}: {e}")
                img = Image.new('RGB', (224, 224), (0, 0, 0))

        img = self.transform(img)

        result = {
            'image': img,
            'label': torch.tensor(sample['label'], dtype=torch.long),
        }

        if self.use_geotemporal:
            result['zone_idx'] = torch.tensor(sample['zone_idx'], dtype=torch.long)
            result['month_enc'] = torch.tensor(
                encode_month_cyclic(sample['month']), dtype=torch.float32
            )

        return result

    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse-frequency class weights for focal loss / sampling."""
        counts = np.zeros(self.num_classes)
        for sample in self.samples:
            counts[sample['label']] += 1
        # Inverse frequency, normalized
        weights = 1.0 / (counts + 1e-6)
        weights = weights / weights.sum() * self.num_classes
        return torch.tensor(weights, dtype=torch.float32)

    def get_class_counts(self) -> np.ndarray:
        """Get per-class sample counts."""
        counts = np.zeros(self.num_classes, dtype=int)
        for sample in self.samples:
            counts[sample['label']] += 1
        return counts

    def get_sampler(self) -> WeightedRandomSampler:
        """Create a weighted random sampler for class-balanced training."""
        counts = self.get_class_counts()
        sample_weights = np.array([
            1.0 / counts[s['label']] for s in self.samples
        ])
        sample_weights = sample_weights / sample_weights.sum()
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(self.samples),
            replacement=True
        )


def create_dataloaders(
    data_root: str,
    batch_size: int = 32,
    img_size: int = 224,
    num_workers: int = 4,
    use_geotemporal: bool = False,
    balanced_sampling: bool = True,
    metadata_file: str = 'metadata_filtered.csv',
) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    """Create train/val/test dataloaders.

    Returns:
        (train_loader, val_loader, test_loader, num_classes)
    """
    train_ds = ButterflyDataset(data_root, 'train', img_size, use_geotemporal, metadata_file)
    val_ds = ButterflyDataset(data_root, 'val', img_size, use_geotemporal, metadata_file)
    test_ds = ButterflyDataset(data_root, 'test', img_size, use_geotemporal, metadata_file)

    # Use same label mapping across splits
    # (species_to_idx built from each split independently — need to unify)
    all_species = set(
        list(train_ds.species_to_idx.keys()) + 
        list(val_ds.species_to_idx.keys()) +
        list(test_ds.species_to_idx.keys())
    )
    # Filter out NaN values and empty strings
    all_species = [sp for sp in all_species if pd.notna(sp) and str(sp).strip() != '' and str(sp) != 'nan']
    all_species.sort()
    unified_mapping = {sp: i for i, sp in enumerate(all_species)}
    num_classes = len(unified_mapping)

    for ds in [train_ds, val_ds, test_ds]:
        ds.species_to_idx = unified_mapping
        ds.idx_to_species = {i: sp for sp, i in unified_mapping.items()}
        ds.num_classes = num_classes
        # Re-map labels and filter out invalid samples
        valid_samples = []
        for sample in ds.samples:
            species = sample['species']
            if species in unified_mapping:
                sample['label'] = unified_mapping[species]
                valid_samples.append(sample)
        ds.samples = valid_samples

    # Sampler for balanced training
    sampler = train_ds.get_sampler() if balanced_sampling else None
    shuffle = not balanced_sampling  # Don't shuffle if using sampler

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler, shuffle=shuffle,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader, num_classes

# create_dataloaders('../sample_butterflies')

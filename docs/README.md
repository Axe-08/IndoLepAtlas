# IndoLepAtlas — Indian Lepidoptera & Host Plants Dataset

A large-scale computer vision dataset of Indian butterflies, moths, and their larval host plants. Sourced from [ifoundbutterflies.org](https://www.ifoundbutterflies.org/) with public CC-licensed photographs.

> **Inspired by:** [iNaturalist](https://www.inaturalist.org/) | **Domain:** Indian Wildlife & Biodiversity

## Dataset Overview

| | Butterflies | Host Plants | Total |
|---|---|---|---|
| **Species** | ~967 | ~127 | ~1,094 |
| **Images** | ~60,000+ | ~700+ | ~60,700+ |
| **Source** | ifoundbutterflies.org | ifoundbutterflies.org | — |

## Motivation

Existing CV datasets (ImageNet, iNaturalist) under-represent Indian biodiversity. India hosts **~1,500 butterfly species** — many endemic and poorly documented in digital datasets. This dataset enables:

- **Species classification** of Indian butterflies and host plants
- **Object detection** with auto-generated bounding boxes
- **Ecological analysis** of butterfly-plant host relationships
- **Diagnostic analysis** using pretrained models

## Directory Structure

```
IndoLepAtlas/
├── data/
│   ├── butterflies/
│   │   ├── raw/              # Original images (with overlays)
│   │   ├── images/           # Clean trimmed images
│   │   └── metadata.csv      # Enriched per-image metadata
│   └── plants/
│       ├── raw/
│       ├── images/
│       └── metadata.csv
├── annotations/
│   ├── butterflies/*.txt     # YOLO format
│   ├── plants/*.txt          # YOLO format
│   ├── annotations.json      # COCO format (combined)
│   └── classes.txt           # Species→ID mapping
├── splits/
│   ├── train.txt (70%)
│   ├── val.txt   (15%)
│   └── test.txt  (15%)
├── docs/
│   ├── class_definitions.json
│   ├── annotation_guide.md
│   ├── data_card.md
│   └── distribution_stats.md
├── README.md
└── LICENSE
```

## Metadata Fields

### Butterflies (`metadata.csv`)
`image_id, filename, species, common_name, family, subfamily, genus, order, life_stage, sex, media_code, location, state, date, credit, source_url, source, split`

### Plants (`metadata.csv`)
`image_id, filename, species, family, genus, image_type, media_code, location, state, date, credit, butterfly_hosts, source_url, source, split`

## Annotation Format

- **YOLO** (per image): `<class_id> <x_center> <y_center> <width> <height>` (normalized)
- **COCO** (`annotations.json`): standard format with `images`, `categories`, `annotations`
- **Classes**: Species-level (~1,094 classes). See `annotations/classes.txt`

## Pipeline Scripts

| Script | Purpose |
|--------|---------|
| `pull_hf_data.py` | Download datasets from HuggingFace |
| `process_images.py` | Trim overlay text bands |
| `enrich_metadata.py` | OCR overlay extraction → metadata CSVs |
| `generate_annotations.py` | Auto bounding boxes (Grounding DINO) |
| `generate_splits.py` | Stratified train/val/test splits |
| `generate_stats.py` | Distribution statistics |
| `crawler_logged.py` | Butterfly species scraper |
| `plant_crawler.py` | Host plant species scraper |

All pipeline scripts are **incremental** — they skip already-processed data and support adding new species non-destructively.

## Ethical Guidelines

- All images are from publicly available sources (ifoundbutterflies.org)
- Photographer credits are preserved in metadata via OCR extraction
- No human faces or personal information in the dataset
- Source URLs and attribution maintained for every image

## Citation

```
@dataset{indolepatlas2026,
  title={IndoLepAtlas: Indian Lepidoptera and Host Plants Dataset},
  year={2026},
  source={ifoundbutterflies.org},
  license={CC BY-NC-SA 4.0}
}
```

## License

This dataset is released under [CC BY-NC-SA 4.0](LICENSE).

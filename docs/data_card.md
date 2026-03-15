# IndoLepAtlas — Data Card

## Dataset Summary

IndoLepAtlas is a computer vision dataset of Indian butterflies, moths, and their larval host plants. It captures the diversity of Indian Lepidoptera across ~1,094 species with images sourced from ifoundbutterflies.org.

**Inspired by:** iNaturalist  
**Domain:** Indian Wildlife & Biodiversity (CSE3292 §3.6)

## Source

- **Website:** [ifoundbutterflies.org](https://www.ifoundbutterflies.org/)
- **License:** Images are publicly available; photographer credits preserved
- **Collection method:** Automated scraping with respectful rate limiting

## Dataset Composition

### Butterflies (~967 species)
- Adult/Unknown life stage photographs
- Early stage (caterpillar, pupa) photographs
- Per-image metadata: species, common name, sex, location, date, credit
- Taxonomic hierarchy: Order → Superfamily → Family → Subfamily → Tribe → Genus → Species

### Host Plants (~127 species)
- Hero images and gallery photographs
- Per-image metadata: species, family, location, date, credit
- Cross-referenced with butterfly species they host

## Geographic Coverage

Images span all major Indian states with concentration in:
- Kerala, Karnataka, Maharashtra (Western Ghats biodiversity hotspot)
- Northeast India (Assam, Arunachal Pradesh, Meghalaya)
- Andaman & Nicobar Islands

## Annotation Process

1. **Auto-trimming:** Image overlays (text labels) removed via fixed-percentage cropping
2. **Auto-detection:** Bounding boxes generated using Grounding DINO (zero-shot object detection)
3. **Classification:** Species labels from source website taxonomy
4. **Metadata enrichment:** OCR extraction of overlay text (location, date, credit, media code)

## Known Limitations

- Some species have very few images (long-tailed distribution)
- OCR may fail on low-quality or unusually formatted overlays → null fields
- Bounding boxes are auto-generated, not manually verified
- Geographic bias toward well-surveyed regions (Western Ghats, Northeast)

## Ethical Considerations

- No human subjects or personal information
- Source is a public biodiversity documentation platform
- Photographer attribution maintained via OCR extraction
- Dataset does not include sensitive or endangered species location details

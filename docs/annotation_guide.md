# Annotation Guide — IndoLepAtlas

## 1. Class Definitions

Each species is its own class. Classes are defined at the **species level** (~1,094 total):
- Butterflies: ~967 species (class IDs 0–966)
- Plants: ~127 species (class IDs 967–1093)

See `annotations/classes.txt` for the full mapping.

### Edge Cases

| Scenario | Rule |
|----------|------|
| Multiple subjects in one image | Annotate all visible subjects with separate bounding boxes |
| Subject partially visible | Annotate if >30% of the subject is visible |
| Very small subject | Annotate if clearly identifiable as the target species |
| Subject occluded by vegetation | Annotate the visible portion |
| Image contains both butterfly and plant | Each gets its own bbox with respective species class |

## 2. Annotation Format

### YOLO (per image `.txt`)
```
<class_id> <x_center> <y_center> <width> <height>
```
All coordinates are **normalized** (0.0 to 1.0) relative to image dimensions.

### COCO (`annotations.json`)
Standard COCO format with `bbox` in `[x, y, width, height]` pixel coordinates (top-left origin).

## 3. Annotation Protocol

### Automated (v1)
- **Grounding DINO** zero-shot detection with text prompts
- Butterfly prompts: `"butterfly . moth . caterpillar . pupa . chrysalis"`
- Plant prompts: `"plant . flower . leaf . tree . shrub"`
- Fallback: full image as bounding box if detection fails
- Species class assigned from directory structure (not model output)

### Quality Verification (recommended)
- Spot-check 100 random images per dataset
- Verify bbox covers the subject adequately
- Flag images where detection clearly failed
- Re-annotate flagged images manually if needed

## 4. Metadata Annotation

Per-image metadata is extracted automatically via OCR:
- Scientific name, common name, family
- Media code (cross-validated against existing records)
- Location, date, photographer credit
- Sex/life stage (butterflies only)

**Missing fields are stored as empty strings**, not dropped.

## 5. Tools Used

| Tool | Purpose |
|------|---------|
| Grounding DINO | Zero-shot object detection for bounding boxes |
| pytesseract | OCR for overlay text extraction |
| Pillow | Image cropping and processing |
| CVAT (optional) | Manual annotation refinement |

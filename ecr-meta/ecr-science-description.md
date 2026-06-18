# BioCLIP2 Species Classifier for Edge Biodiversity Monitoring

## Science

Automated species identification is critical for biodiversity monitoring,
ecological surveys, and conservation management.  Traditional approaches
require expert taxonomists to manually review images — a process that is
slow, expensive, and cannot scale to the millions of images collected by
distributed camera networks.  This plugin brings state-of-the-art
vision-language classification to the edge, enabling real-time species
identification at the point of data collection.

## About BioCLIP2

**BioCLIP2** is a contrastive vision-language model purpose-built for
biological image classification.  Developed by the
[Imageomics Institute](https://imageomics.org), it builds on the SigLIP2
architecture (~430 M parameters) and was trained on the **TreeOfLife-200M**
dataset — over 200 million biological images spanning 450,000+ species
across the full tree of life.

BioCLIP2 performs **zero-shot classification**: it does not need to be
retrained for new species.  The model compares a camera image against
pre-computed text embeddings for every taxon in the TreeOfLife taxonomy,
returning ranked predictions with confidence scores.  This makes it
immediately deployable at any field site without custom training data.

Key capabilities:
- **450,000+ species** recognized out of the box
- **Any taxonomic rank** — from Kingdom down to Species
- **Zero-shot** — no per-site training or fine-tuning needed
- **Fast inference** — ~0.5 seconds per frame on GPU

## How It Works

The plugin captures camera frames (from a Sage node camera, RTSP stream,
or HTTP snapshot URL such as a Reolink IP camera), classifies each frame
at the configured taxonomic rank, and publishes the top-k predictions
with confidence scores.

**Confidence thresholding** controls when results are published and images
are uploaded.  The `--min-confidence` flag (default: 0.1) sets the minimum
confidence for the top prediction before the plugin publishes data and
uploads the annotated image.  Below this threshold, the frame is silently
skipped — no data is published and no image is saved.  This prevents
flooding the data pipeline with low-confidence guesses on empty scenes
(e.g., a feeder with no bird present).

**Annotated image output**: when a prediction exceeds the confidence
threshold, the plugin overlays the top-5 predictions in orange text on
the source image before uploading.  Each line shows the rank, scientific
name, and confidence percentage (e.g., `#1: Archilochus colubris (97.1%)`).
This makes the uploaded images immediately interpretable without needing
to cross-reference the data records.

## Configuration Reference

| Flag               | Type   | Default                      | Description |
|--------------------|--------|------------------------------|-------------|
| `--stream`         | string | `bottom_camera`              | Camera name or RTSP URL |
| `--snapshot-url`   | string | _(none)_                     | HTTP URL returning a JPEG snapshot (e.g. Reolink CGI API). Overrides `--stream`. |
| `--image-dir`      | string | _(none)_                     | Directory of test images for local batch testing |
| `--rank`           | string | `Class`                      | Taxonomic rank: Kingdom, Phylum, Class, Order, Family, Genus, Species |
| `--model`          | string | `hf-hub:imageomics/bioclip-2`| BioCLIP model identifier |
| `--interval`       | int    | `60`                         | Seconds between captures (camera/snapshot-url mode) |
| `--min-confidence` | float  | `0.1`                        | Minimum confidence to publish predictions and upload image |
| `--top-k`          | int    | `5`                          | Number of top predictions to include in output |
| `--continuous`     | string | `Y`                          | `Y` = continuous loop, `N` = single-shot |

## Measurements Published

| Topic                                    | Type   | Description                        |
|------------------------------------------|--------|------------------------------------|
| `env.species.<rank>`                     | string | Top-1 predicted taxon name         |
| `env.species.<rank>.confidence`          | float  | Top-1 confidence score (0–1)       |
| `env.species.top5`                       | string | JSON array of top-5 predictions    |

Annotated JPEG images are uploaded when the top prediction exceeds
`--min-confidence`.  In test mode (`--image-dir`), all images are uploaded
with annotations (including "No confident species prediction" text for
frames below threshold) to facilitate review of every test image.

## Example Use Cases

- **Hummingbird monitoring** — `--rank Species --min-confidence 0.5 --interval 60`
  at a feeder station to identify visiting hummingbird species by their
  binomial name (e.g., *Archilochus colubris* — Ruby-throated Hummingbird).
- **Avian surveys** — `--rank Order --interval 60` at bird feeder stations
  to classify bird families visiting throughout the day.
- **Invasive species detection** — `--rank Species --min-confidence 0.3`
  to flag potential invasive species for rapid response.
- **Pollinator surveys** — `--rank Family` on cameras positioned near
  flowering plants to classify insect visitors.
- **Marine biodiversity** — deploy on underwater camera nodes to classify
  fish and invertebrate species at coral reef monitoring sites.
- **Trail camera biodiversity** — pair with a YOLO object detector:
  YOLO triggers on "bird" or "animal" detections, BioCLIP provides the
  species-level identification for the same frame.

## References

- BioCLIP2 model: https://huggingface.co/imageomics/bioclip-2
- Imageomics Institute: https://imageomics.org
- pybioclip library: https://github.com/Imageomics/pybioclip
- TreeOfLife-200M dataset: https://huggingface.co/datasets/imageomics/TreeOfLife-200M
- Sage Continuum: https://sagecontinuum.org

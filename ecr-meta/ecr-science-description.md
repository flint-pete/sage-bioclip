# BioCLIP 2.5 Species Classifier for Edge Biodiversity Monitoring

## Science

Automated species identification is critical for biodiversity monitoring,
ecological surveys, and conservation management.  Traditional approaches
require expert taxonomists to manually review images — a process that is
slow, expensive, and cannot scale to the millions of images collected by
distributed camera networks.  This plugin brings state-of-the-art
vision-language classification to the edge, enabling real-time species
identification at the point of data collection.

## About BioCLIP 2.5 Huge

**BioCLIP 2.5 Huge** is a contrastive vision-language model purpose-built
for biological image classification.  Developed by the
[Imageomics Institute](https://imageomics.org), it uses a **ViT-H/14**
backbone trained on the **TreeOfLife-200M** dataset — over 219 million
biological images spanning 450,000+ species across the full tree of life.

BioCLIP 2.5 Huge achieves **61.3% mean zero-shot species accuracy** across
10 benchmarks, a +5.7% improvement over BioCLIP 2 (ViT-L/14).  It shows
particularly strong gains on insects (+12.9%), medicinal leaves (+15.4%),
and rare species (+8.7%).

BioCLIP 2.5 performs **zero-shot classification**: it does not need to be
retrained for new species.  The model compares a camera image against
pre-computed text embeddings for every taxon in the TreeOfLife taxonomy,
returning ranked predictions with confidence scores.  This makes it
immediately deployable at any field site without custom training data.

Key capabilities:
- **450,000+ species** recognized out of the box
- **Any taxonomic rank** — from Kingdom down to Species
- **Zero-shot** — no per-site training or fine-tuning needed
- **ViT-H/14 backbone** — larger, more accurate than BioCLIP 2
- **~1-2 seconds per frame** on GPU (Thor / DGX Spark)
- BioCLIP 2 (ViT-L/14) still available via `--model hf-hub:imageomics/bioclip-2`

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
name, and confidence percentage (e.g., `#1: Archilochus colubris (100.0%)`).
This makes the uploaded images immediately interpretable without needing
to cross-reference the data records.

## Configuration Reference

| Flag               | Type   | Default                              | Description |
|--------------------|--------|--------------------------------------|-------------|
| `--stream`         | string | `bottom_camera`                      | Camera name or RTSP URL |
| `--snapshot-url`   | string | _(none)_                             | HTTP URL returning a JPEG snapshot (e.g. Reolink CGI API). Overrides `--stream`. |
| `--image-dir`      | string | _(none)_                             | Directory of test images for local batch testing |
| `--rank`           | string | `Class`                              | Taxonomic rank: Kingdom, Phylum, Class, Order, Family, Genus, Species |
| `--model`          | string | `hf-hub:imageomics/bioclip-2.5-vith14` | BioCLIP model identifier |
| `--interval`       | int    | `60`                                 | Seconds between captures (camera/snapshot-url mode) |
| `--min-confidence` | float  | `0.1`                                | Minimum confidence to publish predictions and upload image |
| `--top-k`          | int    | `5`                                  | Number of top predictions to include in output |
| `--continuous`     | string | `Y`                                  | `Y` = continuous loop, `N` = single-shot |

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

## Resource Requirements

BioCLIP 2.5 Huge (ViT-H/14) requires more memory than BioCLIP 2 (ViT-L/14)
due to the larger model and text embeddings (~3 GB .npy file):

| Model              | Memory Request | Memory Limit | Notes |
|--------------------|---------------|--------------|-------|
| BioCLIP 2 (ViT-L)  | 8 Gi          | 16 Gi        |       |
| BioCLIP 2.5 (ViT-H)| 16 Gi         | 32 Gi        | OOMs at 8/16 Gi |

Both fit comfortably in 128 GB unified memory (DGX Spark / Sage Thor).

## Example Use Cases

- **Hummingbird monitoring** — `--rank Species --min-confidence 0.5 --interval 60`
  at a feeder station to identify visiting hummingbird species by their
  binomial name (e.g., *Archilochus colubris* — Ruby-throated Hummingbird).
  Tested at 99.95% confidence on live feeder camera.
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

- BioCLIP 2.5 Huge model: https://huggingface.co/imageomics/bioclip-2.5-vith14
- BioCLIP 2 model: https://huggingface.co/imageomics/bioclip-2
- BioCLIP 2 paper: https://arxiv.org/abs/2505.23883
- Imageomics Institute: https://imageomics.org
- pybioclip library: https://github.com/Imageomics/pybioclip
- TreeOfLife-200M dataset: https://huggingface.co/datasets/imageomics/TreeOfLife-200M
- Sage Continuum: https://sagecontinuum.org

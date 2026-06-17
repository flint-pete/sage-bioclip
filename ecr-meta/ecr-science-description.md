# BioCLIP2 Species Classifier for Edge Biodiversity Monitoring

## Science

Automated species identification is critical for biodiversity monitoring,
ecological surveys, and conservation management.  Traditional approaches
require expert taxonomists to manually review images — a process that is
slow, expensive, and cannot scale to the millions of images collected by
distributed camera networks.  This plugin brings state-of-the-art
vision-language classification to the edge, enabling real-time species
identification at the point of data collection.

## Model

The plugin uses **BioCLIP2** (`imageomics/bioclip-2`), a ~430 M parameter
contrastive vision-language model built on SigLIP2 and trained on the
**TreeOfLife-200M** dataset — over 200 million biological images spanning
450,000+ species.  BioCLIP2 represents a significant advance over the
original BioCLIP, with improved zero-shot accuracy across the full
taxonomic hierarchy from Kingdom to Species.

Classification can be performed at any taxonomic rank:

| Rank     | Use Case                                    |
|----------|---------------------------------------------|
| Kingdom  | Broad categorisation (Animalia, Plantae)     |
| Phylum   | Major body plan grouping                     |
| Class    | Birds vs Mammals vs Insects (default)        |
| Order    | Fine-grained ecological grouping             |
| Family   | Taxonomic family identification              |
| Genus    | Genus-level field survey                     |
| Species  | Full binomial species identification         |

## Architecture

The plugin loads the BioCLIP2 model and its TreeOfLife text embeddings at
startup (~3 seconds on GPU, ~45 seconds on CPU).  Each cycle it captures a
camera frame, converts to PIL Image, runs the classifier at the configured
taxonomic rank, and publishes the top-k predictions with confidence scores.
The source image is uploaded for verification.

The `pybioclip` library's `TreeOfLifeClassifier` handles all model loading,
tokenisation, and taxonomic embedding management, providing a clean API that
maps directly to ecological use cases.

## Measurements Published

| Topic                                    | Type   | Description                        |
|------------------------------------------|--------|------------------------------------|
| `env.species.<rank>`                     | string | Top-1 predicted taxon name         |
| `env.species.<rank>.confidence`          | float  | Top-1 confidence score (0–1)       |
| `env.species.top5`                       | string | JSON array of top-5 predictions    |

Source JPEG images are uploaded each cycle for downstream verification.

## Example Use Cases

- **Avian monitoring** — `--rank Order --interval 60` at bird feeder stations
  to classify bird families visiting throughout the day.
- **Invasive species detection** — `--rank Species --min-confidence 0.3`
  to flag potential invasive species for rapid response.
- **Pollinator surveys** — `--rank Family` on cameras
  positioned near flowering plants to classify insect visitors.
- **Marine biodiversity** — deploy on underwater camera nodes to classify
  fish and invertebrate species at coral reef monitoring sites.

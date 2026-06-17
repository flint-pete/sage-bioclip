# BioCLIP2 Species Classifier — Plugin Overview

A comprehensive tutorial for understanding, configuring, and deploying the
BioCLIP2 Species Classifier plugin on the Sage Continuum edge computing
platform.

---

## Table of Contents

1. [What This Plugin Does](#what-this-plugin-does)
2. [File Layout](#file-layout)
3. [How It Works (Internals)](#how-it-works-internals)
4. [Configuration Reference](#configuration-reference)
5. [Understanding Taxonomic Ranks](#understanding-taxonomic-ranks)
6. [Local Testing (No Node Required)](#local-testing)
7. [Deployment on a Sage Node](#deployment-on-a-sage-node)
8. [Data Published](#data-published)
9. [Querying Your Data](#querying-your-data)
10. [Key Issues and Pitfalls](#key-issues-and-pitfalls)
11. [Example Scenarios](#example-scenarios)

---

## 1. What This Plugin Does

The BioCLIP2 Species Classifier captures frames from a camera, identifies
biological organisms in the image using the BioCLIP2 vision-language model,
and publishes the top-k taxonomic predictions with confidence scores to the
Sage data store.

Think of it as an automated field biologist: every minute the node looks at
what the camera sees and says "That's a Passeriformes (songbird) with 73%
confidence — could also be Falconiformes (raptor) at 12%."

**What is BioCLIP2?** It is a contrastive vision-language model specifically
trained for biological image recognition.  Unlike general-purpose image
classifiers (ResNet, CLIP), BioCLIP2 was trained on the TreeOfLife-200M
dataset — over 200 million biological images spanning 450,000+ species.
This gives it unmatched zero-shot accuracy for identifying organisms in
natural settings.

**Why at the edge?** Ecological camera networks generate terabytes of
images.  Streaming all of them to the cloud for classification is impractical.
Running BioCLIP2 on-node means only compact taxonomic records travel to the
data store — species name, confidence, and (optionally) the source image
for verification.

---

## 2. File Layout

```
bioclip-species-classifier/
├── app.py                          # Main application (~285 lines)
├── Dockerfile                      # Container build — downloads model at build time
├── requirements.txt                # Python dependencies
├── sage.yaml                       # ECR metadata
├── overview.md                     # This file
├── ecr-meta/                       # ECR submission metadata
│   ├── README                      # Instructions for ECR submission
│   ├── ecr-science-description.md  # Science narrative
│   ├── ecr-credits-license.txt     # Authors, funding, license
│   ├── ecr-project-keywords.txt    # Search keywords
│   └── ecr-project-url.txt         # Project URL
│   (ecr-icon.jpg)                  # 512x512 icon — create before submission
│   (ecr-science-image.jpg)         # 1920x1080 science image — create before submission
├── jobs/                           # Sample Sage job specifications
│   └── bioclip-species-job.yaml    # Deploy on W019 classifying at Order rank
└── tests/                          # Self-contained test suite
    ├── run-tests.sh                # Run all tests for this plugin
    ├── test_bioclip_local.py       # Local validation on your own images
    ├── test_harness.py             # Pywaggle test harness library
    └── test-images/                # Test images (committed, real photos)
```

### What each file does

| File | Purpose |
|------|---------|
| `app.py` | Model loading, camera capture, classification, publishing |
| `Dockerfile` | Single-stage build: installs deps, pre-downloads BioCLIP2 model + runs warmup |
| `requirements.txt` | pywaggle, pybioclip, torch, opencv-python-headless, pillow |
| `sage.yaml` | Plugin name, version, inputs (with descriptions), ontology |
| `ecr-meta/` | Portal display metadata: science description, keywords, credits |

---

## 3. How It Works (Internals)

### Startup Sequence

```
1. Parse command-line arguments
2. Create BioCLIP2Classifier instance
   ├── Instantiate TreeOfLifeClassifier(model_str="hf-hub:imageomics/bioclip-2")
   │   ├── Downloads model weights (~1.7 GB) if not cached
   │   ├── Loads SigLIP2 vision encoder + text encoder
   │   └── Pre-computes text embeddings for all taxa at the target rank
   └── Store the Rank enum for classification
3. Open camera stream via pywaggle Camera
4. Enter main loop
```

### Main Loop (one iteration)

```
┌──────────────────────────────────────────────────────────┐
│ 1. cam.snapshot()                                        │
│    └── Captures a single frame → ImageSample             │
│    └── Convert to PIL Image (RGB)                        │
│                                                          │
│ 2. classifier.classify(image, top_k=5)                   │
│    └── Calls TreeOfLifeClassifier.predict()              │
│    └── Returns list of dicts with all taxonomic ranks    │
│    └── Extracts target rank + score, sorts descending    │
│    └── Returns [{name: "Passeriformes", confidence: 0.73}│
│                  {name: "Falconiformes", confidence: 0.12}│
│                  ...]                                    │
│                                                          │
│ 3. Publish top-1 prediction                              │
│    └── plugin.publish("env.species.order", "Passeriforme"│
│    └── plugin.publish("env.species.order.confidence", 0.7│
│                                                          │
│ 4. Publish top-k as JSON                                 │
│    └── plugin.publish("env.species.top5", json_string)   │
│                                                          │
│ 5. Upload source image                                   │
│    └── Save frame to temp JPEG                           │
│    └── plugin.upload_file()                               │
│                                                          │
│ 6. Sleep(interval) and repeat (if continuous=Y)           │
└──────────────────────────────────────────────────────────┘
```

### Key Classes

**`BioCLIP2Classifier`** (lines 57–108)
- `__init__(rank, model_str)`: loads the TreeOfLifeClassifier, validates
  the rank against `RANK_MAP`
- `classify(image, top_k)`: runs classification, returns `[{name, confidence}]`
- The `RANK_MAP` dict maps string rank names → `bioclip.Rank` enum values

**`TreeOfLifeClassifier`** (from pybioclip library)
- Handles all the heavy lifting: model loading, tokenisation, text embedding
  pre-computation for ~450K species
- The `predict()` method returns raw dicts with keys for every taxonomic
  level (kingdom, phylum, class, order, family, genus, species, common_name,
  score)
- Our wrapper extracts just the target rank level

### The pybioclip Abstraction

We use `pybioclip` rather than raw `open_clip` because:
1. **TreeOfLifeClassifier** pre-computes text embeddings for all taxa —
   without it, you'd need to manually tokenise 450K+ species names
2. **Rank-based classification** is built-in — just pass `rank=Rank.ORDER`
3. **Model loading** handles BioCLIP2's SigLIP2 architecture automatically
4. Result format gives you the full taxonomic hierarchy, not just a class ID

---

## 4. Configuration Reference

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--stream` | string | `bottom_camera` | Camera source name or RTSP URL |
| `--image-dir` | string | (none) | Directory of test images — replaces `--stream` for local batch testing |
| `--rank` | string | `Class` | Taxonomic rank: Kingdom, Phylum, Class, Order, Family, Genus, or Species |
| `--model` | string | `hf-hub:imageomics/bioclip-2` | BioCLIP model string (default: BioCLIP2 from HuggingFace) |
| `--interval` | int | `60` | Seconds between captures (camera mode only, ignored with `--image-dir`) |
| `--min-confidence` | float | `0.1` | Minimum confidence to publish a prediction (0.0–1.0) |
| `--top-k` | int | `5` | Number of top predictions to publish per frame |
| `--continuous` | string | `Y` | `Y` = loop forever, `N` = process once and exit |

### Key Parameter Interactions

- **`--rank` and `--min-confidence`**: Higher-level ranks (Kingdom, Phylum)
  tend to produce higher confidence scores than lower-level ranks (Genus,
  Species) because there are fewer categories to distinguish.  Adjust
  `--min-confidence` accordingly:
  - Kingdom/Phylum: `0.3–0.5` is reasonable
  - Class/Order: `0.1–0.3` is reasonable
  - Species: `0.01–0.1` may be needed (thousands of candidates)

- **`--top-k` and storage**: Each top-k result is published as a JSON array
  in `env.species.top5`.  Higher k values produce larger payloads but
  provide richer data for downstream analysis.

---

## 5. Understanding Taxonomic Ranks

The Linnaean taxonomy used by BioCLIP2:

```
Kingdom → Phylum → Class → Order → Family → Genus → Species
  broadest                                        most specific

Example: American Robin
  Kingdom:  Animalia
  Phylum:   Chordata
  Class:    Aves          ← "it's a bird"
  Order:    Passeriformes ← "it's a songbird"
  Family:   Turdidae      ← "it's a thrush"
  Genus:    Turdus
  Species:  Turdus migratorius
```

**Choosing the right rank depends on your science question:**

| Question | Rank | Why |
|----------|------|-----|
| "Is it an animal, plant, or fungus?" | Kingdom | Broadest separation |
| "Is it a bird, mammal, or insect?" | Class | Common ecological grouping |
| "What kind of bird?" | Order/Family | Finer ecological distinction |
| "What species is it?" | Species | Most specific (requires clear images) |

**Accuracy vs specificity tradeoff**: BioCLIP2 is very accurate at Class
level (>90% on iNaturalist test sets) but accuracy drops at Species level
because of the huge number of candidates (450K+) and visual similarity
between closely related species.

---

## 6. Local Testing (No Node Required)

### Directory Structure

Test images live in the project's shared test directory, organized per plugin:

```
plugins/bioclip-species-classifier/
├── app.py                           ← The plugin
├── tests/
│   ├── test-images/
│   │   ├── test1.jpg                ← Hummingbird (included)
│   │   ├── my-bird.jpg              ← Add your own
│   │   └── my-insect.png            ← Any JPG/PNG/WEBP/BMP/TIFF
│   ├── test_bioclip_local.py        ← Local test runner (recommended)
```

### Quick Start — Local Test Runner

The `test_bioclip_local.py` script is the primary tool for evaluating
BioCLIP2 performance on your own images.  It runs the actual plugin
(app.py) against every image in `tests/test-images/`, validates
the pywaggle output, and prints a detailed report with per-image
predictions, confidence bars, and timing.

```bash
# 1. Activate the test environment
cd /path/to/Sage-agents/plugins/bioclip-species-classifier
source ../../tests/.venv/bin/activate

# 2. Add your test images (any number, any supported format)
cp  field-photo-1.jpg  tests/test-images/
cp  camera-trap.png    tests/test-images/

# 3. Run the test — default is Species rank
python tests/test_bioclip_local.py
```

**Expected output:**

```
======================================================================
  BioCLIP2 LOCAL TEST
======================================================================
  Test images:  tests/test-images
  Image count:  2
    - field-photo-1.jpg  (512 KB)
    - test1.jpg  (971 KB)
  Ranks:        Species
  ...

  Image 1: field-photo-1.jpg
    Rank:       Species
    Prediction: Danaus plexippus
    Confidence: 0.831422  [#################################-------]
    Top-5:
      #1: Danaus plexippus                            0.831422
      #2: Danaus chrysippus                           0.042100
      ...

  PASSED — all images classified successfully
```

### Testing Different Ranks

```bash
# Classify at Class rank (broad: bird, mammal, insect?)
python tests/test_bioclip_local.py --rank Class

# Classify at Order rank
python tests/test_bioclip_local.py --rank Order

# Sweep ALL 7 taxonomic ranks on every image (thorough but slower)
python tests/test_bioclip_local.py --all-ranks
```

The `--all-ranks` sweep is useful for understanding how confidence
degrades from Kingdom (very high) down to Species (lower, because there
are 450K+ candidates).

### Test Runner CLI Options

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--rank` | string | `Species` | Which taxonomic rank to classify at |
| `--all-ranks` | flag | off | Sweep all 7 ranks (Kingdom through Species) |
| `--min-confidence` | float | `0.01` | Minimum confidence to publish (lower = keep more) |
| `--top-k` | int | `5` | How many top predictions to show |
| `--verbose`, `-v` | flag | off | Show full plugin log output |

### Output Files

After a run, results are in `tests/output/bioclip-local/`:

```
tests/output/bioclip-local/
├── species/                    ← Per-rank subdirectory
│   ├── data.ndjson             ← Raw pywaggle measurements
│   └── uploads/                ← Copies of classified images
├── class/                      ← (if --all-ranks was used)
│   └── ...
└── report.json                 ← Machine-readable summary of all results
```

The `data.ndjson` file contains the exact same records the plugin would
publish on a real Sage node — you can inspect these to verify the data
format matches what downstream analysis expects.

### Running the Plugin Directly

You can also run app.py directly against the test image directory.
This is useful for debugging or testing custom flags:

```bash
# Redirect waggle output to a local directory
export PYWAGGLE_LOG_DIR=./my-test-output

# Run single-shot against the test images
python app.py \
    --image-dir tests/test-images \
    --rank Species \
    --continuous N \
    --top-k 5

# Inspect the raw output
cat my-test-output/data.ndjson
ls  my-test-output/uploads/
```

### What the Output Looks Like

Each image produces 3 measurements + 1 upload:

```json
{"name":"env.species.species","value":"Archilochus colubris","meta":{"camera":"test1.jpg","rank":"Species","model":"hf-hub:imageomics/bioclip-2"}}
{"name":"env.species.species.confidence","value":0.9797,"meta":{"camera":"test1.jpg","rank":"Species"}}
{"name":"env.species.top5","value":"[{\"name\":\"Archilochus colubris\",\"confidence\":0.9797},...]","meta":{"camera":"test1.jpg","rank":"Species"}}
{"name":"upload","value":"/path/to/uploads/...-tmp.jpg","meta":{"top_species":"Archilochus colubris","confidence":"0.9797"}}
```

### Running the Test Suite

```bash
cd /path/to/Sage-agents/plugins/bioclip-species-classifier
source ../../tests/.venv/bin/activate

# Real GPU inference against test images
python3 tests/test_bioclip_local.py

# Classify at a specific rank
python3 tests/test_bioclip_local.py --rank Order

# Sweep all 7 taxonomic ranks
python3 tests/test_bioclip_local.py --all-ranks

# Run all plugins (from project root)
cd ../..
bash tests/run-all-tests.sh
```

### Adding Good Test Images

For best results, include a diverse set of images:

1. **Known species** — photos where you already know the answer, so you
   can verify BioCLIP2's identification.  iNaturalist is a good source.
2. **Different taxa** — birds, insects, plants, mammals, fungi — to test
   breadth across the TreeOfLife.
3. **Varying quality** — clear close-ups AND distant/blurry shots, to
   understand where accuracy breaks down.
4. **Edge cases** — night/IR images, multiple organisms in frame,
   human-made objects that look biological.

Images should be standard photo formats (JPG preferred) and can be any
resolution — BioCLIP2 resizes to 224x224 internally.

---

## 7. Deployment on a Sage Node

### Docker Build

The Dockerfile uses a single-stage build with careful layer ordering:

```dockerfile
# Pre-download model weights AND run warmup inference (single layer)
# Placed BEFORE COPY app.py so code edits don't invalidate the
# expensive (~2 GB) model + embeddings download cache layer
RUN python -c "\
from bioclip.predict import TreeOfLifeClassifier; \
c = TreeOfLifeClassifier(model_str='hf-hub:imageomics/bioclip-2'); \
from bioclip import Rank; \
from PIL import Image; import numpy as np; \
dummy = Image.fromarray(np.zeros((224,224,3), dtype=np.uint8)); \
c.predict([dummy], rank=Rank.CLASS, k=1)"

COPY app.py .
```

**Why download at build time?** Edge nodes may have no internet access.
The model must be baked into the container image.  The warmup step
pre-compiles CUDA kernels, reducing first-inference latency from ~45s to ~3s.
The model download is placed *before* `COPY app.py` so that code-only
changes don't invalidate the expensive download layer.

Build locally, then publish via the Sage portal (ECR builds from
your GitHub repo — you do not `docker push` directly):

```bash
docker build --no-cache -t bioclip-species-classifier:0.2.0 .
```

Then register at https://portal.sagecontinuum.org → My Apps →
Create App → enter your GitHub repo URL.

See **DOCKER-BUILD.md** for the full workflow.

### Job YAML Example

A ready-to-use job file is included at `jobs/bioclip-species-job.yaml`.
Here is the structure:

```yaml
name: bioclip-bird-survey
plugins:
  - name: bioclip-species-classifier
    pluginSpec:
      image: registry.sagecontinuum.org/waggle/bioclip-species-classifier:0.2.0
      args:
        - "--stream"
        - "bottom_camera"
        - "--rank"
        - "Order"
        - "--interval"
        - "60"
        - "--min-confidence"
        - "0.15"
nodes:
  W019:
scienceRules:
  - "schedule(bioclip-species-classifier): cronjob('classify-species', '*/5 * * * *')"
successcriteria:
  - WallClock('7day')
```

---

## 8. Data Published

| Topic | Type | Example | Description |
|-------|------|---------|-------------|
| `env.species.<rank>` | string | `"Passeriformes"` | Top-1 taxon name at the configured rank |
| `env.species.<rank>.confidence` | float | `0.734` | Top-1 confidence (0–1) |
| `env.species.top5` | string | `"[{...}]"` | JSON array of top-k predictions |

Plus one uploaded JPEG: the source image for verification.

The `<rank>` in the topic name is lowercase: `env.species.order`,
`env.species.class`, `env.species.species`, etc.

---

## 9. Querying Your Data

```python
import sage_data_client

# Get all species classifications from last hour
df = sage_data_client.query(
    start="-1h",
    filter={"name": "env.species.*", "vsn": "W019"}
)

# Get just the top-1 predictions
df_top1 = df[df["name"].str.endswith(".order")]
print(df_top1[["timestamp", "value"]])
```

---

## 10. Key Issues and Pitfalls

### Model Download is Large

The BioCLIP2 model is ~1.7 GB.  The TreeOfLife text embeddings add
another ~200 MB.  First download takes several minutes.  Always bake into
the Docker image.

### Rank Changes Require Recomputation

Changing the `--rank` parameter triggers re-computation of text embeddings
for all taxa at that rank.  At Species rank, this covers 450,000+ species
and takes ~45 seconds.  At Class rank, it covers ~50 classes and takes
<1 second.  The embeddings are cached in memory for the plugin's lifetime.

### Image Quality Matters

BioCLIP2's accuracy depends heavily on image quality:
- **Clear, centered subjects** → best results
- **Distant/blurry subjects** → classification drops to higher ranks
- **Night/IR images** → very poor results (model trained on RGB daylight)
- **Multiple organisms** → classifies the most prominent one

### pybioclip Version

The plugin requires `pybioclip >= 2.1.0` for BioCLIP2 support.  Earlier
versions only support the original BioCLIP model.

### Meta Values Must Be Strings

Same as all pywaggle plugins: all `meta={}` values must be strings.
The app.py code uses `str()` wrapping throughout.

### HuggingFace Offline Mode

The Dockerfile sets `HF_HUB_OFFLINE=1` after downloading the model.
This prevents accidental internet access at runtime (which would fail on
air-gapped edge nodes and cause cryptic timeout errors).

### Memory Usage

BioCLIP2 uses ~1.5 GB GPU memory.  This is very lightweight compared to
the 128 GB available on Thor nodes, allowing easy co-deployment with YOLO
and even vLLM.

---

## 11. Example Scenarios

### Scenario A: Bird Order Classification at a Feeder

```bash
python3 app.py \
    --stream bottom_camera \
    --rank Order \
    --interval 60 \
    --min-confidence 0.15 \
    --top-k 5
```

Expected output: identifies visiting birds as Passeriformes (songbirds),
Columbiformes (doves), Piciformes (woodpeckers), etc.

### Scenario B: Invasive Species Alert

```bash
python3 app.py \
    --stream bottom_camera \
    --rank Species \
    --interval 30 \
    --min-confidence 0.05 \
    --top-k 10
```

With Species-level classification, downstream analysis can flag detections
matching a known invasive species list (e.g., Sturnus vulgaris — European
Starling in North America).

### Scenario C: Broad Ecological Survey

```bash
python3 app.py \
    --stream bottom_camera \
    --rank Class \
    --interval 120
```

Classifies at the broadest useful level: Aves (birds), Mammalia (mammals),
Insecta (insects), Magnoliopsida (flowering plants).  Good for baseline
biodiversity assessment.

---

## Appendix: Performance Benchmarks (DGX Spark / Sage Thor)

Measured on NVIDIA GB10 (Blackwell), 128 GB unified memory, aarch64:

| Metric | Value |
|--------|-------|
| Model load time | 3.11 s (from cache) |
| Text embedding warmup | 3.73 s (Class rank, ~50 categories) |
| Average inference | ~483 ms per image (Species rank) |
| Class-level inference | ~3.4 s per image |
| GPU memory | ~1.5 GB |

Note: Class-level classification is slower than Species because the text
embeddings are larger at higher taxonomic levels in the TreeOfLife hierarchy.

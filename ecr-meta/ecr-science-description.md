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

**Publishing vs. saving — two independent decisions.** This plugin separates
*what it reports* from *what images it saves*, because they have very different
costs. Publishing a measurement topic is cheap (a few bytes); saving an image to
Beehive is expensive (bandwidth + storage). They are controlled by two different
flags:

- **`--min-confidence`** (default 0.1) is the **reporting floor**: the minimum
  confidence for the top prediction to be *published* as a topic. Raise it to
  reduce noisy, low-confidence reports. It does **NOT** control image saving.
- **`--save-match`** controls **image saving** (upload), and is the **only** way
  an image gets saved. See "Saving Images" below.

Because BioCLIP performs zero-shot classification it has **no reject class** and
can score confidently even on empty frames (it always returns the closest taxon
in the taxonomy). For production deployments such as the H00F hummingcam job,
`--min-confidence` is raised to **0.7** so that only high-confidence species are
reported, suppressing spurious confident predictions on frames with no subject.

**Every cycle publishes something — including a heartbeat.** On every capture the
plugin publishes a `env.species.summary` heartbeat (and, when a prediction clears
`--min-confidence`, the per-rank topics). This means a user can always confirm
the plugin ran from the data plane, even during quiet periods with no confident
detection — distinguishing "running, nothing seen" from "job is dead."

## Saving Images: `--save-match`

By default (when `--save-match` is omitted) **no images are saved at all** — the
plugin only publishes measurement topics. To save annotated images, give
`--save-match` a list of rules describing *what you are looking for*.

A rule is a **name and a confidence**, written `Name:confidence`. Multiple rules
are separated by commas and combined with **OR**. An image is saved if **any**
detection in the frame matches **any** rule:

```
--save-match "Barn Owl:0.5,Northern Cardinal:0.7"
```

This saves the annotated image whenever a Barn Owl is detected at ≥0.5
confidence **or** a Northern Cardinal at ≥0.7. The `Name` is matched
**case-insensitively** and **exactly** against either the **common name** or the
**scientific name**. There is **no substring matching**: `Northern Cardinal`
matches *Cardinalis cardinalis* / "Northern Cardinal", but a bare `Cardinal`
matches nothing. Use the exact names the model emits (see the TreeOfLife
taxonomy / pybioclip).

To reproduce the simple "save anything above a threshold" behavior, use the
**wildcard** `*`:

```
--save-match "*:0.7"      # save the image for any detection at >= 0.7
```

> **IMPORTANT — names must match the rank you publish.** `--save-match` matches
> against whichever taxonomic rank `--rank` is set to. If you run `--rank Order`
> and write a species rule like `Tyto alba:0.5`, it will **never match** (the
> plugin is emitting order names, not species). Match Species names on
> `--rank Species` jobs, Order names on `--rank Order` jobs, and so on.

> **`--save-match` operates on published detections only.** A rule's confidence
> is only meaningful at or above `--min-confidence`. For example, with
> `--min-confidence 0.7` a rule `Barn Owl:0.5` effectively saves Barn Owls at
> ≥0.7, because a 0.55 Barn Owl was never published and is therefore invisible to
> the save logic. To save a species at a low confidence, lower `--min-confidence`
> accordingly.

**What gets saved:** the **annotated** image — the source frame with the top-5
predictions overlaid in orange text (rank, name, confidence), so the uploaded
images are immediately interpretable. In test mode (`--image-dir`), every image
is uploaded regardless of `--save-match` so all test results can be reviewed.

## Windowed GPU Sharing

On nodes with a single GPU, two always-on continuous plugins cannot
co-run without contending for the accelerator.  Sage Thor has **one GPU**
shared between this plugin and a YOLO object detector, so BioCLIP runs in
a **bounded time window** rather than continuously.

The `--max-runtime N` flag (v0.3.2) makes this possible.  When combined
with `--continuous Y`, the plugin loops every `--interval` seconds and
then **self-exits after N seconds**, behaving like one long bounded
single-shot.  Cron launches the plugin once per hour, and the
`--max-runtime` timer cleanly tears it down so the GPU is freed for the
next plugin's window.

On Thor the two plugins are scheduled on **offset hourly windows with
10-minute guard-bands** so they never contend for the GPU:

- **:00** — YOLO object detector runs its window
- **:20** — BioCLIP runs a **10-minute window** (cron starts it with
  `--max-runtime 600 --interval 15`, sampling every 15 s for ~40 frames,
  then exiting)

This uses roughly **20 minutes of GPU time per hour** total, with the
guard-bands ensuring the windows never overlap.

A key benefit of the windowed design is **model warmth**: the ~28 GB
BioCLIP 2.5 ViT-H/14 model loads **once** at the start of each window and
stays resident in GPU memory for all ~40 classifications, so the heavy
load cost is paid once rather than per frame.  A per-frame single-shot
invocation would instead reload the 28 GB model every time, which is
prohibitively expensive.

> **Subtle behavior — `--max-runtime` is WALL-CLOCK, not inference time.** The
> timer starts when the process starts, so the model load counts against the
> window. This matters most for BioCLIP: loading the ~28 GB ViT-H/14 model can
> take a minute or more, and that time is subtracted from your `--max-runtime`
> budget before the first classification happens. So `--max-runtime 600` does
> **not** give 10 minutes of sampling — it gives roughly `600s − model_load`
> of sampling. If you need a guaranteed amount of *inference* time, raise
> `--max-runtime` above your target (e.g. ~660–720 s for a 10-minute sampling
> goal) to absorb the cold start, and keep the guard-band wide enough that a
> slow load can't push the self-exit into the next plugin's window.

## Configuration Reference

| Flag               | Type   | Default                              | Description |
|--------------------|--------|--------------------------------------|-------------|
| `--stream`         | string | `bottom_camera`                      | Camera name or RTSP URL |
| `--snapshot-url`   | string | _(none)_                             | HTTP URL returning a JPEG snapshot (e.g. Reolink CGI API). Overrides `--stream`. |
| `--image-dir`      | string | _(none)_                             | Directory of test images for local batch testing |
| `--rank`           | string | `Class`                              | Taxonomic rank: Kingdom, Phylum, Class, Order, Family, Genus, Species |
| `--model`          | string | `hf-hub:imageomics/bioclip-2.5-vith14` | BioCLIP model identifier |
| `--interval`       | int    | `60`                                 | Seconds between captures (camera/snapshot-url mode) |
| `--min-confidence` | float  | `0.1`                                | **Reporting floor** — minimum confidence to PUBLISH a topic. Raise to reduce noisy reports. Does NOT control image saving. |
| `--save-match`     | string | _(empty)_                            | **Image saving** — OR-list of `Name:confidence` rules, e.g. `"Barn Owl:0.5,Northern Cardinal:0.7"`. Image saved if ANY detection matches ANY rule (exact, case-insensitive, common OR scientific name at the published `--rank`). `"*:0.7"` saves anything ≥0.7. Operates on published detections only. Omit = save nothing. The ONLY way images are saved. |
| `--top-k`          | int    | `5`                                  | Number of top predictions to include in output |
| `--continuous`     | string | `Y`                                  | `Y` = continuous loop, `N` = single-shot |
| `--max-runtime`    | int    | `0`                                  | Maximum seconds to run before self-exiting (`0` = run forever). With `--continuous Y`, loops every `--interval` seconds then exits after N seconds — a bounded window for GPU sharing. Added in v0.3.2. |

## Measurements Published

| Topic                                    | Type   | Description                        |
|------------------------------------------|--------|------------------------------------|
| `env.species.<rank>`                     | string | Top-1 predicted taxon name         |
| `env.species.<rank>.confidence`          | float  | Top-1 confidence score (0–1)       |
| `env.species.top5`                       | string | JSON array of top-5 predictions    |
| `env.species.summary`                    | string | JSON heartbeat published EVERY cycle (even with zero confident detections): `{published_count, top_confidence}`. Proves the cycle ran. |

### Performance Telemetry

Following the standard Sage convention (as used by `avian-diversity-monitoring`
and other production plugins on TAFT nodes), every cycle publishes nanosecond
timing for the three execution phases. These make cold-start cost and per-cycle
latency observable from the data plane — for the ~28 GB BioCLIP 2.5 model this is
especially useful, since `loadmodel` reveals exactly how much of a bounded GPU
window the cold start consumes vs. actual inference.

| Topic | Unit | Frequency | Description |
|-------|------|-----------|-------------|
| `plugin.duration.loadmodel` | ns | once | Construct + load the BioCLIP model |
| `plugin.duration.input`     | ns | per cycle | Snapshot/capture + decode + BGR→PIL |
| `plugin.duration.inference` | ns | per cycle | Run BioCLIP classification |

These publish every cycle regardless of confidence, so they also serve as a
liveness/heartbeat signal even when nothing clears `--min-confidence`.

Annotated JPEG images are uploaded only when a published detection matches a
`--save-match` rule (see "Saving Images" above).  In test mode (`--image-dir`),
all images are uploaded with annotations (including "No confident species
prediction" text for frames below threshold) to facilitate review of every test
image.

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
  Add `--save-match "Archilochus colubris:0.6,Selasphorus rufus:0.6"` to save
  annotated images only when those hummingbirds are seen. Tested at 99.95%
  confidence on live feeder camera.
- **Avian surveys** — `--rank Order --interval 60` at bird feeder stations
  to classify bird families visiting throughout the day. (With `--rank Order`,
  any `--save-match` rules must use Order names, not species.)
- **Invasive species detection** — `--rank Species --min-confidence 0.3
  --save-match "Lymantria dispar:0.3"` to publish all species but save images
  only when a target invasive (here, spongy moth *Lymantria dispar*) is seen,
  enabling rapid response without flooding storage.
- **Pollinator surveys** — `--rank Family` on cameras positioned near
  flowering plants to classify insect visitors.
- **Marine biodiversity** — deploy on underwater camera nodes to classify
  fish and invertebrate species at coral reef monitoring sites.
- **Trail camera biodiversity** — pair with a YOLO object detector:
  YOLO triggers on "bird" or "animal" detections, BioCLIP provides the
  species-level identification for the same frame.

## Testing

The plugin ships two complementary, self-contained test suites (no node,
network, or Beehive access required):

**1. Local classification tests** (`tests/test_bioclip_local.py`) — run the real
BioCLIP model against committed test images (`tests/test-images/`) through the
pywaggle test harness, printing per-image predictions, confidence bars, timing,
and validating the published topics + annotated images.

**2. Save-match unit tests** (`tests/test_save_match.py`) — 29 pure-Python unit
tests (no model load) covering the `--save-match` rule grammar and matching in
`save_match.py`: rule parsing, the `*` wildcard, case-insensitive matching
against **both** common and scientific names (at the published `--rank`), the
OR-of-rules semantics, the no-substring rule, malformed-rule fail-fast (bad
confidence / empty name → clear error, non-zero exit), and out-of-range
confidence rejection.

```bash
python3 tests/test_save_match.py    # => "29 passed, 0 failed (29 total)"
```

> **Note on `save_match.py`:** the matcher module and its test file are kept
> **byte-identical** across the sage-bioclip, birdnet, and sage-yolo repos (the
> three plugins do not share a Python package yet). When changing matcher
> behavior, update all three copies together so they cannot drift.

## References

- BioCLIP 2.5 Huge model: https://huggingface.co/imageomics/bioclip-2.5-vith14
- BioCLIP 2 model: https://huggingface.co/imageomics/bioclip-2
- BioCLIP 2 paper: https://arxiv.org/abs/2505.23883
- Imageomics Institute: https://imageomics.org
- pybioclip library: https://github.com/Imageomics/pybioclip
- TreeOfLife-200M dataset: https://huggingface.co/datasets/imageomics/TreeOfLife-200M
- Sage Continuum: https://sagecontinuum.org

# Detect-then-Classify: YOLO → BioCLIP Pipeline (Design Note)

**Status:** DRAFT for review — design only, no implementation yet.
**Author:** drafted for Pete Beckman, 2026-06-24.
**Scope of this note:** the two-stage **YOLO → bounding-box crop → BioCLIP** path.
A three-stage **YOLO → SAM → BioCLIP** variant is sketched in the "Future: adding
SAM" section but is explicitly out of scope for the first build.

> This document follows the project convention: a reviewable design note
> (decisions + open questions + a staged plan) is committed and reviewed
> **before** any code is written. Nothing here has been implemented.

---

## 1. Motivation — why detect-then-classify

BioCLIP is a **whole-image zero-shot classifier**. It embeds the *entire* frame
and compares it against text embeddings for every taxon in the TreeOfLife
taxonomy, returning the closest match. This has two consequences that the
current production deployment already runs into:

1. **No reject class.** BioCLIP always returns *some* taxon, even on an empty
   feeder frame. We currently paper over this by raising `--min-confidence` to
   0.7 (see the H00F hummingcam job), which suppresses confident-but-wrong calls
   on empty frames but also throws away real low-confidence birds.

2. **Background dominates small subjects.** A hummingbird occupying 2% of a
   1080p frame is a few hundred pixels inside a 2-million-pixel image. After
   BioCLIP resizes the whole frame to 224×224, the bird is a smudge and most of
   the embedding is describing leaves, sky, and the feeder. The model is being
   asked "what is this *image*?" when we want "what is this *animal*?".

The fix is the standard computer-vision pattern: **localize first, classify
second.** A detector (YOLO) finds *where* the animal is and returns a bounding
box; we crop to that box; BioCLIP then classifies a tight image that is almost
entirely the subject. This is how most production wildlife-ID pipelines work
(e.g. MegaDetector → species classifier in camera-trap ecology).

Expected wins:
- **Higher, more honest confidence** — the embedding describes the animal, not
  the scene. We can likely *lower* `--min-confidence` and still suppress
  empty-frame noise (because empty frames produce **no detection** and therefore
  **no crop to classify**).
- **Per-object identification** — two birds in one frame become two crops and
  two species calls, instead of one whole-image guess.
- **YOLO acts as the reject class BioCLIP lacks** — "no animal detected" →
  "nothing to classify," which is exactly the gate we want.

### Why this also fixes the watcher gap (cross-ref ToDo #12)

Today the Slack watcher keys on YOLO `env.count.bird`, which fires ~15×/day,
while BioCLIP whole-image confidently logs ~150 birds/day — a 10× mismatch that
the watcher misses. A fused detect-then-classify record (one topic carrying
*both* "YOLO found a bird here" *and* "BioCLIP says it's a Ruby-throated") gives
the watcher a single, high-quality trigger and removes the YOLO-only blind spot.
The pipeline and the watcher improvement are complementary; this note is about
producing that fused record.

---

## 2. Where this lives — new plugin vs. extend existing

**Decision (proposed): a new plugin, `sage-detect-classify` (working name),
that imports the detector and classifier logic — NOT a modification of the
existing `sage-yolo` or `sage-bioclip` plugins.**

Rationale:
- The existing single-stage plugins are **in production and verified** (jobs
  5667/5668/5670). They should keep working unchanged for users who want plain
  counts or plain whole-image classification.
- The pipeline is a genuinely different science product (per-object species ID),
  not a tweak to either existing product.
- It keeps each plugin's GPU/resource profile legible. The fused plugin loads
  *both* models and has a different memory/timing footprint.

Open alternative (see §9): instead of a third plugin, add a `--classify-crops`
mode to the YOLO plugin. Rejected for now because it would pull the ~28 GB
BioCLIP model into the YOLO image and double its load cost even for users who
only want counts. Documented as an option for the reviewer to override.

**Code reuse without a shared package (yet).** We already copy `save_match.py`
byte-identically across the three repos (ToDo #22). The new plugin would do the
same with the two model wrappers: lift `YOLODetector` from `sage-yolo/app.py` and
`BioCLIP2Classifier` from `sage-bioclip/app.py` as vendored modules. This is
consistent with the current "copy now, refactor into a shared package later"
decision. When ToDo #22's shared-package refactor happens, these wrappers join it.

---

## 3. Pipeline architecture

```
            ┌──────────┐   frame (BGR)   ┌──────────┐  detections[]  ┌───────────────┐
  camera ──▶│  CAPTURE │────────────────▶│  DETECT  │───────────────▶│  CROP + GATE  │
            │ (snapshot│                 │ (YOLO11x)│  {class,conf,   │ per-detection │
            │  /stream)│                 │          │   bbox xyxy}    │  filter+crop  │
            └──────────┘                 └──────────┘                └───────┬───────┘
                                                                             │ crop (PIL), N per frame
                                                                             ▼
                                          ┌──────────────┐   per-crop    ┌───────────────┐
                              publish ◀───│   PUBLISH +   │◀──────────────│   CLASSIFY    │
                              upload  ◀───│   ANNOTATE    │  {name,conf}  │  (BioCLIP)    │
                                          │  fused record │               │  per crop     │
                                          └──────────────┘               └───────────────┘
```

### Data flow, concretely

The two model wrappers already expose exactly the right interfaces — this is the
key feasibility finding from reading the code:

- `YOLODetector.detect(frame_bgr, target_classes) -> [{ "class", "confidence",
  "bbox": [x1,y1,x2,y2] }, ...]` (sage-yolo/app.py). Already returns integer
  pixel xyxy boxes.
- `BioCLIP2Classifier.classify(pil_image, top_k) -> [{ "name", "confidence",
  ... }, ...]` (sage-bioclip/app.py). Accepts **any** PIL image — so handing it
  a crop instead of the whole frame is a one-line change at the call site.

So the new glue is small and well-defined:

```
1. frame = capture()                       # BGR numpy, as today
2. dets  = yolo.detect(frame, classes)     # localize
3. for d in dets:                          # one crop per detection
       if d.confidence < detect_conf:  continue
       if class_filter and d.class not in class_filter: continue
       crop = crop_bbox(frame, d.bbox, pad=pad_frac)   # + optional padding
       crop_pil = to_pil(crop)             # BGR->RGB->PIL
       preds = bioclip.classify(crop_pil, top_k)        # classify the CROP
       result = fuse(d, preds)             # combine detector + classifier
       results.append(result)
4. publish(results); annotate(frame, results); maybe upload
```

`crop_bbox` is the one genuinely new helper. Everything else is wiring two
existing wrappers together.

### Crop helper details (this is where the subtlety is)

```python
def crop_bbox(frame_bgr, bbox, pad_frac=0.10, min_size=32):
    x1, y1, x2, y2 = bbox
    h, w = frame_bgr.shape[:2]
    bw, bh = x2 - x1, y2 - y1
    # Pad outward so BioCLIP sees a little context (helps it disambiguate),
    # but clamp to the frame so we never index out of bounds.
    px, py = int(bw * pad_frac), int(bh * pad_frac)
    x1 = max(0, x1 - px); y1 = max(0, y1 - py)
    x2 = min(w, x2 + px); y2 = min(h, y2 + py)
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.shape[0] < min_size or crop.shape[1] < min_size:
        return None   # too small to be worth classifying
    return crop
```

- **Padding** (`pad_frac`): a tight YOLO box can clip wingtips/tail. A small
  outward pad (~10%) gives BioCLIP enough of the animal to work with without
  re-introducing background. Tunable; needs empirical sweep (§8).
- **Min-size gate** (`min_size`): a 12×8-pixel crop upscaled to 224×224 is pure
  noise; classifying it wastes GPU and produces garbage. Skip it (and record
  *why* — see annotations).
- **Clamping**: padding must clamp to frame bounds or you get an empty / wrong
  slice. Cheap to get wrong, easy to unit-test.

---

## 4. Per-stage on/off toggles (a core requirement)

The pipeline must let each stage be turned on/off independently so it can be
operated, debugged, and compared against the single-stage baselines. Proposed
CLI surface (all default to the full pipeline ON):

| Flag | Default | Effect |
|------|---------|--------|
| `--detect / --no-detect` | on | Run YOLO. If **off**, skip detection and classify the **whole frame** (degenerate to current BioCLIP behavior — useful as an A/B baseline in one binary). |
| `--classify / --no-classify` | on | Run BioCLIP on crops. If **off**, behave like the plain YOLO counter (detector-only; no species). |
| `--detect-conf` | 0.25 | YOLO confidence floor for a box to become a crop. |
| `--detect-classes` | `bird` (job-set) | Only crop+classify these COCO classes (`""` = all). The natural reject filter: only animals get classified. |
| `--classify-min-confidence` | 0.0 (publish all) | BioCLIP reporting floor **per crop** (separate from save). |
| `--pad-frac` | 0.10 | Outward bbox padding before crop. |
| `--min-crop-size` | 32 | Skip crops smaller than this (px, shorter side). |
| `--max-crops-per-frame` | 8 | Cap classify calls per frame (GPU-budget guard). |
| `--save-match` | (job-set) | Same shared matcher as today — but now matched against the **fused** (per-object) results. |

**The two-toggle matrix gives four useful modes from one binary:**

| `--detect` | `--classify` | Behavior | Use |
|-----------|--------------|----------|-----|
| on | on | **Full pipeline** (the point of this plugin) | production |
| on | off | YOLO counter only | parity check vs sage-yolo |
| off | on | Whole-frame BioCLIP | parity check vs sage-bioclip / A-B against the pipeline |
| off | off | Capture + heartbeat only | liveness / camera test |

This matrix is also the **test plan**: each mode must reproduce the behavior of
the corresponding existing plugin (or a trivial subset), which makes the new
plugin verifiable against known-good baselines.

---

## 5. Annotations — what the fused record looks like

Two surfaces: the **published measurement topics** (machine-readable, into
Beehive) and the **annotated image** (human-readable, uploaded selectively).

### 5.1 Published topics (proposed)

The design choice is whether to (a) reuse the existing `env.species.*` and
`env.count.*` topics, or (b) introduce a new namespace for fused per-object
detections. **Proposed: a new `env.detection.object.*` namespace**, because the
semantics are genuinely new (a *located, classified* object) and reusing
whole-image `env.species.*` would silently collide with the single-stage
plugin's records on shared nodes (same lesson as the timing-units note, ToDo
#21: never overload an established topic with different semantics).

Per-frame:
- `env.count.<coco_class>` — kept, for back-compat with the watcher / counters.
- `env.detection.object.summary` — JSON heartbeat **every cycle** (even zero
  objects), so liveness is observable. Carries `{num_objects, classes, top}`.

Per detected-and-classified object (one set per crop):
- `env.detection.object.species` — BioCLIP top-1 name for that crop.
- `env.detection.object.species.confidence` — BioCLIP top-1 confidence.
- meta on each carries the **join keys** that tie detector ↔ classifier:

```json
{
  "detector": "yolo11x.pt",
  "detector_class": "bird",
  "detector_confidence": "0.82",
  "bbox": "[412,233,498,392]",
  "bbox_norm": "[0.32,0.22,0.39,0.36]",
  "classifier": "hf-hub:imageomics/bioclip-2.5-vith14",
  "rank": "Species",
  "common_name": "Ruby-throated Hummingbird",
  "scientific_name": "Archilochus colubris",
  "crop_index": "0",
  "crop_size": "86x159",
  "camera": "http-snapshot"
}
```

> All meta values are **strings** — pywaggle requires `meta` to be a dict of
> str→str. This bit us before in birdnet (the float-meta publish crash, ToDo
> #18); the new plugin must `str()` every meta value. Worth a unit test.

### 5.2 Annotated image

The uploaded JPEG draws, per object, a box labeled with **both** signals:

```
┌─────────────────────────┐
│  ┌───────┐              │   box color = detector class
│  │ bird  │ 0.82          │   line 1 (top): COCO class + detector conf
│  │ ▓▓▓▓▓ │              │   line 2 (bottom): BioCLIP name + conf
│  └───────┘              │
│   Ruby-throated 0.91     │
└─────────────────────────┘
```

This reuses the bioclip annotator's `cv2.getTextSize`-based line spacing (the
0.3.1 overlap fix) so multi-line labels don't collide. A "skipped: too small"
or "no confident species" tag is drawn for crops that gated out, so the image
explains its own decisions.

### 5.3 Provenance / data-quality hook (cross-ref ToDo #20)

Each record's `meta.plugin` already carries the exact image version. The fused
record additionally names *both* models in meta (`detector`, `classifier`), so
the archive is self-describing about which model pair produced each call — the
join key the provenance-markers design (#20) wants.

---

## 6. Pitfalls (the part most likely to bite)

1. **GPU memory: two large models resident at once.** YOLO11x (~4–5 GB at
   1080p) **plus** BioCLIP 2.5 ViT-H (~28 GB). Thor's 128 GB unified memory fits
   both, but: (a) the windowed-GPU-sharing math changes — this one plugin now
   holds the GPU for the whole window with both models loaded; (b) on smaller
   nodes (Xavier NX, 8–16 GB) this pair will **not** co-reside and the design
   must fail-fast with a clear message, not OOM mid-cycle. Decision: detect
   total VRAM at load and refuse to start if below a threshold.

2. **Latency stacks per object.** Cost is `load(YOLO)+load(BioCLIP)` once, then
   per frame `yolo_infer + N×bioclip_infer`. With many objects, N×(1–2 s BioCLIP)
   dominates. `--max-crops-per-frame` caps it; classify the **highest-confidence
   crops first** so the cap drops the least-interesting ones. Telemetry must
   split `plugin.duration.inference.detect` vs `...classify` (don't reuse the
   single `plugin.duration.inference` topic with different meaning — ToDo #21).

3. **YOLO's classes ≠ the animals we care about.** COCO has exactly one bird
   class ("bird"), no per-species, no "hummingbird," and **no "insect"** at all.
   So for the insect-bioclip use case YOLO is nearly useless as a gate (it can't
   localize a bee). Implication: the pipeline helps **birds/mammals** (COCO has
   bird/cat/dog/horse/etc.) but **not insects** — for insects we'd need a
   different detector or fall back to whole-frame (`--no-detect`). This is a real
   scope boundary, not a tuning knob. Document it loudly.

4. **YOLO misses the subject → BioCLIP never runs.** The pipeline's reject-class
   strength is also its failure mode: if YOLO doesn't see the fast/tiny
   hummingbird, there's no crop and no species call — the exact thing the
   whole-image path *did* catch. Mitigation: keep a `--fallback-whole-frame`
   option that classifies the whole frame when YOLO finds nothing, so we don't
   regress the 150/day → 15/day recall cliff. This needs an explicit decision
   (§9, open Q).

5. **Double counting / NMS interplay.** Overlapping boxes (a bird behind a
   feeder bar) can yield two crops of one animal → two species calls. Decide
   whether to dedup by IoU before classifying.

6. **Crop aspect ratio & upscaling.** BioCLIP resizes to a square; a very tall
   crop (perched bird) gets squished. Consider letterbox-pad to square before
   handing to BioCLIP rather than letting it distort. Empirical (§8).

7. **Coordinate-system bugs.** xyxy vs xywh, BGR vs RGB, padding clamp, and the
   normalized-vs-pixel bbox in meta are all classic off-by-one / channel-swap
   traps. Each gets a unit test on a synthetic frame.

8. **Cross-country object store (already learned).** Uploaded annotated images
   propagate with up to ~2-min lag (the watcher fix this session). Anything
   consuming the uploaded crop image downstream must tolerate that — not a
   plugin-side bug but a documented integration constraint.

---

## 7. Telemetry & heartbeat (carry the conventions forward)

Keep the established pattern, but split the inference phase so the two models are
separately observable:

- `plugin.duration.loadmodel` — once. (Could split into `.detect`/`.classify`
  sub-timers; decide in §9.)
- `plugin.duration.input` — per cycle (capture+decode).
- `plugin.duration.inference.detect` — per cycle (YOLO).
- `plugin.duration.inference.classify` — per cycle (sum of N crop classifies).
- `env.detection.object.summary` — per cycle heartbeat (proves liveness even on
  empty frames). Same role as the existing per-plugin summaries.

All integer nanoseconds, matching the ecosystem convention (ToDo #21).

---

## 8. Validation plan (before any production submit)

1. **Unit tests** (no model load), mirroring the save_match suite style:
   `crop_bbox` clamping/padding/min-size, BGR→PIL, bbox-norm math, meta
   stringification, the toggle matrix routing.
2. **Local image-dir A/B** using the committed `tests/test-images/`: run all
   four toggle modes; assert detect-only ≈ sage-yolo and no-detect ≈
   sage-bioclip (parity vs baselines).
3. **Crop-vs-whole-frame confidence study**: for the same test images, compare
   BioCLIP confidence on the whole frame vs. on the YOLO crop. This is the
   quantitative justification for the whole approach — expect the crop to win on
   small-subject frames. Sweep `--pad-frac` ∈ {0, 0.1, 0.25} and letterbox
   on/off here.
4. **Live single-shot on H00F** (pluginctl, not scheduled) before any SES job.
5. **Data-plane verification** (not "Running" status): confirm fused records
   with both join keys actually land in the data API — the project's standing
   "verify in the data plane" rule.

---

## 9. Open questions for the reviewer (Pete)

1. **New plugin vs. `--classify-crops` mode on sage-yolo?** Proposed: new plugin
   (§2). Override?
2. **Fallback to whole-frame when YOLO finds nothing?** (Pitfall #4.) This trades
   the clean reject-class for recall. On/off by default? Proposed:
   `--fallback-whole-frame` **off** by default (clean semantics), on for the
   hummingcam where recall matters.
3. **Topic namespace:** new `env.detection.object.*` (proposed) vs. reuse
   `env.species.*`? Naming bikeshed but it's a permanent archive decision.
4. **Insects:** accept that YOLO can't gate insects and run insect monitoring as
   whole-frame (`--no-detect`)? Or investigate an insect-capable detector later?
5. **Dedup overlapping boxes before classify?** (Pitfall #5.) IoU threshold?
6. **Split `loadmodel` timing** into per-model sub-topics, or one combined?
7. **Min VRAM floor** to refuse startup (Pitfall #1) — what value?

---

## 10. Staged implementation plan (after sign-off)

Mirrors the save-match staging style (small, reviewable, verifiable steps):

- **Stage 0 — sign-off.** Resolve §9 open questions. Lock topic names + CLI.
- **Stage 1 — crop/fuse helpers + unit tests.** `crop_bbox`, BGR→PIL, bbox-norm,
  fuse(), meta stringify. Pure-Python, no models. All green before wiring.
- **Stage 2 — new plugin skeleton.** Vendor `YOLODetector` + `BioCLIP2Classifier`
  wrappers; capture loop; the four-mode toggle matrix; heartbeat + split
  telemetry. Runs `--no-classify` and `--no-detect` first (cheap to verify vs
  baselines).
- **Stage 3 — full pipeline + annotation.** Wire crops→BioCLIP, fused records,
  dual-label annotated image, `--save-match` on fused results.
- **Stage 4 — local A/B + crop-vs-whole study** (§8.2, §8.3). Tune `--pad-frac`,
  letterbox, `--min-crop-size` from data.
- **Stage 5 — build/sideload/catalog/deploy on Thor** (the sideload procedure in
  birdnet/DEPLOY-AND-RUN.md), single-shot first, then a bounded-window SES job.
  Verify in the data plane.
- **Stage 6 — docs (ecr-science-description, overview, Testing section), and
  cross-link this note + the watcher (#12) so the watcher can trigger on the
  fused topic.**

Critical path: 0 → 1 → 2 → 3. Stages 4–6 follow. Stage 1 is independent enough
to start immediately on sign-off.

---

## 11. Future: adding SAM (the three-stage variant) — out of scope for v1

The full vision is **YOLO → SAM → BioCLIP**: YOLO localizes (box), SAM turns the
box prompt into a **pixel-accurate mask**, and BioCLIP classifies the
**masked-and-cropped** subject with background removed entirely (not just
cropped to a rectangle that still contains leaves/sky).

Why it could help beyond box-crop: even a tight box of a perched bird includes
background in the corners; a mask zeroes that out so the embedding is *purely*
the animal. Camera-trap literature shows background removal further lifts
fine-grained ID.

Why it's deferred:
- **Another large model + checkpoint.** SAM ViT-H is ~2.4 GB on top of YOLO +
  BioCLIP; ViT-B (~375 MB) is the realistic edge choice. Three models resident
  stresses even Thor's window and is a non-starter on small nodes.
- **SAM box→mask is cheap per prompt but the image encoder is not.**
  `predictor.set_image()` runs the ViT encoder once per frame (the expensive
  part); each box prompt is then cheap. So SAM cost is ~one extra ViT encode per
  frame, not per object — important for the budget, and a point in its favor.
- **Marginal benefit unproven for our subjects.** We should first *measure*
  whether plain box-crop already gets BioCLIP most of the way (Stage 4 study).
  If box-crop alone lifts confidence enough, SAM may not earn its cost.

Clean integration hook: the pipeline's `crop_bbox` stage becomes a pluggable
"region extractor" with two implementations — `box` (v1) and `box+mask` (SAM,
later) — selected by a `--region-method {box,sam}` flag. The rest of the
pipeline (classify, fuse, annotate, publish) is identical. So v1 should be
written so SAM can slot in as a third stage **without restructuring** — the
`--region-method` seam is the one piece of forward-design we bake in now.

The natural annotation extension: SAM masks publish as a polygon/RLE in meta
(`env.detection.object.mask`), and the annotated image shades the mask instead of
(or in addition to) the box.

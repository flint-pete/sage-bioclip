# Design Note: Decoupling "publish" from "save" — `--save-match` + periodic sampling

**Status:** DRAFT for review (2026-06-23). Open questions flagged inline.
**Scope:** Cross-plugin — applies to `bioclip-species-classifier`, `birdnet-species`,
`yolo-object-counter`, and any future inference plugin. This note lives in the
sage-bioclip repo for convenience but is the shared spec; mirror the final
agreed behavior into each plugin's ECR science description.

---

## 1. Problem

Today a single `--min-confidence` threshold does double duty: it controls both
**what topics get published** and **whether the input media (image/audio clip)
gets uploaded to Beehive**. These are two different concerns with two different
cost profiles:

- **Publishing a topic** is cheap (a few bytes of metadata per detection).
- **Saving media** is expensive (tens to hundreds of KB per artifact) and is the
  real constraint on edge nodes: upload bandwidth and Beehive storage.

A blunt confidence threshold is a poor proxy for "is this worth saving." A
student studying Barn Owls does not want the bucket filled with high-confidence
Robin images. Conversely, lowering the threshold to catch a quiet species floods
storage with everything else.

**Goal:** Separate the two concerns.
- **Publish** topics for *every* detection above a reporting floor (the
  scientific record stays complete).
- **Save** input media *selectively*, driven by an explicit, student-specified
  intent — not a single global number.

---

## 2. The design

### 2.1 `--save-match` — species-aware save list (feature #1)

A single delimited string defining an **OR-list** of `name:confidence` rules.
A clip/frame is uploaded if **any** detection in that execution matches **any**
rule (right name AND confidence ≥ that rule's threshold).

**Format** (single string, easy to read back from logs):

```
--save-match "Barn Owl:0.5,Cardinal:0.7"
```

- Delimiter between rules: `,`
- Delimiter between name and confidence: `:`
- Rationale for single delimited string: the exact save logic for a job is
  visible verbatim in one place in the logs / job spec — no reconstruction from
  repeated flags.

**Wildcard** reproduces the old simple-threshold behavior:

```
--save-match "*:0.7"      # save any detection at or above 0.7
```

**Matching rule (DECIDED):**
- **Case-insensitive EXACT match** on the detection's **common name OR scientific
  name**, at the rank the plugin is currently publishing.
- `*` is the only wildcard: matches any name (still subject to the confidence in
  that rule).
- No substring matching. "Cardinal" matches the taxon whose common name is
  exactly "Cardinal" at the published rank — it does NOT match "Northern
  Cardinal". (If a student wants Northern Cardinal, they write the exact name the
  model emits. This is documented loudly; substring was rejected as a footgun.)

**Rank awareness (DECIDED — must be documented prominently):**
- The name in each rule is matched against **whatever taxonomic rank the plugin
  is configured to publish** (e.g. bioclip `--rank Species` → match species
  names; `--rank Order` → match order names like "Lepidoptera").
- A species-name rule on an order-rank job will simply never match → no saves.
  The docs must make this explicit so students don't silently get nothing.

**Multiple detections per execution (DECIDED):**
- Many models return several detections per input (e.g. one 30s BirdNET clip
  yielded 4 detections; YOLO returns many boxes per frame). The save decision is
  evaluated across **all** detections in the execution: **ANY match triggers a
  save of the whole clip/frame, once.** Document this clearly — it is an OR over
  both the rule list and the detection list.
- Exactly one artifact is uploaded per execution (not one per detection).

### 2.2 Periodic reference sampler (feature #2) — SEPARATE PLUGIN (DECIDED)

Rather than embedding "save one every hour" logic inside each inference plugin
(which complicates the continuous-vs-windowed lifecycle — a windowed GPU pod that
only lives 5 minutes can't meaningfully do "once an hour"), the periodic
reference capture becomes its **own dedicated plugin**:

- Fires on **wall-clock schedule** (e.g. hourly) via the SES science rule.
- Does **no inference** — just grabs an image and/or audio sample and uploads it.
- **Requires no GPU** — schedules freely without contending for the single Thor
  GPU used by the AI pipeline.
- Cleanly identified as a **"sampler"** in the data stream, so adjusting the
  sampling period is a one-line schedule change that never disturbs the
  scientific AI pipeline.
- Uploads carry meta marking them as reference samples (e.g.
  `meta={"trigger": "periodic-sampler"}`), distinct from match-triggered saves.

This keeps each concern in its own clearly-named, independently-tunable unit.
(Spec for the sampler plugin itself is a follow-on; this note records the
architectural decision to split it out.)

---

## 3. Invariants & code-path requirements

1. **Publish always; save selectively.** Every execution publishes its topics
   (per-detection topics + the always-on summary/heartbeat) regardless of
   `--save-match`. Only the **upload** is gated by `--save-match`. The two code
   paths must be unmistakably separate and independently auditable.

2. **Every execution produces datapoints even with zero detections.** The
   summary/heartbeat topic publishes on every run (total_detections may be 0).
   This is the liveness signal and must NOT be gated behind "had a detection."
   (NOTE: a latent bug exists today — birdnet's summary publish is nested inside
   `if detections:`, so quiet cycles publish nothing. This must be fixed as part
   of this work: hoist the summary publish out of the detection branch.)

3. **`--save-match` is the ONLY path that saves input media.** No other code
   path uploads the raw/annotated image or audio clip. Document this explicitly
   in ECR.

4. **`--min-confidence` = reporting floor only.** It governs the minimum
   confidence for a topic to be *published* at all. Its ECR "discussion" text
   explains: raising `--min-confidence` reduces noisy topic reports; it does NOT
   affect what is saved (that's `--save-match`).

---

## 4. Parameter semantics summary

| Parameter | Governs | Effect |
|-----------|---------|--------|
| `--min-confidence` | **Publish** | Floor for emitting a topic. Raise it to cut noisy reports. Does not save media. |
| `--save-match`     | **Save**    | OR-list of `name:confidence` rules. Any match → upload the clip/frame once. `*:c` = save anything ≥ c. Only path that saves input. |

---

## 5. Resolved decisions (formerly open — settled 2026-06-23)

### 5.1 Per-species save threshold vs. the global publish floor — DECIDED: Model (A)
`--save-match` **operates only on PUBLISHED detections** — i.e. detections that
already cleared `--min-confidence`. There is exactly **one floor**.

Consequence (must be documented loudly): a save rule's threshold is only
meaningful *at or above* the publish floor. `--save-match "Barn Owl:0.5"` with
`--min-confidence 0.7` means Barn Owls are effectively saved at ≥0.7, because a
0.55 Barn Owl was never published and therefore is invisible to the save logic.
**To save a species at a low confidence, lower `--min-confidence` accordingly.**
This keeps a clean, single-floor mental model: publish first, then the save-list
selects among what was published.

### 5.2 Other subtleties — DECIDED
- **Annotated vs raw media:** `--save-match` saves the **annotated** image
  (bioclip/yolo: the version with overlaid predictions/boxes). The separate
  periodic **sampler** plugin saves the **raw** sample. (Audio plugins have no
  annotation; they save the captured clip.)
- **YOLO name space:** YOLO has no scientific name. For YOLO, `--save-match`
  matches against the **COCO class name** (e.g. `bird`, `person`, `car`).
  Documented per-plugin: bioclip/birdnet → common OR scientific name at the
  published rank; YOLO → COCO class name.
- **Default when `--save-match` is omitted:** **save nothing.** No image/audio is
  uploaded at all unless `--save-match` is explicitly provided. (Pure
  publish-only is the default; media saving is strictly opt-in.)

### 5.3 Heartbeat invariant — applies to ALL plugins
Every plugin must emit at least the summary/heartbeat datapoint(s) on **every
run**, even with zero detections, so a student can see the plugin ran. This is a
hard requirement across bioclip, birdnet, yolo. (Today birdnet's summary publish
is nested inside `if detections:` — that is a bug to fix as part of this work;
hoist the summary publish out of the detection branch.)

---

## 6. Decisions locked (2026-06-23)
- Matching: case-insensitive EXACT on common OR scientific name + `*` wildcard.
- Match is against the rank the plugin publishes; documented prominently.
- For YOLO, match is against the COCO class name.
- Arg shape: single delimited string, `,` between rules, `:` name/confidence.
- ANY match (over rules × detections) saves the whole clip/frame once.
- `--save-match` saves the ANNOTATED image; the sampler saves RAW.
- Omitting `--save-match` = save nothing (opt-in media saving).
- `--save-match` operates ONLY on published detections (Model A, single floor).
- Periodic sampling = separate, no-GPU, wall-clock "sampler" plugin (saves raw).
- Publish-always / save-selectively as strictly separate code paths.
- Every run emits heartbeat datapoint(s) even with zero detections (all plugins).
- `--save-match` is the sole media-save path; `--min-confidence` = publish floor.
- Backward-compat not a constraint (plugins still in development) — but docs must
  be crystal clear.

---

## 7. Staged implementation plan

Each stage is independently reviewable and (where possible) independently
shippable. Per Pete's conventions: code + version bump + ECR/README + CHANGELOG
in ONE commit per repo; real hardware tests; one plugin per model; verify in the
data plane before declaring done.

### Stage 0 — Shared spec sign-off (no code)
- This design note reviewed and approved.
- Decide the **exact `--save-match` parse grammar** edge cases:
  - whitespace trimming around names and numbers (`"Barn Owl : 0.5"` →
    `("barn owl", 0.5)`);
  - confidence parse/validate (float in [0,1]; reject otherwise with a clear
    error at startup, fail fast — do NOT silently ignore a bad rule);
  - empty/malformed rule handling (fail fast at startup, log the offending rule).
- Decide the heartbeat/summary topic name per plugin (bioclip currently has no
  always-on summary; birdnet has `env.detection.audio.summary`; yolo publishes
  `env.count.total` every cycle which already serves as heartbeat). Document the
  canonical heartbeat per plugin.

### Stage 1 — Shared `save-match` helper + unit tests (foundation)
- Implement a small, dependency-free matcher used identically by all three
  plugins (copy the module into each repo, or a tiny shared snippet — these repos
  don't share a package). Function contract:
  ```
  parse_save_match(spec: str) -> list[Rule]        # Rule = (name_lower, conf) ; "*" allowed
  should_save(rules, detections, name_keys) -> bool
      # detections already filtered to >= min_confidence (published set)
      # name_keys: which fields to match (common+scientific for bioclip/birdnet;
      #            coco class for yolo) — caller supplies the extractor
  ```
- **Pure-Python unit tests** (run locally, no GPU/node): wildcard, exact
  case-insensitive match on common, on scientific, multi-rule OR, multi-detection
  OR, no-match, malformed spec → startup error, empty spec → save nothing.
- Deliverable: helper + green tests. No plugin behavior change yet.

### Stage 2 — bioclip integration (the lead plugin)
- Wire `--save-match` into bioclip `app.py`:
  - Add `--save-match` arg (string, default empty = save nothing).
  - Separate the code paths explicitly:
    1. Publish topics for every published detection (unchanged).
    2. Compute `should_save(...)` over the published detections.
    3. Upload the ANNOTATED image **only if** `should_save` is true.
  - Match against common OR scientific at the configured `--rank`.
- Ensure a heartbeat datapoint publishes every run even with zero detections
  (add bioclip summary if missing — see Stage 0).
- Docs IN THE SAME COMMIT: ecr-science-description (new `--save-match` section +
  rank-awareness warning + "save = annotated, only path that saves, operates on
  published detections only"), CHANGELOG, version bump (0.3.3 → 0.4.0; new
  user-facing feature = minor bump).
- Real test on Thor + verify in data plane: topics still publish for all
  detections; image uploads ONLY when a rule matches; quiet run still emits
  heartbeat; a `*:c` run reproduces "save anything ≥ c".
- Build / sideload / catalog / resubmit / verify (per the established pipeline).

### Stage 3 — birdnet integration
- Same wiring, plus **fix the heartbeat bug**: hoist `env.detection.audio.summary`
  publish out of `if detections:` so it fires on quiet cycles.
- Match against common OR scientific. Save = the captured audio clip (no
  annotation). ANY match over the clip's detections → save the clip once.
- Docs + version bump (0.1.6 → 0.2.0) + CHANGELOG in one commit. Test + verify.

### Stage 4 — yolo integration
- Same wiring; match against COCO class name. Save = annotated frame.
- Confirm yolo already heartbeats every cycle (`env.count.total`); document it as
  the heartbeat. Docs + version bump (0.2.2 → 0.3.0) + CHANGELOG. Test + verify.

### Stage 5 — periodic sampler plugin (new plugin, separate repo/dir)
- New minimal plugin: no inference, no GPU. Grabs an image and/or audio sample
  and uploads the RAW artifact with `meta={"trigger":"periodic-sampler"}`.
- Args: source (camera snapshot URL / stream, and/or audio camera), what to
  capture (image | audio | both). Period is set by the SES science rule
  (`cronjob(... '0 * * * *')`), not a plugin arg — keeps period a one-line
  schedule change.
- Own sage.yaml, ecr-science-description, CHANGELOG, jobs/ example. Build /
  catalog / sideload / submit / verify a raw upload lands hourly.

### Stage 6 — cross-cutting docs + skill
- Add a short "save-match conventions" reference to the sage-waggle skill so the
  pattern (publish-always / save-selectively, single-floor, exact-match rule) is
  reusable for future plugins.
- Cross-link this design note from the shared `~/AI-projects/Sage-potential-
  features.md`.

### Sequencing / parallelism
- Stages 1→2 are the critical path (helper proven, then lead plugin proves the
  end-to-end shape on real hardware). 3 and 4 can follow in either order once 2
  is validated. Stage 5 is independent and can be done anytime after Stage 0.
- GPU contention reminder: bioclip/yolo/birdnet share the single Thor GPU via
  windowed scheduling; the sampler is GPU-free and schedules freely.


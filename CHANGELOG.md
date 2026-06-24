# Changelog

All notable changes to the `bioclip-species-classifier` Sage plugin.

## 0.4.0 — 2026-06-23

### Added
- **`--save-match`: species-aware image saving, decoupled from publishing.**
  Image upload is now governed by an explicit OR-list of `Name:confidence` rules
  (e.g. `"Barn Owl:0.5,Northern Cardinal:0.7"`) instead of being tied to
  `--min-confidence`. An annotated image is saved when ANY published detection
  matches ANY rule. Name matching is case-insensitive and EXACT against the
  common OR scientific name at the published `--rank` (no substring matching).
  The wildcard `"*:0.7"` reproduces the old "save anything ≥ threshold" behavior.
  Implemented via the shared `save_match.py` helper (29 unit tests).
- **`env.species.summary` heartbeat published EVERY cycle**, even with zero
  confident detections (`{published_count, top_confidence}`), so a user can
  always confirm from the data plane that the plugin ran.

### Changed
- **`--min-confidence` is now strictly the reporting floor** (what gets
  *published*), no longer the image-save trigger. Raise it to reduce noisy topic
  reports; it no longer affects which images are saved.
- **Image saving is now strictly opt-in.** With `--save-match` omitted, NO images
  are uploaded (topics still publish). Jobs that want images must set
  `--save-match` (use `"*:<conf>"` to keep prior behavior). NOTE: this is a
  behavior change — a job upgraded to 0.4.0 without a `--save-match` will publish
  topics but stop uploading images until a rule is added.
- Publish and save are now strictly separate code paths in `app.py`.

### Migration
- Add a `--save-match` arg to existing jobs. `"*:0.7"` matches the previous
  "upload when top prediction ≥ 0.7" behavior; a species list saves selectively.

## 0.3.3 — 2026-06-23

### Added
- **Standard `plugin.duration.*` performance telemetry** (matching
  `avian-diversity-monitoring` / TAFT-node convention). Each cycle publishes
  nanosecond phase timings via pywaggle's `plugin.timeit`:
  `plugin.duration.loadmodel` (model construct + load, once),
  `plugin.duration.input` (snapshot/capture + decode + BGR→PIL, per cycle),
  `plugin.duration.inference` (classification, per cycle). Especially valuable
  here: the ~28 GB model's cold-start cost is now directly measurable, so it's
  clear how much of a bounded GPU window is load vs. inference. Doubles as a
  liveness signal even when nothing clears `--min-confidence`. Model load
  refactored into a `load()` method so it can be timed inside the Plugin context.

## 0.3.2 — 2026-06-22

### Added
- **`--max-runtime N` flag for windowed GPU sharing.** When combined with
  `--continuous Y`, the plugin loops every `--interval` seconds and then
  self-exits after N seconds — behaving like one long bounded single-shot.
  Default `0` = run forever (previous behavior, unchanged). On Thor (one GPU)
  BioCLIP runs a bounded 10-minute window at `:20` each hour
  (`cronjob('20 * * * *')`, `--max-runtime 600 --interval 15`, ~40 frames),
  offset from the YOLO plugin's `:00` window with 10-minute guard-bands so the
  two never contend for the GPU. Within the window the ~28 GB model loads once
  and stays warm for all ~40 classifications.

### Changed
- H00F hummingcam job raised `--min-confidence` 0.5 → **0.7** to report only
  high-confidence species. BioCLIP has no "reject" class and can score
  confidently on empty frames, so a higher bar suppresses that noise.
  (System-level species gating belongs in the slack-hummingbird watcher.)
- `DOCKER-BUILD.md` gained a 3-way Continuous / One-shot / Windowed decision
  table with the window-layout diagram.

## 0.3.1 — 2026-06-21

### Fixed
- Annotated-image text-line overlap. Line gap is now computed from
  `cv2.getTextSize` (text height + baseline + padding) instead of a fixed
  `30*scale`, which collided at all scales.

## 0.3.0 and earlier

- See git history. Core: BioCLIP 2.5 Huge ViT-H/14 zero-shot species
  classification at any taxonomic rank, `env.species.*` records, annotated-image
  upload, HTTP-snapshot and camera sources.

---

### Deployment note (arm64 / Thor)

This plugin is built locally and **sideloaded** into the node's k3s containerd
(`docker save | sudo k3s ctr images import -`, ~28 GB) because the ECR portal's
arm64 NVIDIA-base build crashes under QEMU. The ECR **catalog** version is
registered separately via `scripts/register-ecr-version.py` (the metadata record
SES validates against). SES pods use `imagePullPolicy=IfNotPresent`, so the
sideloaded image serves the actual pull. See `DOCKER-BUILD.md` for the full
build → register → sideload → submit workflow.

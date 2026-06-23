# Changelog

All notable changes to the `bioclip-species-classifier` Sage plugin.

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

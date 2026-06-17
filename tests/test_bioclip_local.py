#!/usr/bin/env python3
"""
BioCLIP2 Local Test Runner — Classify real images from a test directory.

Runs the actual BioCLIP2 plugin (app.py) against every image in
tests/test-images/, validates the pywaggle output, and prints
a detailed report with per-image species predictions and timing.

This is the go-to test for checking whether BioCLIP2 is performing well
on your own images before deploying to a Sage node.

Setup:
    # Activate the test venv (must have pybioclip, pywaggle, torch, etc.)
    source tests/.venv/bin/activate

    # Add your test images — any JPG/PNG/WEBP photos of animals, plants,
    # insects, birds, etc.  The more diverse, the better the test.
    cp  my-bird-photo.jpg    tests/test-images/
    cp  my-insect-photo.png  tests/test-images/

Usage:
    # Default: classify at Species rank
    python tests/test_bioclip_local.py

    # Classify at a different taxonomic rank
    python tests/test_bioclip_local.py --rank Class
    python tests/test_bioclip_local.py --rank Order

    # Sweep all 7 ranks on every image (thorough, slower)
    python tests/test_bioclip_local.py --all-ranks

    # Set a minimum confidence threshold (default: 0.01)
    python tests/test_bioclip_local.py --min-confidence 0.5

    # Verbose: show all log output from the plugin
    python tests/test_bioclip_local.py --verbose

Output:
    tests/output/bioclip-local/            — pywaggle data.ndjson + uploads/
    tests/output/bioclip-local/report.json — machine-readable per-image results
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────
TESTS_DIR = Path(__file__).parent
PLUGIN_DIR = TESTS_DIR.parent
PLUGIN_APP = PLUGIN_DIR / "app.py"
TEST_IMAGES = TESTS_DIR / "test-images"
OUTPUT_DIR = TESTS_DIR / "output" / "bioclip-local"

RANK_NAMES = ["Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species"]


def count_images(directory: Path) -> list[Path]:
    """Find all image files in a directory."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
    return sorted(p for p in directory.iterdir()
                  if p.suffix.lower() in exts and p.is_file()
                  and not p.name.startswith("."))


def run_bioclip(rank: str, min_confidence: float, top_k: int,
                verbose: bool) -> tuple[int, str]:
    """
    Run the BioCLIP plugin against the test image directory.
    Returns (exit_code, combined_output).
    """
    # Clean output dir for this run
    rank_output = OUTPUT_DIR / rank.lower()
    if rank_output.exists():
        shutil.rmtree(rank_output)
    rank_output.mkdir(parents=True)

    env = os.environ.copy()
    env["PYWAGGLE_LOG_DIR"] = str(rank_output)

    cmd = [
        sys.executable, str(PLUGIN_APP),
        "--image-dir", str(TEST_IMAGES),
        "--rank", rank,
        "--min-confidence", str(min_confidence),
        "--top-k", str(top_k),
        "--continuous", "N",
    ]

    if verbose:
        print(f"  CMD: {' '.join(cmd)}")
        print(f"  PYWAGGLE_LOG_DIR={rank_output}")

    result = subprocess.run(
        cmd, env=env,
        capture_output=True, text=True, timeout=600,
    )

    output = result.stdout + result.stderr
    return result.returncode, output


def parse_ndjson(rank_output: Path) -> list[dict]:
    """Parse data.ndjson from a run."""
    ndjson = rank_output / "data.ndjson"
    if not ndjson.exists():
        return []
    records = []
    with open(ndjson) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def extract_results(records: list[dict], rank: str) -> list[dict]:
    """
    Group ndjson records by image and extract per-image results.
    Returns list of dicts with image name, top prediction, confidence, top5.
    """
    rank_lower = rank.lower()
    # Records come in groups of 4 per image:
    #   env.species.<rank>, env.species.<rank>.confidence, env.species.top5, upload
    results = []
    species_records = [r for r in records
                       if r["name"] == f"env.species.{rank_lower}"]
    conf_records = [r for r in records
                    if r["name"] == f"env.species.{rank_lower}.confidence"]
    top5_records = [r for r in records
                    if r["name"] == "env.species.top5"]
    upload_records = [r for r in records if r["name"] == "upload"]

    for i, sr in enumerate(species_records):
        entry = {
            "image": sr.get("meta", {}).get("camera", "unknown"),
            "rank": rank,
            "prediction": sr["value"],
            "confidence": conf_records[i]["value"] if i < len(conf_records) else None,
        }
        if i < len(top5_records):
            try:
                entry["top5"] = json.loads(top5_records[i]["value"])
            except (json.JSONDecodeError, TypeError):
                entry["top5"] = []
        if i < len(upload_records):
            entry["uploaded"] = True
        results.append(entry)
    return results


def print_image_report(result: dict, idx: int):
    """Print a single image result."""
    print(f"\n  Image {idx}: {result['image']}")
    print(f"    Rank:       {result['rank']}")
    print(f"    Prediction: {result['prediction']}")
    if result.get("confidence") is not None:
        conf = result["confidence"]
        bar_len = int(conf * 40)
        bar = "#" * bar_len + "-" * (40 - bar_len)
        print(f"    Confidence: {conf:.6f}  [{bar}]")
    if result.get("top5"):
        print(f"    Top-5:")
        for j, p in enumerate(result["top5"], 1):
            c = p["confidence"]
            print(f"      #{j}: {p['name']:40s}  {c:.6f}")


def main():
    parser = argparse.ArgumentParser(
        description="BioCLIP2 local test runner — classify test images and report results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--rank", default="Species", choices=RANK_NAMES,
                        help="Taxonomic rank (default: Species)")
    parser.add_argument("--all-ranks", action="store_true",
                        help="Test all 7 taxonomic ranks (thorough)")
    parser.add_argument("--min-confidence", type=float, default=0.01,
                        help="Minimum confidence to publish (default: 0.01)")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Number of top predictions (default: 5)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show plugin log output")
    args = parser.parse_args()

    # ── Preflight checks ─────────────────────────────────────────────
    if not PLUGIN_APP.exists():
        print(f"ERROR: Plugin not found at {PLUGIN_APP}", file=sys.stderr)
        sys.exit(1)

    images = count_images(TEST_IMAGES)
    if not images:
        print(f"ERROR: No test images found in {TEST_IMAGES}", file=sys.stderr)
        print(f"\nAdd images to test:", file=sys.stderr)
        print(f"  cp your-photo.jpg {TEST_IMAGES}/", file=sys.stderr)
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ranks = RANK_NAMES if args.all_ranks else [args.rank]

    print("=" * 70)
    print("  BioCLIP2 LOCAL TEST")
    print("=" * 70)
    print(f"  Test images:  {TEST_IMAGES}")
    print(f"  Image count:  {len(images)}")
    for img in images:
        size_kb = img.stat().st_size / 1024
        print(f"    - {img.name}  ({size_kb:.0f} KB)")
    print(f"  Ranks:        {', '.join(ranks)}")
    print(f"  Min conf:     {args.min_confidence}")
    print(f"  Top-K:        {args.top_k}")
    print(f"  Output:       {OUTPUT_DIR}")
    print("=" * 70)

    # ── Run classification ───────────────────────────────────────────
    all_results = {}
    all_timings = {}
    overall_pass = True

    for rank in ranks:
        print(f"\n{'─' * 70}")
        print(f"  RANK: {rank}")
        print(f"{'─' * 70}")

        t0 = time.time()
        exit_code, output = run_bioclip(
            rank, args.min_confidence, args.top_k, args.verbose)
        elapsed = time.time() - t0
        all_timings[rank] = round(elapsed, 2)

        if args.verbose:
            for line in output.strip().split("\n"):
                print(f"    | {line}")

        if exit_code != 0:
            print(f"  FAILED (exit code {exit_code})")
            if not args.verbose:
                # Show stderr on failure even without verbose
                for line in output.strip().split("\n")[-10:]:
                    print(f"    | {line}")
            overall_pass = False
            continue

        rank_output = OUTPUT_DIR / rank.lower()
        records = parse_ndjson(rank_output)
        results = extract_results(records, rank)
        all_results[rank] = results

        if not results:
            print(f"  WARNING: No predictions published (all below threshold?)")
            overall_pass = False
        else:
            for idx, r in enumerate(results, 1):
                print_image_report(r, idx)

        print(f"\n  Time: {elapsed:.2f}s  |  Images: {len(images)}  |  "
              f"Published: {len(results)}")

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  SUMMARY")
    print(f"{'=' * 70}")
    for rank in ranks:
        results = all_results.get(rank, [])
        timing = all_timings.get(rank, 0)
        n_published = len(results)
        if results:
            avg_conf = sum(r["confidence"] for r in results
                          if r.get("confidence")) / len(results)
            print(f"  {rank:10s}  {n_published}/{len(images)} images  "
                  f"avg_conf={avg_conf:.4f}  time={timing:.2f}s")
        else:
            print(f"  {rank:10s}  {n_published}/{len(images)} images  "
                  f"time={timing:.2f}s")

    # ── Save report ──────────────────────────────────────────────────
    report = {
        "test_images_dir": str(TEST_IMAGES),
        "image_count": len(images),
        "images": [img.name for img in images],
        "ranks_tested": ranks,
        "results": all_results,
        "timings_s": all_timings,
        "passed": overall_pass,
    }
    report_path = OUTPUT_DIR / "report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    if overall_pass:
        print(f"\n  PASSED — all images classified successfully")
    else:
        print(f"\n  FAILED — check output above for details")

    print(f"  Report: {report_path}")
    print(f"{'=' * 70}")

    sys.exit(0 if overall_pass else 1)


if __name__ == "__main__":
    main()

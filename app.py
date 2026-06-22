"""
BioCLIP 2.5 Species Classifier Plugin for Sage/Waggle
Captures camera frames, classifies biological organisms using BioCLIP 2.5
(imageomics/bioclip-2.5-vith14), and publishes taxonomy predictions.

BioCLIP 2.5 Huge is a CLIP model fine-tuned on the TreeOfLife-200M dataset
covering 450K+ species.  It uses a ViT-H/14 backbone and achieves 61.3%
mean zero-shot species accuracy (+5.7% over BioCLIP 2).

Default model: BioCLIP 2.5 Huge (hf-hub:imageomics/bioclip-2.5-vith14)
  - Architecture: ViT-H/14 vision transformer
  - Training data: TreeOfLife-200M (219M+ biological images)
  - ~5-7 GB GPU memory at inference
  - Fits easily in 128GB unified memory (DGX Spark / Sage Thor)

Measurement topics:
  env.species.<rank>           — top predicted taxon name at chosen rank
  env.species.<rank>.confidence — confidence score (0-1)
  env.species.top5             — JSON of top-5 predictions
  upload                       — annotated camera image (above threshold only)
"""
import argparse
import json
import logging
import os
import tempfile
import time
import urllib.request
import urllib.error

import cv2
import numpy as np
from PIL import Image

from bioclip import Rank
from bioclip.predict import TreeOfLifeClassifier

from waggle.plugin import Plugin
from waggle.data.vision import Camera

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("bioclip-species")

# Map string rank names to bioclip.Rank enum
RANK_MAP = {
    "Kingdom": Rank.KINGDOM,
    "Phylum": Rank.PHYLUM,
    "Class": Rank.CLASS,
    "Order": Rank.ORDER,
    "Family": Rank.FAMILY,
    "Genus": Rank.GENUS,
    "Species": Rank.SPECIES,
}
RANK_NAMES = list(RANK_MAP.keys())


class BioCLIP2Classifier:
    """BioCLIP2 species classifier for Sage edge nodes.

    Uses the pybioclip library's TreeOfLifeClassifier which handles
    model loading, text embeddings, and taxonomic classification.
    """

    def __init__(self, rank: str = "Class",
                 model_str: str = "hf-hub:imageomics/bioclip-2"):
        if rank not in RANK_MAP:
            raise ValueError(f"Invalid rank '{rank}'. Must be one of: {RANK_NAMES}")
        self.rank = rank
        self.rank_enum = RANK_MAP[rank]
        self.model_str = model_str

        logger.info("Loading BioCLIP2 classifier (model=%s, rank=%s)...",
                     model_str, rank)
        self.classifier = TreeOfLifeClassifier(model_str=model_str)
        logger.info("BioCLIP2 classifier loaded successfully")

    def classify(self, image: Image.Image, top_k: int = 5) -> list[dict]:
        """
        Classify an image at the configured taxonomic rank.
        Returns list of {name, confidence} dicts, sorted descending.

        pybioclip returns a list of dicts with keys like:
          file_name, kingdom, phylum, class, order, family, genus,
          species_epithet, species, common_name, score
        """
        results = self.classifier.predict(
            images=[image],
            rank=self.rank_enum,
            k=top_k,
        )

        # Build a display name from the rank-level key
        rank_key = self.rank.lower()
        if rank_key == "species":
            # Species rank has a 'species' key with binomial name
            predictions = [
                {"name": r.get("species", r.get("genus", "Unknown")),
                 "confidence": float(r["score"])}
                for r in results
            ]
        else:
            predictions = [
                {"name": r.get(rank_key, "Unknown"),
                 "confidence": float(r["score"])}
                for r in results
            ]

        return predictions[:top_k]


# ── image sources ────────────────────────────────────────────────────

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def iter_image_dir(directory: str):
    """
    Yield (image_path, frame_bgr, timestamp_ns) for every image in a
    directory.  Used for local testing without a live camera.
    """
    from pathlib import Path

    dir_path = Path(directory)
    if not dir_path.is_dir():
        raise FileNotFoundError(f"Image directory not found: {directory}")

    files = sorted(
        p for p in dir_path.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS and p.is_file()
        and not p.name.startswith(".")
    )
    if not files:
        raise FileNotFoundError(
            f"No image files found in {directory}. "
            f"Supported extensions: {', '.join(sorted(IMAGE_EXTENSIONS))}"
        )

    logger.info("Found %d test images in %s", len(files), directory)
    for img_path in files:
        frame = cv2.imread(str(img_path))
        if frame is None:
            logger.warning("Skipping unreadable file: %s", img_path.name)
            continue
        yield str(img_path), frame, time.time_ns()


def fetch_snapshot(url: str) -> np.ndarray:
    """
    Fetch a JPEG snapshot from an HTTP URL and return as a BGR numpy array.

    Works with Reolink's HTTP API:
      http://IP:PORT/cgi-bin/api.cgi?cmd=Snap&channel=0&rs=abc&user=USER&password=PASS

    Also works with any URL that returns a JPEG image (MJPEG snapshot
    endpoints, generic IP camera snapshot URLs, etc.).
    """
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=15) as resp:
            img_bytes = resp.read()
    except urllib.error.URLError as e:
        raise ConnectionError(f"Failed to fetch snapshot from {url}: {e}") from e

    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError(
            f"Could not decode image from {url} "
            f"({len(img_bytes)} bytes received)"
        )
    return frame


def annotate_predictions(frame: np.ndarray, predictions: list[dict],
                         min_confidence: float) -> np.ndarray:
    """Annotate a frame with species predictions in orange text.

    - Above threshold: show top predictions with confidence
    - Below threshold: show "No confident prediction" message

    Line spacing is derived from the MEASURED glyph height plus padding
    (not a fixed constant), so lines never overlap at any image/font size.
    """
    annotated = frame.copy()
    h, w = annotated.shape[:2]
    # Orange in BGR
    color = (0, 165, 255)
    bg_color = (0, 0, 0)

    # Scale font to image size
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.5, min(w, h) / 1000.0)
    thickness = max(1, int(scale * 2))
    margin = int(10 * scale)

    # Measure a representative glyph height once, then set line spacing from
    # the ACTUAL text height + baseline + padding so lines never collide.
    (_, sample_th), sample_base = cv2.getTextSize("Ag", font, scale, thickness)
    line_gap = sample_th + sample_base + max(6, int(sample_th * 0.5))

    def draw_line(text, y):
        (tw, th), base = cv2.getTextSize(text, font, scale, thickness)
        cv2.rectangle(annotated, (margin, y - th - base - 2),
                      (margin + tw + 6, y + 2), bg_color, -1)
        cv2.putText(annotated, text, (margin + 3, y - base + 1),
                    font, scale, color, thickness)

    if not predictions or predictions[0]["confidence"] < min_confidence:
        # No confident prediction — write message at bottom
        top_conf = predictions[0]["confidence"] if predictions else 0
        draw_line(f"No confident species prediction (best: {top_conf:.1%})",
                  h - margin)
    else:
        # Show top predictions in top-left corner
        y = margin + line_gap
        for i, pred in enumerate(predictions):
            conf = pred["confidence"]
            name = pred["name"]
            draw_line(f"#{i+1}: {name} ({conf:.1%})", y)
            y += line_gap
            if i >= 4:  # Show max 5
                break

    return annotated


# ── main loop ───────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="BioCLIP2 Species Classifier for Sage",
        epilog="""
Examples:
  # Normal mode — capture from camera on a Sage node
  python3 app.py --stream bottom_camera --rank Species

  # HTTP snapshot camera (e.g. Reolink via port-mapped router)
  python3 app.py --snapshot-url "http://IP:PORT/cgi-bin/api.cgi?cmd=Snap&channel=0&rs=snap&user=USER&password=PASS&width=640&height=360" --rank Species

  # Local testing — classify all images in a directory
  export PYWAGGLE_LOG_DIR=./test-output
  python3 app.py --image-dir ./test-images --rank Species --continuous N

  # Local testing — classify a single image file
  python3 app.py --image-dir /path/to/single/image.jpg --rank Class --continuous N
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--stream", default="bottom_camera",
                        help="Camera stream name or RTSP URL (ignored if --image-dir is set)")
    parser.add_argument("--image-dir", default=None,
                        help="Directory of test images (replaces camera input for local testing)")
    parser.add_argument("--snapshot-url", default=None,
                        help="HTTP URL that returns a JPEG snapshot (e.g. Reolink CGI API). "
                             "Overrides --stream. Credentials go in the URL query string. "
                             "Example: http://IP:PORT/cgi-bin/api.cgi?cmd=Snap&channel=0"
                             "&width=640&height=360")
    parser.add_argument("--rank", default="Class",
                        choices=RANK_NAMES,
                        help="Taxonomic rank for classification")
    parser.add_argument("--model", default="hf-hub:imageomics/bioclip-2.5-vith14",
                        help="BioCLIP model string (default: BioCLIP 2.5 Huge ViT-H/14)")
    parser.add_argument("--interval", type=int, default=60,
                        help="Seconds between captures (camera mode only)")
    parser.add_argument("--min-confidence", type=float, default=0.1,
                        help="Minimum confidence to publish")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Number of top predictions to publish")
    parser.add_argument("--continuous", default="Y",
                        help="Y = loop, N = single-shot")
    parser.add_argument("--max-runtime", type=int, default=0,
                        help="When in continuous mode (--continuous Y), exit after this "
                             "many seconds (0 = run forever). Lets a scheduled job behave "
                             "like one long bounded single-shot: e.g. --max-runtime 600 "
                             "--interval 15 samples every 15s for ~10 min then self-exits, "
                             "freeing the GPU for other plugins. Ignored when --continuous N.")
    args = parser.parse_args()

    classifier = BioCLIP2Classifier(
        rank=args.rank,
        model_str=args.model,
    )

    # ── Choose image source ──────────────────────────────────────────
    using_image_dir = args.image_dir is not None
    using_snapshot_url = args.snapshot_url is not None

    if using_image_dir:
        # Local testing mode: read images from a directory
        image_source = iter_image_dir(args.image_dir)
        source_label = f"image-dir:{args.image_dir}"
    elif using_snapshot_url:
        # HTTP snapshot mode: fetch JPEG from URL each cycle
        source_label = args.snapshot_url.split("?")[0]  # log URL without query params
    else:
        # Production mode: capture from camera
        camera = Camera(args.stream)
        source_label = args.stream

    with Plugin() as plugin:
        logger.info("Plugin started — source=%s, rank=%s, model=%s",
                     source_label, args.rank, args.model)

        if not using_image_dir:
            logger.info("Capture interval: %ds", args.interval)

        # Bounded continuous mode: in --continuous Y, optionally self-exit after
        # --max-runtime seconds so a scheduled job runs like one long single-shot
        # and frees the GPU for other plugins. deadline=None means run forever.
        deadline = None
        if args.continuous == "Y" and args.max_runtime > 0 and not using_image_dir:
            deadline = time.monotonic() + args.max_runtime
            logger.info("Max runtime: %ds — will self-exit at the end of the window",
                        args.max_runtime)

        while True:
            try:
                if using_image_dir:
                    # Get next image from directory iterator
                    try:
                        img_path, frame, timestamp = next(image_source)
                    except StopIteration:
                        logger.info("All test images processed")
                        break
                    source_name = os.path.basename(img_path)
                    logger.info("Processing: %s (%dx%d)",
                                source_name, frame.shape[1], frame.shape[0])
                elif using_snapshot_url:
                    frame = fetch_snapshot(args.snapshot_url)
                    timestamp = time.time_ns()
                    source_name = "http-snapshot"
                    logger.info("Snapshot: %dx%d from %s",
                                frame.shape[1], frame.shape[0], source_label)
                else:
                    sample = camera.snapshot()
                    frame = sample.data  # numpy BGR
                    timestamp = sample.timestamp
                    source_name = args.stream

                # Convert BGR -> RGB -> PIL
                pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                predictions = classifier.classify(pil_image, top_k=args.top_k)

                if predictions and predictions[0]["confidence"] >= args.min_confidence:
                    top = predictions[0]

                    # Publish top prediction
                    rank_lower = args.rank.lower()
                    plugin.publish(
                        f"env.species.{rank_lower}",
                        top["name"],
                        timestamp=timestamp,
                        meta={"camera": source_name, "rank": args.rank,
                              "model": args.model},
                    )
                    plugin.publish(
                        f"env.species.{rank_lower}.confidence",
                        top["confidence"],
                        timestamp=timestamp,
                        meta={"camera": source_name, "rank": args.rank},
                    )

                    # Publish top-5 as JSON
                    plugin.publish(
                        f"env.species.top5",
                        json.dumps(predictions),
                        timestamp=timestamp,
                        meta={"camera": source_name, "rank": args.rank},
                    )

                    logger.info("Top prediction: %s (%.4f)", top["name"], top["confidence"])
                    for i, p in enumerate(predictions[1:], 2):
                        logger.info("  #%d: %s (%.4f)", i, p["name"], p["confidence"])

                    # Upload annotated image
                    annotated = annotate_predictions(frame, predictions,
                                                     args.min_confidence)
                    stem = os.path.splitext(source_name)[0]
                    tmp_path = os.path.join(tempfile.gettempdir(),
                                            f"{stem}-classified.jpg")
                    cv2.imwrite(tmp_path, annotated)
                    plugin.upload_file(tmp_path, timestamp=timestamp,
                                       meta={"camera": source_name,
                                             "top_species": top["name"],
                                             "confidence": str(top["confidence"])})
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
                else:
                    logger.info("No confident prediction (top=%.4f, threshold=%.2f)",
                                predictions[0]["confidence"] if predictions else 0,
                                args.min_confidence)

                    # In test mode (--image-dir), still upload annotated image
                    # so all results can be reviewed. In production, skip upload
                    # to avoid flooding storage with low-confidence frames.
                    if using_image_dir:
                        annotated = annotate_predictions(frame, predictions,
                                                         args.min_confidence)
                        stem = os.path.splitext(source_name)[0]
                        tmp_path = os.path.join(tempfile.gettempdir(),
                                                f"{stem}-classified.jpg")
                        cv2.imwrite(tmp_path, annotated)
                        plugin.upload_file(tmp_path, timestamp=timestamp,
                                           meta={"camera": source_name,
                                                 "top_species": "none",
                                                 "confidence": "0"})
                        if os.path.exists(tmp_path):
                            os.unlink(tmp_path)

            except Exception:
                logger.exception("Classification error")

            if args.continuous != "Y" and not using_image_dir:
                break
            # Bounded-window self-exit: stop before sleeping if the next cycle
            # would start at/after the deadline.
            if deadline is not None and time.monotonic() + args.interval >= deadline:
                logger.info("Max runtime reached — self-exiting to free the GPU")
                break
            if not using_image_dir:
                time.sleep(args.interval)


if __name__ == "__main__":
    main()

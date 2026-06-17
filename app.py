"""
BioCLIP2 Species Classifier Plugin for Sage/Waggle
Captures camera frames, classifies biological organisms using BioCLIP2
(imageomics/bioclip-2), and publishes taxonomy predictions.

BioCLIP2 is a CLIP model fine-tuned on the TreeOfLife-200M dataset
covering 450K+ species with significantly improved accuracy over v1.
It classifies at any taxonomic rank:
Kingdom > Phylum > Class > Order > Family > Genus > Species.

Default model: BioCLIP2 (hf-hub:imageomics/bioclip-2)
  - Architecture: SigLIP2-based vision transformer (~430M params)
  - Training data: TreeOfLife-200M (200M+ biological images)
  - ~2.5 GB GPU memory at inference
  - Fits easily in 128GB unified memory (DGX Spark / Sage Thor)

Measurement topics:
  env.species.<rank>           — top predicted taxon name at chosen rank
  env.species.<rank>.confidence — confidence score (0-1)
  env.species.top5             — JSON of top-5 predictions
  upload                       — original camera image
"""
import argparse
import json
import logging
import os
import tempfile
import time

import cv2
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


# ── main loop ───────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="BioCLIP2 Species Classifier for Sage",
        epilog="""
Examples:
  # Normal mode — capture from camera on a Sage node
  python3 app.py --stream bottom_camera --rank Species

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
    parser.add_argument("--rank", default="Class",
                        choices=RANK_NAMES,
                        help="Taxonomic rank for classification")
    parser.add_argument("--model", default="hf-hub:imageomics/bioclip-2",
                        help="BioCLIP model string (default: BioCLIP2)")
    parser.add_argument("--interval", type=int, default=60,
                        help="Seconds between captures (camera mode only)")
    parser.add_argument("--min-confidence", type=float, default=0.1,
                        help="Minimum confidence to publish")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Number of top predictions to publish")
    parser.add_argument("--continuous", default="Y",
                        help="Y = loop, N = single-shot")
    args = parser.parse_args()

    classifier = BioCLIP2Classifier(
        rank=args.rank,
        model_str=args.model,
    )

    # ── Choose image source ──────────────────────────────────────────
    using_image_dir = args.image_dir is not None

    if using_image_dir:
        # Local testing mode: read images from a directory
        image_source = iter_image_dir(args.image_dir)
        source_label = f"image-dir:{args.image_dir}"
    else:
        # Production mode: capture from camera
        camera = Camera(args.stream)
        source_label = args.stream

    with Plugin() as plugin:
        logger.info("Plugin started — source=%s, rank=%s, model=%s",
                     source_label, args.rank, args.model)

        if not using_image_dir:
            logger.info("Capture interval: %ds", args.interval)

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

                    # Upload source image
                    stem = os.path.splitext(source_name)[0]
                    tmp_path = os.path.join(tempfile.gettempdir(),
                                            f"{stem}-classified.jpg")
                    cv2.imwrite(tmp_path, frame)
                    plugin.upload_file(tmp_path, timestamp=timestamp,
                                       meta={"camera": source_name,
                                             "top_species": top["name"],
                                             "confidence": str(top["confidence"])})
                    os.unlink(tmp_path)
                else:
                    logger.info("No confident prediction (top=%.4f, threshold=%.2f)",
                                predictions[0]["confidence"] if predictions else 0,
                                args.min_confidence)

            except Exception:
                logger.exception("Classification error")

            if args.continuous != "Y" and not using_image_dir:
                break
            if not using_image_dir:
                time.sleep(args.interval)


if __name__ == "__main__":
    main()

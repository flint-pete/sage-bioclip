#!/usr/bin/env bash
# run-tests.sh — Run BioCLIP 2.5 Species Classifier local test (GPU required)
#
# Usage:
#   cd sage-bioclip
#   ./tests/run-tests.sh
#
# Requires: pybioclip, pywaggle, torch, opencv-python-headless, Pillow, numpy

set -euo pipefail
TESTS_DIR="$(cd "$(dirname "$0")" && pwd)"
PLUGIN_DIR="$(dirname "$TESTS_DIR")"

# Verify test images exist
if [ ! -d "$TESTS_DIR/test-images" ] || [ -z "$(ls -A "$TESTS_DIR/test-images/" 2>/dev/null)" ]; then
    echo "ERROR: No test images found in $TESTS_DIR/test-images/"
    echo "Add test images (JPG/PNG) to that directory."
    exit 1
fi

echo "=============================================="
echo "  BioCLIP 2.5 Species Classifier — Local Test"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================="
python3 "$TESTS_DIR/test_bioclip_local.py"

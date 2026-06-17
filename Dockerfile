# BioCLIP2 Species Classifier — Sage/Waggle edge plugin
# Default model: BioCLIP2 (imageomics/bioclip-2, ~430M params)
# Text embeddings: TreeOfLife-200M (200M+ biological images, 450K+ species)
# Target: 128GB unified memory ARM64 (DGX Spark / Sage Thor)
#
# Base image: NVIDIA PyTorch 25.04 — CUDA 12.9, PyTorch 2.7, Python 3.12
# Supports Blackwell GPUs (sm_120/sm_121) natively. Requires driver R575+.
FROM nvcr.io/nvidia/pytorch:25.04-py3

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# The NVIDIA base image may ship an opencv compiled against a different numpy.
# Fix: fully remove old opencv (pip uninstall + rm stale files), then
# install a fresh opencv-python-headless matching the current numpy.
RUN pip uninstall -y opencv-python opencv-python-headless 2>/dev/null; \
    rm -rf /usr/local/lib/python3.*/dist-packages/cv2* && \
    pip install --no-cache-dir opencv-python-headless>=4.8.0

# Pre-download BioCLIP2 model weights and TreeOfLife-200M embeddings
# Layer order matters: model weights change rarely, app.py changes often.
# Putting the model download BEFORE COPY app.py means code edits don't
# invalidate the expensive (~2 GB) model + embeddings download layer.
# pybioclip handles caching via HuggingFace Hub
RUN python3 -c "\
from bioclip.predict import TreeOfLifeClassifier; \
c = TreeOfLifeClassifier(model_str='hf-hub:imageomics/bioclip-2'); \
from bioclip import Rank; \
from PIL import Image; \
import numpy as np; \
dummy = Image.fromarray(np.zeros((224,224,3), dtype=np.uint8)); \
c.predict([dummy], rank=Rank.CLASS, k=1); \
print('BioCLIP2 model + embeddings cached')"

ENV HF_HOME=/root/.cache/huggingface
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

COPY app.py .

ENTRYPOINT ["python3", "/app/app.py"]

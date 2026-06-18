# Building and Testing the BioCLIP 2.5 Plugin Docker Image

How to build the Docker image, test it locally, and deploy it to
a Sage node.


## Prerequisites

- A build machine with internet access, Docker, and an NVIDIA GPU
- SSH access to a Sage Thor node (for on-node testing)
- NVIDIA Container Toolkit configured for Docker (see below)

### NVIDIA Container Toolkit Setup

Docker needs the NVIDIA Container Toolkit to pass GPUs into
containers. The toolkit may already be **installed** but not
**configured** — both steps are required.

**Check if it's already working:**

```bash
docker run --rm --runtime=nvidia nvidia/cuda:12.9.0-base-ubuntu24.04 nvidia-smi
```

If that prints your GPU info, you're set. If it fails:

```bash
# Step 1: Install (if not already)
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

# Step 2: Configure Docker to use the nvidia runtime
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Step 3: Verify
docker info | grep -i runtime
#  Runtimes: runc io.containerd.runc.v2 nvidia   ← nvidia must appear
```

This is a one-time setup per machine.


## Base Image

The Dockerfile uses `nvcr.io/nvidia/pytorch:25.08-py3`:

| Component | Version |
|-----------|---------|
| CUDA | 13.0 |
| PyTorch | 2.8 |
| Python | 3.12 |
| Ubuntu | 24.04 |
| Min driver | R575+ |

This image supports both Blackwell GPU variants natively:
- **DGX Spark** (GB10, sm_121)
- **Thor nodes** (NVIDIA Thor / Jetson Thor, sm_110)


## pybioclip Patch

The Dockerfile applies `patch_pybioclip.py` at build time. This
patches pybioclip 2.1.5 to support BioCLIP 2.5 Huge:

- Adds `bioclip-2.5-vith14` to pybioclip's `TOL_MODELS` whitelist
- Redirects text embedding downloads to the model-specific files
  (`txt_emb_bioclip-2.5-vith14.npy/json`) in the TreeOfLife-200M
  dataset repo

This patch is temporary — once pybioclip adds native 2.5 support
upstream, the patch can be removed.


## Building the Image

### Option A: Build on a Machine with Internet (DGX Spark)

```bash
cd ~/sage-bioclip
docker build --no-cache -t bioclip-species:0.3.0 .
```

Build time: ~10-15 minutes (downloads ~4-5 GB model weights +
~3 GB text embeddings on first build; cached thereafter).

Then transfer to Thor (see "Transfer to Thor" below).

### Option B: Build Directly on Thor

If the Thor node has outbound internet access:

```bash
# One-time setup
git clone https://github.com/flint-pete/sage-bioclip.git /tmp/sage-bioclip

# Build (sudo required for Docker on Thor)
cd /tmp/sage-bioclip
sudo docker build --no-cache -t bioclip-species:0.3.0 .
```

To rebuild after code changes:

```bash
cd /tmp/sage-bioclip
git pull
sudo docker build --no-cache -t bioclip-species:0.3.0 .
```

Build time on Thor: ~15-20 minutes (downloads over LTE are slower).


## Testing the Image Locally

### Quick sanity check

```bash
sudo docker run --rm --runtime=nvidia bioclip-species:0.3.0 --help
```

### Batch test with test images

```bash
mkdir -p ~/bioclip-test-output

sudo docker run --rm --runtime=nvidia \
    -e PYWAGGLE_LOG_DIR=/output \
    -v ~/bioclip-test-output:/output \
    -v ~/sage-bioclip/tests/test-images:/images:ro \
    bioclip-species:0.3.0 \
    --image-dir /images --rank Species --min-confidence 0.1 --continuous N

# Check results
cat ~/bioclip-test-output/data.ndjson | python3 -m json.tool
ls -la ~/bioclip-test-output/uploads/
```

### Test with an HTTP snapshot camera (Reolink)

```bash
mkdir -p ~/bioclip-camera-test

# One-shot test
sudo docker run --rm --runtime=nvidia \
    -e PYWAGGLE_LOG_DIR=/output \
    -v ~/bioclip-camera-test:/output \
    bioclip-species:0.3.0 \
    --snapshot-url "http://IP:PORT/cgi-bin/api.cgi?cmd=Snap&channel=0&rs=snap&user=USER&password=PASS&width=640&height=360" \
    --rank Species --min-confidence 0.5 --continuous N

cat ~/bioclip-camera-test/data.ndjson
ls -la ~/bioclip-camera-test/uploads/
```


## Transfer to Thor (Option A only)

```bash
# On the build machine
docker save bioclip-species:0.3.0 | gzip > /tmp/bioclip-species.tar.gz
scp /tmp/bioclip-species.tar.gz beckman@thor-node:~/

# On Thor — load into Docker
sudo docker load < ~/bioclip-species.tar.gz
```


## Deploy via pluginctl (Sage Workflow)

### Step 1: Import the image into k3s

```bash
sudo docker save bioclip-species:0.3.0 | sudo k3s ctr images import -
```

This takes ~8-10 minutes for the full image (~30 GB). Verify:

```bash
sudo k3s ctr images ls | grep bioclip
```

### Step 2: Deploy

```bash
sudo pluginctl deploy -n bioclip-hummingcam \
    --resource 'memory=16Gi,limit.memory=32Gi' \
    docker.io/library/bioclip-species:0.3.0 \
    -- --snapshot-url "http://IP:PORT/cgi-bin/api.cgi?cmd=Snap&channel=0&rs=snap&user=USER&password=PASS&width=640&height=360" \
       --rank Species --interval 60 --min-confidence 0.5 --continuous Y
```

**Important:** BioCLIP 2.5 Huge requires `memory=16Gi,limit.memory=32Gi`.
The ViT-H/14 model + 3 GB text embeddings exceed the 16 Gi limit that
worked for BioCLIP 2. OOMKilled at 8Gi/16Gi.

### Step 3: Monitor

```bash
sudo pluginctl ps
sudo pluginctl logs bioclip-hummingcam
sudo kubectl get pod bioclip-hummingcam
```

### Step 4: Stop

```bash
sudo pluginctl rm bioclip-hummingcam
```

### Step 5: Rebuild and redeploy

```bash
cd /tmp/sage-bioclip
git pull
sudo docker build --no-cache -t bioclip-species:0.3.0 .
sudo docker save bioclip-species:0.3.0 | sudo k3s ctr images import -
sudo pluginctl rm bioclip-hummingcam
sudo pluginctl deploy -n bioclip-hummingcam \
    --resource 'memory=16Gi,limit.memory=32Gi' \
    docker.io/library/bioclip-species:0.3.0 \
    -- --snapshot-url "..." --rank Species --interval 60 --min-confidence 0.5 --continuous Y
```


## Publish to Sage ECR (Production)

The Sage Edge Code Repository (ECR) is **not** a Docker registry.
You do not `docker push`. ECR pulls from GitHub and builds for you.

1. Go to https://portal.sagecontinuum.org
2. Sign in → My Apps → Create App
3. Enter: `https://github.com/flint-pete/sage-bioclip`
4. ECR builds the image and assigns a registry tag

**Note:** ECR multi-arch arm64 builds currently fail with the NVIDIA
base image (QEMU emulation crashes on `import torch`). See the
infrastructure issues report for details.


## Troubleshooting

**OOMKilled (exit code 137) via pluginctl**
  → BioCLIP 2.5 needs `memory=16Gi,limit.memory=32Gi`. The ViT-H/14
  model + 3 GB text embeddings exceed 16 Gi during initialization.

**"TreeOfLife predictions are only supported for..."**
  → The pybioclip patch didn't apply. Rebuild with `--no-cache`.

**"unknown or invalid runtime name: nvidia"**
  → NVIDIA Container Toolkit not configured. Run:
  `sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker`

**"No CUDA GPUs are available" inside container**
  → Missing `--runtime=nvidia` flag on docker run.

**"permission denied" on docker commands (Thor)**
  → Use `sudo docker ...` — Thor's Docker socket is root-only.

**Slow inference (~4s/frame instead of ~1-2s)**
  → First frame is slow (model warmup + torch.compile). Subsequent
  frames should be ~1-2s. If consistently slow, check GPU is being
  used: `sudo docker run --rm --runtime=nvidia bioclip-species:0.3.0
  python3 -c "import torch; print(torch.cuda.is_available())"`

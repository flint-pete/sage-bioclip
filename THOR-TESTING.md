# Testing on Thor Nodes

How to test the BioCLIP 2.5 Species Classifier plugin on a Sage Thor node.


## Quick Start: Build on Thor and Test

The fastest workflow — clone, build, and test all on Thor:

```bash
# One-time: clone the repo
git clone https://github.com/flint-pete/sage-bioclip.git /tmp/sage-bioclip

# Build the Docker image
cd /tmp/sage-bioclip
sudo docker build --no-cache -t bioclip-species:0.3.0 .

# Run against test images
mkdir -p ~/bioclip-test-output
sudo docker run --rm --runtime=nvidia \
    -e PYWAGGLE_LOG_DIR=/output \
    -v ~/bioclip-test-output:/output \
    -v /tmp/sage-bioclip/tests/test-images:/images:ro \
    bioclip-species:0.3.0 \
    --image-dir /images --rank Species --min-confidence 0.1 --continuous N

# Check results
cat ~/bioclip-test-output/data.ndjson | python3 -m json.tool
ls -la ~/bioclip-test-output/uploads/
```

To iterate after code changes:

```bash
cd /tmp/sage-bioclip
git pull
sudo docker build --no-cache -t bioclip-species:0.3.0 .
```

For the full Docker build reference (base image, pybioclip patch,
NVIDIA Container Toolkit setup), see **DOCKER-BUILD.md**.


## Testing with an HTTP Snapshot Camera

For cameras behind a port-mapped router (e.g. Reolink with
only HTTP port forwarded):

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

The `&width=640&height=360` parameters request low-resolution
snapshots (~12KB vs ~445KB at 4K) — critical for LTE-connected
cameras. BioCLIP resizes to 224x224 anyway.

### Continuous monitoring

```bash
sudo docker run --rm --runtime=nvidia \
    -e PYWAGGLE_LOG_DIR=/output \
    -v ~/bioclip-camera-test:/output \
    bioclip-species:0.3.0 \
    --snapshot-url "http://IP:PORT/cgi-bin/api.cgi?..." \
    --rank Species --interval 60 --min-confidence 0.5 --continuous Y
```

Monitor results in another terminal:

```bash
tail -f ~/bioclip-camera-test/data.ndjson
ls -la ~/bioclip-camera-test/uploads/
```

Press Ctrl-C to stop.


## Testing via pluginctl (Sage Infrastructure)

For testing with the full Sage stack (data publishing):

```bash
# Import into k3s (if not already done)
sudo docker save bioclip-species:0.3.0 | sudo k3s ctr images import -

# Deploy — note the higher memory limits for ViT-H/14
sudo pluginctl deploy -n bioclip-hummingcam \
    --resource 'memory=16Gi,limit.memory=32Gi' \
    docker.io/library/bioclip-species:0.3.0 \
    -- --snapshot-url "http://..." \
       --rank Species --interval 60 --min-confidence 0.5 --continuous Y

# Monitor
sudo pluginctl ps
sudo pluginctl logs bioclip-hummingcam

# Stop
sudo pluginctl rm bioclip-hummingcam
```

**Important:** BioCLIP 2.5 requires `memory=16Gi,limit.memory=32Gi`.
OOMKilled at the 8Gi/16Gi limits that worked for BioCLIP 2.


## Inspecting Output

All output from Docker testing is captured locally via
`PYWAGGLE_LOG_DIR`:

```bash
# Pretty-print published measurements
cat ~/bioclip-test-output/data.ndjson | python3 -m json.tool

# View uploaded annotated images
ls -la ~/bioclip-test-output/uploads/

# Copy annotated images to your dev machine
scp beckman@thor-node:~/bioclip-test-output/uploads/* ./
```

Each line in `data.ndjson` looks like:

```json
{"timestamp":"...","name":"env.species.species","value":"Archilochus colubris","meta":{"camera":"http-snapshot","model":"hf-hub:imageomics/bioclip-2.5-vith14"}}
```

Annotated images have orange text overlay with top-5 predictions
and confidence percentages.


## Clean Up

```bash
# Remove test output
rm -rf ~/bioclip-test-output ~/bioclip-camera-test

# Remove the Docker image (to free disk space)
sudo docker rmi bioclip-species:0.3.0

# Remove the cloned repo
rm -rf /tmp/sage-bioclip
```


## Troubleshooting

**OOMKilled (exit code 137)**
  → BioCLIP 2.5 needs `memory=16Gi,limit.memory=32Gi`.

**"permission denied" on docker commands**
  → Use `sudo docker ...` — Thor's Docker socket is root-only.

**"No CUDA GPUs are available"**
  → Missing `--runtime=nvidia` on the docker run command.

**torch.cuda.is_available() returns False (direct execution)**
  → Your user is not in the `video` group. `/dev/nvmap` is owned
  by `root:video`. Use Docker instead (containers get GPU access
  automatically via `--runtime=nvidia`).

**"No confident prediction" on every frame**
  → Lower `--min-confidence` (e.g. 0.1) to see what BioCLIP is
  predicting. If the top confidence is very low (<5%), the camera
  may not be showing a clear biological subject.

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


## Production: Scheduled SES Jobs on Thor (arm64)

This is the production deployment path — a scheduler-managed job that
survives reboots and is visible to the scheduler, instead of a hand-deployed
`pluginctl` pod. There are **two modes**, and the choice matters:

### Continuous vs One-shot vs Windowed — choose before you deploy

| | **Windowed** (default for birds) | **Continuous** (always-on) | **One-shot** (cron) |
|---|---|---|---|
| Job file | `jobs/bioclip-hummingcam-h00f.yaml` | (git history / hand-edit) | `jobs/bioclip-hummingcam-h00f-oneshot.yaml` |
| Args | `--continuous Y --interval 15 --max-runtime 600` | `--continuous Y --interval 60` | `--continuous N` |
| Science rule | `cronjob(..., '20 * * * *')` | `schedule(...): True` | `cronjob(..., '*/10 * * * *')` |
| Sampling | every 15 s for 10 min/hour, then self-exit | every 60 s, forever | once per cron tick |
| Model load | once per window, **warm** within it | once, stays **warm** | **cold start** (~28 GB) every tick |
| GPU | ~10 min/hour (shares with other plugins) | held 24/7 | freed between ticks |
| Best for | birds on a **single-GPU node** shared with YOLO | birds on a node with a dedicated GPU | slow scenes, 10-min cadence is plenty |

**Why Windowed is the default on Thor (single-GPU sharing).** Thor has ONE GPU,
and two always-on continuous plugins cannot co-run — a held GPU blocks the
second pod from scheduling at all. So BioCLIP and YOLO each take a bounded
10-minute window per hour instead of holding the GPU 24/7:

```
:00–:10  YOLO     (--max-runtime 600, samples every 15s)
:10–:20  guard-band
:20–:30  BioCLIP  (--max-runtime 600, samples every 15s)
:30–:00  guard-band
```

The **`--max-runtime`** flag (added in 0.3.2) makes this work: in `--continuous Y`
mode the plugin loops every `--interval` seconds, then self-exits after
`--max-runtime` seconds — like one long bounded single-shot — freeing the GPU. A
cron starts each window; the plugin ends it. Within the 10-min window the ~28 GB
model loads **once and stays warm** for all ~40 classifications (a one-shot cron
would cold-start it every tick). The 10-min guard-bands absorb model-load overrun
so the two plugins never collide on the single GPU. Net: ~20 min/hour (~1/3).

The windowed job also raises `--min-confidence` to **0.7**: BioCLIP has no
"reject" class and can score confidently on empty frames, so a high bar
suppresses most of that noise (system-level species gating still belongs in the
slack-hummingbird watcher).

**Why this matters — detection coverage.** When the companion YOLO cam ran as a
`*/10` one-shot cron, bird detections collapsed from ~15/day to ~0 — a
hummingbird is in-frame only a few seconds, so 10-min sampling almost never
catches one. Windowed mode samples every 15s *within* its window, restoring
coverage while still sharing the GPU.

**Rule of thumb:** brief/unpredictable subject + shared GPU → windowed; brief
subject + dedicated GPU → continuous; slowly-changing scene → one-shot. To
switch modes, deploy the other job file (see "Create + submit" below).

### Why the normal ECR portal build does NOT work for this plugin

The documented Sage workflow is "Create App → Register and Build App" and
the ECR portal builds the image from your GitHub repo. **That build fails
for any arm64 plugin on the NVIDIA base image**, and here is why:

- The ECR/Jenkins build pipeline runs on **x86_64** hardware.
- To produce a `linux/arm64` image it cross-builds under **QEMU emulation**.
- The NVIDIA base (`nvcr.io/nvidia/pytorch:25.08-py3`) contains aarch64
  binaries QEMU cannot emulate; the build crashes on `import torch` /
  `pip install` with `qemu: uncaught target signal 6 (Aborted) - core
  dumped`, build exit 134.

So the portal build is a dead end for Thor-targeted NVIDIA plugins until
the ECR pipeline gets a **native arm64 builder**. (BioCLIP 2.5's ViT-H/14
image is ~28 GB, so it's the heaviest of the three to build/sideload.)

### Why `docker push` to the registry also does NOT work (yet)

Build natively on Thor (arm64, no QEMU), then push to
`registry.sagecontinuum.org`? The build succeeds, but the push is denied:

```
denied: requested access to the resource is denied
```

`docker login registry.sagecontinuum.org` with a Sage portal access token
**authenticates** (login succeeds) but the token is **read/pull-only** — it
lacks push/write scope to the `beckman` namespace. Registry writes are
reserved for the Jenkins build pipeline. Getting push access (or a native
arm64 builder) is an ECR-team request — see "Systemic fix" below.

### The working workaround: build locally + sideload into k3s

Because SES pods on Thor use **`imagePullPolicy: IfNotPresent`**, the
scheduler uses a locally-cached image if one is already present in k3s
containerd under the exact registry-qualified name — it never has to pull
from the registry. So we build natively on Thor, tag with the full
registry path, and import it straight into k3s. No registry push needed.

**Step 1 — build natively on Thor (arm64, no QEMU):**

```bash
cd ~/sage-bioclip
git pull
sudo docker build -t registry.sagecontinuum.org/beckman/bioclip-species-classifier:0.3.1 .
```

Note the tag is the **full registry path**, not the bare
`bioclip-species:0.3.0`. It must exactly match the `image:` field in the
job YAML so k3s finds the cached copy. (The bare local image name used by
the old `pluginctl` workflow was `bioclip-species`; the ECR app / registry
name is `bioclip-species-classifier` — make sure the tag uses the latter.)

**Step 2 — sideload into k3s containerd** (this is large, ~28 GB; allow
several minutes):

```bash
sudo docker save registry.sagecontinuum.org/beckman/bioclip-species-classifier:0.3.1 \
  | sudo k3s ctr images import -
```

**Step 3 — verify it landed (and is CRI-managed):**

```bash
sudo k3s ctr images ls | grep bioclip-species-classifier
# Expect registry.sagecontinuum.org/beckman/bioclip-species-classifier:0.3.1
# with io.cri-containerd.image=managed  (that label = k8s/SES can see it)
```

### Step 4 — register the version in the ECR catalog (metadata only)

SES validates a job's image against the ECR app **catalog**
(ecr.sagecontinuum.org), NOT against the raw Docker registry or the image
you sideloaded. If the catalog has no record for your exact version,
`sesctl submit` fails with:

```
[registry.sagecontinuum.org/beckman/bioclip-species-classifier:0.3.1 does not exist in ECR]
```

You do **not** need the portal UI (and you do **not** need the portal
*build* to succeed — for this NVIDIA-base plugin it crashes under QEMU
anyway) — you only need the catalog metadata record. Register it directly
via the ECR API with the included helper script:

```bash
python3 scripts/register-ecr-version.py \
    --namespace beckman \
    --name bioclip-species-classifier \
    --from-version 0.3.0 \
    --version 0.3.1 \
    --git-url https://github.com/flint-pete/sage-bioclip.git \
    --token "$SAGE_TOKEN"
```

(For a *new* version, set `--from-version` to an already-registered
version to clone, and `--version` to the new one.)

The script clones an existing version's catalog record, bumps the version
and git source, and POSTs it to `/api/submit` using the
`Authorization: Sage <token>` header. It's idempotent (re-running a version
that already exists is a no-op) and prints the resulting catalog listing.

> **Under the hood:** that's a `POST https://ecr.sagecontinuum.org/api/submit`
> with your Sage portal token. The one required field the API insists on is
> `description`. The catalog record only satisfies SES validation — the
> actual image still comes from your sideloaded copy via
> `imagePullPolicy: IfNotPresent`.

If you prefer the UI: Portal → My Apps → the app → add the version from
GitHub. The portal *build* will fail (QEMU), but the catalog record still
gets created, which is all SES needs.

Either way, make the app **public** or SES returns
`registry does not exist in ECR`.

**Step 5 — create + submit the SES job** (needs a write-scoped SES
token in your interactive shell). **Pick the job file for your mode** (see
"Continuous vs One-shot" above):

- Continuous (default, for hummingbirds): `jobs/bioclip-hummingcam-h00f.yaml`
- One-shot cron (slow scenes): `jobs/bioclip-hummingcam-h00f-oneshot.yaml`

```bash
# Continuous (recommended for the bird cam — warm model, 60s sampling):
sesctl --server https://es.sagecontinuum.org --token "$SES_USER_TOKEN" \
    create -f jobs/bioclip-hummingcam-h00f.yaml   # returns a numeric job ID
sesctl --server https://es.sagecontinuum.org --token "$SES_USER_TOKEN" \
    submit -j <job-id>

# …or one-shot cron instead (swap the file):
#   create -f jobs/bioclip-hummingcam-h00f-oneshot.yaml
```

To switch an already-running job between modes: suspend + remove the old
job (`sesctl ... rm -s <id>` then `sesctl ... rm <id>`), then create +
submit the other job file.

**Step 6 — verify it fires and publishes.** The pod appears in the `ses`
namespace each tick, runs (with a cold-start model load each cycle, since
one-shot), publishes, exits, and is GC'd — invisible between ticks.
Confirm via the data API:

```bash
curl -s -X POST https://data.sagecontinuum.org/api/v1/query \
  -H 'Content-Type: application/json' \
  -d '{"start":"-15m","filter":{"vsn":"H00F","name":"env.species.species"}}'
```

The proof it's the SES job (not a leftover hand-deployed pod) is in the
record metadata: `"job": "bioclip-species-classifier-<id>"` and
`"plugin": "registry.sagecontinuum.org/beckman/bioclip-species-classifier:0.3.1"`
("already present on machine" in the pod events confirms the sideload hit).

> ⚠️ **Cold-start caveat for BioCLIP:** as a one-shot, BioCLIP 2.5 ViT-H/14
> reloads the model every cycle. At a 10-min cadence this is acceptable, but
> if you ever tighten the schedule, measure the per-cycle load time first —
> a continuous pod (warm model) may be the better trade for high frequency.

### Re-deploying after a code change (new version)

Bump the version everywhere (sage.yaml, Makefile, job YAML), then repeat
build → sideload with the new tag. Because the tag changes, k3s uses the
new local image on the next tick automatically; no job re-submit needed if
the job YAML already points at the new tag (otherwise update + re-submit).

### Systemic fix (escalate to the ECR/cyberinfra team)

The sideload workaround is manual and per-node. The durable fix is one of:

- **(a)** Grant push/write access to `registry.sagecontinuum.org/beckman/`
  for a Sage portal token, so `docker push` works after a native Thor build; or
- **(b)** Add a **native arm64 build node** to the Jenkins ECR pipeline so
  the portal "Register and Build" path works without QEMU.

Either unblocks every Thor-targeted NVIDIA plugin (yolo, bioclip, birdnet)
and removes the manual sideload step entirely.

See: https://sagecontinuum.org/docs/tutorials/edge-apps/publishing-to-ecr


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

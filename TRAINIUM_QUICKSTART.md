# Trainium Quick Start — Speech Enhancement Arena

**Last updated: May 9, 2026**  
**Neuron SDK: 2.29.1 (latest)**

Complete step-by-step guide for training and deploying the speech enhancement arena on AWS Trainium from scratch.

---

## Prerequisites

- AWS account with access to Trainium instances (trn1 family)
- Understanding of PyTorch training workflows
- S3 bucket for NEFF cache (optional but highly recommended)

---

## Part 1: Launch and Setup

### Step 1: Choose Your Instance

**For first-time compilation:**
- **trn1.32xlarge** — 32 NeuronCores, 512 GB RAM, $21.50/hr
  - Required for initial model compilation (neuronx-cc needs >32GB RAM)
  - Use this if you haven't compiled before or don't have cached NEFFs

**For inference or cached training:**
- **trn1.2xlarge** — 2 NeuronCores, 32 GB RAM, $1.34/hr
  - Sufficient once NEFFs are cached in S3
  - Good for inference/serving workloads

**Cost-optimized compilation (advanced):**
- **r7i.24xlarge** — 768 GB RAM, $6.36/hr, x86_64 (no Neuron hardware)
  - neuronx-cc is a cross-compiler; works on x86 with Neuron SDK
  - 70% cost savings vs trn1.32xlarge for compilation-only workflows
  - See "Advanced: Cross-Compilation" section below

### Step 2: Launch Instance with Neuron DLAMI

**AMI:** Search for "Deep Learning AMI Neuron PyTorch" in your region  
**Latest:** Neuron SDK 2.29.1 (as of May 2026)  
**Recommended regions:** us-west-2, us-east-1

```bash
# Example launch command (adjust key-name, subnet-id, security-group)
aws ec2 run-instances \
    --image-id ami-0xxxxxx \
    --instance-type trn1.32xlarge \
    --key-name your-key \
    --subnet-id subnet-xxxxx \
    --security-group-ids sg-xxxxx \
    --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=512}' \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=arena-trn1}]'
```

**Storage:** Minimum 512 GB for compilation artifacts + datasets + checkpoints.

### Step 3: SSH and Activate Environment

```bash
ssh -i your-key.pem ubuntu@<instance-public-ip>

# Activate the PyTorch Neuron environment (comes with DLAMI)
source /opt/aws_neuronx_venv_pytorch_2_9/bin/activate

# Verify installation
python -c "import torch; import torch_xla; import torch_neuronx; print('Neuron ready')"
```

### Step 4: Clone Repository and Install Dependencies

```bash
git clone https://github.com/scttfrdmn/speech-enhancement-arena.git
cd speech-enhancement-arena

# Install project dependencies
pip install -r requirements.txt

```

### Step 5: Configure NEFF Caching (Highly Recommended)

NEFF caching provides **24× speedup** on subsequent runs. First compile takes hours; cached runs take ~2 minutes.

```bash
# Create S3 bucket for cache (one-time setup)
aws s3 mb s3://your-neuron-cache-bucket

# Set environment variable (add to ~/.bashrc for persistence)
export NEURON_COMPILE_CACHE_URL=s3://your-neuron-cache-bucket/arena-cache/

# Verify
echo $NEURON_COMPILE_CACHE_URL
```

### Step 6: Set Compilation Flags (Optional)

For very large models, you may need to adjust compilation settings:

```bash
# Add to ~/.bashrc or export before training (if needed)
export NEURON_CC_FLAGS="--optlevel 2"
```

---

## Part 2: Training on Trainium

### Using TorchNeuron Native Backend (Recommended)

The modern PyTorch integration for Trainium uses the native `neuron` device with minimal code changes from CUDA.

```bash
# Single model test (ConvMask, 3 epochs)
python train.py \
    --model conv_mask \
    --device neuron \
    --epochs 3 \
    --batch-size 16 \
    --run-id trn1_test

# Full 4-model arena (parallel, 4 NeuronCores)
python arena.py --device neuron --epochs 30
```

**Expected behavior:**
1. Console shows: `[device] Trainium via TorchNeuron native (Neuron SDK 2.28+)`
2. First run: compilation happens during first forward pass (similar to CUDA JIT)
3. Cached runs: NEFFs load from S3, compilation skipped
4. Uses eager mode + `torch.compile` backend for optimization

**Code differences from CUDA:**
- Device: `.to('neuron')` instead of `.to('cuda')`
- Mixed precision: `torch.autocast(device_type="neuron")`
- Distributed: Works with standard FSDP/DDP (no XLA-specific changes)

**Note:** The legacy XLA backend (`--device xla`) is also supported but not recommended for new projects.

---

## Part 3: Pre-compilation (Optional but Recommended)

Pre-compile graphs before full training to avoid watching compilation during epochs:

```bash
# Compile all models with a short run to populate NEFF cache
python arena.py --device neuron --epochs 1 --num-samples 160 --run-id precompile

# Or use neuron_parallel_compile for faster parallel compilation
neuron_parallel_compile python train.py \
    --model conv_mask \
    --device neuron \
    --epochs 1 \
    --num-samples 160
```

This populates the NEFF cache. Subsequent full runs will load from cache.

---

## Part 4: Monitoring Compilation

### Check if Compilation is Stuck or Still Running

```bash
# Find latest compilation log
ls -t /tmp/ubuntu/neuroncc_compile_workdir/*/log-neuron-cc.txt | head -1

# Tail it to see progress
tail -f $(ls -t /tmp/ubuntu/neuroncc_compile_workdir/*/log-neuron-cc.txt | head -1)
```

**Signs of progress:**
- Timestamps advancing
- `/proc/<pid>/io` shows `write_bytes` increasing
- CPU at ~98% on one core (expected; neuronx-cc is single-threaded)

**Signs of stuck:**
- No new log lines for 30+ minutes
- `write_bytes` frozen AND CPU at 0%
- Kill and restart with `--optlevel 1` if not already set

---

## Part 5: Serving Inference (Streaming Demo)

Once models are trained, serve the streaming inference server:

```bash
# On the same trn1 instance (or a trn1.2xlarge with cached NEFFs)
python stream/server/inference.py \
    --checkpoint-dir checkpoints_libri \
    --device neuron \
    --port 8765 \
    --context 0.5 \
    --hop 0.1
```

**Client connection:**
```bash
# On your local machine, forward port 8765
ssh -L 8765:localhost:8765 ubuntu@<trn1-ip>

# Open browser to http://localhost:8765
```

**Or remote connection:**
- Open port 8765 in security group
- Client connects via `http://<trn1-public-ip>:8765?server=<trn1-public-ip>:8765`

---

## Part 6: Advanced — Cross-Compilation Workflow

**Goal:** Compile on cheap x86, run on cheap Trainium. Saves ~70% on compilation costs.

### Step 1: Emit HLO Graphs on trn1.2xlarge

```bash
# Launch small instance ($1.34/hr)
# Run training briefly to generate compilation artifacts
python train.py --model conv_mask --device neuron --epochs 1 --num-samples 64

# Compilation artifacts are cached locally during first run
# Package and copy off if needed for manual cache seeding
tar czf neuron_cache.tar.gz /tmp/neuroncc_compile_workdir/
scp neuron_cache.tar.gz your-workstation:~/
```

### Step 2: Compile on r7i.24xlarge (x86, $6.36/hr)

```bash
# Launch r7i.24xlarge with Neuron DLAMI (same AMI, no Neuron hardware needed)
ssh ubuntu@<r7i-ip>
source /opt/aws_neuronx_venv_pytorch_2_9/bin/activate

# Extract HLO graphs
tar xzf hlo_graphs.tar.gz

# Compile each graph
export NEURON_CC_FLAGS="--optlevel 1"
for pb in /tmp/ubuntu/neuroncc_compile_workdir/*/model.*.hlo_module.pb; do
    neuronx-cc compile \
        --framework=XLA \
        --target=trn1 \
        --output ${pb%.pb}.neff \
        $pb
done
```

### Step 3: Upload NEFFs to S3

```bash
# Upload to your cache bucket
aws s3 sync /tmp/ubuntu/neuroncc_compile_workdir/ \
    s3://your-neuron-cache-bucket/arena-cache/ \
    --exclude "*" --include "*.neff"
```

### Step 4: Run on trn1.2xlarge with Cache

```bash
# Launch trn1.2xlarge, point at cache
export NEURON_COMPILE_CACHE_URL=s3://your-neuron-cache-bucket/arena-cache/
python arena.py --device neuron --epochs 30
```

**Result:** Zero compilation, models load in ~2 min, training starts immediately.

---

## Known Gotchas (As of SDK 2.29.1)

See [TRAINIUM_NOTES.md](TRAINIUM_NOTES.md) for full details. Key ones:

1. **32GB RAM insufficient for first compile** — use trn1.32xlarge or r7i.24xlarge
2. **`--optlevel 1` required** — FFT models won't finish at default `optlevel 3`
3. **`XLA_PARAMETER_WRAPPING_THREADSHOLD=50000`** — typo in env var name is correct
4. **No `torch.unfold` on XLA** — use explicit indexing with torch.arange
5. **BiGRU split required** — use two unidirectional GRUs (already fixed in our code)
7. **Single-threaded compilation** — 98% CPU on one core is normal, not stuck

---

## Troubleshooting

### "Passing arguments as a tuple is currently unsupported"
```bash
export XLA_PARAMETER_WRAPPING_THREADSHOLD=50000
```

### "Too many instructions after unroll for function sg0000"
```bash
export NEURON_CC_FLAGS="--optlevel 1"
```


### OOM during compilation
Use trn1.32xlarge (512GB) or r7i.24xlarge (768GB) for first compile.

### Compilation stuck for hours
Check latest log: `tail -f $(ls -t /tmp/ubuntu/neuroncc_compile_workdir/*/log-neuron-cc.txt | head -1)`

---

## Cost Summary (us-west-2 pricing)

| Workflow | Instance | Duration | Cost |
|----------|----------|----------|------|
| First compile (no cache) | trn1.32xlarge | ~8-12 hrs | ~$172-258 |
| Cached training | trn1.2xlarge | ~4 hrs | ~$5.36 |
| Cross-compile | r7i.24xlarge | ~6-8 hrs | ~$38-51 |
| Inference/serving | trn1.2xlarge | per hour | $1.34/hr |

**Best practice:** Compile once on r7i.24xlarge, cache to S3, run everything else on trn1.2xlarge.

---

## Next Steps

1. **Test locally first**: `python arena.py --device cpu --epochs 5 --num-samples 500`
2. **Launch trn1.32xlarge**: Use Neuron DLAMI 2.29.1
3. **Set env vars**: `NEURON_COMPILE_CACHE_URL`, `NEURON_CC_FLAGS`, `XLA_PARAMETER_WRAPPING_THREADSHOLD`
4. **Pre-compile**: Short run to populate NEFF cache
5. **Full training**: `python arena.py --device xla --epochs 30`
6. **Serve**: `python stream/server/inference.py --device xla`

For production, migrate to cross-compile workflow and trn1.2xlarge instances.

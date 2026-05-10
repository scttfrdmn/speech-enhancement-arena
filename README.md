# Speech Enhancement Arena

**4 models. 4 GPU slices. 1 winner. You be the judge.**

A self-contained demo for the ASPIRE group workshop that trains and compares four speech enhancement architectures — simultaneously on MIG-partitioned GPUs or Trainium NeuronCores — then serves a live web demo where you speak into a microphone and hear each model's enhancement in real-time.

> **📖 New to ML hardware selection?** Read [**Hardware Selection Guide**](docs/HARDWARE_SELECTION.md) — learn why L4 spot at $0.39/hr often beats waiting 4 days for a "free" H200, and when to use Trainium vs GPU for training.

## Architecture

```
                    ┌─────────────────────────────────────────────────┐
                    │         NVIDIA G7e (--device cuda)               │
                    │  ┌──────────┬──────────┬──────────┬──────────┐  │
                    │  │MIG 24 GB │MIG 24 GB │MIG 24 GB │MIG 24 GB │  │
                    │  │ConvMask  │  CRMNet  │Attention │   GRU    │  │
                    │  └──────────┴──────────┴──────────┴──────────┘  │
                    └─────────────────────────────────────────────────┘

                    ┌─────────────────────────────────────────────────┐
                    │      Trainium (--device neuron or --device xla)  │
                    │  ┌──────────┬──────────┬──────────┬──────────┐  │
                    │  │NeuronCor │NeuronCor │NeuronCor │NeuronCor │  │
                    │  │    e0    │    e1    │    e2    │    e3    │  │
                    │  │ConvMask  │  CRMNet  │Attention │   GRU    │  │
                    │  └──────────┴──────────┴──────────┴──────────┘  │
                    └─────────────────────────────────────────────────┘
                                         │
                              arena.py orchestrates
                            4 parallel train.py processes
                                         │
                                    ┌────┴────┐
                                    │ serve.py│──→ Browser: record/upload → hear it
                                    └─────────┘
```

## The Four Models

| Model | Architecture | Params | Inspired By |
|-------|-------------|--------|-------------|
| **ConvMaskNet** | Conv encoder-decoder, magnitude mask | 1.2M | Baseline |
| **CRMNet** | Complex Ratio Mask estimator | 2.8M | Williamson & Wang (IEEE TASLP, 2017) |
| **AttentionMask** | Multi-head self-attention mask | 3.6M | SWIM (ASPIRE, 2024) |
| **GatedRecurrent** | Bidirectional GRU mask | 2.1M | Nayem & Williamson (IEEE TASLP, 2024) |

All models operate in the STFT domain, estimate masks, and reconstruct via iSTFT — exactly the ASPIRE pipeline.

## Quick Start (CPU, for testing)

```bash
pip install -r requirements.txt

# Train all 4 models (CPU, fast test)
python arena.py --device cpu --epochs 5 --num-samples 500

# Launch the live demo
python serve.py --checkpoint-dir checkpoints --device cpu

# Open http://localhost:8000 → record audio → hear the difference
```

## AWS g7e with MIG (Recommended for Research)

**Instance**: `g7e.2xlarge` (NVIDIA RTX Pro 6000, Blackwell, 96GB)  
**Cost**: $3.36/hr on-demand, ~$1.35/hr spot (May 2026 us-west-2)  
**MIG**: 4× slices (24GB each) = $0.34/hr per experiment

```bash
# Launch AWS g7e.2xlarge instance with Deep Learning AMI

# 1. Enable MIG and create 4 partitions
sudo nvidia-smi -i 0 -mig 1
sudo nvidia-smi mig -cgi 14,14,14,14 -C

# 2. Verify
nvidia-smi -L  # Should show 4x MIG 1g.24gb

# 3. Install
pip install -r requirements.txt

# 4. Launch the arena (auto-discovers MIG slices, runs 4 models in parallel)
python arena.py --device cuda --epochs 30

# 5. Serve the demo (loads best checkpoints)
python serve.py --checkpoint-dir checkpoints --device cuda
```

**Why MIG?** Run all 4 models simultaneously on one instance. Faster than sequential, same cost.

## AWS Trainium — Production Training

**Instance**: `trn1.2xlarge` (Trainium 1st gen, 2 NeuronCores, 32GB)  
**Cost**: $1.34/hr on-demand, ~$0.54/hr spot (May 2026 us-west-2)  
**Best for**: Production training of finalized models (after architecture is decided)

> **Before you launch on Trainium**, read [TRAINIUM_QUICKSTART.md](TRAINIUM_QUICKSTART.md) for setup, and [TRAINIUM_NOTES.md](TRAINIUM_NOTES.md) for compile gotchas.


```bash
# On a trn1.2xlarge or trn1.32xlarge with Neuron DLAMI (SDK 2.28+)

# 1. Activate Neuron environment
source /opt/aws_neuronx_venv_pytorch_2_9/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# Optional: persist compiled NEFFs to S3 across instances
export NEURON_COMPILE_CACHE_URL=s3://your-bucket/neuron-cache/

# 3. Launch arena on NeuronCores (eager mode)
python arena.py --device neuron --epochs 30

# 4. Or with torch.compile — Neuron compiler JIT-compiles your
#    PyTorch graph into optimized Trainium instructions.
#    First epoch is slower (compilation), rest are faster.
python arena.py --device neuron --compile --epochs 30

# 5. Serve (CPU inference is fine for the demo)
python serve.py --checkpoint-dir checkpoints --device cpu
```

### What `--device neuron` Does Differently

TorchNeuron is the native PyTorch backend for Trainium (Neuron SDK 2.28+).
It's fundamentally different from the legacy PyTorch/XLA path:

| | `--device neuron` | `--device xla` (legacy) |
|-|-------------------|------------------------|
| **Execution** | True eager mode — ops dispatch immediately | Lazy tracing — builds graph, flushes on `mark_step()` |
| **torch.compile** | Yes — Neuron compiler backend | Not supported |
| **mark_step()** | Not needed | Required after every backward pass |
| **Debugging** | Standard PyTorch debugging (pdb, print, etc.) | Graph-level only — print triggers compilation |
| **SDK version** | Neuron SDK 2.28+ (PyTorch 2.9+) | Older Neuron SDK |
| **Future** | Active investment, transitioning to native PyTorch 2.10+ | Being deprecated |

### Why torch.compile Matters for Speech Enhancement

When you pass `--compile`, PyTorch's TorchDynamo captures the forward + backward
computation graphs and hands them to the Neuron compiler, which:

1. Maps dense matrix ops (your Conv1d, GRU, attention layers) to the **tensor engine's systolic array**
2. Maps element-wise ops (activation functions, masking) to the **vector engine**
3. Fuses operations to minimize data movement between engines
4. Schedules memory access patterns optimized for your fixed-size STFT spectrograms

The regular access patterns of speech/EEG data (fixed sample rate, fixed channel count,
fixed STFT window) are ideal for the Neuron compiler — no dynamic shapes to fight.

## AWS Trainium — Legacy PyTorch/XLA Path

```bash
# For older Neuron SDK (< 2.28) that doesn't support TorchNeuron native

source /opt/aws_neuronx_venv_pytorch/bin/activate
pip install -r requirements.txt

python arena.py --device xla --epochs 30
python serve.py --checkpoint-dir checkpoints --device cpu

# NOTE: This path requires xm.mark_step() after each backward pass
# (handled automatically by the training script). Consider upgrading
# to Neuron SDK 2.28+ and using --device neuron instead.
```

## Using Real Speech Data

By default, the arena generates synthetic speech-like audio (procedural formants + noise). For better results, point it at real clean speech files:

```bash
# Download a small subset of LibriSpeech or VCTK
# Then:
python arena.py --device cuda --clean-dir /path/to/clean/wavs --epochs 30
```

## Project Structure

```
speech-enhancement-arena/
├── arena.py              # Orchestrator — launches 4 parallel training runs
├── train.py              # Single-model training script (MIG/Trainium aware)
├── serve.py              # FastAPI inference server + web UI
├── requirements.txt
├── models/
│   ├── __init__.py
│   └── architectures.py  # All 4 model architectures
├── utils/
│   ├── __init__.py
│   └── data.py           # Synthetic data generation + WAV loading
├── static/
│   └── index.html        # Web UI (mic recording, spectrograms, A/B audio)
├── logs/                  # Training logs (JSONL, read by dashboard)
└── checkpoints/           # Model checkpoints
```

## Workshop Flow

1. **Train on NVIDIA (15 min):** Run `arena.py --device cuda` — watch 4 models race on 4 MIG slices
2. **Compare (5 min):** See the scoreboard — which architecture won?
3. **Listen (10 min):** Run `serve.py` — record your voice, add noise, hear each model clean it
4. **Train on Trainium (15 min):** Run `arena.py --device neuron --compile` — same code, different silicon
5. **Compare across hardware (5 min):** NVIDIA vs. Trainium — training time, cost, quality
6. **Discuss:** Why did CRM beat magnitude masking? Why is attention slower but sometimes better? What would torch.compile + NKI kernels unlock?

## Loss Functions

- **SI-SDR** (Scale-Invariant Signal-to-Distortion Ratio) — the standard
- **Multi-Resolution STFT Loss** — spectral convergence + log magnitude at 512/1024/2048 FFT
- Combined: `loss = SI-SDR + 0.5 * MR-STFT`

## Key Design Decisions

- **STFT domain, not time domain:** Matches ASPIRE's published approach
- **All standard PyTorch ops:** No custom CUDA kernels → Trainium compatible out of the box
- **Three device backends:** `cuda` (NVIDIA/MIG), `neuron` (TorchNeuron native), `xla` (legacy)
- **torch.compile support:** On `neuron`, JIT-compiles to Trainium instructions via Neuron compiler; on `cuda`, uses inductor backend
- **No mark_step on neuron:** TorchNeuron native runs in true eager mode — the legacy `xla` path is the only one needing `xm.mark_step()`
- **Synthetic data by default:** No dataset downloads needed for the workshop
- **JSONL logging:** Training dashboard reads logs in real-time via polling
- **Web Audio API:** Mic recording works in Chrome/Firefox, no plugins needed

## Extending This

- Add your own model: implement in `models/architectures.py`, add to `MODEL_REGISTRY`
- Custom loss functions: edit `train.py` — try PESQ approximation, STOI optimization
- NKI kernels: write a custom Trainium kernel for the loss computation (see workshop Part 3)
- Scale up: use `--clean-dir` with VCTK/LibriSpeech for publication-quality results

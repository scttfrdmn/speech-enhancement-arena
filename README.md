# Speech Enhancement Arena

**4 models. 4 GPU slices. 1 winner. You be the judge.**

A self-contained demo for the ASPIRE group workshop that trains and compares four speech enhancement architectures — simultaneously on MIG-partitioned GPUs or Trainium NeuronCores — then serves a live web demo where you speak into a microphone and hear each model's enhancement in real-time.

> **📖 New to ML hardware selection?** Read [**Hardware Selection Guide**](docs/HARDWARE_SELECTION.md) — learn why L4 spot at $0.39/hr often beats waiting 4 days for a "free" H200, and when to use Trainium vs GPU for training.
>
> **📺 Want to try the demo?** Train models with `arena.py`, then run the streaming inference server (`stream/server/inference.py`) and open the browser UI — mic → enhancement → speakers, live. See the "Workshop Flow" section below.

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
                                    ┌────────────────────────────┐
                                    │ stream/server/inference.py │──→ Browser: mic → enhanced speakers (live)
                                    └────────────────────────────┘
```

## The Four Models

| Model | Architecture | Params (small / large) | Inspired By |
|-------|-------------|------------------------|-------------|
| **ConvMaskNet** | Conv U-Net encoder-decoder, magnitude mask | 5.0M / 47.5M | Baseline |
| **CRMNet** | Dilated TCN, Complex Ratio Mask | 2.8M / 25.2M | Williamson & Wang (IEEE TASLP, 2017) |
| **AttentionMask** | Transformer mask estimator | 3.6M / 33.1M | SWIM (ASPIRE, 2024) |
| **GatedRecurrent** | Deep BiGRU mask | 2.2M / 15.0M | Nayem & Williamson (IEEE TASLP, 2024) |

All models operate in the STFT (Short-Time Fourier Transform) domain: the noisy waveform is decomposed into time-frequency bins, the model predicts a per-bin mask that suppresses noise, and the masked spectrogram is inverted back to audio via iSTFT — exactly the ASPIRE pipeline. `--scale small` (n_fft=512) is the default; `--scale large` (n_fft=1024) is for production-quality benchmarks.

## Quick Start (CPU, for testing)

```bash
# Install via uv (recommended — reads pyproject.toml)
uv sync
# Or pip:  pip install -r requirements.txt

# Train all 4 models (CPU, fast test) — checkpoints written to ./checkpoints/
uv run python arena.py --device cpu --epochs 5 --num-samples 500

# Launch the live streaming demo (mic → model → speakers)
uv run python stream/server/inference.py --checkpoint-dir checkpoints --device cpu --port 8765

# Open http://localhost:8765 → click Start → speak into mic, hear enhancement
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

# 5. Serve the live streaming demo on the same instance
uv run python stream/server/inference.py --checkpoint-dir checkpoints --device cuda --port 8765
```

Open `http://<host>:8765` in a browser. Click **Start** to stream from your mic; the model runs in real time and the enhanced audio plays back through your speakers.

**Why MIG?** Run all 4 models simultaneously on one instance. Faster than sequential, same cost.

## AWS Trainium — Production Training

**Instance**: `trn1.2xlarge` (Trainium 1st gen, 2 NeuronCores, 32GB)  
**Cost**: $1.34/hr on-demand, ~$0.54/hr spot (May 2026 us-west-2)  
**Best for**: Production training of finalized models (after architecture is decided)

> ⚠️ **First-time compile on `trn1.2xlarge` will OOM.** Its 32 GB host RAM isn't enough to hold the Neuron compiler's whole-graph IR. **Compile on a cheap x86 instance (`r7i.24xlarge`, 768 GB RAM, $6.36/hr OD), upload NEFFs to S3, then run on `trn1.2xlarge`.** See [`docs/TRAINIUM_PRACTICAL_NOTES.md`](docs/TRAINIUM_PRACTICAL_NOTES.md) for the full cross-compile workflow, plus the Bedrock-Distillation and Neuron-simulator paths.
>
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

# 5. Serve the live streaming demo on Trainium
uv run python stream/server/inference.py --checkpoint-dir checkpoints --device neuron --port 8765
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
uv run python stream/server/inference.py --checkpoint-dir checkpoints --device xla --port 8765

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
├── serve.py              # Legacy file-upload demo (kept for offline A/B testing)
├── stream/
│   ├── server/inference.py   # Live streaming WebSocket demo (canonical)
│   └── client/index.html     # Streaming web UI (mic, pipeline canvas, spectrograms)
├── requirements.txt
├── models/
│   ├── __init__.py
│   └── architectures.py  # All 4 model architectures
├── utils/
│   ├── __init__.py
│   └── data.py           # Synthetic data generation + WAV loading
├── static/
│   └── index.html        # Legacy file-upload web UI (paired with serve.py)
├── logs/                  # Training logs (JSONL, per-epoch metrics + final summary)
├── checkpoints/           # Model checkpoints
└── economics.py           # Cost / right-sizing analysis for arena runs
```

## Documentation

Audience-facing guides live in `docs/`. Suggested reading order:

- [`HARDWARE_SELECTION.md`](docs/HARDWARE_SELECTION.md) — top-level guide for picking AWS hardware. Start here.
- [`ACADEMIC_HARDWARE_GUIDE.md`](docs/ACADEMIC_HARDWARE_GUIDE.md) — researcher-focused decision frameworks, common mistakes, real-talk FAQ.
- [`GPU_SELECTION_GUIDE.md`](docs/GPU_SELECTION_GUIDE.md) — decision tree for L4 vs L40S vs RTX Pro 6000.
- [`WHEN_TO_USE_PREMIUM_HARDWARE.md`](docs/WHEN_TO_USE_PREMIUM_HARDWARE.md) — when H100 / Trainium actually pay off vs. L4.
- [`CORE_PRINCIPLES.md`](docs/CORE_PRINCIPLES.md) — why bigger isn't always better for research.
- [`SPECS_VS_REALITY.md`](docs/SPECS_VS_REALITY.md) — why peak FLOPS don't predict real training speed.
- [`RESEARCH_VS_PRODUCTION_MODELS.md`](docs/RESEARCH_VS_PRODUCTION_MODELS.md) — when to optimize for time-to-publication vs. cost-at-scale.
- [`COMPILATION_FAQ.md`](docs/COMPILATION_FAQ.md) — why Trainium compilation takes hours (and CUDA doesn't).
- [`TRAINIUM_PRACTICAL_NOTES.md`](docs/TRAINIUM_PRACTICAL_NOTES.md) — three things that change the Trainium cost picture (Bedrock alternative, cross-compile on x86, Neuron simulator).
- [`TRAINIUM_QUICKSTART.md`](TRAINIUM_QUICKSTART.md) — step-by-step Trainium setup.
- [`TRAINIUM_NOTES.md`](TRAINIUM_NOTES.md) — pre-flight checklist and compile gotchas.
- [`BENCHMARK_RESULTS.md`](BENCHMARK_RESULTS.md) — measured throughput, SI-SDR, and cost across hardware.

## Workshop Flow

1. **Train on NVIDIA (15 min):** Run `arena.py --device cuda` — watch 4 models race on 4 MIG slices
2. **Compare (5 min):** See the scoreboard — which architecture won?
3. **Listen (10 min):** Run `stream/server/inference.py` — speak into the mic, hear each model clean it in real time.
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
- **JSONL logging:** Per-epoch metrics streamed to `logs/arena_*.jsonl`; `arena.py` parses them live to drive the terminal scoreboard
- **WebSocket streaming:** Browser mic captured at 48 kHz, resampled to 16 kHz, sent to the inference server in 100 ms hops; enhanced PCM streams back on the same WebSocket

## Extending This

- Add your own model: implement in `models/architectures.py`, add to `MODEL_REGISTRY`
- Custom loss functions: edit `train.py` — try PESQ approximation, STOI optimization
- NKI kernels: write a custom Trainium kernel for the loss computation (see workshop Part 3)
- Scale up: use `--clean-dir` with VCTK/LibriSpeech for publication-quality results

# Speech Enhancement Arena — Benchmark Results & Analysis

**Date:** April 11, 2026
**Models:** 4 architectures (20-50M params), STFT-domain speech enhancement
**Config:** 30 epochs, 5K synthetic samples, 3-second audio, batch_size=16, AMP (CUDA), n_fft=1024

---

## 1. Hardware Tested

| Hardware | GPU | VRAM | CPU | Real Cores | $/hr OD | $/hr Spot |
|---|---|---|---|---|---|---|
| **RTX 5090** (ceres, local) | GeForce RTX 5090 | 32GB | Ryzen 9 9950X3D | 16 | $0.80 amortized | — |
| **g7e.2xlarge** (AWS) | RTX PRO 6000 Blackwell | 96GB | Xeon 8559C | 4 | $3.36 | ~$1.35 |
| **g7e.8xlarge** (AWS) | RTX PRO 6000 Blackwell | 96GB | Xeon 8559C | 16 | ~$8.15 | — |
| **g7e.2xlarge MIG** (AWS) | 4x MIG 1g.24GB | 4x24GB | Xeon 8559C | 4 | $3.36 | ~$1.35 |
| **g6e.xlarge** (AWS) | L40S | 48GB | — | 2 | $1.86 | $0.75 |
| **g6.xlarge** (AWS) | L4 | 24GB | — | 2 | $0.98 | $0.39 |
| **g5.xlarge** (AWS) | A10G | 24GB | — | 2 | $1.21 | $0.48 |
| **M4 Pro** (orion, local) | Apple MPS | Unified 48GB | Apple M4 Pro | 14 | $0.80 amortized | — |
| **trn1.2xlarge** (AWS) | Trainium1 | 32GB | — | 4 | $1.34 | $0.54 |

---

## 2. Model Architectures

| Model | Params | Architecture | Description |
|---|---|---|---|
| **ConvMaskNet** | 47.5M | U-Net encoder-decoder with skip connections | Magnitude mask, baseline |
| **CRMNet** | 25.2M | Dilated TCN with residual blocks | Complex Ratio Mask (Williamson & Wang 2017) |
| **AttentionMask** | 33.1M | 10-layer Transformer + positional encoding | SWIM-inspired (ASPIRE 2024) |
| **GatedRecurrent** | 19.7M | 4-layer BiGRU with Conv pre/post-nets | Nayem & Williamson style |

All models operate in the STFT domain with n_fft=1024, estimate masks, and reconstruct via iSTFT.

---

## 3. Training Results (Best Loss / SI-SDR)

ConvMaskNet consistently wins across all hardware. Model quality is hardware-independent (as expected — same math, different speed).

| Model | RTX 5090 | g7e Full | g7e MIG | L40S | L4 | A10G |
|---|---|---|---|---|---|---|
| ConvMask | -19.51 | -19.39 | -19.40 | -19.34 | -19.52 | -19.31 |
| CRM | -18.01 | -18.14 | -17.96 | -17.78 | -17.81 | -17.90 |
| Attention | -15.02 | -6.59* | -10.77* | -10.12* | -12.69 | -7.05* |
| GRU | -19.04 | -18.68 | -19.07 | -18.67 | -18.71 | -18.96 |

*Attention model convergence varied across runs — likely AMP precision sensitivity on certain hardware, not a hardware defect.

---

## 4. GPU & System Utilization

| Hardware | GPU Compute Util | VRAM Peak | VRAM % Used | CPU Load | Throughput (samp/s) |
|---|---|---|---|---|---|
| **RTX 5090** | **79-81%** | 1.5GB / 32GB | 4.4% | 2-3% | 633-1156 |
| **L4** | **77-86%** | 1.4GB / 24GB | 4.2-6.0% | 24-25% | 247-308 |
| **A10G** | **70-82%** | 1.4GB / 24GB | 3.1-6.0% | 24-25% | 234-327 |
| **g7e.2xlarge** (full) | 56-57% | 1.4GB / 96GB | 1.0-1.4% | 12.5% | 327-704 |
| **g7e.8xlarge** (full) | 56-63% | 1.4GB / 96GB | 0.7-1.4% | 3.1% | 329-697 |
| **L40S** | 48-56% | 1.4GB / 48GB | 1.5-3.0% | 24-25% | 312-482 |

### Key Utilization Insights

1. **Smaller GPUs = higher utilization.** L4 (24GB) at 83%, vs g7e (96GB) at 56%. Right-sizing matters.
2. **VRAM is massively wasted everywhere.** Peak 1.5GB on GPUs with 24-96GB. These models need a 4-8GB GPU.
3. **CPU is NOT the bottleneck.** g7e.8xlarge (16 cores) had same GPU util as g7e.2xlarge (4 cores). Adding 4x CPU cores changed nothing.
4. **The bottleneck is GPU clock speed** — see Section 6.

---

## 5. Cost Analysis

### Cost Per Result (4-model sweep)

This is the metric that matters — not $/hr, but **$/answer**.

| Hardware | Mode | Wall Clock | Raw $/hr | Eff. $/hr* | **Cost/Run** |
|---|---|---|---|---|---|
| **L4 Spot** | serial | 37 min | $0.39 | $0.47 | **$0.24** |
| **g7e MIG Spot** | 4-way parallel | 9.3 min | ~$1.35 | $2.41 | $0.21 |
| **A10G Spot** | serial | 36 min | $0.48 | $0.63 | $0.29 |
| **RTX 5090** (40hr/wk) | serial | 11 min | $0.80 | $1.00 | $0.18 |
| **L40S Spot** | serial | 27 min | $0.75 | $1.44 | $0.34 |
| **g7e OD** (full) | serial | 20 min | $3.36 | $6.00 | $1.12 |
| **M4 Pro** (40hr/wk) | serial | 161 min | $0.80 | — | $2.15 |
| **OSC A100** | serial | 2-8hr wait + 33min | "$0" | hidden | $52-416** |

*Effective $/hr = raw price / GPU utilization %
**OSC cost = student wait time at $52/hr fully-loaded RA cost

### Local Hardware TCO (RTX 5090 system, $5,000, 3-year life)

| Usage Pattern | Amortized $/hr | Effective $/hr (at 80% GPU util) |
|---|---|---|
| 40 hr/week (heavy) | $0.80 | $1.00 |
| 20 hr/week (moderate) | $1.60 | $2.00 |
| 10 hr/week (light) | $3.21 | $4.01 |
| 5 hr/week (occasional) | $6.41 | $8.01 |

At <15 hr/week usage, the L4 Spot at $0.47/effective-hr is cheaper than owning.

---

## 6. Why RTX 5090 Beats RTX PRO 6000 (Root Cause Analysis)

Same Blackwell architecture, same 600W TDP, but consumer card is 1.6-2x faster:

| Spec | RTX 5090 (consumer) | RTX PRO 6000 (server) |
|---|---|---|
| **GPU clock** | **3135 MHz** | 2430 MHz |
| **Memory clock** | **14001 MHz** | 12481 MHz |
| VRAM | 32GB | 96GB |
| PCIe | Gen4 x16 (motherboard limited) | Gen5 x16 |

### Microbenchmark Results

| Test | RTX 5090 | RTX PRO 6000 | Ratio |
|---|---|---|---|
| CPU FFT (1000x 48K samples) | 8237 FFTs/s | 8623 FFTs/s | 1.0x (same) |
| **GPU FFT** (1000x 48K samples) | **71,972 FFTs/s** | 36,821 FFTs/s | **2.0x** |
| **Data gen** (16 samples on GPU) | **1283 samp/s** | 392 samp/s | **3.3x** |

**Conclusion:** CPU performance is identical. GPU clock speed (29% higher on consumer card) drives 2x FFT throughput. The RTX PRO 6000 is intentionally downclocked for datacenter thermal/reliability requirements. More VRAM, enterprise drivers, ECC — but lower single-GPU throughput.

**Adding more CPU cores did not help.** g7e.8xlarge (16 real cores) produced identical GPU utilization (56-63%) as g7e.2xlarge (4 real cores). CPU load dropped from 12.5% to 3.1% — proving the CPU was never the bottleneck.

---

## 7. Trainium Compatibility

**STFT-domain speech enhancement is incompatible with Trainium.**

The Neuron compiler (NCC) does not support complex data types:
```
[ERROR] [NCC_EVRF004] Complex data types are not supported but found to be used 
with operator multiply.
```

Both the data generation (`torch.fft.rfft/irfft`) and the model forward pass (`torch.stft/istft`, `torch.polar`) use complex tensors. This is fundamental to STFT-domain processing.

**Workarounds under investigation:**
- Real-valued STFT decomposition (separate real/imag channels)
- Custom NKI kernels for STFT/iSTFT on Trn2
- Time-domain model architectures (Conv-TasNet style) that avoid complex ops entirely

---

## 8. MIG Analysis (g7e.2xlarge)

### Full GPU vs MIG (same instance, same workload)

| Mode | Training Time (sum) | Wall Clock | Speedup | Cost (OD) |
|---|---|---|---|---|
| Full GPU (serial) | 1218s | 1218s (20 min) | 1.0x | $1.12 |
| **MIG 4x24GB (parallel)** | 1960s | **555s (9.3 min)** | **2.2x** | **$0.52** |

MIG delivers 2.2x wall-clock speedup (not 4x due to GRU bottleneck — sequential BiGRU dominates parallel wall time).

### MIG improves VRAM utilization

| Mode | VRAM per "GPU" | Model VRAM Usage | Utilization |
|---|---|---|---|
| Full GPU | 96GB | 1.4GB | 1.4% |
| MIG slice | 24GB | 1.4GB | 5.8% |

MIG is the right-sizing solution for datacenter GPUs — it turns one 96GB GPU into four 24GB GPUs that better match small-model workloads.

### When MIG makes sense

- You have **4+ experiments to run simultaneously** (lab with multiple students)
- You need **fastest wall-clock time** for a sweep
- The models are **small enough to fit in a MIG slice** (24GB for 1g profile)

### When MIG doesn't help

- Single experiment (full GPU is faster per model)
- Models that need >24GB VRAM
- OSC or similar HPC centers that don't configure MIG

---

## 9. The OSC Problem

Ohio Supercomputing Center has ~984 GPUs (A100s, H100s, V100s) serving all Ohio research universities. OSC reports ~30% average GPU utilization.

### Why 30% utilization + long queues coexist

| What happens | Why |
|---|---|
| Jobs are **allocated** full GPUs | Slurm assigns whole GPUs, not slices |
| Jobs **use** 1.5GB of 40-80GB VRAM | Speech enhancement models are small |
| Jobs hold allocation for **1-4 hours** | Default Slurm time limits |
| GPU is **70% idle** during allocation | No MIG, no right-sizing |
| Queue shows **full** | All GPUs allocated (even if barely used) |
| Researchers **wait hours** | Queue depth, not actual compute demand |

### The math

- 984 GPUs at 30% utilization = **295 GPUs of actual work**
- With MIG (7 slices per A100): 984 × 7 = **6,888 effective instances** for small models
- Queue wait times would collapse

### What it costs

ASPIRE lab (10 grad students), estimating 2hr/week average queue wait:
- 10 students × 2 hr/wk × 52 wk × $52/hr (fully-loaded RA cost) = **$54,080/year in lost productivity**
- Cloud compute to eliminate all wait: **~$2,500/year**

---

## 10. The Effective Cost Framework

### Stop measuring $/GPU-hour. Measure $/result.

| Metric | What it measures | Who it serves |
|---|---|---|
| $/GPU-hr | Hardware cost | Procurement, facilities |
| $/effective-GPU-hr | Cost adjusted for utilization | Financial planning |
| **$/result** | Cost to get an answer | **The researcher** |
| **Time-to-result** | Wall clock from submit to answer | **The researcher** |

### The hidden costs nobody reports

| Cost | Visible? | OSC | Cloud |
|---|---|---|---|
| Hardware purchase | No (federal/state) | $10M+ | $0 |
| Operations (power, cooling, staff) | No (overhead) | $5M+/yr | $0 |
| GPU-hour charge to researcher | Barely | ~$0.50 | $0.39-3.36 |
| Queue wait (researcher time) | **No** | **$52/hr** | **$0** |
| VRAM waste (overprovisioned) | No | 94-99% | 94-96% |
| Compute waste (low utilization) | No | 70% | 17-48% |
| **Invoice** | — | None | **Yes** |

Cloud's "disadvantage" is that every cost is visible on the invoice. OSC's "advantage" is that the real costs are hidden across federal grants, state budgets, and student time.

**The cloud invoice is a feature, not a bug.** It enables informed decisions.

---

## 11. The Resource Portfolio

No single resource is optimal for everything. The right approach is a portfolio:

| Task | Best Resource | Why |
|---|---|---|
| **Write code, debug** | Laptop (any CPU) | Fast iteration, zero cost |
| **Verify training works** | Laptop MPS or CPU | Our models run on M4 Pro |
| **Quick validation** | Local RTX 5090 (sunk cost) | If idle, zero marginal cost |
| **Production sweep (4+ experiments)** | Cloud g7e MIG or L4 Spot | Parallel results, $0.21-0.24 |
| **Deadline crunch (10 students)** | Cloud (any) | M/M/infinity — everyone gets a GPU instantly |
| **Foundation model (200hr dataset)** | Cloud p5 H100 or OSC | Actually needs the big iron |

### Cloud doesn't replace anything — it fills the gaps

- OSC is great **when you can get on it** — keep it for jobs that need 4x A100 NVLink
- Local GPU is great **for one person** — keep it for daily development
- Cloud eliminates the **queue wait** and **scaling constraint**

### The queueing theory argument

- **1 local GPU** = M/M/1 queue. At >70% utilization, wait times explode.
  - 10 students, 3 jobs/hr → ρ=0.55. Average wait: 13 min on top of 11 min run.
  - During deadlines: ρ=0.95. Average wait: **over 1 hour**.
- **Cloud** = M/M/infinity. Every arrival gets a server. Wait time = 0. Always.
- **OSC** = M/M/c with c=984 but huge demand. Wait times are hours despite massive capacity.

### Power and scaling constraints (local hardware)

| Constraint | Local RTX 5090 | Cloud |
|---|---|---|
| Power draw | 500-600W sustained | Their problem |
| Circuit requirement | 15-20A dedicated | None |
| Max systems per 20A circuit | 1 | Unlimited |
| Scaling to 10 simultaneous users | Impossible | Click a button |

You **cannot** horizontally scale local GPU systems without upgrading electrical infrastructure. Cloud scales instantly.

---

## 12. Recommendations for ASPIRE Lab

**Prof. Williamson's group:** 10 grad students + 1 visiting student, STFT-domain speech enhancement, NSF-funded.

### Immediate actions

1. **Try L4 Spot ($0.39/hr)** for routine experiments — highest utilization, lowest cost, 37 min per 4-model sweep
2. **Try g7e MIG ($0.52/run)** for deadline-critical sweeps — 9.3 min wall clock for 4 parallel experiments
3. **Keep using OSC** for large distributed training that actually needs multi-GPU NVLink
4. **Students develop on laptops** — CPU is fine for code/debug, MPS for validation

### Budget impact

| Item | Monthly Cost |
|---|---|
| 40 experiments/day × L4 Spot | ~$200 |
| 10 deadline sweeps/month × g7e MIG | ~$5 |
| OSC allocation | $0 (existing) |
| **Total cloud augmentation** | **~$205/month** |
| **Recovered productivity** (eliminated queue wait) | **~$4,500/month** |

### What NOT to do

- Don't buy multiple local GPU systems (power/scaling constraints)
- Don't run 1.5GB-VRAM models on 80GB A100s at OSC (96% VRAM waste)
- Don't optimize for $/GPU-hr (optimize for $/result)
- Don't treat student time as free (it costs $52/hr fully loaded)
- Don't try Trainium for STFT-domain work (complex ops not supported — yet)

---

## 13. Future Work

1. **Real speech data** — Rerun benchmarks with LibriSpeech train-clean-100 instead of synthetic data (already supported: `--dataset librispeech`)
2. **Trainium STFT** — Develop real-valued STFT wrapper or NKI kernels for Trn2 complex op support
3. **p5.4xlarge** (single H100 with MIG) — Test 7-way MIG for maximum parallelism
4. **MIG vs full-GPU comparison script** — Automate the back-to-back comparison
5. **AI workload advisor** — Given model code, recommend optimal hardware and estimate cost-per-result
6. **Attention convergence investigation** — Debug AMP precision issue on certain GPUs

---

## Appendix: Raw Instance Pricing (us-west-2, April 2026)

| Instance | GPU | VRAM | vCPU | $/hr OD | $/hr Spot |
|---|---|---|---|---|---|
| g6.xlarge | 1x L4 | 24GB | 4 | $0.98 | $0.39 |
| g5.xlarge | 1x A10G | 24GB | 4 | $1.21 | $0.48 |
| trn1.2xlarge | 1x Trainium1 | 32GB | 8 | $1.34 | $0.54 |
| g6e.xlarge | 1x L40S | 48GB | 4 | $1.86 | $0.75 |
| g7e.2xlarge | 1x RTX PRO 6000 | 96GB | 8 | $3.36 | ~$1.35 |
| g7e.8xlarge | 1x RTX PRO 6000 | 96GB | 32 | ~$8.15 | — |
| p5.4xlarge | 1x H100 | 80GB | 16 | ~$7-8 | ~$3 |
| p5.48xlarge | 8x H100 | 8x80GB | 192 | $98.32 | ~$39 |

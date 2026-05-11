# Hardware Selection Guide: Research vs Production Training

**Last Updated: May 9, 2026**  
**Pricing: AWS us-west-2 as of May 2026**

This guide helps you choose the right hardware for different phases of ML research and production. Based on real benchmarks from the speech enhancement arena.

---

## TL;DR

- **Research iteration**: AWS g7e MIG or L4 spot — fast feedback, no queues, elastic parallelism
- **Production training**: AWS Trainium (trn1) — lower $/FLOP for finalized models at scale
- **Inference**: AWS Inferentia (inf2) or Trainium — optimized for serving

**Don't use**: Shared H100/H200 clusters with multi-day queues for iterative research.

---

## The Real Bottleneck: Time to Results

### What Actually Happens at Research Institutions

**Scenario: University with premium GPU cluster**

Hardware:
- 32× NVIDIA H200 GPUs (Hopper, 144GB HBM3e)
- $100M+ investment
- 500 graduate students and researchers
- Average queue wait: **3-5 days**

**PhD Student Experience:**

```
Monday 9am:     Submit experiment to queue
                Position: 47th in line
                Estimated wait: 4 days

Friday 11am:    Job starts, runs for 18 minutes
                Result: Found a bug in data loader

Friday 11:30am: Fix bug, resubmit
                Position: 52nd in line  
                Estimated wait: 4 days

[Next] Thursday: Job completes
                 Need to try different learning rate

Thursday 3pm:   Resubmit with new LR
                Position: 61st in line

[14 DAYS LATER]

Three experiments complete.
Conference deadline missed.
```

**Time for 3 iterations: 14+ days**

### Alternative: Cloud Spot Instances

**Same researcher, AWS spot instances**

Hardware:
- AWS g6.xlarge (NVIDIA L4, Ada Lovelace, 24GB)
- $0.39/hr spot pricing
- No queue, launches in 30 seconds

```
Monday 9am:     Launch instance, run experiment
                Completes in 35 minutes
                Find data loader bug

Monday 9:40am:  Fix bug, relaunch immediately
                Completes in 35 minutes
                Try different learning rate

Monday 10:20am: Relaunch with new LR
                Completes in 35 minutes

Monday 11am:    Three experiments complete
                Cost: $0.58 total
```

**Time for 3 iterations: 2 hours**

**The cloud is 168× faster to results** despite using "slower" hardware.

---

## Hardware Comparison: May 2026

### AWS GPU Instances for ML Training

| Instance | GPU | Arch | VRAM | Spot $/hr | OD $/hr | **Best For** |
|----------|-----|------|------|-----------|---------|--------------|
| **g7e.2xlarge** | RTX Pro 6000 | Blackwell | 96GB | $1.35 | $3.36 | Large models, single GPU |
| **g7e.2xlarge (MIG)** | 4× MIG 1g.24gb | Blackwell | 4×24GB | $1.35 ($0.34/slice) | $3.36 | **Parallel experiments** |
| **g6e.xlarge** | L40S | Ada | 48GB | $0.75 | $1.86 | Medium models, good VRAM |
| **g6.xlarge** | L4 | Ada | 24GB | $0.39 | $0.98 | **Research iteration** |
| **g5.xlarge** | A10G | Ampere | 24GB | $0.48 | $1.21 | Legacy workloads |
| **p5.48xlarge** | 8× H100 | Hopper | 8×80GB | ~$20 | $98.32 | Multi-GPU training |
| **p4d.24xlarge** | 8× A100 | Ampere | 8×40GB | ~$12 | $32.77 | Multi-GPU (older) |

### AWS Custom Silicon

| Instance | Accelerator | Cores | RAM | Spot $/hr | OD $/hr | **Best For** |
|----------|-------------|-------|-----|-----------|---------|--------------|
| **trn1.2xlarge** | Trainium (1st gen) | 2 NC | 32GB | $0.54 | $1.34 | Inference, cached training |
| **trn1.32xlarge** | Trainium (1st gen) | 32 NC | 512GB | $8.60 | $21.50 | First-time compilation |
| **inf2.xlarge** | Inferentia2 | 2 NC | 32GB | $0.30 | $0.76 | **Production inference** |
| **inf2.48xlarge** | Inferentia2 | 96 NC | 384GB | $9.00 | $22.40 | Large-scale inference |

**NC = NeuronCore** (Trainium/Inferentia compute unit)

### Reference: Local Hardware

| Hardware | Arch | VRAM | Approx Cost | **Best For** |
|----------|------|------|-------------|--------------|
| RTX 4090 | Ada | 24GB | $2,000 | Local debugging |
| RTX 5090 | Blackwell | 32GB | $2,500 | Local development |

**Amortized cost**: ~$0.20/hr (assumes 3-year lifespan, 8hr/day usage)

---

## The MIG Advantage: Parallel Experiments on One Instance

**MIG (Multi-Instance GPU)** partitions a single physical GPU into isolated slices.

### AWS g7e.2xlarge with MIG

**One g7e.2xlarge instance** ($1.35/hr spot) gives you:
- 4× isolated MIG slices (1g.24gb each)
- 24GB VRAM per slice
- Full isolation (no interference between slices)
- **Run 4 experiments in parallel for $0.34/hr per experiment**

### Example: 4-Model Arena Benchmark

**Without MIG** (sequential):
```bash
# Run 4 models one at a time on g6.xlarge (L4)
ConvMask:   40 min × $0.39/hr = $0.26
CRM:        40 min × $0.39/hr = $0.26  
Attention:  40 min × $0.39/hr = $0.26
GRU:        40 min × $0.39/hr = $0.26

Total time: 160 minutes
Total cost: $1.04
```

**With MIG** (parallel):
```bash
# Run all 4 models simultaneously on g7e.2xlarge MIG
All 4 models: 40 min × $1.35/hr = $0.90

Total time: 40 minutes
Total cost: $0.90
```

**4× faster for 15% less cost**, plus you get Blackwell architecture.

### When to Use MIG:

✅ **Yes - Use MIG for:**
- Running multiple small models in parallel
- Hyperparameter sweeps (each config on one slice)
- Multi-model comparisons (like this arena)
- Shared team environments (give each person a slice)

❌ **No - Don't use MIG for:**
- Models that need >24GB VRAM per run
- Multi-GPU training (use p5.48xlarge instead)
- When you need the full 96GB on one model

---

## Cost Analysis: 10 Experiments

**Scenario**: You need to run 10 full training experiments (hyperparameter sweep).

### Option 1: University H200 Cluster (Shared)

- **Hardware**: NVIDIA H200 (Hopper, 144GB), shared cluster
- **Queue wait**: 4 days average
- **Job time**: 15 minutes per experiment
- **Total time**: 10 experiments × (4 days + 15 min) ≈ **40+ days**
- **Cost to you**: $0 (institution pays)
- **Cost of your time**: 40 days of blocked research

### Option 2: AWS g7e MIG (Blackwell, 24GB per slice)

- **Hardware**: g7e.2xlarge with 4× MIG slices
- **Launch time**: 30 seconds
- **Job time**: 35 minutes per experiment
- **Parallelism**: 4 slices = 4 experiments at once
- **Total time**: 3 batches × 35 min = **105 minutes**
- **Cost**: 1.75 hours × $1.35/hr = **$2.36**

### Option 3: AWS L4 Spot (Ada, 24GB)

- **Hardware**: g6.xlarge (NVIDIA L4)
- **Launch time**: 30 seconds  
- **Job time**: 40 minutes per experiment
- **Parallelism**: Launch 10 instances
- **Total time**: **40 minutes**
- **Cost**: 10 instances × 0.67 hours × $0.39/hr = **$2.61**

### Option 4: Local RTX 4090 (2 GPUs)

- **Hardware**: 2× RTX 4090 (24GB each)
- **Setup time**: 0 (already owned)
- **Job time**: 25 minutes per experiment
- **Parallelism**: 2 GPUs
- **Total time**: 5 batches × 25 min = **125 minutes**
- **Cost**: $0 (amortized over 3 years)

### Comparison Table

| Option | Time to Results | Direct Cost | **Time Value** | **Best For** |
|--------|-----------------|-------------|----------------|--------------|
| H200 cluster | 40+ days | $0 | **$48,000** @ $50/hr | Never (if queued) |
| g7e MIG | 105 min | $2.36 | $87 | Team with budget |
| L4 spot (parallel) | 40 min | $2.61 | $33 | **Research iteration** |
| Local 4090s | 125 min | $0 | $104 | Solo researcher |

**Time value** = hours blocked × $50/hr (grad student cost)

**The L4 parallel approach gives you results 1,440× faster than the queue** for $2.61.

---

## The Queue Problem: Why "Best" Hardware Slows You Down

### Utilization vs Throughput

Universities optimize for **GPU utilization** (% of time GPU is busy).  
Researchers need **research throughput** (experiments per day).

**These are opposite goals.**

**H200 cluster at "95% utilization":**
- Every GPU is busy 24/7 (admins happy)
- Every researcher waits 4 days (researchers blocked)
- Total experiments per day: LOW

**L4 cluster at "30% utilization":**
- Most GPUs idle most of the time (admins concerned)
- Zero researchers wait (researchers productive)
- Total experiments per day: HIGH

### The Formula That Actually Matters

```
Research velocity = experiments per day ÷ days blocked
```

Not:
```
Research velocity = peak FLOPS
```

An H200 has more FLOPS than an L4. But if you wait 4 days to use it, the L4 gives you results **96× faster** (4 days vs 1 hour).

---

## Recommendations by Research Phase

### Phase 1: Model Architecture Exploration

**Goal**: Try different architectures, layer counts, attention mechanisms, etc.  
**Need**: Fast iteration (minutes between experiments)

**Recommended hardware:**
1. **AWS g6.xlarge** (L4, 24GB) — $0.39/hr spot
   - Launch instantly, no queue
   - 24GB sufficient for most research models
   - Scale to 10+ instances for parallel sweeps

2. **AWS g7e.2xlarge MIG** (Blackwell, 4×24GB) — $0.34/hr per slice
   - Latest architecture
   - Run 4 experiments in parallel
   - Better $/performance than L4

3. **Local RTX 4090/5090** (24-32GB) — if you own it
   - Zero latency (no SSH)
   - Good for quick debugging
   - Limited to 1-2 GPUs

**Don't use:**
- ❌ Shared cluster with >1 day queue
- ❌ H100/H200 (overkill for exploration)
- ❌ Trainium (compilation overhead kills iteration speed)

### Phase 2: Hyperparameter Tuning

**Goal**: Fixed architecture, sweep learning rates, batch sizes, etc.  
**Need**: Parallel execution, cost efficiency

**Recommended hardware:**
1. **AWS L4 spot in parallel** (10× g6.xlarge) — $3.90/hr total
   - Launch 10 instances
   - Run 10 configs simultaneously
   - $0.39/hr per experiment

2. **AWS g7e MIG** (multiple instances) — scalable
   - Each instance = 4 parallel experiments
   - Blackwell performance
   - Clean isolation

**Don't use:**
- ❌ Sequential execution on single GPU (10× slower)
- ❌ Premium instances (A100/H100) for small sweeps

### Phase 3: Final Training Run (Production)

**Goal**: Train finalized model on full dataset for days/weeks  
**Need**: Cost efficiency, lower $/FLOP

**Recommended hardware:**
1. **AWS trn1.2xlarge** (Trainium, 2 NeuronCores) — $0.54/hr spot
   - Compile once (2-8 hours on trn1.32xlarge)
   - Cache NEFFs to S3
   - Training cost 40-60% lower than GPU
   - Best for: models that will train for >20 hours

2. **AWS g6e.xlarge** (L40S, 48GB) — $0.75/hr spot
   - No compilation overhead
   - Good VRAM for larger models
   - Best for: 10-20 hour training runs

3. **Reserved p5.48xlarge** (8× H100) — for multi-GPU
   - Reserved capacity = no queue
   - Best for: models that need 80GB+ VRAM or multi-node training

**When to use Trainium:**
- ✅ Model architecture is frozen
- ✅ Training run >20 hours
- ✅ You have S3 NEFF cache populated
- ❌ Still experimenting with model design

### Phase 4: Production Inference

**Goal**: Serve finalized model to users at scale  
**Need**: Lowest $/inference, high throughput

**Recommended hardware:**
1. **AWS inf2.xlarge** (Inferentia2, 2 NeuronCores) — $0.30/hr spot
   - Purpose-built for inference
   - Lowest cost per inference
   - Compile once, serve millions

2. **AWS trn1.2xlarge** (Trainium) — $0.54/hr spot
   - If you already have Trainium NEFFs cached
   - Training + inference on same silicon

3. **AWS g6.xlarge** (L4) — $0.39/hr spot
   - If you need CUDA ecosystem
   - No compilation overhead

**Cost comparison** (1M inferences):
- Inferentia2: ~$8
- L4: ~$12
- H100: ~$45

---

## Case Study: Speech Enhancement Arena

This repository's benchmarks provide real-world data across hardware platforms.

### Models Tested

- **ConvMaskNet**: 47.5M params, U-Net encoder-decoder
- **CRMNet**: 25.2M params, Complex Ratio Mask
- **AttentionMask**: 33.1M params, 10-layer Transformer
- **GatedRecurrent**: 19.7M params, 4-layer BiGRU

All models: STFT-domain, n_fft=1024, 30 epochs, batch_size=16

### Training Time & Cost (Single Model)

| Instance | GPU/Accelerator | Time | Spot Cost | **$/Model** |
|----------|-----------------|------|-----------|-------------|
| g7e.2xlarge | RTX Pro 6000 (Blackwell, 96GB) | 2.3h | $1.35/hr | $3.11 |
| g7e.2xlarge MIG | 4× MIG 1g.24gb (Blackwell) | 2.5h | $0.34/slice | $0.85 |
| g6e.xlarge | L40S (Ada, 48GB) | 3.0h | $0.75/hr | $2.25 |
| g6.xlarge | L4 (Ada, 24GB) | 4.0h | $0.39/hr | $1.56 |
| Local Mac | M4 Pro MPS (48GB unified) | 2.7h | $0/hr | $0 |
| trn1.32xlarge | Trainium (32 NC, first compile) | 9.5h | $8.60/hr | $81.70 |
| trn1.2xlarge | Trainium (2 NC, cached) | 4.0h | $0.54/hr | $2.16 |

### Key Findings

1. **L4 spot is the research sweet spot**: $1.56 per model, 4 hours, instant availability

2. **MIG enables parallelism**: g7e.2xlarge with 4 slices trains all 4 models in 2.5 hours for $3.40 total (vs 16 hours sequential on L4)

3. **Trainium first compile is expensive**: $82 including compilation time — only worth it if you'll reuse the cache

4. **Trainium cached is competitive**: $2.16 per model, similar to L40S, lower than g7e

5. **Local M4 Pro is free but limited**: Can't scale beyond 1 GPU

### Recommendations from This Benchmark

**For this workload (speech enhancement models):**

- **Development**: L4 spot ($0.39/hr) — fast enough, dirt cheap
- **Parallel testing**: g7e MIG ($1.35/hr for 4 experiments)
- **Production training**: Trainium cached ($0.54/hr) — if training many models
- **One-off jobs**: L4/L40S — compilation overhead not worth it

---

## Common Mistakes to Avoid

### ❌ Mistake 1: Using Premium Hardware for Small Models

**Bad**: Training 2GB model on H100 (80GB VRAM)
- 98% VRAM wasted
- Paying for capacity you don't use
- Blocking others from using the H100

**Good**: Training 2GB model on L4 (24GB VRAM)
- Right-sized hardware
- 10× cheaper
- H100 free for someone who needs it

### ❌ Mistake 2: Waiting in Queue for "Better" Hardware

**Bad**: Wait 4 days for H200, job runs in 15 minutes
- Total time: 4 days 15 minutes
- Your time wasted: 96 hours

**Good**: Launch L4 spot immediately, job runs in 30 minutes
- Total time: 30 minutes
- Saved: 4 days of your life

### ❌ Mistake 3: Sequential Execution When You Could Parallelize

**Bad**: Run 10 experiments one at a time on single GPU
- Time: 10 × 40 min = 400 minutes
- Cost: 6.7 hours × $0.39 = $2.61

**Good**: Launch 10× L4 spot instances in parallel
- Time: 40 minutes
- Cost: 10 × 0.67 hours × $0.39 = $2.61
- **Same cost, 10× faster**

### ❌ Mistake 4: Using Trainium for Model Exploration

**Bad**: Try different architectures on Trainium
- Each architecture change: 2-8 hours recompilation
- Can't iterate quickly

**Good**: Explore on L4/L40S, then port finalized model to Trainium
- Exploration: instant feedback
- Production: cost-optimized training

### ❌ Mistake 5: Buying Local GPUs Thinking It Scales

**Bad**: Buy 2× RTX 4090 for $4,000
- Limited to 2 parallel experiments
- No VRAM elasticity (stuck at 24GB)
- You're the sysadmin

**Good**: Use that $4,000 for cloud compute
- $4,000 ÷ $0.39/hr = 10,256 L4 hours
- Or: 1,285 8-hour days of research
- Or: 3.5 years of full-time L4 access
- Scale to 100+ GPUs when needed

---

## Budget Planning

### Individual Researcher (PhD Student)

**Monthly cloud budget: $100**

**What you can do:**
- 256 hours of L4 spot ($0.39/hr)
- Or: 128 hours of L40S spot ($0.75/hr)  
- Or: 32 8-hour days of continuous L4 access
- Or: 10× L4 instances for 25 hours (parallel sweeps)

**Enough for**: 50-100 full training runs per month

### Research Lab (5 students)

**Monthly cloud budget: $500**

**What you can do:**
- 1,282 hours of L4 spot
- Or: 50× L4 instances running 24/7
- Or: 100× L4 instances for hyperparameter sweeps

**Enough for**: 250-500 training runs per month, 50-100 per student

### Institutional Cluster Alternative

**Instead of**: $100M on 32× H200 cluster
- Serves 500 users with 4-day queue
- High utilization, low throughput

**Consider**: $50M on cloud compute + $50M on other research
- 500 users × $100k/year = same capacity
- Zero queue time
- Elastic scaling
- Latest hardware (Blackwell available now)

---

## Action Items

### If You're a Researcher

1. **Stop waiting in queues**
   - If queue >1 day, use cloud spot instead
   - Your time is worth $50-100/hr

2. **Right-size your hardware**
   - Use L4 ($0.39/hr) for most research work
   - Use L40S ($0.75/hr) if you need 48GB VRAM
   - Use g7e MIG ($1.35/hr) for parallel experiments

3. **Parallelize experiments**
   - Don't run 10 experiments sequentially
   - Launch 10× instances, finish 10× faster

4. **Save premium hardware for when you need it**
   - H100/H200: multi-GPU training, >80GB models
   - Trainium: production training of finalized models
   - Not: iterative development

### If You're Managing a Cluster

1. **Measure research throughput, not GPU utilization**
   - Track: experiments per day per researcher
   - Not: % of time GPUs are busy

2. **Provide more smaller GPUs, not fewer bigger GPUs**
   - 320× L4s serves more researchers than 32× H200s
   - Same total budget
   - No queues

3. **Make spot instances easy**
   - Provide templates, scripts, training
   - Pre-populate Docker images
   - Setup billing/quotas

4. **Reserve premium hardware for specific use cases**
   - Multi-GPU training
   - Models that truly need 80GB+ VRAM
   - Not: every experiment

### If You're Buying Hardware

**For research (exploration):**
- ✅ Many smaller GPUs (L4, L40S)
- ✅ MIG-capable instances (g7e)
- ❌ Few large GPUs with long queues

**For production (deployment):**
- ✅ Trainium/Inferentia for cost optimization
- ✅ Reserved H100s for guaranteed capacity
- ✅ Mix of instance types for different workloads

---

## Comparison to Other Platforms

### AWS vs Google Cloud

**Google Cloud offers:**
- TPU v5e/v5p (Google's custom silicon, similar role to Trainium)
- NVIDIA GPUs (A100, H100, L4)
- Similar pricing structure

**Key differences:**
- TPU has similar AOT compilation model as Trainium
- Google's JAX ecosystem vs AWS's PyTorch/Neuron
- AWS has broader instance selection

**Recommendation**: Both are viable. Choose based on:
- Existing cloud ecosystem
- Framework preference (JAX vs PyTorch)
- Regional availability

### AWS vs On-Premise Cluster

**On-premise pros:**
- No network latency
- Data never leaves your control
- Fixed cost (no usage billing)

**On-premise cons:**
- Hardware obsoletes in 2-3 years
- You're the sysadmin
- Can't scale beyond what you bought
- Queue times at high utilization

**Recommendation**: 
- On-premise: Regulated industries, data sovereignty requirements
- Cloud: Everyone else (flexibility, latest hardware, elasticity)

---

## Summary

### The Core Principle

**Match hardware to the phase:**

| Phase | Goal | Right Hardware | Wrong Hardware |
|-------|------|----------------|----------------|
| Exploration | Fast iteration | L4 spot ($0.39/hr) | Queued H200 |
| Tuning | Parallel sweeps | 10× L4 or g7e MIG | Sequential on 1 GPU |
| Production training | Cost/FLOP | Trainium cached ($0.54/hr) | On-demand H100 |
| Inference | Cost/inference | Inferentia2 ($0.30/hr) | GPU |

### The Cloud Superpower

It's not raw FLOPS. It's **elastic parallelism**.

- 1 GPU → 10 GPUs → 100 GPUs → back to 0
- Launch in 30 seconds
- Pay only for what you use
- Always have the latest hardware

### The Bottom Line

**For research: Use AWS L4 spot at $0.39/hr until you have a reason not to.**

Most research doesn't need more. When you do, the cloud scales instantly.

---

**Questions?** See [TRAINIUM_QUICKSTART.md](../TRAINIUM_QUICKSTART.md) for Trainium-specific guidance, or [COMPILATION_FAQ.md](COMPILATION_FAQ.md) for understanding AOT vs JIT compilation.

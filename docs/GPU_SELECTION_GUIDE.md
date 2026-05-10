# GPU Selection Guide: L4 vs L40S vs RTX Pro 6000

**Last Updated: May 9, 2026**  
**Pricing: AWS us-west-2 spot**

When to use which NVIDIA GPU for academic research.

---

## The Three Research GPUs

| Instance | GPU | Architecture | VRAM | Spot $/hr | **Use When** |
|----------|-----|--------------|------|-----------|--------------|
| **g6.xlarge** | L4 | Ada Lovelace | 24GB | $0.39 | 90% of research (default choice) |
| **g6e.xlarge** | L40S | Ada Lovelace | 48GB | $0.75 | Models 24-45GB, 2× VRAM needed |
| **g7e.2xlarge** | RTX Pro 6000 | Blackwell | 96GB | $1.35 | Large models 45-90GB, or 4× MIG |

---

## Decision Tree: Which GPU?

### Step 1: Measure Your Model Size

**Run this in Python:**
```python
import torch

# Your model
model = YourModel()

# Model parameters only
params_gb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9
print(f"Model parameters: {params_gb:.2f} GB")

# During training (rough estimate):
# - FP32 training: params × 4 (model + gradients + optimizer states)
# - Mixed precision (AMP): params × 2-3 (FP16 forward, FP32 optimizer)
# - Plus: activations (depends on batch size)

# Conservative estimate for training:
training_gb = params_gb * 4  # Assume FP32
print(f"Estimated training memory: {training_gb:.2f} GB")

# Add batch size overhead (rule of thumb):
# Small batch (8-16): + 2-4GB
# Medium batch (32-64): + 6-10GB
# Large batch (128+): + 15-20GB
```

### Step 2: Apply the Decision Rules

```
Model + training overhead < 20GB?
├─ YES → L4 ($0.39/hr) - 24GB sufficient
└─ NO → Continue

Model + training overhead < 43GB?
├─ YES → L40S ($0.75/hr) - 48GB sufficient
└─ NO → Continue

Model + training overhead < 88GB?
├─ YES → RTX Pro 6000 ($1.35/hr) - 96GB sufficient
└─ NO → Need multi-GPU (p5.48xlarge with H100s)
```

---

## Detailed Use Cases

### L4 (24GB) - The Default Choice

**AWS g6.xlarge: $0.39/hr spot**

**Use for:**
- ✅ Most research models (<10B parameters)
- ✅ Fine-tuning up to 7B models (with LoRA/QLoRA)
- ✅ Image models (ResNet, EfficientNet, ViT-Base)
- ✅ NLP models (BERT, RoBERTa, T5-Base, GPT-2)
- ✅ Speech models (Whisper-Base, Wav2Vec2)
- ✅ Batch size 8-32 (typical research)

**Concrete examples:**
- BERT-Large (340M params): ~5GB + training = 8GB total ✅
- ResNet-50 (25M params): ~0.5GB + activations = 4GB total ✅
- GPT-2 (1.5B params): ~6GB + training = 12GB total ✅
- ViT-Base (86M params): ~2GB + activations = 6GB total ✅
- Llama-2-7B with LoRA (LoRA params only): ~50MB trainable = 10GB total ✅

**When L4 is NOT enough:**
- ❌ Large batch sizes (>64, activations don't fit)
- ❌ Models >7B full fine-tuning
- ❌ High-resolution images (2048×2048+)
- ❌ Very long sequences (>4096 tokens)

**Why this is the default:**
- Instant availability (abundant capacity)
- Lowest cost ($0.39/hr)
- Sufficient for 90% of academic research
- Easy to parallelize (10× L4 = $3.90/hr for parallel sweeps)

---

### L40S (48GB) - The Middle Ground

**AWS g6e.xlarge: $0.75/hr spot**

**Use for:**
- ✅ Models 7-13B parameters (full fine-tuning)
- ✅ Larger batch sizes (64-128)
- ✅ High-resolution images (1024-2048px)
- ✅ Long sequences (4096-8192 tokens)
- ✅ 3D medical imaging (CT/MRI volumes, small batches)

**Concrete examples:**
- Llama-2-13B full fine-tune: ~26GB + training = 38GB total ✅
- Stable Diffusion (860M params, high-res): ~3GB + activations = 25GB ✅
- ViT-Large (300M params, large batch): ~6GB + batch 64 = 28GB ✅
- 3D U-Net (50M params, volumetric data): ~2GB + 3D activations = 35GB ✅
- BERT-Large with batch size 128: ~5GB + large batch = 32GB ✅

**When to choose L40S over L4:**
- Hit 24GB limit on L4 (OOM errors)
- Batch size matters for your research (studying scaling)
- High-resolution required (not just "nice to have")
- **Not:** "I want faster training" (use parallel L4 instead)

**Why not always use L40S:**
- 2× the cost of L4
- If model fits on L4, paying extra for unused VRAM
- Better to run 2× L4 in parallel (same cost, faster results)

---

### RTX Pro 6000 (96GB) - The Large Model Option

**AWS g7e.2xlarge: $1.35/hr spot**

**Use for:**

#### Option A: Single Large Model
- ✅ Models 13-30B parameters (full fine-tuning)
- ✅ Very large batch sizes (256+)
- ✅ Highest-resolution images (4096px+)
- ✅ 3D medical imaging (large volumes, batch size >1)
- ✅ Multi-modal models (vision + language)

**Concrete examples:**
- Llama-2-30B full fine-tune: ~60GB + training = 70GB total ✅
- Stable Diffusion XL (2.6B params, high-res): ~10GB + activations = 40GB ✅
- 3D U-Net (batch size 4): ~2GB × 4 batches = 50GB ✅
- CLIP-Large + GPT-3 (combined): ~15GB + training = 55GB ✅

#### Option B: MIG (4× Parallel Experiments)
- ✅ Run 4 models simultaneously (4× 24GB slices)
- ✅ Parallel hyperparameter search
- ✅ Multi-model comparison (like your arena demo)
- ✅ Team shared resource (4 researchers, 1 instance)

**MIG breakdown:**
```
g7e.2xlarge with MIG enabled:
├─ Total: 96GB, $1.35/hr
├─ Slice 1: 24GB (MIG 1g.24gb) = $0.34/hr effective
├─ Slice 2: 24GB (MIG 1g.24gb) = $0.34/hr effective
├─ Slice 3: 24GB (MIG 1g.24gb) = $0.34/hr effective
└─ Slice 4: 24GB (MIG 1g.24gb) = $0.34/hr effective

Use case: Run 4 models in parallel
- Each model <20GB
- Finish 4× faster than sequential
- Cost: $1.35/hr for all 4 (vs $1.56/hr for 4× L4)
```

**When to choose g7e:**
- Model genuinely needs >48GB (doesn't fit on L40S)
- Running 4 experiments in parallel (MIG mode)
- Latest architecture wanted (Blackwell vs Ada)
- **Not:** "I want the best" (usually overkill)

**Why not always use g7e:**
- 3.5× the cost of L4
- Most research models fit on L4 or L40S
- If running one small model, wasting 70GB of VRAM
- Better to parallelize on cheaper instances

---

## Common Scenarios

### Scenario 1: Fine-Tuning Language Models

**Llama-2-7B (LoRA):**
- Memory: ~10GB
- **Use:** L4 ($0.39/hr) ✅
- Why: LoRA only trains small adapter, fits easily

**Llama-2-7B (full fine-tune):**
- Memory: ~28GB
- **Use:** L40S ($0.75/hr) ✅
- Why: Full model + gradients + optimizer = 28GB

**Llama-2-13B (full fine-tune):**
- Memory: ~52GB
- **Use:** RTX Pro 6000 ($1.35/hr) ✅
- Why: Doesn't fit on L40S (48GB)

**Llama-2-70B (any method):**
- Memory: >200GB
- **Use:** p5.48xlarge (8× H100) with model parallelism
- Why: Requires multi-GPU

---

### Scenario 2: Computer Vision

**Image classification (ImageNet, 224px):**
- ResNet-50, batch 32
- Memory: ~6GB
- **Use:** L4 ($0.39/hr) ✅

**Object detection (COCO, 640px):**
- YOLOv8-Large, batch 16
- Memory: ~12GB
- **Use:** L4 ($0.39/hr) ✅

**Semantic segmentation (high-res, 1024px):**
- DeepLabv3, batch 8
- Memory: ~28GB
- **Use:** L40S ($0.75/hr) ✅

**Medical imaging (CT scans, 3D volumes):**
- 3D U-Net, 512×512×200, batch 2
- Memory: ~45GB
- **Use:** L40S ($0.75/hr) ✅

**Video generation (1024px, 16 frames):**
- Diffusion model, batch 1
- Memory: ~70GB
- **Use:** RTX Pro 6000 ($1.35/hr) ✅

---

### Scenario 3: Hyperparameter Search

**Need to test 10 configurations:**

**Option A: Sequential on L4**
```
10 experiments × 3 hours each = 30 hours
Cost: 30 × $0.39 = $11.70
Time: 30 hours
```

**Option B: Parallel on 10× L4**
```
10 experiments in parallel = 3 hours
Cost: 10 × 3 × $0.39 = $11.70
Time: 3 hours (10× faster)
```

**Option C: Parallel on 3× g7e MIG (12 slices)**
```
10 experiments on 12 slices = 3 hours (2 idle)
Cost: 3 × 3 × $1.35 = $12.15
Time: 3 hours
```

**Winner:** Option B (10× L4) - same cost as sequential, 10× faster

**Lesson:** For parallelizable work, use many cheap GPUs, not one expensive GPU.

---

### Scenario 4: Batch Size Experiments

**Research question:** "How does batch size affect convergence?"

**Need to test:** batch sizes 8, 16, 32, 64, 128, 256

**Measured memory:**
- Batch 8: 8GB → L4 ✅
- Batch 16: 12GB → L4 ✅
- Batch 32: 20GB → L4 ✅
- Batch 64: 35GB → L40S ✅
- Batch 128: 60GB → RTX Pro 6000 ✅
- Batch 256: 110GB → Need multi-GPU

**Right approach:**
```
Experiments 1-3: L4 (3 × 4 hours × $0.39 = $4.68)
Experiment 4: L40S (1 × 4 hours × $0.75 = $3.00)
Experiment 5: RTX Pro 6000 (1 × 4 hours × $1.35 = $5.40)
Total: $13.08 (right-sized for each experiment)
```

**Wrong approach:**
```
All on RTX Pro 6000:
6 experiments × 4 hours × $1.35 = $32.40
Waste: $19.32 on experiments that fit on cheaper GPUs
```

**Lesson:** Start with L4, upgrade only when you hit VRAM limit.

---

## Special Case: MIG for Teams

**Scenario:** 4 PhD students sharing compute

**Option A: Each gets their own L4 (4× g6.xlarge)**
```
4 instances × $0.39/hr = $1.56/hr
Each student: full 24GB, full isolation
Cost per student: $0.39/hr
```

**Option B: Shared g7e with MIG (1× g7e.2xlarge)**
```
1 instance × $1.35/hr = $1.35/hr
Each student: 24GB MIG slice, full isolation
Cost per student: $0.34/hr
Savings: $0.21/hr (13% cheaper)
```

**When MIG makes sense:**
- Team wants shared resource
- All working on similar-sized models (<20GB)
- Admin wants single instance to manage
- Latest architecture (Blackwell) desired

**When separate instances better:**
- Students need flexibility (different VRAM needs)
- Want independent start/stop
- One person needs >24GB occasionally

---

## The "Default Then Upgrade" Strategy

### Start Here

**For 90% of research: Start with L4**
```
Launch: g6.xlarge spot ($0.39/hr)
Try: Your model + typical batch size
If: Fits in 24GB → You're done
If: OOM error → Continue
```

### First Upgrade

**If L4 OOMs: Try L40S**
```
Launch: g6e.xlarge spot ($0.75/hr)
Try: Same model, larger batch or higher res
If: Fits in 48GB → You're done
If: Still OOM → Continue
```

### Second Upgrade

**If L40S OOMs: Try RTX Pro 6000**
```
Launch: g7e.2xlarge spot ($1.35/hr)
Try: Same model, maximum batch
If: Fits in 96GB → You're done
If: Still OOM → Need multi-GPU (p5.48xlarge)
```

**This strategy:**
- Minimizes cost (use smallest that fits)
- Avoids premature optimization (don't guess)
- Quick to test (launch, try, upgrade if needed)

---

## Common Mistakes

### Mistake 1: "I'll Just Use the Best GPU"

**Thought process:**
> "RTX Pro 6000 is the best, I'll use that for everything"

**Problem:**
- Paying $1.35/hr for model that fits on $0.39/hr L4
- Wasting $0.96/hr = $23/day = $690/month
- Same results, 3.5× the cost

**Better:**
> "Start with L4, upgrade only if needed"

---

### Mistake 2: "I Need More VRAM for Speed"

**Thought process:**
> "L40S has 48GB, 2× the VRAM = 2× faster training"

**Problem:**
- VRAM capacity ≠ training speed
- If model fits, extra VRAM sits idle
- Paying 2× for no speed benefit

**Reality:**
- L4 and L40S similar training speed for same model
- Extra VRAM only helps if you use it (larger batch)
- For speed: use multiple GPUs in parallel, not one big GPU

---

### Mistake 3: "I'll Get Big GPU Just in Case"

**Thought process:**
> "I might need 96GB later, so I'll use g7e from the start"

**Problem:**
- Paying premium while prototyping small models
- When you actually need 96GB (maybe never), upgrade then
- Premature optimization wastes budget

**Better:**
> "Use L4 for prototyping, upgrade when I hit limits"

---

### Mistake 4: "Batch Size Doesn't Matter"

**Thought process:**
> "My model fits on L4, so I'm fine"

**Problem:**
- Batch size 1: Underutilizing GPU (slow training)
- Batch size 8: Good GPU utilization
- Batch size 32: Better convergence (in some cases)
- Batch size 64: Need L40S (doesn't fit L4)

**Better approach:**
- Start with reasonable batch size (8-16)
- If training too slow, try larger batch
- Upgrade GPU only if larger batch helps (research-dependent)

---

## Summary Table

| Your Model | VRAM Needed | Right GPU | Cost/hr | Wrong GPU | Waste |
|------------|-------------|-----------|---------|-----------|-------|
| BERT-Base (110M) | ~5GB | L4 | $0.39 | g7e | $0.96/hr |
| ResNet-50 (25M) | ~6GB | L4 | $0.39 | L40S | $0.36/hr |
| GPT-2 (1.5B) | ~12GB | L4 | $0.39 | g7e | $0.96/hr |
| Llama-2-7B LoRA | ~10GB | L4 | $0.39 | L40S | $0.36/hr |
| Llama-2-7B full | ~28GB | L40S | $0.75 | g7e | $0.60/hr |
| Stable Diffusion | ~25GB | L40S | $0.75 | g7e | $0.60/hr |
| Llama-2-13B full | ~52GB | g7e | $1.35 | p5 (overkill) | $50+/hr |
| 3D U-Net (medical) | ~45GB | L40S | $0.75 | g7e | $0.60/hr |

**Key insight:** Most waste comes from using expensive GPU for small models.

---

## Quick Reference Card

**Print this, keep handy:**

```
┌─────────────────────────────────────────────────┐
│ GPU SELECTION QUICK REFERENCE                   │
├─────────────────────────────────────────────────┤
│ Model <20GB           → L4 ($0.39/hr)          │
│ Model 20-43GB         → L40S ($0.75/hr)        │
│ Model 43-88GB         → g7e ($1.35/hr)         │
│ Model >88GB           → p5 multi-GPU           │
│                                                 │
│ Parallel 4 experiments → g7e MIG ($0.34/slice) │
│ Parallel 10 experiments → 10× L4 ($3.90/hr)    │
│                                                 │
│ DEFAULT: Start L4, upgrade only if OOM         │
└─────────────────────────────────────────────────┘
```

---

## For Your BD Conversations

### When researcher says: "Which GPU should I use?"

**Your response:**

> "Start with L4 ($0.39/hr). It handles 90% of academic research:
> - Models <10B parameters
> - Batch sizes 8-32
> - Standard image resolutions
> - Most fine-tuning tasks
>
> Launch it, try your model. If you get OOM (out of memory), upgrade to L40S ($0.75/hr, 48GB). Still OOM? Then g7e ($1.35/hr, 96GB).
>
> Don't guess—test. Better to start cheap and upgrade than pay for unused VRAM."

### When researcher says: "I want the RTX Pro 6000 (best GPU)"

**Your response:**

> "RTX Pro 6000 is great for large models (45-90GB). But if your model fits on L4 (24GB), you'll get the same results for 3.5× less cost.
>
> **Try this:** Start with L4 ($0.39/hr). If it works, you just saved $0.96/hr = $23/day. If you OOM, upgrade to L40S or g7e.
>
> The 'best' GPU is the cheapest one that fits your model."

### When researcher says: "Should I use MIG?"

**Your response:**

> "MIG makes sense if:
> - Running 4 experiments in parallel (each <20GB)
> - Team sharing one instance (4 students)
> - Want Blackwell architecture
>
> Cost: g7e MIG = $0.34/hr per slice (4 slices)
>
> Otherwise: Just launch separate L4 instances ($0.39/hr each). More flexible, same cost."

---

## Bottom Line

**90% of academic research: L4 ($0.39/hr)**
- Default choice
- Start here
- Upgrade only if needed

**9% of academic research: L40S ($0.75/hr)**
- When L4 OOMs (24GB not enough)
- Models 20-45GB
- High-res or large batch

**1% of academic research: g7e ($1.35/hr)**
- When L40S OOMs (48GB not enough)
- Models 45-90GB
- Or: 4× parallel experiments (MIG)

**Test, don't guess. Start cheap, upgrade if needed.**

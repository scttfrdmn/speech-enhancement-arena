# When to Actually Use Trainium and Large NVIDIA GPUs

**Last Updated: May 9, 2026**

Most ML research fits on L4 ($0.39/hr, 24GB). But some workloads genuinely need premium hardware. Here's when.

---

## When to Use AWS Trainium

### The Pattern: Compile Once, Run Many Times

Trainium makes sense when:
1. Architecture is **finalized** (no changes between runs)
2. Training is **long** (>20 hours per run)
3. You'll **reuse** the compiled model (multiple runs, scheduled retraining)

**Break-even analysis:**
- Compilation: 2-8 hours (one-time cost)
- Training on Trainium: 40-60% cheaper than GPU
- Need: ~3+ long runs to amortize compilation overhead

---

### Concrete Use Case 1: **Model Distillation**

**Scenario:** You have a large teacher model (e.g., GPT-3.5), want to train smaller student models.

**Why Trainium fits:**
- **Teacher model frozen** (no architecture changes)
- **Student architecture finalized** (decided after initial experiments)
- **Multiple distillation runs:** Train students for different domains/tasks
- **Long runs:** Distillation requires many epochs (50-100+)
- **Compile once:** Teacher forward pass + student training compiled together

**Workflow:**
```
Phase 1: Develop student architecture on L4 (fast iteration)
├─ Try different student sizes (12-layer, 24-layer, etc.)
├─ Test distillation loss functions
├─ Validate approach on small dataset
└─ Finalize architecture: 12-layer, 512 hidden, distillation + task loss

Phase 2: Production distillation on Trainium
├─ Compile: 4 hours (one-time)
├─ Train student for domain 1: 40 hours → $21.60 (vs $36 on L40S)
├─ Train student for domain 2: 40 hours → $21.60 (reuse compilation)
├─ Train student for domain 3: 40 hours → $21.60 (reuse compilation)
└─ Total: 4 + 120 hours = 124 hours, saved $43.20 vs GPU
```

**Instance:** trn1.2xlarge ($0.54/hr spot)

**AWS cost comparison:**
- Trainium: $67 (including compilation)
- L40S GPU: $110 (no compilation, but higher $/hr)
- **Savings: $43** (39%)

**Typical domains:**
- Medical: General biomedical model → specialized for radiology, pathology, genomics
- Legal: General legal model → specialized for contract law, IP, litigation
- Finance: General financial model → specialized for trading, risk, compliance

---

### Concrete Use Case 2: **Fine-Tuning at Scale**

**Scenario:** You have a base model (e.g., Llama-3-8B), need to fine-tune for 20 different customer domains.

**Why Trainium fits:**
- **Base model frozen** (using official weights)
- **Fine-tuning recipe finalized** (LoRA, QLoRA, or full fine-tune decided)
- **Multiple runs:** Same recipe, different datasets
- **Scheduled:** New customer domains arrive predictably

**Workflow:**
```
Phase 1: Develop fine-tuning recipe on L4
├─ Try LoRA vs QLoRA vs full fine-tune
├─ Test on 2-3 customer datasets
├─ Tune learning rate, rank, alpha
└─ Finalize recipe: LoRA rank=32, alpha=64, LR=3e-4

Phase 2: Scale on Trainium
├─ Compile: 3 hours (one-time for LoRA + Llama-3-8B)
├─ Customer 1: 8 hours → $4.32
├─ Customer 2: 8 hours → $4.32 (reuse compilation)
├─ ... (customers 3-20)
└─ Total: 3 + 160 hours = 163 hours @ $0.54/hr = $88
```

vs **GPU alternative:**
- L40S: 160 hours @ $0.75/hr = $120 (no compilation)
- **Savings: $32** (27%)

**When this doesn't work:**
- Customers have wildly different requirements (different architectures)
- Fine-tuning takes <4 hours each (compilation overhead dominates)
- Exploring which fine-tuning method works (still in phase 1)

---

### Concrete Use Case 3: **Continual Learning / Scheduled Retraining**

**Scenario:** News recommendation model retrained nightly on fresh articles.

**Why Trainium fits:**
- **Model architecture fixed** (CNN + BERT encoder decided months ago)
- **Predictable schedule:** Runs every night at 2am
- **Long runs:** Each retraining takes 6 hours
- **Compile once:** Same graph every night

**Workflow:**
```
Initial setup:
├─ Compile model once: 2 hours
└─ Save NEFF to S3

Nightly pipeline (automated):
├─ Download NEFF from S3: 30 seconds
├─ Load yesterday's checkpoint
├─ Train on today's articles: 6 hours
├─ Save new checkpoint
└─ Deploy to inference

Cost per month (30 retrainings):
├─ Compilation (once): 2 hours × $1.34 = $2.68
├─ Training (30×): 180 hours × $0.54 = $97.20
└─ Total: $99.88 @ Trainium vs $165 @ L40S GPU
```

**Savings: $65/month** (39%)

**Over 1 year:** $780 saved vs GPU

**Similar workloads:**
- Weather forecasting: Retrain on new observations daily
- Stock prediction: Retrain on market data nightly
- Fraud detection: Retrain on transaction data weekly
- Recommendation systems: Retrain on user behavior continuously

---

### Concrete Use Case 4: **Hyperparameter Search (After Architecture Fixed)**

**Scenario:** Architecture is finalized, now searching for optimal LR, batch size, warmup schedule.

**Why Trainium *might* fit:**
- **Architecture frozen** (no recompilation)
- **Long runs:** Each config needs to train to convergence (20+ hours)
- **Many runs:** Grid search over 50 configs

**Trade-off analysis:**

**Option A: Trainium sequential**
```
├─ Compile once: 4 hours
├─ Config 1: 24 hours
├─ Config 2: 24 hours (reuse compilation)
├─ ... (configs 3-50)
└─ Total: 4 + (50 × 24) = 1,204 hours @ $0.54/hr = $650
```

**Option B: L4 parallel (10 instances)**
```
├─ No compilation
├─ Configs 1-10: 24 hours parallel
├─ Configs 11-20: 24 hours parallel
├─ ... (5 batches)
└─ Total: 5 × 24 = 120 hours × 10 instances × $0.39/hr = $468
```

**Winner: L4 parallel** (27% cheaper, 10× faster to completion)

**When Trainium would win:**
- Runs are >50 hours each (compilation amortizes better)
- Can't afford 10× parallel capacity
- Cost per hour matters more than time to completion

---

### Concrete Use Case 5: **Large-Scale Batched Inference**

**Scenario:** Process 1 million images/documents through a model (one-time batch job, not serving).

**Why Trainium fits:**
- **Model frozen** (inference only, no training)
- **Large batch:** Process millions of samples
- **Compile once:** Model graph never changes
- **Run for hours:** Even at 1000 samples/sec, takes >15 minutes
- **Cost matters:** Doing this on H100 costs 6× more

**Workflow:**
```
Setup (one-time):
├─ Compile model for inference on Trainium: 1-2 hours
├─ Optimize batch size for throughput
└─ Save NEFF to S3

Batch job:
├─ Load NEFF: 30 seconds
├─ Process 1M images through ResNet-50: 3 hours
└─ Save predictions to S3

Cost:
├─ Trainium: 3 hours × $0.54/hr = $1.62
└─ vs H100: 1 hour × $3.13/hr = $3.13 (faster but 93% more expensive)
└─ vs Inferentia2: 6 hours × $0.30/hr = $1.80 (slower but cheaper)
```

**When this makes sense:**
- ✅ Batch processing (not real-time serving)
- ✅ Large dataset (>10K samples)
- ✅ Don't need sub-second latency
- ✅ Cost optimization matters

**When this doesn't make sense:**
- ❌ Real-time serving (use Inferentia2 instead)
- ❌ Small dataset (<1K samples) - compilation overhead dominates
- ❌ Model changes between batches (constant recompilation)

**Academic examples:**
- **Genomics:** Annotate 1M protein sequences with AlphaFold
- **Astronomy:** Classify 500K galaxy images from telescope survey
- **Social science:** Sentiment analysis on 10M historical tweets
- **Medical research:** Screen 100K pathology slides for anomalies
- **Climate science:** Process 50 years of satellite imagery

**Key insight:** Trainium is cheaper than GPU for batch inference when:
- Batch takes >2 hours (amortizes compilation)
- Cost matters more than speed
- But use Inferentia2 if workload is pure inference (no training ever)

---

### Concrete Use Case 6: **Research on Trainium Itself**

**Scenario:** Computer architecture research studying custom silicon.

**Why Trainium fits:**
- You're **studying Trainium** (the point is to use it)
- Comparing tile-based vs systolic architectures
- Publishing on compiler optimizations
- Educational: teaching ML systems courses

**Examples:**
- "NeuronCore vs TPU systolic array: A comparative study"
- "Optimizing BERT for Trainium: Compilation strategies"
- Course: "CS 395: ML Systems on Custom Silicon"

**This is <1% of academic ML research**, but legitimate use case.

---

### When Trainium Does NOT Fit

❌ **Exploring architectures**
- Changing layer counts, attention mechanisms daily
- Each change = 2-8 hour recompilation
- Use L4 instead

❌ **Debugging convergence**
- "Why isn't my model learning?"
- Need rapid iteration, add print statements, try fixes
- Compilation overhead kills iteration speed

❌ **Short experiments (<10 hours)**
- Compilation overhead (2-8 hours) is significant fraction of total time
- Economics don't work out

❌ **One-off training**
- Train model once for paper, never again
- Compilation cost not amortized
- GPU faster to results

❌ **Prototype development**
- Still figuring out what works
- Architecture changes frequently
- Trainium's strength (compilation optimization) becomes weakness

---

## When to Use Large NVIDIA GPUs (H100/H200/B200)

### The Pattern: Workloads That Don't Fit in 24GB

H100+ makes sense when:
1. Model **>24GB** (doesn't fit on L4)
2. Need **multi-GPU** (model parallel, data parallel)
3. Genuinely **memory-bound** (not just "want faster")

---

### Concrete Use Case 1: **Multi-GPU Model Parallelism**

**Scenario:** Training 70B parameter model (e.g., Llama-3-70B fine-tuning).

**Why H100 needed:**
- **Model size:** ~140GB in FP16 (model weights + gradients + optimizer states)
- **Doesn't fit:** 24GB L4 ❌, 48GB L40S ❌
- **Needs:** 2× H100 (80GB each) with tensor parallelism

**Workflow:**
```
Model sharding:
├─ Layer 0-35: GPU 0 (70GB)
├─ Layer 36-70: GPU 1 (70GB)
└─ NVLink communication between GPUs

Training:
├─ Forward: Pipeline through both GPUs
├─ Backward: Pipeline back through both GPUs
└─ 8-16 hours training time
```

**Instance:** p5.48xlarge (8× H100, but you use 2)

**AWS cost:**
- Reserved p5.48xlarge: ~$25/hr (entire instance, 8 GPUs)
- 10 hours training: $250
- **No cheaper alternative** (model doesn't fit on smaller GPUs)

**Similar workloads:**
- LLM fine-tuning: >30B parameters
- Multimodal models: Vision + language, large
- Diffusion models: High-res image generation (Stable Diffusion XL+)

---

### Concrete Use Case 2: **Large Batch Training for Research**

**Scenario:** Studying scaling laws, need to test batch sizes up to 1024.

**Why H100 needed:**
- **Batch size 1024:** ~60GB activation memory
- **Doesn't fit:** L4 (24GB) ❌
- **Needs:** H100 (80GB) ✅

**Workflow:**
```
Research question: "How does batch size affect convergence?"

Experiments:
├─ Batch 32: L4 (fits in 24GB)
├─ Batch 64: L4 (fits in 24GB)
├─ Batch 128: L4 (fits in 24GB)
├─ Batch 256: L40S (fits in 48GB)
├─ Batch 512: H100 (fits in 80GB) ← Need H100
└─ Batch 1024: H100 (fits in 80GB) ← Need H100
```

**AWS cost:**
- Experiments 1-3 on L4: 3 × 4 hours × $0.39 = $4.68
- Experiment 4 on L40S: 4 hours × $0.75 = $3.00
- Experiments 5-6 on H100: 2 × 4 hours × $3.13 = $25.04
- **Total: $32.72** (use right-sized instance for each)

vs **All on H100:**
- 6 × 4 hours × $3.13 = $75.12
- **Waste: $42.40** on experiments that fit on cheaper GPUs

**Key insight:** Use H100 only for experiments that need it, not all experiments.

---

### Concrete Use Case 3: **Medical Imaging (3D Volumetric Data)**

**Scenario:** Training on CT/MRI scans (3D volumes, not 2D images).

**Why H100 needed:**
- **3D volumes:** 512×512×512 voxels typical
- **Activation memory:** ~40-60GB for reasonable batch size
- **Doesn't fit:** L4 (24GB) ❌
- **Barely fits:** L40S (48GB) - tight
- **Comfortable:** H100 (80GB) ✅

**Workflow:**
```
Model: 3D U-Net for tumor segmentation

Data:
├─ Input: CT scan (512×512×200 slices)
├─ Output: Tumor mask (same size)
└─ Batch size 2 → ~50GB memory

Training:
├─ 100 epochs on hospital dataset
├─ 40 hours total
└─ Need H100 to fit batch size >1
```

**AWS cost:**
- H100: 40 hours × $3.13/hr (p5 spot) = $125
- **No cheaper option** (doesn't fit on L40S with batch size >1)

**Could you use L4?**
- Yes, with batch size 1
- But: Training would be 4× slower (GPU underutilized)
- And: Batch normalization doesn't work well with batch size 1
- **Conclusion:** H100 is right choice here

**Similar workloads:**
- Microscopy: High-res whole-slide imaging
- Satellite imagery: Large multispectral images
- Climate modeling: 3D atmospheric data
- Materials science: Molecular dynamics (large systems)

---

### Concrete Use Case 4: **High-Resolution Video Generation**

**Scenario:** Training video generation model (e.g., Sora-style).

**Why H100 needed:**
- **Video data:** 16 frames × 1024×1024 × 3 channels
- **Temporal attention:** Attention across frames = huge memory
- **Activation memory:** ~70GB
- **Doesn't fit:** Anything <80GB

**Workflow:**
```
Model: Diffusion transformer for video

Training:
├─ Dataset: 100K video clips
├─ Resolution: 1024×1024, 16 frames
├─ Batch size 1 per GPU → needs 8× H100 for data parallel
└─ 200 hours training
```

**AWS cost:**
- p5.48xlarge (8× H100): 200 hours × $98/hr = $19,600
- **No cheaper alternative** (memory requirements)

**This is frontier research.** Most academic labs can't afford this. But when they can (large grant), H100 is necessary.

---

### Concrete Use Case 5: **Mixture of Experts (MoE) Models**

**Scenario:** Training Sparse MoE with 64 experts.

**Why H100 needed:**
- **All experts in memory:** 64 experts × 2GB each = 128GB
- **Doesn't fit:** L4 (24GB) ❌, L40S (48GB) ❌
- **Needs:** 2× H100 (160GB total) with expert parallelism

**Workflow:**
```
Model: MoE Transformer (like Switch Transformer)

Sharding:
├─ Experts 1-32: GPU 0 (64GB)
├─ Experts 33-64: GPU 1 (64GB)
└─ Route tokens to experts dynamically

Training:
├─ 100 epochs
├─ 60 hours
└─ Need multi-H100 for all experts in memory
```

**AWS cost:**
- p5.48xlarge: 60 hours × $98/hr = $5,880
- **No cheaper alternative** (architectural requirement)

**Could you use fewer experts to fit on L4?**
- Yes: 8 experts would fit on L4
- But: Research question is about scaling to 64 experts
- **Conclusion:** H100 necessary for this specific research

---

### Concrete Use Case 6: **Conference Deadline Panic Mode**

**Scenario:** Paper due in 2 days, need final results NOW.

**Why H100:**
- **Speed matters more than cost**
- Experiment runs 10 hours on L4 vs 3 hours on H100
- Willing to pay 8× more to save 7 hours

**Workflow:**
```
Friday 10am: Realize final experiment needed
├─ L4 option: Start now, finish 8pm (miss deadline)
├─ H100 option: Start now, finish 1pm (make deadline)
└─ Decision: Pay $25 for H100 instead of $4 for L4

Result: Paper submitted on time
```

**Cost:**
- H100: 3 hours × $3.13 = $9.39
- L4: Would have been $1.56, but paper rejected
- **ROI:** Accepted paper worth >$9.39

**This is valid use case**, but should be rare. If you're always in panic mode, fix workflow.

---

### When H100 Does NOT Fit

❌ **Small models that fit on L4**
- 90% of academic research
- Paying 8× more for 3× speedup = bad economics
- Use L4 until you hit 24GB limit

❌ **Development/debugging**
- Iterating on architecture
- H100 availability is lower (more contended)
- Use L4 for iteration, H100 for final runs

❌ **Parallel hyperparameter search**
- Better to run 10× L4 in parallel
- Faster to completion AND cheaper
- Unless each run genuinely needs >24GB

❌ **When you're just "trying something"**
- Exploratory experiment
- Not sure if it will work
- Start cheap (L4), scale up if promising

❌ **Prestige/bragging rights**
- "I used H100" sounds impressive
- But doesn't make research better
- Use what fits the problem

---

## Decision Tree: Which Hardware?

### Step 1: What's your model size?

```
Model + activations + optimizer < 20GB?
├─ YES → L4 ($0.39/hr) - start here
└─ NO → Continue to step 2

Model + activations + optimizer < 45GB?
├─ YES → L40S ($0.75/hr)
└─ NO → Continue to step 3

Model > 80GB?
├─ YES → Multi-GPU (H100) required
└─ NO → Single H100 ($3.13/hr spot)
```

### Step 2: What's your workflow?

```
Exploring architectures (changes daily)?
├─ YES → GPU (L4/L40S) - avoid Trainium
└─ NO → Continue to step 3

Architecture finalized, training >20 hours?
├─ YES → Consider Trainium ($0.54/hr)
└─ NO → GPU better fit

Multiple runs of same architecture?
├─ YES → Trainium amortizes compilation
└─ NO → GPU faster to single result
```

### Step 3: What's your parallelization strategy?

```
Can parallelize experiments?
├─ YES → 10× L4 ($3.90/hr) often beats 1× H100
└─ NO → Consider premium hardware

Need results urgently (conference deadline)?
├─ YES → H100 (pay for speed)
└─ NO → Use right-sized, cheaper hardware
```

---

## Real-World Examples from Academia

### Example 1: PhD Student - Biology (Typical)

**Workload:** Protein structure prediction (AlphaFold-style)
- Model: 25M parameters, fits in 16GB
- Experiments: Trying different architectures weekly
- Duration: 2-6 hours per training run

**Right hardware:** L4 spot ($0.39/hr)
- Instant availability
- Sufficient VRAM
- Cost per run: $0.78-2.34

**Wrong hardware:** 
- Trainium ❌ (architecture changes too often)
- H100 ❌ (doesn't need 80GB, paying 8× more for no benefit)

---

### Example 2: Postdoc - Computer Vision (Scaling Study)

**Workload:** Studying how image resolution affects model accuracy
- Models: ResNet variants at 256px, 512px, 1024px, 2048px
- 2048px version needs 35GB

**Right hardware:** 
- 256px, 512px, 1024px → L4 ($0.39/hr)
- 2048px → L40S ($0.75/hr)

**Total cost:** 3 × $2 (L4) + 1 × $4 (L40S) = $10

**Wrong hardware:**
- All on H100 ❌ → $40 (4× waste on small experiments)
- All on L4 ❌ → Can't run 2048px experiment

---

### Example 3: Research Lab - NLP (Production-ish)

**Workload:** Fine-tuning language models for hospital EHR system
- Model: Finalized Bio-LLM
- Task: Fine-tune for 15 hospital specialties
- Duration: 8 hours per specialty

**Right hardware:** Trainium ($0.54/hr)
- Compile once: 3 hours = $4.05
- 15 specialties: 120 hours = $64.80
- Total: $68.85

**vs GPU:**
- L40S: 120 hours × $0.75 = $90
- **Savings: $21.15** (24%)

**Why this works:**
- Architecture finalized (not exploring)
- Multiple runs (15 specialties)
- Long enough (8 hours each) to amortize compilation

---

### Example 4: Undergrad Research - Climate Science (Learning)

**Workload:** Predicting temperature from satellite data
- Model: Small CNN, 5M parameters
- Learning ML + climate science
- Short experiments (30 min each)

**Right hardware:** L4 spot ($0.39/hr)
- Cost per experiment: $0.20
- Budget: $20 → 100 experiments
- Perfect for learning

**Wrong hardware:**
- H100 ❌ → $1.56 per experiment, only 12 experiments for $20
- Trainium ❌ → Compilation longer than experiments

---

### Example 5: CS Professor - Systems Research (Novel)

**Workload:** Studying Trainium compiler optimizations
- Research question: "Can we improve BERT compilation?"
- Publishing on NeuronCore architecture

**Right hardware:** Trainium ($0.54/hr)
- **The point is to study Trainium**
- Not about cost, about research contribution
- Budget from systems research grant

**This is <1% of academic ML**, but legitimate.

---

## Summary Table

| Workload | Model Size | Duration | Changes | Right Hardware | Wrong Hardware |
|----------|-----------|----------|---------|----------------|----------------|
| **Exploration** | <20GB | Any | Daily | L4 ($0.39/hr) | H100 (waste), Trainium (too slow) |
| **Hyperparameter search** | <20GB | <10hr each | None | 10× L4 parallel | Sequential H100 |
| **Model distillation** | <20GB | >20hr each | None | Trainium ($0.54/hr) | GPU (works, but 40% more) |
| **Fine-tuning at scale** | <20GB | >10hr each | None | Trainium if >5 runs | GPU if <5 runs |
| **Continual learning** | <20GB | >5hr each | None | Trainium ($0.54/hr) | GPU (works, but costs more) |
| **Large model training** | >80GB | Any | Any | Multi-H100 | Nothing else fits |
| **3D medical imaging** | 30-60GB | Any | Any | H100 ($3.13/hr) | L4 (doesn't fit) |
| **Video generation** | >60GB | Any | Any | Multi-H100 | Nothing else fits |
| **Conference deadline** | Any | Any | Any | H100 (pay for speed) | L4 (too slow) |
| **Systems research** | Any | Any | Any | Platform being studied | N/A |

---

## The Bottom Line

### Most Academic Research: L4

**90% of academic ML research should use L4 spot ($0.39/hr):**
- Models <20GB
- Exploring architectures
- Need fast iteration
- Time-sensitive (graduations, deadlines)

### Some Academic Research: H100

**5% of academic ML research needs H100 ($3.13/hr spot):**
- Models >24GB that don't fit on L4
- Multi-GPU required
- Genuinely memory-bound
- Conference deadline emergencies

### Rare Academic Research: Trainium

**<5% of academic ML research fits Trainium ($0.54/hr spot):**
- Finalized architecture
- Multiple long runs (>20 hours each)
- Production-like workflows
- Studying Trainium itself (systems research)

### The Key Question

**Before using premium hardware, ask:**

> "Could I develop this on L4 first, then scale up only if needed?"

**Answer is almost always: Yes.**

Start with L4. Prove it works. Then scale to premium hardware if the workload genuinely requires it.

**Don't start with premium hardware "just in case."** That's how budgets get wasted.

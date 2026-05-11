# Research Models vs Production Models

Academic researchers training models face very different constraints than industry deploying models at scale. Hardware choices that make sense for one are wrong for the other. This document spells out the distinction so you can pick hardware that fits your actual situation, not someone else's.

---

## The Two Types of Model Development

### Type 1: Research Models (Academic)

**Goal:** Publish papers, prove concepts, graduate.

**Model characteristics:**
- Trained on manageable datasets (10K–1M samples)
- Proves a scientific hypothesis or new technique
- Lifespan: until paper is published, then mostly abandoned
- Audience: maybe 5–50 other researchers who cite the paper

**Inference scale:**
- Generate figures for the paper (100–1K predictions)
- Maybe release code + weights for reproducibility
- Total lifetime inference: <10K requests

**Success metrics:**
- Does it work better than baseline?
- Can I publish at NeurIPS / ICML / ACL / domain venue?
- Can I graduate?
- Time to publication.

**Optimize hardware for:** fast iteration, time to results, researcher productivity.

**Don't care about:** cost per inference at scale, serving millions of users, 99.9% uptime, sub-100ms latency.

---

### Type 2: Production Models (Industry)

**Goal:** Deploy at scale, serve millions of users, generate revenue.

**Model characteristics:**
- Trained on all available data (often billions of samples)
- Continuously improved (retraining pipeline)
- Lifespan: years (ongoing maintenance)
- Audience: millions of end users

**Inference scale:**
- Real users 24/7
- Millions to billions of requests per day
- Total lifetime inference: trillions of requests

**Success metrics:** cost per inference, latency, uptime, model quality.

**Optimize hardware for:** $/inference, throughput, reliability.

**Care deeply about:** 10% cost reductions (millions in savings at scale), every 50 ms of latency, power consumption.

---

## The "Frontier Model" Reality Check

Many academic researchers describe their work as "frontier model training." It's worth being honest about what that means in context.

**What academic researchers actually train:**
- 1–10 GPUs for 2–7 days
- 10K–1M samples (maybe 10B tokens for NLP)
- Total compute: $500–$5K
- Model size: 100M–7B parameters

**This is good research.** It just isn't *frontier*.

**What frontier models actually are (GPT-4 / Claude 3.5 / Gemini Ultra class):**
- 10,000–25,000 GPUs for weeks to months
- Trillions of tokens — the open internet plus proprietary data
- Total compute: $50M–$200M
- Model size: 200B–1.7T parameters (estimated)
- Multi-node training, custom data pipelines (petabytes)
- Teams of 100+ researchers and engineers
- Continuous improvement, not one-shot training

Academic work that calls itself "frontier" usually sets unrealistic expectations, drives wrong hardware choices, and invites comparison to inappropriate baselines. Better framing: "I'm doing research that advances the state of the art in [specific area]." That's accurate *and* respectable.

---

## Hardware Implications

### For research models (~99% of academic work)

**Optimize for time to publication.**

Right hardware:
- **L4 spot ($0.39/hr)** for ~95% of work — fast iteration, instant launch, 24 GB VRAM sufficient. Cost per experiment: $0.50–$5. Conference deadline coming? Launch 10× in parallel.
- **L40S spot ($0.75/hr)** for larger models that need 48 GB.
- **H100 reserved** for the rare case of true multi-GPU need or >80 GB models — final benchmark runs, not daily exploration.

Wrong hardware:
- Trainium — optimizing for production cost you'll never deploy at.
- H100 for everything — paying 8× more for no benefit.
- On-premise cluster with multi-day queue — kills research velocity.

The reasoning: your model will be used by maybe 50 people. Lifetime inference will be under 10K requests. If production deployment happens later, someone else will do it. Your bottleneck is time to publication, not cost at scale.

### For production models (industry, not academic)

**Optimize for cost per inference.**

Right hardware:
- **Development:** L4 / L40S (architecture iteration phase)
- **Training:** Trainium (40–60% cheaper for long runs, amortizes compilation)
- **Inference:** Inferentia2 (70% cheaper than GPU for serving)

The reasoning: serving millions of users means every 1% cost saving compounds into significant dollars. The 2–8 hour compilation overhead disappears when amortized over years of weekly retraining.

---

## Real Examples: Research vs Production

### Example 1: Protein Structure Prediction

**Research (academic PhD student):**

| | |
|---|---|
| Goal | Publish a new attention mechanism for proteins |
| Dataset | 10K protein structures (AlphaFold subset) |
| Training | 3 days on 1× L4 |
| Cost | 72 hours × $0.39 = **$28** |
| Inference | 1K predictions for paper figures |
| Outcome | Paper accepted; PhD student graduates |
| Hardware | L4 spot |

**Production (DeepMind / industry equivalent):**

| | |
|---|---|
| Goal | Predict all human proteins for drug discovery |
| Dataset | All known proteins (100M+) |
| Training | Weeks on 1000s of GPUs/TPUs |
| Cost | $10M+ |
| Inference | Billions of predictions for pharma |
| Outcome | Commercial service, real revenue |
| Hardware | Custom TPUs + specialized infrastructure |

Academic model proves the concept. Industry model scales it. Different hardware for different phases.

### Example 2: Large Language Model

**Research (university lab):**

| | |
|---|---|
| Goal | Show sparse attention works for long documents |
| Dataset | 100K documents (Books3 subset) |
| Model | 1B parameter GPT-style |
| Training | 5 days on 4× L40S |
| Cost | 120 hours × 4 × $0.75 = **$360** |
| Inference | Generate examples for paper |
| Outcome | EMNLP paper, ~50 citations |
| Hardware | 4× L40S spot |

**Production (OpenAI / Anthropic / Google):**

| | |
|---|---|
| Goal | Deploy LLM to millions of users |
| Dataset | Entire internet (trillions of tokens) |
| Model | 200B–1.7T parameters |
| Training | Months on 10,000+ GPUs |
| Cost | $50M–$200M |
| Inference | Billions of requests per day |
| Outcome | Commercial product, hundreds of millions in revenue |
| Hardware | Thousands of H100s + custom inference infra |

### Example 3: Medical Imaging

**Research (hospital resident + CS collaborator):**

| | |
|---|---|
| Goal | Detect rare disease in CT scans |
| Dataset | 1,000 CT scans from one hospital |
| Model | 3D U-Net, 25M parameters |
| Training | 2 days on 1× L40S |
| Cost | 48 hours × $0.75 = **$36** |
| Inference | 200 test cases for paper |
| Outcome | Radiology journal, clinical validation |
| Hardware | L40S spot |

**Production (healthcare-tech company):**

| | |
|---|---|
| Goal | Screen all CT scans nationwide |
| Dataset | 10M CT scans (licensed from hospitals) |
| Model | Ensemble, 100M+ parameters |
| Training | Continuous, weekly retraining on new data |
| Cost | ~$1M/year compute budget |
| Inference | 100K scans/day, millions per year |
| Outcome | FDA-approved device, hospital contracts |
| Hardware | Trainium for training, Inferentia2 for inference |

Resident proved clinical value on 1K scans. Company scales to millions. Academic proof-of-concept doesn't need production infrastructure.

---

## The Workflow Gap

What actually happens after a paper publishes:

**Phase 1 — Academic research:**
```
University researcher:
├─ Proves a new technique works (small dataset, L4)
├─ Publishes paper
└─ Moves to the next research question

Model either:
├─ Abandoned, or
└─ Open-sourced (10–100 stars on GitHub)
```

**Phase 2 — Industry adoption (years later, different people):**
```
Company sees the paper:
├─ Decides the technique is promising
├─ Reimplements from scratch (not using academic code)
├─ Scales the dataset 1000× (proprietary data)
├─ Trains a production model (Trainium / H100 clusters)
├─ Builds inference infrastructure (Inferentia2, load balancing)
├─ Deploys to millions of users
└─ Maintains for years (retraining, monitoring, updates)
```

**The academic researcher never touches the production infrastructure.** Company engineers handle that phase, on their own timeline, with their own constraints, on their own hardware. Your hardware choice for the research paper doesn't propagate to the production deployment.

### Why this matters for hardware choice

A common bad reasoning chain:
> "My model might be used in production someday, so I should optimize for production deployment costs now."

This is wrong for three reasons:

1. **You won't be the one deploying it.** Industry reimplements; they don't use your code.
2. **Your optimization is premature.** Most research doesn't get adopted. If it does, it's years later on different hardware. Industry will re-optimize for their constraints anyway.
3. **You have the wrong bottleneck.** Academic bottleneck = time to publication. Industry bottleneck = cost at scale. Optimizing for the wrong one slows you down.

Better reasoning: "I need to prove this works and publish. Optimize for time to publication."

Practical implication: use L4 (fast iteration, low cost per experiment). Don't pay the Trainium compile tax for a production deployment you won't be doing.

---

## The "But My Model Will Be Used at Scale" Counter

Sometimes the claim is "if my model gets adopted at scale, I should optimize training for production costs now." Why this is wrong:

- **You won't be the one deploying it.** Industry reimplements for their stack.
- **Your optimization is premature.** Adoption is unlikely; if it happens, it's years later.
- **Your bottleneck is different.** Academic = time to publication. Industry = cost at scale.
- **The opportunity cost is real.** Compilation overhead = 2–8 hours per experiment. That's 10 L4 experiments you could have run in the same time. More experiments → better paper → higher real impact.

**The narrow case where it's valid:** your academic lab itself is doing production deployment — a hospital deploying your model internally, a government using your model for policy, a university providing the model as a service to other researchers. Then yes, optimize for production. This is well under 1% of academic ML work.

---

## When To Go Directly to Trainium (Skipping L4)

Most academic work should start on L4 (exploration) and only move to Trainium for genuinely production-like runs. But some academic workloads can skip L4 entirely. The honest list:

### 1. Distillation with a known student architecture

You're distilling a fixed teacher (e.g., GPT-3.5) into a standard student architecture (e.g., a 6-layer Transformer) for 5–10 domains. Architecture isn't changing; you'll do many long runs. Trainium fits.

Caveat: For most domain-distillation use cases, **Bedrock Distillation** (managed service) is even better — no infrastructure at all. See `docs/TRAINIUM_PRACTICAL_NOTES.md`.

### 2. Replication studies of Trainium-specific papers

If the paper you're reproducing studied Trainium behavior (compiler optimizations, NeuronCore architecture), you need to match the hardware. If the paper studied an ML *technique* that should generalize across hardware, use L4 — failure to replicate on different hardware is itself a finding.

### 3. Teaching production ML workflows

Professional ML systems courses where the educational goal is learning AOT compilation, NeuronCore optimization, production deployment skills. Trainium is the curriculum, not just the compute.

### 4. Grant-mandated hardware

Some grants or institutional partnerships specify particular hardware (Trainium in some AWS-backed academic programs, specific GPU types from other sources). When you're already constrained, pick projects that fit the constraint — for Trainium, that means standard architectures, multiple long runs, production-like workflows — rather than fighting the hardware with novel-architecture exploration.

### 5. Industry collaboration with deployment plan

Academic lab partnering with a company that will deploy on Trainium. The deliverable *is* a Trainium-optimized model. End user dictates hardware.

For everything else — exploratory research, architecture iteration, typical PhD work — L4 is the right answer.

---

## Summary Table

| Dimension | Research Model (Academic) | Production Model (Industry) |
|---|---|---|
| Goal | Publish papers | Serve users at scale |
| Dataset | 10K–1M samples | 10M–1B+ samples |
| Training cadence | Once, or 5–10 ablations | Continuous, weekly retraining |
| Inference volume | 1K–10K predictions total | 1B+ per month |
| Lifespan | Until paper publishes | Years |
| Users | 10–50 researchers | Millions of end users |
| Total compute cost | $100–$5K | $1M+/year |
| Primary success metric | Paper accepted | Revenue, retention |
| Primary bottleneck | Time to publication | Cost per inference |
| Right hardware | L4 spot ($0.39/hr) | Trainium training, Inferentia2 inference |
| Wrong hardware | Trainium (premature optimization) | L4 (too expensive at scale) |

---

## Three Questions to Ask Yourself

1. **Will my model serve millions of users?**
   - No (~99% of research) → Use L4. Optimize for research velocity.
   - Yes (rare) → You're doing production deployment, not research.

2. **Will I be the one deploying to production?**
   - No (~99% of researchers) → Your hardware choice doesn't affect production.
   - Yes (rare) → You're an ML engineer, not a researcher. Optimize accordingly.

3. **What's my actual bottleneck?**
   - Time to publication → L4 (fast iteration).
   - Cost at billion-inference scale → Trainium / Inferentia (you're in industry territory).

---

## Bottom Line

For ~99% of academic ML research: you're training a **research model** that proves a concept. You're not deploying to millions of users; you're not serving billions of inferences. Your model will be cited by maybe 50 other researchers. If it's genuinely valuable, a company may adopt the idea and reimplement for production 3–5 years from now — but you won't be doing that deployment.

**Optimize for research velocity, not production cost.**

- Use L4 spot ($0.39/hr)
- Fast iteration (instant launch, 24 GB sufficient)
- Focus: time to publication, not cost per inference

Don't use Trainium. You're not in the production phase — you're in the research phase. Use research hardware.

**The 1% exception:** if you're actually deploying at scale (hospital using your model in the clinic, government policy tool, commercial product), then yes, use production hardware. But be honest with yourself first: are you really deploying to millions of users, or are you doing research?

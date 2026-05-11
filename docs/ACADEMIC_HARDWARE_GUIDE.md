# Academic Hardware Selection Guide

**Audience:** Researchers, PhD students, postdocs, faculty using ML in their work — biology, physics, economics, CS, anyone doing exploratory model development.

This guide answers the practical questions: *Which AWS instance fits my workload? When does Trainium actually make sense? How do I avoid the most common mistakes?*

---

## The Reality Check

### What Academic ML Research Actually Looks Like

Most academic ML work is:

- **Exploratory** — trying new architectures, changing things daily
- **Small-scale** — most models fit in 24 GB
- **Iterative** — fast feedback loops drive productivity
- **Educational** — teaching students, building intuition
- **One-off** — grant-funded projects with specific milestones

What it usually isn't:

- Production model training at scale
- Weekly retraining pipelines
- Finalized models running for months
- 24/7 inference serving
- Cost-optimized deployment

### What This Means for Trainium

Trainium is purpose-built for production training: compile-once, run-many. Long stable training runs, finalized architectures, predictable schedules. The compilation cost (2–8 hours for the first run) amortizes to zero when you train the same model weekly for a year.

Academic exploration is the opposite pattern: change the model, retrain, look at results, change again. The compilation cost dominates if the architecture changes daily.

**Conclusion: Trainium is a poor fit for most academic exploration work, but excellent for the narrow band of academic workflows that resemble production.**

### The "Build on Trainium" Caveat

AWS runs a "Build on Trainium" program at universities that provides credits and resources for Trainium adoption. This is genuinely useful for:

- Systems researchers studying custom silicon
- ML systems courses teaching AOT compilation
- Computer architecture students learning NeuronCore design

It's a poor fit for:

- Domain scientists using ML as a tool (biologists, physicists, etc.)
- PhD students exploring model architectures
- Most academic ML research that needs fast iteration

If your work falls into the second group, take the credits but use them for L4 / L40S / Trainium-when-it-fits — not as a Trainium-only deployment.

---

## Decision Heuristics

You don't need to understand CUDA memory hierarchy, XLA internals, or NeuronCore architecture to make a good choice. Three rules cover most cases:

1. **If your model fits in 24 GB → start with L4 (g6.xlarge).** Spot price ~$0.39/hr.
2. **If you change architecture daily → don't use Trainium.** Compilation overhead kills iteration.
3. **If you're waiting more than a day in a cluster queue → cloud is both faster AND cheaper** when you account for researcher time.

The honest trade-offs:

- **H100 has 26× more memory bandwidth than L4**, but if it's queued 4 days and L4 launches instantly, L4 wins on time-to-results.
- **Trainium compiles for 2–8 hours on first run.** Excellent for production retraining (compile once, run weekly). Poor for trying new architectures.

---

## Self-Assessment: Does Your Research Need H100?

A five-minute questionnaire. Answer honestly based on your *typical* workload, not your worst case.

```
1. What's your model size?
   ☐ <10 GB        → L4 (g6.xlarge, $0.39/hr) sufficient
   ☐ 10–40 GB      → L40S (g6e.xlarge, $0.75/hr) recommended
   ☐ 40–80 GB      → Try L40S first, then H100 if needed
   ☐ >80 GB        → H100 likely needed

2. How often do you change model architecture?
   ☐ Daily/weekly  → Use GPU (L4/L40S), avoid Trainium
   ☐ Monthly       → GPU for development; Trainium only if production-like
   ☐ Rarely        → Consider Trainium if runs >20 hours

3. Typical experiment duration?
   ☐ <1 hour       → Fast iteration matters; L4 spot
   ☐ 1–10 hours    → L4/L40S fine; queue time matters more than speed
   ☐ >10 hours     → Only consider Trainium if architecture is finalized

4. Experiments per week?
   ☐ 1–5           → Single instance sufficient
   ☐ 5–20          → Parallel instances pay off (10× L4 = $3.90/hr total)
   ☐ 20+           → Definitely cloud-parallel

5. Cluster queue time?
   ☐ <1 hour       → Use cluster for development
   ☐ 1–4 days      → Cloud faster for iteration
   ☐ >4 days       → Cloud essential for productivity

Interpretation:
- Mostly left column           → L4 spot is your answer
- Mixed                        → Portfolio: L4 + reserved H100 when needed
- Mostly right column          → Reserved capacity / multi-GPU territory
```

You don't need to understand memory-bound vs compute-bound. If your model trains, the hardware is sufficient. Optimization is a separate conversation.

---

## Common Mistakes

These are the patterns that produce bad first experiences with cloud ML — almost always avoidable.

**Mistake 1 — Starting with Trainium for exploratory work.**  
*Problem:* 2–8 hour compile per architecture change.  
*Fix:* Develop on L4. Only move to Trainium if your workflow resembles production retraining.

**Mistake 2 — Buying / reserving H100 for small models.**  
*Problem:* 95% of academic jobs fit on 24 GB, which an L4 has at 20× lower cost than an H100.  
*Fix:* Start with L4 spot. Upgrade only if you hit a real limit.

**Mistake 3 — Running experiments sequentially.**  
*Problem:* Wall-clock waste when the experiments are independent.  
*Fix:* Launch 10× L4 instances in parallel for the same total cost as 1× L4 sequential.

**Mistake 4 — Waiting in cluster queues.**  
*Problem:* Researcher time is the actual scarce resource. PhD-student fully loaded cost is ~$50–100/hr. Four-day queue ≈ $4,800 in researcher time.  
*Fix:* Cloud for iteration, cluster for final long runs.

**Mistake 5 — Not using spot.**  
*Problem:* On-demand is ~3× the spot price for the same hardware.  
*Fix:* Spot for research. Checkpointing handles preemption.

**Mistake 6 — Following "Build on Trainium" literally for non-systems research.**  
*Problem:* Compilation overhead kills iteration speed.  
*Fix:* Use Trainium only if you're studying systems or have a production-like workflow.

---

## Real-Talk FAQ

### "I'm a [biology / physics / economics] researcher using ML. Do I understand hardware enough?"

Probably not — and that's fine. You're an expert in your domain. Vendors use jargon that assumes systems expertise. Here's what you actually need to know:

- Your model size in GB: `sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9`
- Whether you're exploring (architecture changes) or producing (retraining a stable model)
- How long your typical experiment takes

That's it. Everything else is details.

### "NVIDIA says I need H100. AWS says try Trainium. Who's right?"

Both oversimplify. Real answer depends on your phase.

- **H100 excellent for:** models that don't fit in 24 GB, multi-GPU training, reserved capacity (no queue)
- **Trainium excellent for:** weekly retraining of the same model, training runs >20 hours, systems-research workloads
- **L4 excellent for:** ~90% of academic research — iterative development, anything that fits in 24 GB

Start with L4. Upgrade if you hit limits.

### "My institution bought a $2M H100 cluster. Should I use it or AWS?"

Depends on queue time.

- Queue <1 hour → use the cluster
- Queue 1–2 days → AWS for iteration, cluster for final runs
- Queue >2 days → AWS for everything

Your time has value. Don't wait four days to save $2 in compute.

### "I tried Trainium and compilation took six hours. Is something wrong?"

No, that's expected. Trainium uses Ahead-of-Time (AOT) compilation, not Just-in-Time (JIT) like CUDA.

- CUDA ≈ Python interpreter (start fast, decent speed)
- Trainium ≈ C++ compiler (slow compile, fast execution)

Trainium pays off when you compile once and run many times. NOT when you're trying new architectures daily, debugging convergence, or running quick experiments.

For most academic research: don't use Trainium for your main work.

### "AWS has a 'Build on Trainium' program at my university. Should I participate?"

Depends what you're researching.

Good fit:
- Studying ML systems or computer architecture
- Teaching a course on custom accelerators
- Production-like ML workflows
- Interested in learning AOT compilation as a skill

Poor fit:
- Domain research with ML as a tool (bio, physics, etc.)
- Exploring model architectures
- Early PhD stages (need fast iteration)
- Focused on results, not on studying systems

### "How do I know if my model is memory-bound or compute-bound?"

You probably don't need to. If your model trains, the hardware is sufficient. Most academic research isn't bottlenecked by hardware — it's bottlenecked by queues, sequential experiments, and insufficient hyperparameter sweeps. Fix those first.

### "When does Trainium actually make sense for academic research?"

Rarely, but specifically:

1. **Systems / architecture research** — you're studying custom silicon, NeuronCore architecture, or compiler optimizations.
2. **Production-like workflows** — weekly retraining on new data, long training runs (>100 hours) with a stable architecture, grant-funded compute where cost optimization matters.
3. **Never:** exploratory research with frequent architecture changes, typical PhD-student work, short experiments (<10 hours), or one-off "let me try this."

Realistically, well under 5% of academic ML research is a good fit.

---

## When Trainium Actually Fits in Academia

There are real Trainium use cases in academic research. Here are the four shapes they typically take:

### 1. Computer Architecture Research

You're researching Trainium itself: NeuronCore compiler optimizations, tile-based vs systolic architectures, compilation models. These researchers *should* use Trainium — they're studying it.

### 2. ML Engineering Programs

Professional masters programs training ML engineers. Teaching production ML workflows, compilation, optimization. Educational value is learning production skills, not maximizing research velocity.

### 3. Rare Production-Like Academic Workflows

Labs with ongoing data collection plus weekly retraining on a stable model. Examples:

- Climate lab retraining on new satellite data weekly
- Genomics lab running the same model on new sequences monthly
- Astronomy reprocessing new telescope data on schedule

Requirements: architecture is finalized, training runs >20 hours, predictable schedule, cost optimization matters.

This is maybe 1–2% of academic labs.

### 4. When It Doesn't Fit (~95% of academic work)

- Typical PhD-student workflows: exploring architectures, debugging convergence, short experiments, unpredictable schedule
- Typical postdoc/professor workflows: multiple projects, different models, grant-deadline-driven timing
- Typical undergrad research: learning ML, small datasets, rapid iteration, low budget

None of these match Trainium's compilation model.

---

## Summary Recommendations

**For ~90% of academic ML research:**

Use **g6.xlarge** (NVIDIA L4, ~$0.39/hr spot) — instant launch, 24 GB VRAM, elastic scaling, fast iteration.

**For ~5% of academic ML research:**

Use **p5.48xlarge** (8× NVIDIA H100, reserved) — models >80 GB, multi-GPU training, genuine premium-hardware need.

**For <5% of academic ML research:**

Use **trn1.2xlarge** (Trainium, ~$0.54/hr spot) — computer architecture research, production-like workflows (weekly retraining of stable models), ML systems education.

**For production inference / serving:**

Use **inf2.xlarge** (Inferentia, ~$0.30/hr spot) — purpose-built for inference, lowest $/inference.

Match the instance to the workload phase. Most academic work matches L4. Don't pay H100 prices for L4 needs, and don't pay Trainium compile-tax for work that changes daily.

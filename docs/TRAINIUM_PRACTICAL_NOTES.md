# Trainium Practical Notes

Three practical things to know about Trainium development that aren't obvious from the quickstart. They change the cost picture and lower the barrier to entry significantly.

---

## 1. Bedrock Distillation Often Beats DIY

If your goal is "I want a smaller, faster, cheaper model for my domain by distilling Claude or Llama," **you probably don't need to train anything yourself.**

### Amazon Bedrock Distillation

- Managed service that distills large models (Claude, Llama, Titan) into smaller custom models.
- You provide: examples of inputs/outputs from your domain.
- Bedrock handles: teacher inference, student training, infrastructure, compilation.
- No instance selection, no hardware decisions, no NEFF caching.

### When Bedrock is the right answer

- Distilling **from** a model that's already on Bedrock (Claude, Llama, Titan).
- You want the resulting smaller model for your domain.
- You don't care about distillation methods themselves — you just want the result.

### When DIY is the right answer

- You're **researching distillation methods** (the contribution is the method itself).
- The teacher model is custom and not on Bedrock.
- You're publishing on distillation algorithms.

The default for "I'm doing distillation" should be Bedrock. DIY distillation on Trainium (or anywhere) is a research workflow, not a production-application workflow.

---

## 2. Compile on Cheap x86, Run on Trainium

The Neuron compiler (`neuronx-cc`) is a **cross-compiler**: it runs on any x86_64 machine with the Neuron SDK installed, and emits Trainium machine code (NEFF files). Trainium hardware is not required for compilation.

This is the same model as `aarch64-gcc` on an x86 host emitting ARM binaries, or Rust cross-compiling to embedded targets.

### Why it matters

If you compile on Trainium itself (`trn1.32xlarge`, $21.50/hr OD), you pay the premium Trainium price while doing CPU work. If you compile on cheap x86 (`r7i.24xlarge`, $6.36/hr OD), you save ~70%.

### Why `r7i.24xlarge` specifically

Compilation needs a lot of host RAM. The Neuron compiler builds the full computation graph in memory:

- Loads every op in the graph
- Builds intermediate representations (HLO, XLA graphs) that can run 100–400 GB
- Models with FFT / STFT operations are especially memory-hungry (lots of unrolled ops)

Practical thresholds:

| Instance | RAM | Cost (OD) | Verdict |
|---|---|---|---|
| `r7i.12xlarge` | 384 GB | $3.18/hr | May OOM on large models |
| `r7i.24xlarge` | 768 GB | $6.36/hr | Sufficient for arena-scale models |
| `trn1.32xlarge` | 512 GB | $21.50/hr | Works but 3.4× more expensive |
| `trn1.2xlarge` | 32 GB | $1.34/hr | **Cannot compile** moderately complex models — OOM-killed mid-Tensorizer |

### Canonical workflow

```
Compile on r7i.24xlarge (cheap x86, 768 GB RAM):
  - 2–4 hours of compilation for typical workloads
  - Upload resulting NEFFs to S3

Run on trn1.2xlarge (cheap Trainium):
  - Set NEURON_COMPILE_CACHE_URL=s3://your-bucket/neuron-cache/
  - Trainium pulls compiled NEFFs from S3
  - Run inference / training at $0.54/hr spot
```

You pay for Trainium only when you're actually executing on Trainium.

---

## 3. Neuron Simulator Lets You Develop Without Hardware

AWS Neuron ships with a simulator that runs Trainium code on CPU. It's not a performance simulator (timing will be wrong), but it's a **functional simulator** — it catches the same compilation errors, XLA issues, and correctness bugs that you'd hit on real Trainium.

### What this enables

- **Develop on your laptop.** Install the Neuron SDK on macOS/Linux. Write PyTorch. Run with the simulator. Iterate quickly without spinning up cloud instances.
- **Validate correctness before deployment.** Catch the XLA-incompatible op or the shape mismatch on CPU, not on a $1.34/hr Trainium.
- **Lower the learning curve.** You can learn the Trainium development model — `mark_step()` semantics, what compiles and what doesn't, how the cache works — without spending anything on cloud compute.

### Limitations

- **Performance is wrong.** The simulator is CPU; it doesn't tell you how fast the real Trainium will be.
- **Some hardware edge cases differ.** Rare, but real — a few low-level Neuron primitives behave differently.
- **No multi-core simulation of NeuronCore parallelism.**

Use the simulator for "does this code work at all" and Trainium for "how fast is it."

---

## 4. Two paths for STFT/FFT models: XLA whole-graph vs. native-with-CPU-partition

For workloads that include FFT/STFT, the choice between the legacy XLA path and the current `torch_neuronx.trace()` native path is not just "old vs new" — it's a meaningful **runtime vs compile-time** tradeoff. Both work; they cost differently.

### Path A — Legacy XLA (`--device xla`)

torch_xla emits HLO IR for every op, including the Cooley-Tukey butterflies of `torch.stft`. The Neuron compiler unrolls this into a graph of hundreds of thousands of ops and produces a single NEFF that runs **entirely on NeuronCore**, FFT included.

- **Pros:** Full computation on Trainium silicon. No host↔Neuron transitions per forward. Best steady-state inference throughput.
- **Cons:** Compile is brutal — hours to days. The arena's GRU compile hit 14 hours with `--optlevel 3` before we capped it at `--optlevel 1` and got bounded (still-slow) compilation. See §3 of `TRAINIUM_NOTES.md`.
- **Mitigation:** Compile once on `r7i.24xlarge`, cache NEFFs in S3 (`NEURON_COMPILE_CACHE_URL=s3://...`). Subsequent loads are seconds. **This is what the arena's `s3://aws-arena-neuron-cache-scttfrdmn/trn1-xla/` cache is** — 509 MB of NEFFs from April 2026, 4-models-fully-on-Neuron.
- **Status:** Deprecated. AWS is moving off `torch_xla` for Trainium in favor of native `torch_neuronx`.

### Path B — Native `torch_neuronx.trace()` (current)

`torch.stft` isn't lowerable to NeuronCore in the native path. You handle this one of two ways:

1. **`PartitionerConfig` (automatic):** `torch_neuronx.PartitionerConfig(ops_to_partition={"aten::stft", "aten::istft", ...})` automatically ships the unsupported ops to host CPU, compiles the rest for Neuron, wires them together as a single ScriptModule.
2. **Manual core extraction (cleaner):** Refactor each model into a `Core` submodule that takes a magnitude/[real|imag] tensor and returns a mask — no STFT inside. Trace just the core. The host runs STFT/iSTFT on CPU around the traced call. See `scripts/trace_for_neuron.py`.

Either way, FFT runs on host CPU, not NeuronCore.

- **Pros:** Compile time drops to minutes (no whole-graph FFT unrolling). Fast iteration. No `r7i.24xlarge` needed for compilation.
- **Cons:** Per-forward overhead: STFT/iSTFT on CPU + host↔Neuron data transfers across the partition boundary. Empirically ~10–15 ms added per forward for this workload (out of ~100 ms total — 10–15%).
- **Status:** Supported. The path AWS is investing in.

### Which path when

| You care most about... | Path |
|---|---|
| Steady-state inference $/request, FFT in the model, will serve for months | A (XLA) — pay the compile cost once, get full Trainium runtime |
| Fast iteration, model architecture still changing, exploratory work | B (native) — minutes-not-hours compile, accept the partition overhead |
| Reproducibility across SDK upgrades | A (NEFFs are stable artifacts) or B + pin AMI per §5 |
| Just want it to work tomorrow | B with manual core extraction (`scripts/trace_for_neuron.py`) |

The deprecation of `torch_xla` for Trainium is a real **capability tradeoff**, not a pure improvement. The native API is faster to develop against; the XLA API was more capable for ops outside the dense-linear-algebra family. Both have legitimate use cases. Workshop attendees evaluating Trainium for FFT-heavy workloads should hear this honestly.

---

## 5. Pin a Personal AMI to Lock the Compiler Version

The NEFF cache is keyed on compiler version: `s3://your-bucket/.../neuronxcc-<VERSION>+<HASH>/MODULE_<hash>/...`. When AWS releases a new Neuron SDK (which they do every few weeks), the public DLAMI ships the new compiler, the cache key changes, and your existing NEFFs are stranded. A fresh `trn1.2xlarge` from the latest DLAMI will trigger a full recompile.

**That recompile fails on `trn1.2xlarge` for non-trivial workloads.** The 32 GB host RAM can't hold the Neuron compiler's whole-graph IR; you hit OOM in `neuronx-cc` mid-Tensorizer and the only recovery is cross-compiling on an `r7i.24xlarge` and uploading fresh NEFFs. Hours of work, no demo.

### The fix

Once you have a working setup — DLAMI + compiled NEFFs in S3 — **snapshot the instance as a private AMI** and pin to that AMI ID in your launch scripts. The compiler version is frozen, the cache keys stay valid, and a future launch is 30 seconds instead of a multi-hour rebuild.

```bash
# After your r7i.24xlarge compile run finishes and NEFFs are in S3:
aws ec2 create-image \
  --instance-id i-xxx \
  --name "arena-trn1-pinned-$(date +%Y%m%d)" \
  --description "DLAMI + compiled toolchain. Neuron SDK 2.29.1. NEFF cache: s3://aws-arena-neuron-cache-scttfrdmn/trn1-xla-2.29.1/" \
  --no-reboot

# Use the resulting AMI ID for all future demo launches:
aws ec2 run-instances --image-id ami-<your-snapshot> --instance-type trn1.2xlarge ...
```

The AMI captures the Neuron SDK, `neuronx-cc` compiler, Python venv, and any other preinstalled dependencies. When you launch from it, the compiler version stays frozen at 2.29.1 (or whatever you snapshot) regardless of what AWS ships in the public DLAMI catalog.

### When to re-cut the AMI

Only when you *intentionally* want a new compiler version:
- A bugfix lands in a newer Neuron SDK that affects your model
- You're starting a fresh project and want the current toolchain
- The AMI accumulates other staleness you care about (security patches, etc.)

Otherwise leave it. **Compiler stability is more valuable than running the latest SDK** for any workload that lives off a NEFF cache. This is the one place where "stale but working" beats "current but broken."

---

## Updated Decision Tree

Given these three tools, the practical Trainium workflow for academic researchers is:

```
What's your goal?

├─ Distill a model for my domain
│   ├─ Teacher on Bedrock? → Bedrock Distillation (managed)
│   └─ Custom teacher? → Continue below
│
├─ Research distillation methods
│   ├─ Functional testing? → Neuron simulator (free, on your laptop)
│   ├─ Performance check? → L4 ($0.39/hr spot)
│   └─ Production runs? → Compile on r7i.24xlarge, run on trn1.2xlarge
│
├─ Learn the Trainium / Neuron stack
│   ├─ Develop with Neuron simulator (free)
│   ├─ Compile on r7i.24xlarge ($6.36/hr OD)
│   └─ Run on trn1.2xlarge ($0.54/hr spot)
│
└─ Regular research (exploring ideas, iterating)
    └─ L4 (g6.xlarge, $0.39/hr spot) — stick with GPU; Trainium's compile tax
       isn't worth it for daily iteration
```

---

## Cost Comparison (DIY Distillation, 10 Domain Variants)

For grounding: distilling one teacher into 10 domain-specific student models.

| Path | Setup | Per-run | Total |
|---|---|---|---|
| **L4 only** | none | 40 hr × $0.39 spot = $15.60 | **$156** |
| **Trainium with cross-compile** | r7i.24xlarge × 4 hr = $25 | 40 hr × $0.54 spot = $21.60 | **$241** |
| **Bedrock Distillation** | none | per-distillation pricing | varies — usually cheapest if it fits your use case |

L4 wins on raw cost for this workload. Trainium pays off when (a) you're training the same model many times (compile amortized over far more than 10 runs), (b) you need NeuronCore-specific performance characteristics, or (c) you're learning the production-ML stack.

---

## Summary

- **Default for distillation:** Bedrock Distillation. Don't build infrastructure you don't need.
- **Compilation belongs on cheap x86**, not on Trainium. `r7i.24xlarge` is the right instance — it has the RAM for the compiler's whole-graph IR.
- **Neuron simulator** lets you develop and validate on a laptop. Use it before paying for any cloud compute.
- **Trainium is for production-like execution**, not for first-time exploration or compilation.

These three tools change Trainium from "expensive specialty hardware" to "one piece of a pipeline that's mostly cheap CPU work." The barrier to learning is much lower than it looks from the SDK docs alone.

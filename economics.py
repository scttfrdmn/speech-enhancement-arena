"""
Speech Enhancement Arena — Cost & Performance Economics

Reads arena training results and computes:
  - Cost per experiment (instance-hours × price)
  - MIG consolidation savings (4 experiments / 1 GPU vs. 4 GPUs)
  - Cross-instance comparison (G7e vs Trn1 vs Trn2 vs G5 vs P5)
  - Spot vs On-Demand economics
  - Scale projections for larger workloads (EEG foundation models)
  - Cost per quality unit ($/dB of SI-SDR improvement)

Pricing is baked in as constants for the workshop — update INSTANCE_CATALOG
if prices change.

Usage:
    # After running arena.py:
    python economics.py --log-dir logs

    # Hypothetical comparison across instance types:
    python economics.py --compare --training-hours 10

    # Scale projection for EEG foundation model:
    python economics.py --project --model-params 50e6 --dataset-hours 200 --epochs 100
"""

import argparse
import json
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Instance Catalog — update pricing here
# Prices are On-Demand in us-east-1 as of Q1 2026
# ---------------------------------------------------------------------------

@dataclass
class InstanceSpec:
    name: str
    price_on_demand: float   # $/hr
    price_spot: float        # $/hr (approximate, varies)
    accelerator: str         # GPU/chip name
    accel_count: int         # number of GPUs/chips
    accel_memory_gb: float   # per-accelerator memory
    fp16_tflops: float       # per-accelerator FP16 TFLOPS (approximate)
    supports_mig: bool       # MIG support
    mig_slices: int          # max MIG instances (1g profile)
    mig_memory_gb: float     # memory per MIG slice
    family: str              # "nvidia" or "trainium"
    notes: str


INSTANCE_CATALOG = {
    # ---- NVIDIA ----
    "g7e.2xlarge": InstanceSpec(
        name="g7e.2xlarge",
        price_on_demand=3.36,
        price_spot=1.35,       # ~60% discount typical
        accelerator="RTX PRO 6000 Blackwell",
        accel_count=1,
        accel_memory_gb=96,
        fp16_tflops=209,       # Blackwell RTX PRO 6000
        supports_mig=True,
        mig_slices=4,
        mig_memory_gb=24,
        family="nvidia",
        notes="New Blackwell, MIG 4x24GB, best for small-model training + ablations",
    ),
    "g6e.2xlarge": InstanceSpec(
        name="g6e.2xlarge",
        price_on_demand=1.86,
        price_spot=0.75,
        accelerator="L40S",
        accel_count=1,
        accel_memory_gb=48,
        fp16_tflops=91.6,
        supports_mig=False,
        mig_slices=0,
        mig_memory_gb=0,
        family="nvidia",
        notes="Previous gen, no MIG, good price/perf for single experiments",
    ),
    "g5.2xlarge": InstanceSpec(
        name="g5.2xlarge",
        price_on_demand=1.21,
        price_spot=0.48,
        accelerator="A10G",
        accel_count=1,
        accel_memory_gb=24,
        fp16_tflops=31.2,
        supports_mig=False,
        mig_slices=0,
        mig_memory_gb=0,
        family="nvidia",
        notes="Budget option, 24GB sufficient for speech enhancement models",
    ),
    "p5.48xlarge": InstanceSpec(
        name="p5.48xlarge",
        price_on_demand=32.77,
        price_spot=13.11,
        accelerator="H100 SXM",
        accel_count=8,
        accel_memory_gb=80,
        fp16_tflops=989.4,      # per GPU
        supports_mig=True,
        mig_slices=7,           # per GPU
        mig_memory_gb=10,       # 1g.10gb profile
        family="nvidia",
        notes="Overkill for speech enhancement, designed for frontier models",
    ),
    # ---- TRAINIUM ----
    "trn1.2xlarge": InstanceSpec(
        name="trn1.2xlarge",
        price_on_demand=1.34,
        price_spot=0.54,
        accelerator="Trainium1",
        accel_count=1,           # 1 chip, 2 NeuronCores
        accel_memory_gb=32,
        fp16_tflops=47.5,        # per chip (BF16)
        supports_mig=False,
        mig_slices=0,
        mig_memory_gb=0,
        family="trainium",
        notes="Best $/hr for Trainium, 2 NeuronCores, good for small models",
    ),
    "trn1.32xlarge": InstanceSpec(
        name="trn1.32xlarge",
        price_on_demand=21.50,
        price_spot=8.60,
        accelerator="Trainium1",
        accel_count=16,          # 16 chips, 32 NeuronCores
        accel_memory_gb=32,
        fp16_tflops=47.5,        # per chip
        supports_mig=False,
        mig_slices=0,
        mig_memory_gb=0,
        family="trainium",
        notes="Full Trn1 node, 32 NeuronCores, good for distributed training",
    ),
    "trn2.48xlarge": InstanceSpec(
        name="trn2.48xlarge",
        price_on_demand=16.02,
        price_spot=6.41,
        accelerator="Trainium2",
        accel_count=16,          # 16 chips, 64 NeuronCores
        accel_memory_gb=96,      # per chip
        fp16_tflops=190.0,       # per chip (BF16, approximate)
        supports_mig=False,
        mig_slices=0,
        mig_memory_gb=0,
        family="trainium",
        notes="Latest gen, 64 NeuronCores, best for foundation model training",
    ),
}


# ---------------------------------------------------------------------------
# Arena result reader
# ---------------------------------------------------------------------------

def read_arena_results(log_dir):
    """Read arena summary and per-model logs."""
    log_dir = Path(log_dir)
    summary_file = log_dir / "arena_summary.json"

    if summary_file.exists():
        with open(summary_file) as f:
            return json.load(f)

    # Fall back to reading individual logs
    results = []
    for log_file in sorted(log_dir.glob("arena_*.jsonl")):
        final = None
        header = None
        with open(log_file) as f:
            for line in f:
                entry = json.loads(line.strip())
                if entry["type"] == "header":
                    header = entry
                elif entry["type"] == "final":
                    final = entry
        if final and header:
            results.append({
                "model": header.get("model", log_file.stem),
                "best_loss": final.get("best_loss"),
                "total_time": final.get("total_time_sec"),
                "params": final.get("params"),
            })

    return {"results": results} if results else None


# ---------------------------------------------------------------------------
# Cost calculations
# ---------------------------------------------------------------------------

def compute_arena_cost(arena_results, instance_type="g7e.2xlarge", use_spot=False):
    """
    Compute the cost of the arena run on a given instance.

    Key insight: with MIG, all 4 models ran simultaneously on 1 instance,
    so the cost is max(training_times) × instance_price, not sum.
    Without MIG (sequential), cost is sum(training_times) × instance_price.
    """
    spec = INSTANCE_CATALOG.get(instance_type)
    if not spec:
        print(f"Unknown instance: {instance_type}")
        return None

    price = spec.price_spot if use_spot else spec.price_on_demand
    results = arena_results.get("results", [])
    if not results:
        return None

    times = [r.get("total_time", 0) for r in results]
    max_time_hrs = max(times) / 3600
    sum_time_hrs = sum(times) / 3600

    return {
        "instance": instance_type,
        "pricing": "Spot" if use_spot else "On-Demand",
        "price_per_hr": price,
        # MIG / parallel: you pay for wall-clock time of the longest run
        "parallel_time_hrs": max_time_hrs,
        "parallel_cost": max_time_hrs * price,
        # Sequential: you pay for total compute time
        "sequential_time_hrs": sum_time_hrs,
        "sequential_cost": sum_time_hrs * price,
        # Savings
        "mig_savings_pct": (1 - max_time_hrs / sum_time_hrs) * 100 if sum_time_hrs > 0 else 0,
        "cost_per_experiment_parallel": (max_time_hrs * price) / len(results),
        "cost_per_experiment_sequential": (sum_time_hrs * price) / len(results),
    }


def cross_instance_comparison(training_hours=1.0, num_experiments=4):
    """
    Compare cost across instance types for a given workload.
    Assumes models are small enough (1-4M params) to fit on any instance.

    Key insight: MIG slices don't run at 1/N speed for small models.
    Small models (<10M params) are memory-bandwidth and data-loading bound,
    not compute bound. A MIG 1g.24gb slice typically runs a 2M param speech
    model at ~75% the speed of the full GPU. We use a conservative 0.70x
    factor (the "MIG efficiency" for small models).

    For Trainium, each NeuronCore can run an independent model, so
    trn1.2xlarge (2 NeuronCores) can run 2 experiments in parallel.
    """
    MIG_EFFICIENCY = 0.70  # small model on MIG slice vs full GPU

    comparisons = []

    for inst_name, spec in INSTANCE_CATALOG.items():
        # Relative speed vs g7e baseline (single full GPU)
        baseline_tflops = INSTANCE_CATALOG["g7e.2xlarge"].fp16_tflops
        relative_speed = spec.fp16_tflops / baseline_tflops

        # Single-experiment time on the full accelerator
        single_exp_hours = training_hours / relative_speed

        if spec.supports_mig and spec.mig_slices >= num_experiments:
            # MIG: each slice runs at MIG_EFFICIENCY × full speed
            per_slice_hours = single_exp_hours / MIG_EFFICIENCY
            # All experiments run in parallel → wall clock = longest = per_slice_hours
            wall_clock_hours = per_slice_hours
            experiments_parallel = min(spec.mig_slices, num_experiments)
        elif spec.family == "trainium":
            # Trainium: NeuronCores can run independent models
            # trn1.2xlarge: 2 NeuronCores → 2 parallel
            # trn1.32xlarge: 32 NeuronCores → all 4 parallel (and then some)
            neuron_cores = spec.accel_count * (2 if "Trainium1" in spec.accelerator else 4)
            experiments_parallel = min(neuron_cores, num_experiments)
            batches = -(-num_experiments // experiments_parallel)  # ceil division
            wall_clock_hours = single_exp_hours * batches
        else:
            # Sequential: one experiment at a time
            experiments_parallel = 1
            wall_clock_hours = single_exp_hours * num_experiments

        cost_od = wall_clock_hours * spec.price_on_demand
        cost_spot = wall_clock_hours * spec.price_spot

        comparisons.append({
            "instance": inst_name,
            "accelerator": spec.accelerator,
            "family": spec.family,
            "relative_speed": round(relative_speed, 2),
            "mig_available": spec.supports_mig,
            "experiments_parallel": experiments_parallel,
            "wall_clock_hrs": round(wall_clock_hours, 3),
            "cost_on_demand": round(cost_od, 2),
            "cost_spot": round(cost_spot, 2),
            "cost_per_experiment_od": round(cost_od / num_experiments, 2),
            "notes": spec.notes,
        })

    comparisons.sort(key=lambda x: x["cost_on_demand"])
    return comparisons


def project_foundation_model(model_params=50e6, dataset_hours=200,
                             epochs=100, batch_size=32, sr=16000):
    """
    Project training cost for a larger workload (e.g., EEG foundation model).

    Rough estimation based on:
    - Samples per epoch = dataset_hours * 3600 * sr / (sr * 1.0)  [1-sec clips]
    - Steps per epoch = samples / batch_size
    - Time per step ≈ scaled from arena benchmarks
    """
    samples_per_epoch = int(dataset_hours * 3600)  # 1-second clips
    steps_per_epoch = samples_per_epoch / batch_size
    total_steps = steps_per_epoch * epochs

    # Rough: 50M param model, ~0.05s/step on g7e, ~0.06s/step on trn1
    projections = []

    for inst_name in ["g7e.2xlarge", "trn1.2xlarge", "trn2.48xlarge", "p5.48xlarge"]:
        spec = INSTANCE_CATALOG[inst_name]

        # Very rough step-time estimate based on TFLOPS
        # Calibrated: g7e ≈ 0.05s/step for 50M params
        base_step_time = 0.05  # seconds, on g7e
        base_tflops = INSTANCE_CATALOG["g7e.2xlarge"].fp16_tflops
        step_time = base_step_time * (base_tflops / spec.fp16_tflops)

        # For multi-chip instances, data parallelism helps
        effective_step_time = step_time / spec.accel_count

        total_time_hrs = (total_steps * effective_step_time) / 3600
        cost_od = total_time_hrs * spec.price_on_demand
        cost_spot = total_time_hrs * spec.price_spot

        projections.append({
            "instance": inst_name,
            "accelerator": spec.accelerator,
            "chips": spec.accel_count,
            "training_hours": round(total_time_hrs, 1),
            "training_days": round(total_time_hrs / 24, 1),
            "cost_on_demand": round(cost_od, 0),
            "cost_spot": round(cost_spot, 0),
        })

    return {
        "model_params": model_params,
        "dataset_hours": dataset_hours,
        "epochs": epochs,
        "total_steps": int(total_steps),
        "projections": projections,
    }


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_arena_economics(arena_results, instance_type="g7e.2xlarge"):
    """Print a complete economic analysis of the arena run."""

    print("\n" + "=" * 72)
    print("  COST & PERFORMANCE ECONOMICS")
    print("=" * 72)

    # Arena results on the actual instance
    for pricing_mode in ["On-Demand", "Spot"]:
        cost = compute_arena_cost(arena_results, instance_type,
                                  use_spot=(pricing_mode == "Spot"))
        if not cost:
            continue

        print(f"\n  [{instance_type} — {pricing_mode} @ ${cost['price_per_hr']:.2f}/hr]")
        print(f"  {'─' * 60}")
        print(f"  Parallel (MIG):     {cost['parallel_time_hrs']*60:.1f} min  →  "
              f"${cost['parallel_cost']:.2f}")
        print(f"  Sequential:         {cost['sequential_time_hrs']*60:.1f} min  →  "
              f"${cost['sequential_cost']:.2f}")
        print(f"  MIG savings:        {cost['mig_savings_pct']:.0f}%")
        print(f"  Cost/experiment:    ${cost['cost_per_experiment_parallel']:.3f} (parallel) "
              f"vs ${cost['cost_per_experiment_sequential']:.3f} (sequential)")

    # Cross-instance comparison
    print(f"\n\n  CROSS-INSTANCE COMPARISON (4 experiments, normalized to arena time)")
    print(f"  {'─' * 68}")
    print(f"  {'Instance':<18} {'Accelerator':<22} {'Parallel':>8} {'OD Cost':>9} {'Spot':>9} {'$/Exp':>8}")
    print(f"  {'─' * 68}")

    # Use the actual arena wall-clock time as baseline
    results = arena_results.get("results", [])
    if results:
        max_time = max(r.get("total_time", 0) for r in results)
        baseline_hrs = max_time / 3600
    else:
        baseline_hrs = 0.5  # default 30 min

    comparisons = cross_instance_comparison(
        training_hours=baseline_hrs,
        num_experiments=len(results) if results else 4,
    )

    for c in comparisons:
        flag = " ◀" if c["instance"] == instance_type else ""
        print(f"  {c['instance']:<18} {c['accelerator']:<22} "
              f"{c['wall_clock_hrs']*60:>6.1f}m "
              f"${c['cost_on_demand']:>8.2f} "
              f"${c['cost_spot']:>8.2f} "
              f"${c['cost_per_experiment_od']:>7.2f}{flag}")

    # Key takeaways
    cheapest = comparisons[0]
    print(f"\n  Cheapest (On-Demand): {cheapest['instance']} at "
          f"${cheapest['cost_on_demand']:.2f} total")

    # Find Trainium option
    trn_options = [c for c in comparisons if "trn" in c["instance"]]
    nvidia_options = [c for c in comparisons if c["instance"] == instance_type]
    if trn_options and nvidia_options:
        trn = trn_options[0]
        nv = nvidia_options[0]
        savings = (1 - trn["cost_on_demand"] / nv["cost_on_demand"]) * 100
        print(f"  Trainium savings:    {trn['instance']} is "
              f"{'cheaper' if savings > 0 else 'more expensive'} by "
              f"{abs(savings):.0f}% vs {nv['instance']}")

    # The MIG argument
    if results:
        n = len(results)
        total_sequential = sum(r.get("total_time", 0) for r in results) / 3600
        wall_clock = max(r.get("total_time", 0) for r in results) / 3600
        spec = INSTANCE_CATALOG.get(instance_type)
        if spec:
            print(f"\n  THE MIG ARGUMENT:")
            print(f"  Without MIG: {n} instances × {total_sequential/n*60:.0f} min = "
                  f"${total_sequential * spec.price_on_demand:.2f}")
            print(f"  With MIG:    1 instance × {wall_clock*60:.0f} min = "
                  f"${wall_clock * spec.price_on_demand:.2f}")
            print(f"  You just saved ${(total_sequential - wall_clock) * spec.price_on_demand:.2f} "
                  f"and {(total_sequential - wall_clock)*60:.0f} minutes of wall-clock time.")

    print()


def print_scale_projection():
    """Print cost projections for EEG foundation model training."""

    print("\n" + "=" * 72)
    print("  SCALE PROJECTION: EEG FOUNDATION MODEL")
    print("=" * 72)
    print("  Scenario: 50M parameter transformer, 200 hours of EEG data,")
    print("  100 epochs, batch size 32\n")

    proj = project_foundation_model()

    print(f"  Total training steps: {proj['total_steps']:,}")
    print(f"\n  {'Instance':<18} {'Chips':>6} {'Hours':>8} {'Days':>6} {'OD Cost':>10} {'Spot Cost':>10}")
    print(f"  {'─' * 64}")

    for p in proj["projections"]:
        print(f"  {p['instance']:<18} {p['chips']:>6} {p['training_hours']:>8.1f} "
              f"{p['training_days']:>6.1f} "
              f"${p['cost_on_demand']:>9,.0f} "
              f"${p['cost_spot']:>9,.0f}")

    # NSF budget context
    cheapest = min(proj["projections"], key=lambda p: p["cost_on_demand"])
    most_exp = max(proj["projections"], key=lambda p: p["cost_on_demand"])

    print(f"\n  For an NSF CAREER budget:")
    print(f"    Cheapest path: {cheapest['instance']} at ${cheapest['cost_on_demand']:,.0f} On-Demand "
          f"(${cheapest['cost_spot']:,.0f} Spot)")
    print(f"    vs GPU path:   {most_exp['instance']} at ${most_exp['cost_on_demand']:,.0f} On-Demand")
    print(f"    Savings:       ${most_exp['cost_on_demand'] - cheapest['cost_on_demand']:,.0f} "
          f"({(1 - cheapest['cost_on_demand']/most_exp['cost_on_demand'])*100:.0f}%)")

    print(f"\n  With Spot + checkpointing, the EEG foundation model")
    print(f"  could train for ~${cheapest['cost_spot']:,.0f} — potentially within")
    print(f"  a single NSF supplement request.\n")


def print_cost_per_quality(arena_results, instance_type="g7e.2xlarge"):
    """Cost per dB of SI-SDR improvement — the researcher's metric."""

    results = arena_results.get("results", [])
    if not results:
        return

    spec = INSTANCE_CATALOG.get(instance_type)
    if not spec:
        return

    print("\n  COST PER QUALITY")
    print(f"  {'─' * 56}")
    print(f"  {'Model':<25} {'SI-SDR':>8} {'Time':>8} {'Cost':>8} {'$/dB':>8}")
    print(f"  {'─' * 56}")

    for r in sorted(results, key=lambda x: x.get("best_loss", 0)):
        time_hrs = r.get("total_time", 0) / 3600
        cost = time_hrs * spec.price_on_demand
        # SI-SDR isn't directly in the summary — estimate from loss
        # (combined loss ≈ -SI-SDR + 0.5*STFT_loss, so rough SI-SDR ≈ -loss)
        si_sdr_est = -r.get("best_loss", 0)
        cost_per_db = cost / max(si_sdr_est, 0.1)

        model_name = r.get("description", r.get("model", "?"))[:25]
        print(f"  {model_name:<25} {si_sdr_est:>+7.1f} "
              f"{r.get('total_time', 0)/60:>7.1f}m "
              f"${cost:>7.3f} "
              f"${cost_per_db:>7.4f}")

    print()


def print_right_sizing():
    """
    The core argument: use appropriately-sized resources for the job.

    Researchers default to requesting the biggest GPU available because
    that's how HPC allocation works — you get a node, you use whatever's
    on it. Cloud flips this: you choose the resource that fits the workload,
    and you stop paying the moment you're done.

    This analysis shows what happens when you match instance to workload.
    """

    print("\n" + "=" * 72)
    print("  RIGHT-SIZING: MATCH THE RESOURCE TO THE WORKLOAD")
    print("=" * 72)

    # Speech enhancement models (1-4M params) — the ASPIRE workload
    print("\n  Your models are 1-4M parameters. Here's what they actually need:\n")

    models = [
        ("ConvMaskNet",   1.4e6, 0.08),  # params, VRAM in GB (model + batch + grads)
        ("CRMNet",        2.8e6, 0.14),
        ("AttentionMask", 3.3e6, 0.22),
        ("GatedRecurrent",2.1e6, 0.12),
    ]

    # Header
    print(f"  {'Model':<20} {'Params':>8} {'VRAM Used':>10} {'% of GPU Memory':>16}")
    print(f"  {'─' * 58}")

    instances_to_show = [
        ("H100 (80GB)",     80.0),
        ("RTX PRO 6000 (96GB)", 96.0),
        ("L40S (48GB)",     48.0),
        ("MIG 1g.24gb",    24.0),
        ("A10G (24GB)",    24.0),
        ("Trainium1 (32GB)", 32.0),
    ]

    # Show utilization for the largest model (AttentionMask, 3.3M)
    largest_vram = 0.22  # GB for AttentionMask with batch_size=32
    # Real VRAM: model weights + optimizer state (2x for Adam) + activations + batch
    # 3.3M params × 4 bytes = 13.2 MB weights, ×3 for Adam state = ~40 MB
    # Plus activations/gradients for batch_size=32, 1-sec audio: ~200 MB
    # Total: ~250 MB = 0.25 GB
    real_vram = 0.25

    for model_name, params, _ in models:
        # More realistic VRAM: params × 12 bytes (fp32 weights + Adam m/v) + activations
        param_bytes = params * 12  # weights + optimizer state
        activation_bytes = params * 4 * 32  # rough: proportional to params × batch_size
        total_bytes = param_bytes + activation_bytes
        total_gb = total_bytes / 1e9

        utilizations = []
        for gpu_name, gpu_mem in instances_to_show:
            pct = (total_gb / gpu_mem) * 100
            utilizations.append((gpu_name, pct))

        # Just show vs H100 and MIG
        h100_pct = total_gb / 80.0 * 100
        mig_pct = total_gb / 24.0 * 100
        print(f"  {model_name:<20} {params/1e6:>6.1f}M  {total_gb:>8.2f} GB   "
              f"H100: {h100_pct:.1f}%  |  MIG-24GB: {mig_pct:.1f}%")

    print(f"\n  {'─' * 72}")
    print(f"  What this means:\n")

    print(f"  ┌─────────────────────────────────────────────────────────────────┐")
    print(f"  │  Instance          GPU Mem   Your Model Uses   You're Paying   │")
    print(f"  │                              of Available       For Idle        │")
    print(f"  ├─────────────────────────────────────────────────────────────────┤")
    print(f"  │  p5.48xlarge       640 GB    < 0.1%            99.9% idle      │")
    print(f"  │    8× H100                                     $32.77/hr       │")
    print(f"  │                                                                │")
    print(f"  │  g7e.2xlarge       96 GB     ~1%               99% idle        │")
    print(f"  │    1× RTX PRO 6000                             $3.36/hr        │")
    print(f"  │                                                                │")
    print(f"  │  g7e.2xlarge MIG   24 GB     ~4%               96% idle        │")
    print(f"  │    1× MIG slice    (×4)      (but 4 models!)   $0.84/hr/model  │")
    print(f"  │                                                                │")
    print(f"  │  g5.2xlarge        24 GB     ~4%               96% idle        │")
    print(f"  │    1× A10G                                     $1.21/hr        │")
    print(f"  │                                                                │")
    print(f"  │  trn1.2xlarge      32 GB     ~3%               97% idle        │")
    print(f"  │    1× Trainium1                                $1.34/hr        │")
    print(f"  └─────────────────────────────────────────────────────────────────┘")

    print(f"\n  None of these instances are fully utilized by a 3M param model.")
    print(f"  But the cost difference is 25×: $32.77/hr vs $1.21/hr.")
    print(f"  With MIG, you get 4 experiments for $3.36/hr total — $0.84 each.")
    print()
    print(f"  The right question isn't 'what's the fastest GPU?'")
    print(f"  It's 'what's the cheapest resource that doesn't slow me down?'")
    print()
    print(f"  For 1-4M param speech enhancement models:")
    print(f"    • Ablation sweeps → G7e MIG (4 parallel, $0.84/hr each)")
    print(f"    • Single experiment → G5 Spot ($0.48/hr) or Trn1 ($0.54/hr)")
    print(f"    • Final training run → G7e full GPU ($3.36/hr) for fastest time")
    print(f"    • Foundation model → Trn2 ($16/hr) — this is where the big iron pays off")
    print()
    print(f"  This is the ladder: small resource for exploration, big resource for")
    print(f"  production. Cloud lets you step up and down. HPC allocations don't.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Speech Enhancement Arena — Economics")
    parser.add_argument("--log-dir", type=str, default="logs",
                        help="Directory with arena training logs")
    parser.add_argument("--instance", type=str, default="g7e.2xlarge",
                        help="Instance type used for the arena run")
    parser.add_argument("--compare", action="store_true",
                        help="Show cross-instance cost comparison")
    parser.add_argument("--project", action="store_true",
                        help="Show EEG foundation model cost projection")
    parser.add_argument("--training-hours", type=float, default=0.5,
                        help="Baseline training hours for comparison")
    args = parser.parse_args()

    # Read arena results if available
    arena_results = read_arena_results(args.log_dir)

    if arena_results and arena_results.get("results"):
        print_arena_economics(arena_results, args.instance)
        print_cost_per_quality(arena_results, args.instance)
    elif not args.compare and not args.project:
        print("[info] No arena results found in", args.log_dir)
        print("[info] Run arena.py first, or use --compare / --project for hypothetical analysis")

    if args.compare or not (arena_results and arena_results.get("results")):
        print("\n" + "=" * 72)
        print("  INSTANCE COMPARISON (hypothetical)")
        print("=" * 72)
        comps = cross_instance_comparison(training_hours=args.training_hours)
        print(f"\n  Scenario: 4 experiments, {args.training_hours:.1f}hr each on g7e baseline\n")
        print(f"  {'Instance':<18} {'Accelerator':<22} {'Wall':>7} {'OD Cost':>9} {'Spot':>9} {'MIG?':>5}")
        print(f"  {'─' * 68}")
        for c in comps:
            mig = f"{c['experiments_parallel']}x" if c["mig_available"] else "—"
            print(f"  {c['instance']:<18} {c['accelerator']:<22} "
                  f"{c['wall_clock_hrs']*60:>5.0f}m "
                  f"${c['cost_on_demand']:>8.2f} "
                  f"${c['cost_spot']:>8.2f} "
                  f"{mig:>5}")

    if args.project:
        print_scale_projection()

    # Always show right-sizing analysis
    print_right_sizing()

    # Always show the perishable rectangle insight
    print("\n" + "─" * 72)
    print("  THE PERISHABLE RECTANGLE")
    print("─" * 72)
    print("  Cloud GPU-hours are perishable inventory. Unlike on-prem HPC")
    print("  allocations (use-or-lose quarterly), you rent exactly what you")
    print("  need, when you need it. The economics favor bursty research:")
    print()
    print("    On-prem (OSC):  Wait 4 hours in queue → run 30 min → results")
    print("    Cloud (G7e):    Launch instantly → run 30 min → terminate → $1.68")
    print("    Cloud (Spot):   Launch instantly → run 30 min → terminate → $0.67")
    print()
    print("  Bank your steady-state work on OSC. Burst to cloud for deadlines,")
    print("  ablation sweeps, and anything that can't wait for a queue.")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()

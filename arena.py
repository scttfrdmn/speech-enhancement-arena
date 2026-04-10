#!/usr/bin/env python3
"""
Speech Enhancement Arena — Orchestrator

Launches 4 competing speech enhancement models simultaneously:
  - On NVIDIA G7e with MIG: 1 model per MIG 1g.24gb slice
  - On Trainium (TorchNeuron native): 1 model per NeuronCore (--device neuron)
  - On Trainium (legacy XLA): 1 model per NeuronCore (--device xla)
  - On CPU: 4 parallel processes (for testing)

This is the "holy shit" moment: 4 different architectures racing
each other in real-time, each on its own isolated GPU partition.

Usage:
    # NVIDIA with MIG (4x 1g.24gb slices already configured)
    python arena.py --device cuda

    # Trainium — TorchNeuron native (recommended, Neuron SDK 2.28+)
    python arena.py --device neuron

    # Trainium — TorchNeuron native with torch.compile
    python arena.py --device neuron --compile

    # Trainium — Legacy PyTorch/XLA path
    python arena.py --device xla

    # CPU testing
    python arena.py --device cpu --epochs 3

    # With real speech data
    python arena.py --device cuda --clean-dir /path/to/clean/wavs
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


MODELS = ["conv_mask", "crm", "attention", "gru"]

DESCRIPTIONS = {
    "conv_mask":  "Conv Encoder-Decoder (Magnitude Mask)",
    "crm":        "Complex Ratio Mask (Williamson & Wang 2017)",
    "attention":  "Multi-Head Attention Mask (SWIM-inspired)",
    "gru":        "Bidirectional GRU Mask (Nayem & Williamson)",
}


def discover_mig_uuids():
    """Discover MIG instance UUIDs from nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True, text=True, timeout=10,
        )
        uuids = []
        for line in result.stdout.strip().split("\n"):
            if "MIG" in line and "UUID" in line:
                # Extract MIG UUID
                uuid = line.split("UUID: ")[1].rstrip(")")
                uuids.append(uuid)
        return uuids
    except Exception as e:
        print(f"[warn] Could not discover MIG UUIDs: {e}")
        return []


def launch_arena(args):
    print("=" * 70)
    print("  SPEECH ENHANCEMENT ARENA")
    print("  4 models. 4 GPU slices. 1 winner.")
    print("=" * 70)
    print()

    # Setup directories
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Clean previous logs
    for f in log_dir.glob("arena_*.jsonl"):
        f.unlink()

    # Discover MIG slices (NVIDIA only)
    mig_uuids = []
    if args.device == "cuda":
        mig_uuids = discover_mig_uuids()
        if mig_uuids:
            print(f"[arena] Found {len(mig_uuids)} MIG instances:")
            for i, uuid in enumerate(mig_uuids):
                print(f"         Slice {i}: {uuid}")
        else:
            print("[arena] No MIG instances found — using ordinal device indices")
            print("[arena] (For MIG, run: sudo nvidia-smi mig -cgi 14,14,14,14 -C)")

    print()
    for i, model_name in enumerate(MODELS):
        print(f"  [{i}] {DESCRIPTIONS[model_name]:50s}")
    print()

    # Build launch commands
    train_script = str(Path(__file__).parent / "train.py")
    processes = []

    for i, model_name in enumerate(MODELS):
        run_id = f"arena_{model_name}"

        cmd = [
            sys.executable, train_script,
            "--model", model_name,
            "--run-id", run_id,
            "--device", args.device,
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--lr", str(args.lr),
            "--num-samples", str(args.num_samples),
            "--log-dir", str(log_dir),
            "--checkpoint-dir", str(ckpt_dir),
            "--num-workers", str(args.num_workers),
        ]

        if args.clean_dir:
            cmd.extend(["--clean-dir", args.clean_dir])

        if args.compile:
            cmd.append("--compile")

        # Set environment for device isolation
        env = os.environ.copy()

        if args.device == "cuda":
            if mig_uuids and i < len(mig_uuids):
                env["CUDA_VISIBLE_DEVICES"] = mig_uuids[i]
            else:
                env["CUDA_VISIBLE_DEVICES"] = str(i)

        elif args.device == "neuron":
            # TorchNeuron native: isolate to one NeuronCore per model
            env["NEURON_RT_VISIBLE_CORES"] = str(i)

        elif args.device == "xla":
            # Legacy XLA: same isolation mechanism
            env["NEURON_RT_VISIBLE_CORES"] = str(i)

        # Log file for stdout
        stdout_file = log_dir / f"{run_id}_stdout.log"

        print(f"[arena] Launching {model_name} on slice {i}...")
        with open(stdout_file, "w") as f_out:
            p = subprocess.Popen(cmd, env=env, stdout=f_out, stderr=subprocess.STDOUT)
            processes.append((model_name, p, stdout_file))

    print()
    print(f"[arena] All {len(processes)} models launched! Monitoring...")
    print("-" * 70)

    # Monitor progress
    start_time = time.time()
    completed = set()

    while len(completed) < len(processes):
        time.sleep(3)
        elapsed = time.time() - start_time

        status_line = f"\r  [{elapsed:6.1f}s] "
        for model_name, p, _ in processes:
            ret = p.poll()
            if ret is not None:
                completed.add(model_name)
                status_line += f" {model_name}:DONE"
            else:
                # Try to read latest metric
                log_file = log_dir / f"arena_{model_name}.jsonl"
                latest = _read_latest_metric(log_file)
                if latest:
                    status_line += f" {model_name}:E{latest.get('epoch','?')}({latest.get('si_sdr','?'):+.1f}dB)"
                else:
                    status_line += f" {model_name}:starting"

        print(status_line, end="", flush=True)

    total_time = time.time() - start_time
    print()
    print("-" * 70)
    print(f"\n[arena] All models finished in {total_time:.1f}s\n")

    # Collect results
    results = []
    for model_name, p, stdout_file in processes:
        log_file = log_dir / f"arena_{model_name}.jsonl"
        final = _read_final_metric(log_file)
        if final:
            results.append({
                "model": model_name,
                "description": DESCRIPTIONS[model_name],
                "best_loss": final.get("best_loss", float("inf")),
                "total_time": final.get("total_time_sec", 0),
                "params": final.get("params", 0),
                "exit_code": p.returncode,
            })
        else:
            results.append({
                "model": model_name,
                "description": DESCRIPTIONS[model_name],
                "best_loss": float("inf"),
                "total_time": 0,
                "params": 0,
                "exit_code": p.returncode,
            })

    # Sort by loss (lower = better)
    results.sort(key=lambda r: r["best_loss"])

    # Print scoreboard
    print("=" * 70)
    print("  ARENA RESULTS")
    print("=" * 70)
    print(f"  {'Rank':4s}  {'Model':20s}  {'Loss':>8s}  {'Time':>8s}  {'Params':>8s}")
    print("  " + "-" * 54)

    for i, r in enumerate(results):
        medal = ["🥇", "🥈", "🥉", "  "][i]
        params_str = f"{r['params']/1e6:.1f}M" if r['params'] > 0 else "?"
        print(f"  {medal}{i+1}    {r['description'][:20]:20s}  "
              f"{r['best_loss']:8.4f}  {r['total_time']:7.1f}s  {params_str:>8s}")

    print()
    winner = results[0]
    print(f"  WINNER: {winner['description']}")
    winner_model = winner["model"]
    print(f"  Checkpoint: {ckpt_dir / f'arena_{winner_model}_best.pt'}")
    print()
    print(f"  Total arena time: {total_time:.1f}s")
    print(f"  (Sequential would have taken: ~{sum(r['total_time'] for r in results):.1f}s)")
    print(f"  Speedup from parallel MIG: {sum(r['total_time'] for r in results) / total_time:.1f}x")
    print("=" * 70)

    # Write summary
    summary_file = log_dir / "arena_summary.json"
    with open(summary_file, "w") as f:
        json.dump({
            "results": results,
            "total_time_sec": total_time,
            "device": args.device,
            "epochs": args.epochs,
        }, f, indent=2)
    print(f"\n  Summary written to {summary_file}")

    return results


def _read_latest_metric(log_file):
    """Read the most recent epoch entry from a JSONL log."""
    if not log_file.exists():
        return None
    latest = None
    try:
        with open(log_file) as f:
            for line in f:
                entry = json.loads(line.strip())
                if entry.get("type") == "epoch":
                    latest = entry
    except (json.JSONDecodeError, IOError):
        pass
    return latest


def _read_final_metric(log_file):
    """Read the final summary entry from a JSONL log."""
    if not log_file.exists():
        return None
    try:
        with open(log_file) as f:
            for line in f:
                entry = json.loads(line.strip())
                if entry.get("type") == "final":
                    return entry
    except (json.JSONDecodeError, IOError):
        pass
    return None


def main():
    parser = argparse.ArgumentParser(description="Speech Enhancement Arena")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "mps", "neuron", "xla", "cpu"])
    parser.add_argument("--compile", action="store_true",
                        help="Pass --compile to each training process (torch.compile)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num-samples", type=int, default=5000)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--clean-dir", type=str, default=None)
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    args = parser.parse_args()
    launch_arena(args)


if __name__ == "__main__":
    main()

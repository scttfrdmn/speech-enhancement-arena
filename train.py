"""
Speech Enhancement Arena — Training Script

Trains a single speech enhancement model on a specified device.
Designed to be launched multiple times in parallel on MIG slices
or NeuronCores by the arena orchestrator.

Usage:
    # NVIDIA (specific MIG slice via CUDA_VISIBLE_DEVICES)
    CUDA_VISIBLE_DEVICES=0 python train.py --model crm --run-id crm_run1

    # Trainium — TorchNeuron native (Neuron SDK 2.28+, recommended)
    NEURON_RT_VISIBLE_CORES=0 python train.py --model crm --run-id crm_run1 --device neuron

    # Trainium — TorchNeuron native with torch.compile
    NEURON_RT_VISIBLE_CORES=0 python train.py --model crm --run-id crm_run1 --device neuron --compile

    # Trainium — Legacy PyTorch/XLA path (older Neuron SDK)
    NEURON_RT_VISIBLE_CORES=0 python train.py --model crm --run-id crm_run1 --device xla

    # CPU (testing)
    python train.py --model conv_mask --run-id test --device cpu --epochs 2
"""

import argparse
import json
import os
import time
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from models import get_model, count_params, list_models
from utils.data import make_dataloader


# ---------------------------------------------------------------------------
# Loss functions (standard in speech enhancement research)
# ---------------------------------------------------------------------------

def si_sdr_loss(estimate, target, eps=1e-8):
    """
    Scale-Invariant Signal-to-Distortion Ratio (SI-SDR).
    Negative because we minimize (higher SI-SDR = better).
    Standard metric in speech separation/enhancement.
    """
    target = target - target.mean(dim=-1, keepdim=True)
    estimate = estimate - estimate.mean(dim=-1, keepdim=True)

    dot = (target * estimate).sum(dim=-1, keepdim=True)
    s_target_energy = (target ** 2).sum(dim=-1, keepdim=True) + eps
    proj = dot * target / s_target_energy

    noise = estimate - proj
    si_sdr = 10 * torch.log10(
        (proj ** 2).sum(dim=-1) / ((noise ** 2).sum(dim=-1) + eps) + eps
    )
    return -si_sdr.mean()  # negative because we minimize


def multi_resolution_stft_loss(estimate, target, fft_sizes=(512, 1024, 2048)):
    """
    Multi-resolution STFT loss — combines spectral convergence and
    log magnitude loss at multiple resolutions. Better for capturing
    both fine and coarse spectral structure.
    """
    loss = 0.0
    for n_fft in fft_sizes:
        hop = n_fft // 4
        window = torch.hann_window(n_fft, device=estimate.device)

        est_stft = torch.stft(estimate, n_fft, hop, n_fft, window=window, return_complex=True)
        tgt_stft = torch.stft(target, n_fft, hop, n_fft, window=window, return_complex=True)

        est_mag = est_stft.abs()
        tgt_mag = tgt_stft.abs()

        # Spectral convergence
        sc = torch.norm(tgt_mag - est_mag, p="fro") / (torch.norm(tgt_mag, p="fro") + 1e-8)
        # Log magnitude loss
        log_mag = nn.functional.l1_loss(
            torch.log(est_mag + 1e-8),
            torch.log(tgt_mag + 1e-8)
        )
        loss += sc + log_mag

    return loss / len(fft_sizes)


def combined_loss(estimate, target):
    """SI-SDR + multi-resolution STFT — the good stuff."""
    return si_sdr_loss(estimate, target) + 0.5 * multi_resolution_stft_loss(estimate, target)


# ---------------------------------------------------------------------------
# Device setup
# ---------------------------------------------------------------------------

def setup_device(device_str):
    """
    Set up compute device. Supports:
      - "cuda"   — NVIDIA GPU (uses CUDA_VISIBLE_DEVICES for MIG slice selection)
      - "mps"    — Apple Silicon GPU (Metal Performance Shaders)
      - "neuron" — TorchNeuron native backend (Neuron SDK 2.28+, PyTorch 2.9+)
                   Supports eager mode, torch.compile, and native distributed APIs.
                   This is the recommended path for Trainium going forward.
      - "xla"    — Legacy PyTorch/XLA path (older Neuron SDK < 2.28)
                   Requires xm.mark_step() after each backward pass.
                   Will be deprecated when TorchNeuron transitions fully in PyTorch 2.10+.
      - "cpu"    — Fallback for testing without accelerators.
    """
    if device_str == "mps":
        if not torch.backends.mps.is_available():
            print("[warn] MPS not available, falling back to CPU")
            return torch.device("cpu"), "cpu"
        device = torch.device("mps")
        print("[device] Apple Silicon GPU via Metal Performance Shaders")
        return device, "mps"

    elif device_str == "cuda":
        if not torch.cuda.is_available():
            print("[warn] CUDA not available, falling back to CPU")
            return torch.device("cpu"), "cpu"
        device = torch.device("cuda:0")
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"[device] {name} ({mem:.1f} GB)")
        return device, "cuda"

    elif device_str == "neuron":
        # TorchNeuron native path (Neuron SDK 2.28+)
        # Registers a "neuron" backend with PyTorch's dispatcher.
        # Device is "xla" but the execution path is fundamentally different
        # from legacy PyTorch/XLA: eager mode works out of the box,
        # torch.compile uses the Neuron compiler backend, and no
        # xm.mark_step() is needed.
        try:
            import torch_neuronx
            device = torch.device("xla")
            print("[device] Trainium via TorchNeuron native (Neuron SDK 2.28+)")
            print("[device] Eager mode active — use --compile to enable torch.compile")
            return device, "neuron"
        except ImportError:
            print("[warn] torch_neuronx not available.")
            print("[warn] Install Neuron SDK 2.28+: pip install torch-neuronx")
            print("[warn] Falling back to CPU")
            return torch.device("cpu"), "cpu"

    elif device_str == "xla":
        # Legacy PyTorch/XLA path (Neuron SDK < 2.28)
        # This path requires xm.mark_step() to flush the XLA graph
        # after each training step. Being deprecated in favor of TorchNeuron.
        try:
            import torch_xla.core.xla_model as xm
            device = xm.xla_device()
            print("[device] Trainium via PyTorch/XLA (legacy path)")
            print("[device] NOTE: This path is being deprecated. Consider --device neuron")
            return device, "xla"
        except ImportError:
            print("[warn] torch_xla not available.")
            print("[warn] For Neuron SDK 2.28+, use --device neuron instead")
            print("[warn] Falling back to CPU")
            return torch.device("cpu"), "cpu"

    else:
        return torch.device("cpu"), "cpu"


# ---------------------------------------------------------------------------
# Metrics logger (writes JSON for the dashboard)
# ---------------------------------------------------------------------------

class MetricsLogger:
    """Writes per-epoch metrics to a shared JSON file for live dashboard."""

    def __init__(self, log_dir, run_id, model_name):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / f"{run_id}.jsonl"
        self.run_id = run_id
        self.model_name = model_name

        # Write header
        with open(self.log_file, "w") as f:
            json.dump({
                "type": "header",
                "run_id": run_id,
                "model": model_name,
                "timestamp": time.time(),
            }, f)
            f.write("\n")

    def log(self, epoch, metrics):
        entry = {
            "type": "epoch",
            "run_id": self.run_id,
            "model": self.model_name,
            "epoch": epoch,
            "timestamp": time.time(),
            **metrics,
        }
        with open(self.log_file, "a") as f:
            json.dump(entry, f)
            f.write("\n")

    def log_final(self, metrics):
        entry = {
            "type": "final",
            "run_id": self.run_id,
            "model": self.model_name,
            "timestamp": time.time(),
            **metrics,
        }
        with open(self.log_file, "a") as f:
            json.dump(entry, f)
            f.write("\n")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    # Device
    device, device_type = setup_device(args.device)

    # Model
    model = get_model(args.model).to(device)
    n_params = count_params(model)
    print(f"[model] {model.name} — {n_params/1e6:.1f}M parameters")
    print(f"[model] {model.description}")

    # Optional torch.compile
    # On TorchNeuron: uses the Neuron compiler backend to JIT-compile
    #   the forward+backward FX graphs into optimized Trainium instructions.
    #   First epoch is slower (compilation), subsequent epochs are faster.
    # On CUDA: uses torch.compile's default inductor backend.
    # On XLA (legacy): not supported — skip.
    if args.compile and device_type in ("neuron", "cuda", "mps"):
        print(f"[compile] Wrapping model with torch.compile (backend={'neuron' if device_type == 'neuron' else 'inductor'})")
        try:
            backend = None  # use default (inductor for CUDA, neuron for TorchNeuron)
            model = torch.compile(model)
            print(f"[compile] OK — first epoch will be slower due to compilation")
        except Exception as e:
            print(f"[compile] Failed: {e} — continuing without compilation")
    elif args.compile and device_type == "xla":
        print(f"[compile] torch.compile not supported on legacy XLA path — skipping")

    # Data
    loader = make_dataloader(
        clean_dir=args.clean_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_samples=args.num_samples,
        duration=args.duration,
        sr=args.sr,
        snr_range=(args.snr_min, args.snr_max),
    )

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Logger
    logger = MetricsLogger(args.log_dir, args.run_id, model.name)

    # Checkpoint dir
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Training
    print(f"\n[train] Starting {args.epochs} epochs, batch_size={args.batch_size}")
    print(f"[train] Loss: SI-SDR + Multi-Resolution STFT")
    print(f"[train] LR: {args.lr}, Run ID: {args.run_id}")
    print("-" * 60)

    best_loss = float("inf")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_si_sdr = 0.0
        num_batches = 0

        for batch_idx, (noisy, clean) in enumerate(loader):
            noisy = noisy.to(device)
            clean = clean.to(device)

            enhanced = model(noisy)

            # Ensure same length
            min_len = min(enhanced.shape[-1], clean.shape[-1])
            enhanced = enhanced[..., :min_len]
            clean = clean[..., :min_len]

            loss = combined_loss(enhanced, clean)
            si_sdr_val = -si_sdr_loss(enhanced, clean)  # positive = better

            optimizer.zero_grad()
            loss.backward()

            # Legacy XLA requires mark_step to flush the computation graph.
            # TorchNeuron native ("neuron") does NOT need this — it runs
            # in true eager mode where ops execute immediately.
            if device_type == "xla":
                try:
                    import torch_xla.core.xla_model as xm
                    xm.mark_step()
                except ImportError:
                    pass

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_si_sdr += si_sdr_val.item()
            num_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(num_batches, 1)
        avg_si_sdr = epoch_si_sdr / max(num_batches, 1)
        elapsed = time.time() - start_time
        lr = scheduler.get_last_lr()[0]

        metrics = {
            "loss": round(avg_loss, 4),
            "si_sdr": round(avg_si_sdr, 2),
            "lr": lr,
            "elapsed_sec": round(elapsed, 1),
        }
        logger.log(epoch, metrics)

        # Console output
        print(f"  Epoch {epoch:3d}/{args.epochs} | "
              f"Loss: {avg_loss:.4f} | "
              f"SI-SDR: {avg_si_sdr:+.2f} dB | "
              f"LR: {lr:.2e} | "
              f"Time: {elapsed:.0f}s")

        # Checkpoint best
        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt_path = ckpt_dir / f"{args.run_id}_best.pt"
            torch.save({
                "epoch": epoch,
                "model_name": args.model,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": best_loss,
                "si_sdr": avg_si_sdr,
                "params": n_params,
            }, ckpt_path)

    total_time = time.time() - start_time
    logger.log_final({
        "best_loss": round(best_loss, 4),
        "total_time_sec": round(total_time, 1),
        "params": n_params,
    })

    print("-" * 60)
    print(f"[done] {model.name} | Best loss: {best_loss:.4f} | "
          f"Total time: {total_time:.1f}s")
    print(f"[done] Checkpoint: {ckpt_dir / f'{args.run_id}_best.pt'}")

    return best_loss


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train a speech enhancement model")
    parser.add_argument("--model", type=str, default="conv_mask",
                        choices=["conv_mask", "crm", "attention", "gru"],
                        help="Model architecture")
    parser.add_argument("--run-id", type=str, default="run_0",
                        help="Unique run identifier (used for logging and checkpoints)")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "mps", "neuron", "xla", "cpu"],
                        help="Device: cuda (NVIDIA/MIG), mps (Apple Silicon), "
                             "neuron (TorchNeuron native), xla (legacy PyTorch/XLA), cpu")
    parser.add_argument("--compile", action="store_true",
                        help="Wrap model with torch.compile for JIT optimization. "
                             "On neuron: uses Neuron compiler backend. "
                             "On cuda: uses inductor. Not supported on legacy xla.")

    # Data
    parser.add_argument("--clean-dir", type=str, default=None,
                        help="Directory with clean .wav files (optional, uses synthetic if not set)")
    parser.add_argument("--num-samples", type=int, default=5000,
                        help="Number of synthetic samples per epoch")
    parser.add_argument("--duration", type=float, default=1.0,
                        help="Audio clip duration in seconds")
    parser.add_argument("--sr", type=int, default=16000, help="Sample rate")
    parser.add_argument("--snr-min", type=float, default=-5, help="Min SNR (dB)")
    parser.add_argument("--snr-max", type=float, default=15, help="Max SNR (dB)")

    # Training
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num-workers", type=int, default=2)

    # Output
    parser.add_argument("--log-dir", type=str, default="logs",
                        help="Directory for training logs (JSONL)")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                        help="Directory for model checkpoints")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()

"""
Speech Enhancement Arena — Training Script

Trains a single speech enhancement model on a specified device.
Supports small (1-4M params) and large (20-50M params) model scales,
mixed precision training, and per-step throughput logging.

Usage:
    # Large-scale benchmark on NVIDIA
    python train.py --model attention --scale large --device cuda --amp --epochs 40

    # MIG slice
    CUDA_VISIBLE_DEVICES=MIG-xxx python train.py --model crm --scale large --device cuda --amp

    # Quick test
    python train.py --model conv_mask --scale small --device cpu --epochs 2
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
import math

sys.path.insert(0, str(Path(__file__).parent))

from models import get_model, count_params, list_models
from utils.data import make_dataloader


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def si_sdr_loss(estimate, target, eps=1e-8):
    """Scale-Invariant Signal-to-Distortion Ratio (negative for minimization)."""
    target = target - target.mean(dim=-1, keepdim=True)
    estimate = estimate - estimate.mean(dim=-1, keepdim=True)

    dot = (target * estimate).sum(dim=-1, keepdim=True)
    s_target_energy = (target ** 2).sum(dim=-1, keepdim=True) + eps
    proj = dot * target / s_target_energy

    noise = estimate - proj
    si_sdr = 10 * torch.log10(
        (proj ** 2).sum(dim=-1) / ((noise ** 2).sum(dim=-1) + eps) + eps
    )
    return -si_sdr.mean()


def multi_resolution_stft_loss(estimate, target, fft_sizes=(512, 1024, 2048)):
    """Multi-resolution STFT loss: spectral convergence + log magnitude."""
    loss = 0.0
    for n_fft in fft_sizes:
        hop = n_fft // 4
        window = torch.hann_window(n_fft, device=estimate.device)

        est_stft = torch.stft(estimate, n_fft, hop, n_fft, window=window, return_complex=True)
        tgt_stft = torch.stft(target, n_fft, hop, n_fft, window=window, return_complex=True)

        est_mag = est_stft.abs()
        tgt_mag = tgt_stft.abs()

        sc = torch.norm(tgt_mag - est_mag, p="fro") / (torch.norm(tgt_mag, p="fro") + 1e-8)
        log_mag = nn.functional.l1_loss(
            torch.log(est_mag + 1e-8),
            torch.log(tgt_mag + 1e-8)
        )
        loss += sc + log_mag

    return loss / len(fft_sizes)


def combined_loss(estimate, target):
    """SI-SDR + multi-resolution STFT."""
    return si_sdr_loss(estimate, target) + 0.5 * multi_resolution_stft_loss(estimate, target)


# ---------------------------------------------------------------------------
# Device setup
# ---------------------------------------------------------------------------

def setup_device(device_str):
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
        props = torch.cuda.get_device_properties(0)
        mem = getattr(props, 'total_memory', getattr(props, 'total_mem', 0)) / 1e9
        print(f"[device] {name} ({mem:.1f} GB)")
        return device, "cuda"

    elif device_str == "neuron":
        try:
            import torch_neuronx
            device = torch.device("xla")
            print("[device] Trainium via TorchNeuron native (Neuron SDK 2.28+)")
            return device, "neuron"
        except ImportError:
            print("[warn] torch_neuronx not available, falling back to CPU")
            return torch.device("cpu"), "cpu"

    elif device_str == "xla":
        try:
            import torch_xla.core.xla_model as xm
            device = xm.xla_device()
            print("[device] Trainium via PyTorch/XLA (legacy path)")
            return device, "xla"
        except ImportError:
            print("[warn] torch_xla not available, falling back to CPU")
            return torch.device("cpu"), "cpu"

    else:
        return torch.device("cpu"), "cpu"


# ---------------------------------------------------------------------------
# System metrics collector
# ---------------------------------------------------------------------------

def collect_system_metrics(device_type):
    """Collect CPU, system memory, and GPU utilization metrics."""
    metrics = {}

    # CPU utilization (average over all cores since last call)
    try:
        import resource
        # Load average (1 min) — works on Linux and macOS
        load1, load5, _ = os.getloadavg()
        cpu_count = os.cpu_count() or 1
        metrics["cpu_load_1min"] = round(load1, 2)
        metrics["cpu_load_pct"] = round(load1 / cpu_count * 100, 1)
        metrics["cpu_count"] = cpu_count
    except Exception:
        pass

    # System memory
    try:
        if sys.platform == "linux":
            with open("/proc/meminfo") as f:
                meminfo = {}
                for line in f:
                    parts = line.split()
                    meminfo[parts[0].rstrip(":")] = int(parts[1])
                total = meminfo.get("MemTotal", 0) / 1024  # MB
                avail = meminfo.get("MemAvailable", 0) / 1024
                metrics["sys_mem_total_mb"] = round(total, 0)
                metrics["sys_mem_used_mb"] = round(total - avail, 0)
                metrics["sys_mem_util_pct"] = round((total - avail) / total * 100, 1) if total > 0 else 0
        elif sys.platform == "darwin":
            import subprocess
            result = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=2)
            lines = result.stdout.strip().split("\n")
            page_size = 16384  # Apple Silicon default
            stats = {}
            for line in lines[1:]:
                parts = line.split(":")
                if len(parts) == 2:
                    stats[parts[0].strip()] = int(parts[1].strip().rstrip("."))
            active = stats.get("Pages active", 0) * page_size / 1e6
            wired = stats.get("Pages wired down", 0) * page_size / 1e6
            compressed = stats.get("Pages occupied by compressor", 0) * page_size / 1e6
            metrics["sys_mem_used_mb"] = round(active + wired + compressed, 0)
    except Exception:
        pass

    # GPU metrics (CUDA)
    if device_type == "cuda":
        try:
            metrics["gpu_util_pct"] = torch.cuda.utilization(0)
            metrics["gpu_mem_used_mb"] = round(torch.cuda.memory_allocated(0) / 1e6, 1)
            props = torch.cuda.get_device_properties(0)
            total = getattr(props, 'total_memory', getattr(props, 'total_mem', 0))
            metrics["gpu_mem_total_mb"] = round(total / 1e6, 1)
            metrics["gpu_mem_peak_mb"] = round(torch.cuda.max_memory_allocated(0) / 1e6, 1)
        except Exception:
            pass

    return metrics


# ---------------------------------------------------------------------------
# Metrics logger
# ---------------------------------------------------------------------------

class MetricsLogger:
    def __init__(self, log_dir, run_id, model_name):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / f"{run_id}.jsonl"
        self.run_id = run_id
        self.model_name = model_name

        with open(self.log_file, "w") as f:
            json.dump({
                "type": "header",
                "run_id": run_id,
                "model": model_name,
                "timestamp": time.time(),
            }, f)
            f.write("\n")

    def log(self, epoch, metrics):
        entry = {"type": "epoch", "run_id": self.run_id, "model": self.model_name,
                 "epoch": epoch, "timestamp": time.time(), **metrics}
        with open(self.log_file, "a") as f:
            json.dump(entry, f)
            f.write("\n")

    def log_final(self, metrics):
        entry = {"type": "final", "run_id": self.run_id, "model": self.model_name,
                 "timestamp": time.time(), **metrics}
        with open(self.log_file, "a") as f:
            json.dump(entry, f)
            f.write("\n")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    device, device_type = setup_device(args.device)

    # Model
    model_kwargs = {}
    if args.n_fft:
        model_kwargs["n_fft"] = args.n_fft
    model = get_model(args.model, scale=args.scale, **model_kwargs).to(device)
    n_params = count_params(model)
    print(f"[model] {model.name} — {n_params/1e6:.1f}M parameters (scale={args.scale})")
    print(f"[model] {model.description}")

    # torch.compile
    if args.compile and device_type in ("neuron", "cuda", "mps"):
        print(f"[compile] Wrapping model with torch.compile")
        try:
            model = torch.compile(model)
            print(f"[compile] OK — first epoch will be slower due to compilation")
        except Exception as e:
            print(f"[compile] Failed: {e} — continuing without compilation")

    # Data
    loader = make_dataloader(
        clean_dir=args.clean_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_samples=args.num_samples,
        duration=args.duration,
        sr=args.sr,
        snr_range=(args.snr_min, args.snr_max),
        device=device,
        dataset=args.dataset,
    )

    # Optimizer with warmup + cosine decay
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    warmup_epochs = min(args.warmup_epochs, args.epochs // 4)

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / max(warmup_epochs, 1)
        progress = (epoch - warmup_epochs) / max(args.epochs - warmup_epochs, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # AMP setup
    use_amp = args.amp and device_type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    if use_amp:
        print(f"[amp] Mixed precision training enabled (FP16)")

    # Logger
    logger = MetricsLogger(args.log_dir, args.run_id, model.name)

    # Checkpoint dir
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Training
    print(f"\n[train] Starting {args.epochs} epochs, batch_size={args.batch_size}")
    print(f"[train] Loss: SI-SDR + Multi-Resolution STFT")
    print(f"[train] LR: {args.lr} (warmup={warmup_epochs} epochs), Run ID: {args.run_id}")
    print("-" * 60)

    best_loss = float("inf")
    start_time = time.time()
    total_samples = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_si_sdr = 0.0
        num_batches = 0
        epoch_start = time.time()
        epoch_samples = 0

        for batch_idx, (noisy, clean) in enumerate(loader):
            noisy = noisy.to(device)
            clean = clean.to(device)
            batch_size_actual = noisy.shape[0]

            optimizer.zero_grad()

            if use_amp:
                with torch.amp.autocast("cuda"):
                    enhanced = model(noisy)
                    min_len = min(enhanced.shape[-1], clean.shape[-1])
                    enhanced = enhanced[..., :min_len]
                    clean_trimmed = clean[..., :min_len]
                    loss = combined_loss(enhanced, clean_trimmed)

                si_sdr_val = -si_sdr_loss(enhanced.float(), clean_trimmed.float())
                scaler.scale(loss).backward()

                if device_type == "xla":
                    try:
                        import torch_xla.core.xla_model as xm
                        xm.mark_step()
                    except ImportError:
                        pass

                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                enhanced = model(noisy)
                min_len = min(enhanced.shape[-1], clean.shape[-1])
                enhanced = enhanced[..., :min_len]
                clean_trimmed = clean[..., :min_len]
                loss = combined_loss(enhanced, clean_trimmed)
                si_sdr_val = -si_sdr_loss(enhanced, clean_trimmed)

                loss.backward()

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
            epoch_samples += batch_size_actual

        scheduler.step()
        avg_loss = epoch_loss / max(num_batches, 1)
        avg_si_sdr = epoch_si_sdr / max(num_batches, 1)
        elapsed = time.time() - start_time
        epoch_time = time.time() - epoch_start
        lr = optimizer.param_groups[0]["lr"]
        total_samples += epoch_samples
        samples_per_sec = epoch_samples / max(epoch_time, 1e-6)

        # System & GPU utilization
        sys_metrics = collect_system_metrics(device_type)

        metrics = {
            "loss": round(avg_loss, 4),
            "si_sdr": round(avg_si_sdr, 2),
            "lr": lr,
            "elapsed_sec": round(elapsed, 1),
            "epoch_time_sec": round(epoch_time, 2),
            "samples_per_sec": round(samples_per_sec, 1),
            "total_samples": total_samples,
            **sys_metrics,
        }
        logger.log(epoch, metrics)

        # Console output with utilization info
        util_parts = []
        if "gpu_util_pct" in sys_metrics:
            util_parts.append(f"GPU:{sys_metrics['gpu_util_pct']}%")
        if "gpu_mem_used_mb" in sys_metrics:
            util_parts.append(f"VRAM:{sys_metrics['gpu_mem_used_mb']:.0f}/{sys_metrics['gpu_mem_total_mb']:.0f}MB")
        if "cpu_load_pct" in sys_metrics:
            util_parts.append(f"CPU:{sys_metrics['cpu_load_pct']:.0f}%")
        util_str = f" | {' '.join(util_parts)}" if util_parts else ""

        print(f"  Epoch {epoch:3d}/{args.epochs} | "
              f"Loss: {avg_loss:.4f} | "
              f"SI-SDR: {avg_si_sdr:+.2f} dB | "
              f"LR: {lr:.2e} | "
              f"{samples_per_sec:.0f} samp/s | "
              f"Time: {elapsed:.0f}s{util_str}")

        # Checkpoint best
        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt_path = ckpt_dir / f"{args.run_id}_best.pt"
            torch.save({
                "epoch": epoch,
                "model_name": args.model,
                "scale": args.scale,
                "n_fft": args.n_fft,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": best_loss,
                "si_sdr": avg_si_sdr,
                "params": n_params,
            }, ckpt_path)

    total_time = time.time() - start_time
    avg_throughput = total_samples / max(total_time, 1e-6)
    final_sys = collect_system_metrics(device_type)
    final_metrics = {
        "best_loss": round(best_loss, 4),
        "total_time_sec": round(total_time, 1),
        "params": n_params,
        "total_samples": total_samples,
        "avg_samples_per_sec": round(avg_throughput, 1),
        "scale": args.scale,
        **final_sys,
    }
    if device_type == "cuda":
        try:
            final_metrics["gpu_name"] = torch.cuda.get_device_name(0)
            peak = final_sys.get("gpu_mem_peak_mb", 0)
            total = final_sys.get("gpu_mem_total_mb", 1)
            if total > 0:
                final_metrics["gpu_mem_utilization_pct"] = round(peak / total * 100, 1)
        except Exception:
            pass
    logger.log_final(final_metrics)

    print("-" * 60)
    print(f"[done] {model.name} | Best loss: {best_loss:.4f} | "
          f"Total time: {total_time:.1f}s | "
          f"Throughput: {avg_throughput:.0f} samp/s")
    print(f"[done] Checkpoint: {ckpt_dir / f'{args.run_id}_best.pt'}")

    return best_loss


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train a speech enhancement model")
    parser.add_argument("--model", type=str, default="conv_mask",
                        choices=["conv_mask", "crm", "attention", "gru"])
    parser.add_argument("--scale", type=str, default="small",
                        choices=["small", "large"],
                        help="Model scale: small (1-5M) or large (20-50M)")
    parser.add_argument("--n-fft", type=int, default=None,
                        help="Override STFT FFT size (default: from scale config)")
    parser.add_argument("--run-id", type=str, default="run_0")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "mps", "neuron", "xla", "cpu"])
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--amp", action="store_true",
                        help="Enable mixed precision training (CUDA only)")

    # Data
    parser.add_argument("--dataset", type=str, default=None,
                        choices=["librispeech"],
                        help="Auto-download dataset (currently: librispeech)")
    parser.add_argument("--clean-dir", type=str, default=None)
    parser.add_argument("--num-samples", type=int, default=5000)
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--snr-min", type=float, default=-5)
    parser.add_argument("--snr-max", type=float, default=15)

    # Training
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=2)

    # Output
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()

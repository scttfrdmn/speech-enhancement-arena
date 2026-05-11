#!/usr/bin/env python3
"""Trace each arena model's FFT-free "core" for AWS Trainium inference.

Why this exists
---------------
`torch_neuronx.trace()` cannot trace models that call `torch.stft` (the
PyTorch 2.9 `as_strided` lowering of `stft_center` fails the XLA bounds
check). The supported AWS approach is to keep STFT/iSTFT on the host CPU
and only compile the dense-linear-algebra parts ("the core") for
NeuronCore. This script does that extraction + trace + S3 upload.

Each model's core takes a real-valued tensor (magnitude spectrogram, or
[real|imag] stack) and returns a real-valued mask. STFT and iSTFT are
done on the host CPU around the traced core at inference time.

Usage
-----
    # On a trn1.2xlarge with the Neuron PyTorch DLAMI:
    source /opt/aws_neuronx_venv_pytorch_2_9/bin/activate

    # Trace ConvMaskNet from a trained checkpoint, save locally + S3:
    python scripts/trace_for_neuron.py \\
        --checkpoint checkpoints/arena_conv_mask_best.pt \\
        --output traced/conv_mask_small.pt \\
        --s3 s3://your-bucket/traced-cores-2.29.1/conv_mask_small.pt

    # Or trace all four models from an arena run's checkpoint dir:
    python scripts/trace_for_neuron.py --all --checkpoint-dir checkpoints/ \\
        --output-dir traced/ \\
        --s3-prefix s3://aws-arena-neuron-cache-scttfrdmn/traced-cores-2.29.1/
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Repo on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models import get_model, MODEL_REGISTRY  # noqa: E402
from models.architectures import SCALE_CONFIGS  # noqa: E402


# ---------------------------------------------------------------------------
# Core wrappers — pure tensor → tensor, no STFT/iSTFT, traceable on Neuron
# ---------------------------------------------------------------------------

class ConvMaskCore(nn.Module):
    """ConvMaskNet's FFT-free core: takes magnitude, returns mask."""
    def __init__(self, m):
        super().__init__()
        self.encoder_blocks = m.encoder_blocks
        self.downsample = m.downsample
        self.bottleneck = m.bottleneck
        self.upsample = m.upsample
        self.decoder_blocks = m.decoder_blocks
        self.output_proj = m.output_proj

    def forward(self, mag):
        skips = []
        h = mag
        for enc, down in zip(self.encoder_blocks, self.downsample):
            h = enc(h)
            skips.append(h)
            h = down(h)
        h = self.bottleneck(h)
        for up, dec, skip in zip(self.upsample, self.decoder_blocks, reversed(skips)):
            h = up(h)
            if h.shape[-1] != skip.shape[-1]:
                h = F.pad(h, (0, skip.shape[-1] - h.shape[-1]))
            h = dec(torch.cat([h, skip], dim=1))
        mask = self.output_proj(h)
        return mask[..., :mag.shape[-1]]


class CRMCore(nn.Module):
    """CRMNet's core: takes [real|imag] concat, returns [mask_real|mask_imag]."""
    def __init__(self, m):
        super().__init__()
        self.input_proj = m.input_proj
        self.blocks = m.blocks
        self.output_proj = m.output_proj

    def forward(self, x_ri):
        h = self.input_proj(x_ri)
        for b in self.blocks:
            h = b(h)
        return torch.tanh(self.output_proj(h))


class AttnCore(nn.Module):
    """AttentionMask's core: takes magnitude, returns mask."""
    def __init__(self, m):
        super().__init__()
        self.input_proj = m.input_proj
        self.pos_enc = m.pos_enc
        self.transformer = m.transformer
        self.output_proj = m.output_proj

    def forward(self, mag):
        h = self.input_proj(mag)
        h = h.transpose(1, 2)
        h = self.pos_enc(h)
        h = self.transformer(h)
        mask = self.output_proj(h)
        return mask.transpose(1, 2)


class GRUCore(nn.Module):
    """GatedRecurrent's core: takes magnitude, returns mask.
    Uses _GRUOutputOnly wrappers (in architectures.py) so the GRU tuple
    return doesn't break the JIT tracer.
    """
    def __init__(self, m):
        super().__init__()
        self.pre_net = m.pre_net
        self.gru_fwd = m.gru_fwd     # already _GRUOutputOnly-wrapped
        self.gru_rev = m.gru_rev
        self.post_net = m.post_net

    def forward(self, mag):
        h = self.pre_net(mag)
        h = h.transpose(1, 2)
        h_fwd = self.gru_fwd(h)
        h_rev = self.gru_rev(torch.flip(h, dims=[1]))
        h = torch.cat([h_fwd, torch.flip(h_rev, dims=[1])], dim=-1)
        mask = self.post_net(h)
        return mask.transpose(1, 2)


CORE_BUILDERS = {
    "conv_mask": ConvMaskCore,
    "crm":       CRMCore,
    "attention": AttnCore,
    "gru":       GRUCore,
}


# ---------------------------------------------------------------------------
# Example-input shape per model + scale (must match the streaming server's
# context window: 0.5s at 16 kHz = 8000 samples, n_fft from scale config).
# ---------------------------------------------------------------------------

def make_example_input(model_key: str, scale: str, audio_samples: int = 8000):
    """Build a representative magnitude/RI tensor matching what the streaming
    server's CPU STFT will produce at inference time."""
    cfg = SCALE_CONFIGS[scale][model_key]
    n_fft = cfg.get("n_fft", 512)
    hop = n_fft // 4
    n_frames = (audio_samples - n_fft) // hop + 1 + 1  # +1 for center=True
    freq_bins = n_fft // 2 + 1
    # CRM uses [real|imag] = 2*freq_bins channels; others use mag = freq_bins
    if model_key == "crm":
        return torch.randn(1, 2 * freq_bins, n_frames)
    return torch.randn(1, freq_bins, n_frames).abs()  # magnitude is non-negative


# ---------------------------------------------------------------------------
# Trace
# ---------------------------------------------------------------------------

def trace_one(model_key: str, checkpoint_path: Path, output_path: Path,
              audio_samples: int = 8000) -> None:
    import torch_neuronx

    print(f"[trace] {model_key}: loading {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    scale = ckpt.get("scale", "small")
    n_fft = ckpt.get("n_fft") or SCALE_CONFIGS[scale][model_key].get("n_fft", 512)
    model = get_model(model_key, scale=scale, n_fft=n_fft).eval()
    model.load_state_dict(ckpt["model_state_dict"])

    core = CORE_BUILDERS[model_key](model).eval()
    example = make_example_input(model_key, scale, audio_samples)
    print(f"[trace] {model_key}: example input {tuple(example.shape)}, scale={scale}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        traced = torch_neuronx.trace(core, example)
    traced.save(str(output_path))
    size_mb = output_path.stat().st_size / 1e6
    print(f"[trace] {model_key}: saved {output_path} ({size_mb:.1f} MB)")


def upload_to_s3(local_path: Path, s3_uri: str) -> None:
    print(f"[s3]   uploading {local_path} -> {s3_uri}")
    subprocess.run(
        ["aws", "s3", "cp", str(local_path), s3_uri, "--quiet"],
        check=True,
        env={**os.environ, "AWS_DEFAULT_REGION": os.environ.get("AWS_DEFAULT_REGION", "us-west-2")},
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--checkpoint", type=Path,
                   help="Single checkpoint (e.g. checkpoints/arena_conv_mask_best.pt)")
    g.add_argument("--all", action="store_true",
                   help="Trace all four arena models from --checkpoint-dir")

    ap.add_argument("--model", choices=list(MODEL_REGISTRY),
                    help="With --checkpoint: which model key the checkpoint is for")
    ap.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"),
                    help="With --all: directory containing arena_<model>_best.pt")
    ap.add_argument("--output", type=Path,
                    help="With --checkpoint: where to write the traced .pt")
    ap.add_argument("--output-dir", type=Path, default=Path("traced"),
                    help="With --all: directory for traced .pt files")
    ap.add_argument("--s3", type=str,
                    help="With --checkpoint: full s3:// URI to upload to")
    ap.add_argument("--s3-prefix", type=str,
                    help="With --all: s3:// prefix; per-model files appended")
    ap.add_argument("--audio-samples", type=int, default=8000,
                    help="Example input length in samples (default: 8000 = 0.5s @ 16kHz)")

    args = ap.parse_args()

    if args.checkpoint:
        if not args.model:
            ap.error("--checkpoint requires --model")
        out = args.output or Path(f"traced/{args.model}_traced.pt")
        trace_one(args.model, args.checkpoint, out, args.audio_samples)
        if args.s3:
            upload_to_s3(out, args.s3)
        return

    # --all path
    for key in MODEL_REGISTRY:
        ckpt = args.checkpoint_dir / f"arena_{key}_best.pt"
        if not ckpt.exists():
            print(f"[skip] {key}: no checkpoint at {ckpt}")
            continue
        out = args.output_dir / f"{key}_traced.pt"
        trace_one(key, ckpt, out, args.audio_samples)
        if args.s3_prefix:
            ckpt_dict = torch.load(ckpt, map_location="cpu", weights_only=False)
            scale = ckpt_dict.get("scale", "small")
            upload_to_s3(out, f"{args.s3_prefix.rstrip('/')}/{key}_{scale}.pt")


if __name__ == "__main__":
    main()

"""
Speech Enhancement Arena — Data Pipeline

Supports three data sources:
  1. Synthetic formant-based speech (GPU-accelerated, no downloads)
  2. LibriSpeech train-clean-100 (~6GB, 28K utterances, auto-download)
  3. Custom WAV/FLAC directory (--clean-dir)

All sources add noise on-the-fly at random SNR levels.
"""

import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
import os
import random
import subprocess
import tarfile
from pathlib import Path
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Noise generators
# ---------------------------------------------------------------------------

def white_noise(length, device="cpu"):
    return torch.randn(length, device=device)


def pink_noise(length, device="cpu"):
    """Approximate 1/f noise via spectral shaping (vectorized)."""
    white = torch.randn(length, device=device)
    S = torch.fft.rfft(white)
    freqs = torch.fft.rfftfreq(length, device=device)
    freqs[0] = 1.0
    S = S / freqs.sqrt()
    S[0] = 0
    pink = torch.fft.irfft(S, n=length)
    pink = pink / (pink.abs().max() + 1e-8)
    return pink


def babble_noise(length, num_speakers=6, device="cpu"):
    """Simulated multi-talker babble from overlapping formant signals."""
    babble = torch.zeros(length, device=device)
    for _ in range(num_speakers):
        babble += _generate_formant_signal(length, device=device)
    return babble / num_speakers


def _generate_formant_signal(length, sr=16000, device="cpu"):
    """Generate a single synthetic speech-like signal with formant structure."""
    t = torch.linspace(0, length / sr, length, device=device)

    f0 = random.uniform(80, 300)
    formants = torch.tensor([
        random.uniform(200, 900),
        random.uniform(800, 2500),
        random.uniform(1800, 3500),
    ], device=device)
    bandwidths = torch.tensor([200.0, 300.0, 400.0], device=device)

    ks = torch.arange(1, 30, device=device, dtype=torch.float32)
    amps = 1.0 / (ks ** 1.2)
    phases = 2 * np.pi * f0 * ks.unsqueeze(1) * t.unsqueeze(0)
    signal = (amps.unsqueeze(1) * torch.sin(phases)).sum(0)

    freqs = torch.fft.rfftfreq(length, 1/sr, device=device)
    S = torch.fft.rfft(signal)
    formant_env = torch.zeros_like(freqs)
    for i in range(3):
        formant_env = formant_env + torch.exp(-0.5 * ((freqs - formants[i]) / bandwidths[i]) ** 2)
    signal = torch.fft.irfft(S * formant_env, n=length)

    envelope_freq = random.uniform(2, 6)
    envelope = 0.5 + 0.5 * torch.sin(2 * np.pi * envelope_freq * t + random.uniform(0, 2*np.pi))
    signal = signal * envelope

    signal = signal / (signal.abs().max() + 1e-8)
    return signal


# ---------------------------------------------------------------------------
# LibriSpeech download
# ---------------------------------------------------------------------------

def download_librispeech(data_dir="/data/librispeech", split="train-clean-100"):
    """Download LibriSpeech split if not already present. Returns path to audio files."""
    data_dir = Path(data_dir)
    audio_dir = data_dir / "LibriSpeech" / split

    if audio_dir.exists() and any(audio_dir.rglob("*.flac")):
        n_files = sum(1 for _ in audio_dir.rglob("*.flac"))
        print(f"[data] LibriSpeech {split} already downloaded ({n_files} files)")
        return str(audio_dir)

    data_dir.mkdir(parents=True, exist_ok=True)
    url = f"https://www.openslr.org/resources/12/{split}.tar.gz"
    tar_path = data_dir / f"{split}.tar.gz"

    print(f"[data] Downloading LibriSpeech {split} (~6GB)...")
    print(f"[data] URL: {url}")

    try:
        subprocess.run(
            ["wget", "-c", "-q", "--show-progress", "-O", str(tar_path), url],
            check=True, timeout=1800,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fallback to curl
        subprocess.run(
            ["curl", "-L", "-C", "-", "-o", str(tar_path), url],
            check=True, timeout=1800,
        )

    print(f"[data] Extracting...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=str(data_dir))

    # Clean up tarball to save disk
    tar_path.unlink(missing_ok=True)

    n_files = sum(1 for _ in audio_dir.rglob("*.flac"))
    print(f"[data] LibriSpeech {split} ready ({n_files} files)")
    return str(audio_dir)


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class SyntheticSpeechDataset(Dataset):
    """On-the-fly synthetic noisy/clean speech pairs."""

    def __init__(self, num_samples=5000, duration=1.0, sr=16000,
                 snr_range=(-5, 15), noise_types=None):
        self.num_samples = num_samples
        self.length = int(duration * sr)
        self.sr = sr
        self.snr_range = snr_range
        self.noise_types = noise_types or ["white", "pink", "babble"]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        clean = _generate_formant_signal(self.length)
        clean = clean / (clean.abs().max() + 1e-8) * 0.9

        noise_type = random.choice(self.noise_types)
        if noise_type == "white":
            noise = white_noise(self.length)
        elif noise_type == "pink":
            noise = pink_noise(self.length)
        elif noise_type == "babble":
            noise = babble_noise(self.length)
        else:
            noise = white_noise(self.length)

        snr_db = random.uniform(*self.snr_range)
        snr_linear = 10 ** (snr_db / 20)
        clean_rms = clean.pow(2).mean().sqrt()
        noise_rms = noise.pow(2).mean().sqrt()
        noise = noise * (clean_rms / (noise_rms * snr_linear + 1e-8))

        noisy = clean + noise
        peak = max(noisy.abs().max(), 1e-8)
        return noisy / peak, clean / peak


class AudioFileDataset(Dataset):
    """
    Loads clean speech files (.wav, .flac) from a directory and adds noise on-the-fly.
    Works with LibriSpeech (FLAC), VCTK (WAV), DNS Challenge, or any speech corpus.
    """

    def __init__(self, clean_dir, duration=3.0, sr=16000,
                 snr_range=(-5, 15), noise_types=None):
        self.files = sorted(
            list(Path(clean_dir).rglob("*.wav")) +
            list(Path(clean_dir).rglob("*.flac"))
        )
        if not self.files:
            raise ValueError(f"No .wav or .flac files found in {clean_dir}")
        self.duration = duration
        self.sr = sr
        self.length = int(duration * sr)
        self.snr_range = snr_range
        self.noise_types = noise_types or ["white", "pink", "babble"]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        wav, orig_sr = torchaudio.load(self.files[idx])
        wav = wav[0]  # mono

        if orig_sr != self.sr:
            wav = torchaudio.functional.resample(wav, orig_sr, self.sr)

        # Random crop or pad
        if wav.shape[0] >= self.length:
            start = random.randint(0, wav.shape[0] - self.length)
            clean = wav[start:start + self.length]
        else:
            clean = F.pad(wav, (0, self.length - wav.shape[0]))

        clean = clean / (clean.abs().max() + 1e-8) * 0.9

        noise_type = random.choice(self.noise_types)
        if noise_type == "white":
            noise = white_noise(self.length)
        elif noise_type == "pink":
            noise = pink_noise(self.length)
        else:
            noise = babble_noise(self.length)

        snr_db = random.uniform(*self.snr_range)
        snr_linear = 10 ** (snr_db / 20)
        clean_rms = clean.pow(2).mean().sqrt()
        noise_rms = noise.pow(2).mean().sqrt()
        noise = noise * (clean_rms / (noise_rms * snr_linear + 1e-8))

        noisy = clean + noise
        peak = max(noisy.abs().max(), 1e-8)
        return noisy / peak, clean / peak


# Keep old name as alias for backward compatibility
WavFileDataset = AudioFileDataset


# ---------------------------------------------------------------------------
# GPU batch generator — generates entire batches on-device
# ---------------------------------------------------------------------------

def _generate_batch_on_device(batch_size, length, sr, snr_range, device):
    """Generate a full batch of noisy/clean pairs directly on the accelerator."""
    clean_batch = torch.zeros(batch_size, length, device=device)
    for b in range(batch_size):
        clean_batch[b] = _generate_formant_signal(length, sr=sr, device=device)
    clean_batch = clean_batch / (clean_batch.abs().amax(dim=-1, keepdim=True) + 1e-8) * 0.9

    noise = torch.randn(batch_size, length, device=device)
    pink_mask = torch.rand(batch_size, device=device) > 0.5
    if pink_mask.any():
        S = torch.fft.rfft(noise[pink_mask])
        freqs = torch.fft.rfftfreq(length, device=device)
        freqs[0] = 1.0
        S = S / freqs.sqrt()
        S[:, 0] = 0
        noise[pink_mask] = torch.fft.irfft(S, n=length)

    snr_db = torch.empty(batch_size, 1, device=device).uniform_(snr_range[0], snr_range[1])
    snr_linear = 10 ** (snr_db / 20)
    clean_rms = clean_batch.pow(2).mean(dim=-1, keepdim=True).sqrt()
    noise_rms = noise.pow(2).mean(dim=-1, keepdim=True).sqrt()
    noise = noise * (clean_rms / (noise_rms * snr_linear + 1e-8))

    noisy = clean_batch + noise
    peak = noisy.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    return noisy / peak, clean_batch / peak


class GPUBatchGenerator:
    """
    Iterator that generates synthetic batches directly on the accelerator.
    Bypasses DataLoader entirely — no CPU-to-GPU transfer, no worker processes.
    """

    def __init__(self, num_samples=5000, batch_size=32, duration=1.0,
                 sr=16000, snr_range=(-5, 15), device="cpu"):
        self.num_batches = num_samples // batch_size
        self.batch_size = batch_size
        self.length = int(duration * sr)
        self.sr = sr
        self.snr_range = snr_range
        self.device = device

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        for _ in range(self.num_batches):
            yield _generate_batch_on_device(
                self.batch_size, self.length, self.sr,
                self.snr_range, self.device,
            )


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def make_dataloader(clean_dir=None, batch_size=32, num_workers=4,
                    device="cpu", **kwargs):
    """Create a dataloader.

    - If clean_dir is provided: uses AudioFileDataset with DataLoader
    - If dataset="librispeech": downloads and uses LibriSpeech train-clean-100
    - Otherwise: GPU batch generator for synthetic data
    """
    dataset_name = kwargs.pop("dataset", None)

    # LibriSpeech auto-download
    if dataset_name == "librispeech" and not clean_dir:
        clean_dir = download_librispeech()

    if clean_dir and Path(clean_dir).exists():
        duration = kwargs.get("duration", 3.0)
        sr = kwargs.get("sr", 16000)
        snr_range = kwargs.get("snr_range", (-5, 15))
        dataset = AudioFileDataset(clean_dir, duration=duration, sr=sr,
                                   snr_range=snr_range)
        print(f"[data] {len(dataset)} audio files from {clean_dir}")
        return DataLoader(
            dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, drop_last=True,
        )
    else:
        num_samples = kwargs.get("num_samples", 5000)
        duration = kwargs.get("duration", 1.0)
        sr = kwargs.get("sr", 16000)
        snr_range = kwargs.get("snr_range", (-5, 15))
        gen = GPUBatchGenerator(
            num_samples=num_samples, batch_size=batch_size,
            duration=duration, sr=sr, snr_range=snr_range,
            device=device,
        )
        print(f"[data] GPU batch generator ({num_samples} samples, "
              f"{len(gen)} batches on {device})")
        return gen


if __name__ == "__main__":
    ds = SyntheticSpeechDataset(num_samples=10)
    noisy, clean = ds[0]
    print(f"Noisy: {noisy.shape}, Clean: {clean.shape}")
    print(f"Noisy range: [{noisy.min():.3f}, {noisy.max():.3f}]")
    print(f"Clean range: [{clean.min():.3f}, {clean.max():.3f}]")
    print("Data generation OK.")

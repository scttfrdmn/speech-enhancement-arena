"""
Speech Enhancement Arena — Data Generation

Generates synthetic noisy speech pairs for training without external datasets.
For the workshop demo, this creates realistic-enough training data that the
model generalizes to real speech captured from a microphone.

Supports:
  - Synthetic formant-based "speech" (procedural, no files needed)
  - Loading real .wav files from a directory (for proper training)
  - Multiple noise types: white, pink, babble, environmental
  - Configurable SNR range
"""

import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
import os
import random
from pathlib import Path
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Noise generators
# ---------------------------------------------------------------------------

def white_noise(length, device="cpu"):
    return torch.randn(length, device=device)


def pink_noise(length, device="cpu"):
    """Approximate 1/f noise via filtering white noise."""
    white = torch.randn(length + 1024, device=device)
    # Simple IIR approximation of pink noise
    b = torch.tensor([0.049922035, -0.095993537, 0.050612699, -0.004709510], device=device)
    a = torch.tensor([1.0, -2.494956002, 2.017265875, -0.522189400], device=device)
    # Manual filter (avoid scipy dependency)
    pink = torch.zeros(length + 1024, device=device)
    for i in range(4, len(pink)):
        pink[i] = (b[0]*white[i] + b[1]*white[i-1] + b[2]*white[i-2] + b[3]*white[i-3]
                    - a[1]*pink[i-1] - a[2]*pink[i-2] - a[3]*pink[i-3])
    return pink[1024:1024+length]


def babble_noise(length, num_speakers=6, device="cpu"):
    """Simulated multi-talker babble from overlapping formant signals."""
    babble = torch.zeros(length, device=device)
    for _ in range(num_speakers):
        babble += _generate_formant_signal(length, device=device)
    return babble / num_speakers


def _generate_formant_signal(length, sr=16000, device="cpu"):
    """Generate a single synthetic speech-like signal with formant structure."""
    t = torch.linspace(0, length / sr, length, device=device)

    # Random fundamental frequency (pitch)
    f0 = random.uniform(80, 300)  # Hz

    # Random formants (vowel-like)
    formants = [
        random.uniform(200, 900),   # F1
        random.uniform(800, 2500),  # F2
        random.uniform(1800, 3500), # F3
    ]
    bandwidths = [80, 120, 150]

    # Generate glottal pulse train
    signal = torch.zeros_like(t)
    for k in range(1, 30):  # harmonics
        amp = 1.0 / (k ** 1.2)  # spectral tilt
        signal += amp * torch.sin(2 * np.pi * f0 * k * t)

    # Apply formant resonances (bandpass-like amplitude shaping)
    # This is a simplification — real formants are IIR filters
    # But for training data, amplitude modulation is sufficient
    for fc, bw in zip(formants, bandwidths):
        formant_env = torch.exp(-0.5 * ((torch.fft.rfftfreq(length, 1/sr, device=device) - fc) / bw) ** 2)
        S = torch.fft.rfft(signal)
        S = S * formant_env
        signal = torch.fft.irfft(S, n=length)

    # Random amplitude envelope (syllable-like rhythm)
    envelope_freq = random.uniform(2, 6)  # syllable rate
    envelope = 0.5 + 0.5 * torch.sin(2 * np.pi * envelope_freq * t + random.uniform(0, 2*np.pi))
    signal = signal * envelope

    # Normalize
    signal = signal / (signal.abs().max() + 1e-8)
    return signal


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SyntheticSpeechDataset(Dataset):
    """
    Generates on-the-fly noisy/clean speech pairs.
    No external files needed — perfect for workshop demos.
    """

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

        # Random noise type
        noise_type = random.choice(self.noise_types)
        if noise_type == "white":
            noise = white_noise(self.length)
        elif noise_type == "pink":
            noise = pink_noise(self.length)
        elif noise_type == "babble":
            noise = babble_noise(self.length)
        else:
            noise = white_noise(self.length)

        # Random SNR
        snr_db = random.uniform(*self.snr_range)
        snr_linear = 10 ** (snr_db / 20)

        clean_rms = clean.pow(2).mean().sqrt()
        noise_rms = noise.pow(2).mean().sqrt()
        noise = noise * (clean_rms / (noise_rms * snr_linear + 1e-8))

        noisy = clean + noise
        # Normalize to prevent clipping
        peak = max(noisy.abs().max(), 1e-8)
        noisy = noisy / peak
        clean = clean / peak

        return noisy, clean


class WavFileDataset(Dataset):
    """
    Loads clean .wav files from a directory and adds noise on-the-fly.
    Use this with real speech corpora (e.g., VCTK, LibriSpeech).
    """

    def __init__(self, clean_dir, duration=1.0, sr=16000,
                 snr_range=(-5, 15), noise_types=None):
        self.files = sorted(Path(clean_dir).glob("**/*.wav"))
        if not self.files:
            raise ValueError(f"No .wav files found in {clean_dir}")
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

        # Add noise (same as synthetic)
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


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def make_dataloader(clean_dir=None, batch_size=32, num_workers=4, **kwargs):
    """Create a dataloader — synthetic if no clean_dir, real files otherwise."""
    if clean_dir and Path(clean_dir).exists():
        dataset = WavFileDataset(clean_dir, **kwargs)
        print(f"[data] Loaded {len(dataset)} wav files from {clean_dir}")
    else:
        dataset = SyntheticSpeechDataset(**kwargs)
        print(f"[data] Using synthetic data ({len(dataset)} samples)")

    return DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )


if __name__ == "__main__":
    # Quick test
    ds = SyntheticSpeechDataset(num_samples=10)
    noisy, clean = ds[0]
    print(f"Noisy: {noisy.shape}, Clean: {clean.shape}")
    print(f"Noisy range: [{noisy.min():.3f}, {noisy.max():.3f}]")
    print(f"Clean range: [{clean.min():.3f}, {clean.max():.3f}]")
    print("Data generation OK.")

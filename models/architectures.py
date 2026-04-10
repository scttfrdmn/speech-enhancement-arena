"""
Speech Enhancement Arena — Model Architectures

Four architectures representing the spectrum of approaches in modern speech
enhancement research (aligned with ASPIRE group publications):

1. ConvMaskNet:   Conv encoder-decoder, magnitude mask (baseline)
2. CRMNet:        Complex Ratio Mask estimator (Williamson & Wang, 2017 style)
3. AttentionMask: Attention-based mask estimation (SWIM-inspired, 2024)
4. GatedRecurrent: GRU-based sequential mask (Nayem & Williamson style)

All models operate in the STFT domain, estimate masks, and reconstruct
enhanced audio — exactly the pipeline ASPIRE uses.

Each model is ~1-4M parameters, suitable for MIG 1g.24gb partitions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ---------------------------------------------------------------------------
# Shared STFT front-end / back-end
# ---------------------------------------------------------------------------

class STFTProcessor:
    """Shared STFT/iSTFT logic. Not a module — just config + methods."""

    def __init__(self, n_fft=512, hop_length=128, win_length=512):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

    def stft(self, x):
        """x: (B, T) -> complex spectrogram (B, F, N)"""
        window = torch.hann_window(self.win_length, device=x.device)
        return torch.stft(
            x, self.n_fft, self.hop_length, self.win_length,
            window=window, return_complex=True
        )

    def istft(self, X, length=None):
        """X: complex (B, F, N) -> (B, T)"""
        window = torch.hann_window(self.win_length, device=X.device)
        return torch.istft(
            X, self.n_fft, self.hop_length, self.win_length,
            window=window, length=length
        )


# ---------------------------------------------------------------------------
# Model 1: ConvMaskNet — Convolutional magnitude mask estimator
# ---------------------------------------------------------------------------

class ConvMaskNet(nn.Module):
    """
    Convolutional encoder-decoder that estimates a magnitude mask.
    The simplest baseline — purely feedforward, no recurrence.
    ~1.2M parameters.
    """
    name = "ConvMaskNet"
    description = "Conv encoder-decoder, magnitude mask"

    def __init__(self, n_fft=512):
        super().__init__()
        F = n_fft // 2 + 1  # 257 frequency bins

        self.encoder = nn.Sequential(
            nn.Conv1d(F, 256, 5, padding=2),
            nn.BatchNorm1d(256),
            nn.PReLU(),
            nn.Conv1d(256, 256, 5, padding=2),
            nn.BatchNorm1d(256),
            nn.PReLU(),
            nn.Conv1d(256, 256, 5, padding=2),
            nn.BatchNorm1d(256),
            nn.PReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(256, 256, 5, padding=2),
            nn.BatchNorm1d(256),
            nn.PReLU(),
            nn.Conv1d(256, F, 1),
            nn.Sigmoid(),  # magnitude mask in [0, 1]
        )
        self.stft_proc = STFTProcessor(n_fft=n_fft)

    def forward(self, noisy_wav):
        """noisy_wav: (B, T) -> enhanced_wav: (B, T)"""
        T = noisy_wav.shape[-1]
        X = self.stft_proc.stft(noisy_wav)          # (B, F, N) complex
        mag = X.abs()                                 # (B, F, N)
        phase = X.angle()

        mask = self.decoder(self.encoder(mag))        # (B, F, N)
        enhanced_mag = mag * mask
        enhanced = torch.polar(enhanced_mag, phase)   # back to complex
        return self.stft_proc.istft(enhanced, length=T)


# ---------------------------------------------------------------------------
# Model 2: CRMNet — Complex Ratio Mask estimator
# ---------------------------------------------------------------------------

class CRMNet(nn.Module):
    """
    Estimates a complex ratio mask (real + imaginary components).
    Directly inspired by Williamson & Wang (IEEE TASLP, 2017):
    "Time-frequency masking in the complex domain for speech
    dereverberation and denoising."
    ~2.8M parameters.
    """
    name = "CRMNet"
    description = "Complex Ratio Mask (Williamson & Wang 2017)"

    def __init__(self, n_fft=512, hidden=384):
        super().__init__()
        F = n_fft // 2 + 1

        # Input: concatenated real + imag (2*F channels)
        self.net = nn.Sequential(
            nn.Conv1d(2 * F, hidden, 5, padding=2),
            nn.BatchNorm1d(hidden),
            nn.PReLU(),
            nn.Conv1d(hidden, hidden, 5, padding=2),
            nn.BatchNorm1d(hidden),
            nn.PReLU(),
            nn.Conv1d(hidden, hidden, 3, padding=1),
            nn.BatchNorm1d(hidden),
            nn.PReLU(),
            nn.Conv1d(hidden, hidden, 3, padding=1),
            nn.BatchNorm1d(hidden),
            nn.PReLU(),
            nn.Conv1d(hidden, 2 * F, 1),  # output: real + imag mask
        )
        # Bounded complex mask using tanh compression
        self.stft_proc = STFTProcessor(n_fft=n_fft)

    def forward(self, noisy_wav):
        T = noisy_wav.shape[-1]
        X = self.stft_proc.stft(noisy_wav)

        # Stack real and imaginary as channels
        x_ri = torch.cat([X.real, X.imag], dim=1)   # (B, 2F, N)
        mask_ri = torch.tanh(self.net(x_ri))          # bounded [-1, 1]

        F = X.shape[1]
        mask_real = mask_ri[:, :F, :]
        mask_imag = mask_ri[:, F:, :]

        # Complex mask application
        enhanced_real = X.real * mask_real - X.imag * mask_imag
        enhanced_imag = X.real * mask_imag + X.imag * mask_real
        enhanced = torch.complex(enhanced_real, enhanced_imag)

        return self.stft_proc.istft(enhanced, length=T)


# ---------------------------------------------------------------------------
# Model 3: AttentionMask — Self-attention mask estimator
# ---------------------------------------------------------------------------

class AttentionMask(nn.Module):
    """
    Attention-based mask estimation using multi-head self-attention
    over time frames. Inspired by ASPIRE's SWIM model (2024):
    "An Attention-Only Model for Speech Quality Assessment."
    ~3.6M parameters.
    """
    name = "AttentionMask"
    description = "Multi-head attention mask (SWIM-inspired)"

    def __init__(self, n_fft=512, d_model=256, nhead=4, num_layers=4):
        super().__init__()
        F = n_fft // 2 + 1

        self.input_proj = nn.Sequential(
            nn.Conv1d(F, d_model, 1),
            nn.PReLU(),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1, batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_proj = nn.Sequential(
            nn.Linear(d_model, F),
            nn.Sigmoid(),
        )
        self.stft_proc = STFTProcessor(n_fft=n_fft)

    def forward(self, noisy_wav):
        T = noisy_wav.shape[-1]
        X = self.stft_proc.stft(noisy_wav)
        mag = X.abs()
        phase = X.angle()

        h = self.input_proj(mag)           # (B, d_model, N)
        h = h.transpose(1, 2)             # (B, N, d_model) for transformer
        h = self.transformer(h)            # (B, N, d_model)
        mask = self.output_proj(h)         # (B, N, F)
        mask = mask.transpose(1, 2)        # (B, F, N)

        enhanced = torch.polar(mag * mask, phase)
        return self.stft_proc.istft(enhanced, length=T)


# ---------------------------------------------------------------------------
# Model 4: GatedRecurrent — GRU-based sequential mask estimator
# ---------------------------------------------------------------------------

class GatedRecurrent(nn.Module):
    """
    Bidirectional GRU for mask estimation with intra-spectral dependencies.
    Inspired by Nayem & Williamson (ICASSP 2020, IEEE TASLP 2024):
    "Attention-Based Speech Enhancement Using Human Quality Perception."
    ~2.1M parameters.
    """
    name = "GatedRecurrent"
    description = "Bidirectional GRU mask (Nayem & Williamson)"

    def __init__(self, n_fft=512, hidden=256, num_layers=2):
        super().__init__()
        F = n_fft // 2 + 1

        self.input_bn = nn.BatchNorm1d(F)
        self.gru = nn.GRU(
            input_size=F, hidden_size=hidden,
            num_layers=num_layers, batch_first=True,
            bidirectional=True, dropout=0.1,
        )
        self.output_proj = nn.Sequential(
            nn.Linear(hidden * 2, F),
            nn.Sigmoid(),
        )
        self.stft_proc = STFTProcessor(n_fft=n_fft)

    def forward(self, noisy_wav):
        T = noisy_wav.shape[-1]
        X = self.stft_proc.stft(noisy_wav)
        mag = X.abs()
        phase = X.angle()

        h = self.input_bn(mag)            # (B, F, N)
        h = h.transpose(1, 2)            # (B, N, F)
        h, _ = self.gru(h)               # (B, N, 2*hidden)
        mask = self.output_proj(h)        # (B, N, F)
        mask = mask.transpose(1, 2)       # (B, F, N)

        enhanced = torch.polar(mag * mask, phase)
        return self.stft_proc.istft(enhanced, length=T)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    "conv_mask":   ConvMaskNet,
    "crm":         CRMNet,
    "attention":   AttentionMask,
    "gru":         GatedRecurrent,
}

def get_model(name, **kwargs):
    return MODEL_REGISTRY[name](**kwargs)

def list_models():
    return [(k, v.name, v.description) for k, v in MODEL_REGISTRY.items()]


# ---------------------------------------------------------------------------
# Parameter count utility
# ---------------------------------------------------------------------------

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("Speech Enhancement Arena — Model Architectures\n")
    for key, cls in MODEL_REGISTRY.items():
        m = cls()
        n = count_params(m)
        # Quick forward pass test
        x = torch.randn(2, 16000)  # 1 second of 16kHz audio
        y = m(x)
        print(f"  {cls.name:20s} | {n/1e6:.1f}M params | in={x.shape} -> out={y.shape} | {cls.description}")
    print("\nAll models OK.")

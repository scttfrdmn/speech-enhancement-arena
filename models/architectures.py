"""
Speech Enhancement Arena — Model Architectures

Four architectures representing the spectrum of approaches in modern speech
enhancement research (aligned with ASPIRE group publications):

1. ConvMaskNet:   Conv U-Net encoder-decoder, magnitude mask (baseline)
2. CRMNet:        Dilated TCN with Complex Ratio Mask (Williamson & Wang, 2017 style)
3. AttentionMask: Transformer-based mask estimation (SWIM-inspired, 2024)
4. GatedRecurrent: Deep Bidirectional GRU mask (Nayem & Williamson style)

All models operate in the STFT domain, estimate masks, and reconstruct
enhanced audio — exactly the pipeline ASPIRE uses.

Scale modes:
  - "small"  — 1-4M params (quick testing, workshop demos)
  - "large"  — 25-80M params (real benchmarks, GPU stress testing)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ---------------------------------------------------------------------------
# Shared STFT front-end / back-end
# ---------------------------------------------------------------------------

# trnfft provides Trainium-compatible STFT + ComplexTensor
# (trnfft incorporates neuron-complex-ops; see https://github.com/trnsci/trnfft)
try:
    import trnfft
    from trnfft import ComplexTensor
    _HAS_TRNFFT = True
except ImportError:
    _HAS_TRNFFT = False


def _is_xla_device(device):
    """Check if a device is XLA (Trainium/TPU)."""
    if device is None:
        return False
    if isinstance(device, torch.device):
        return device.type == "xla"
    return "xla" in str(device)


class STFTProcessor:
    """Shared STFT/iSTFT logic with pluggable backend.

    backend="native": uses torch.stft/istft (requires complex tensor support — CUDA, CPU, MPS)
    backend="trnfft": uses trnfft.stft/istft (works on Trainium; split real/imag)
    backend="auto":   uses trnfft backend if available and device is xla, else native
    """

    def __init__(self, n_fft=512, hop_length=None, win_length=None, backend="auto"):
        self.n_fft = n_fft
        self.hop_length = hop_length or n_fft // 4
        self.win_length = win_length or n_fft
        if backend == "neuron":
            backend = "trnfft"
        self.backend = backend

    def _use_trnfft(self, device=None):
        if self.backend == "trnfft":
            return True
        if self.backend == "native":
            return False
        return _HAS_TRNFFT and _is_xla_device(device)

    def stft(self, x):
        """x: (B, T) -> complex spectrogram (B, F, N)

        Returns torch complex tensor (native) or ComplexTensor (trnfft).
        Models should use .abs(), .angle(), .real, .imag which work on both.
        """
        window = torch.hann_window(self.win_length, device=x.device)
        if self._use_trnfft(x.device):
            return trnfft.stft(x, n_fft=self.n_fft, hop_length=self.hop_length,
                               win_length=self.win_length, window=window)
        return torch.stft(
            x, self.n_fft, self.hop_length, self.win_length,
            window=window, return_complex=True,
        )

    def istft(self, X, length=None):
        """X: complex (B, F, N) or ComplexTensor -> (B, T)"""
        device = X.real.device if hasattr(X, 'real') else getattr(X, 'device', None)
        window = torch.hann_window(self.win_length, device=device)
        if self._use_trnfft(device):
            return trnfft.istft(X, n_fft=self.n_fft, hop_length=self.hop_length,
                                win_length=self.win_length, window=window, length=length)
        return torch.istft(
            X, self.n_fft, self.hop_length, self.win_length,
            window=window, length=length,
        )


def _polar(mag, phase):
    """Create complex from magnitude and phase — works with both backends."""
    if _HAS_TRNFFT and _is_xla_device(mag.device):
        return ComplexTensor.from_polar(mag, phase)
    return torch.polar(mag, phase)


def _complex(real, imag):
    """Create complex from real and imag — works with both backends."""
    if _HAS_TRNFFT and _is_xla_device(real.device):
        return ComplexTensor(real, imag)
    return torch.complex(real, imag)


# ---------------------------------------------------------------------------
# Scale configs
# ---------------------------------------------------------------------------

SCALE_CONFIGS = {
    "small": {
        "conv_mask":  {"n_fft": 512, "hidden": 256, "depth": 3},
        "crm":        {"n_fft": 512, "hidden": 384, "num_layers": 4},
        "attention":  {"n_fft": 512, "d_model": 256, "nhead": 4, "num_layers": 4},
        "gru":        {"n_fft": 512, "hidden": 256, "num_layers": 2},
    },
    "large": {
        "conv_mask":  {"n_fft": 1024, "hidden": 512, "depth": 8},
        "crm":        {"n_fft": 1024, "hidden": 768, "num_layers": 10},
        "attention":  {"n_fft": 1024, "d_model": 512, "nhead": 8, "num_layers": 10},
        "gru":        {"n_fft": 1024, "hidden": 512, "num_layers": 4},
    },
}


# ---------------------------------------------------------------------------
# Model 1: ConvMaskNet — Convolutional U-Net magnitude mask estimator
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """Conv1d + BatchNorm + PReLU."""
    def __init__(self, in_ch, out_ch, kernel_size=5):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.PReLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ConvMaskNet(nn.Module):
    """
    Convolutional U-Net encoder-decoder that estimates a magnitude mask.
    Skip connections between encoder and decoder at each depth level.

    Small: ~1.4M params | Large: ~25M params
    """
    name = "ConvMaskNet"
    description = "Conv U-Net encoder-decoder, magnitude mask"

    def __init__(self, n_fft=512, hidden=256, depth=3):
        super().__init__()
        freq_bins = n_fft // 2 + 1

        # Encoder: progressive downsampling via stride-2 convolutions
        self.encoder_blocks = nn.ModuleList()
        self.downsample = nn.ModuleList()
        in_ch = freq_bins
        for i in range(depth):
            self.encoder_blocks.append(ConvBlock(in_ch, hidden, kernel_size=5))
            self.downsample.append(nn.Conv1d(hidden, hidden, 3, stride=2, padding=1))
            in_ch = hidden

        # Bottleneck
        self.bottleneck = nn.Sequential(
            ConvBlock(hidden, hidden * 2, kernel_size=3),
            ConvBlock(hidden * 2, hidden, kernel_size=3),
        )

        # Decoder: progressive upsampling with skip connections
        self.upsample = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        for i in range(depth):
            self.upsample.append(nn.ConvTranspose1d(hidden, hidden, 3, stride=2, padding=1, output_padding=1))
            # Skip connection doubles input channels
            self.decoder_blocks.append(ConvBlock(hidden * 2, hidden, kernel_size=5))

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Conv1d(hidden, freq_bins, 1),
            nn.Sigmoid(),
        )
        self.stft_proc = STFTProcessor(n_fft=n_fft)

    def forward(self, noisy_wav):
        T = noisy_wav.shape[-1]
        X = self.stft_proc.stft(noisy_wav)
        mag = X.abs()
        phase = X.angle()

        # Encoder with skip connections
        skips = []
        h = mag
        for enc, down in zip(self.encoder_blocks, self.downsample):
            h = enc(h)
            skips.append(h)
            h = down(h)

        h = self.bottleneck(h)

        # Decoder with skip connections
        for up, dec, skip in zip(self.upsample, self.decoder_blocks, reversed(skips)):
            h = up(h)
            # Match temporal dimension from skip
            if h.shape[-1] != skip.shape[-1]:
                h = F.pad(h, (0, skip.shape[-1] - h.shape[-1]))
            h = dec(torch.cat([h, skip], dim=1))

        mask = self.output_proj(h)
        # Match original spectrogram length
        mask = mask[..., :mag.shape[-1]]
        enhanced = _polar(mag * mask, phase)
        return self.stft_proc.istft(enhanced, length=T)


# ---------------------------------------------------------------------------
# Model 2: CRMNet — Dilated TCN with Complex Ratio Mask
# ---------------------------------------------------------------------------

class DilatedResBlock(nn.Module):
    """Dilated convolution residual block (TCN-style)."""
    def __init__(self, channels, kernel_size=3, dilation=1):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding=padding),
            nn.BatchNorm1d(channels),
            nn.PReLU(),
            nn.Conv1d(channels, channels, 1),
            nn.BatchNorm1d(channels),
        )
        self.act = nn.PReLU()

    def forward(self, x):
        return self.act(x + self.net(x))


class CRMNet(nn.Module):
    """
    Dilated Temporal Convolutional Network estimating a complex ratio mask.
    Inspired by Williamson & Wang (IEEE TASLP, 2017) and Conv-TasNet.

    Small: ~2.8M params | Large: ~45M params
    """
    name = "CRMNet"
    description = "Dilated TCN, Complex Ratio Mask (Williamson & Wang 2017)"

    def __init__(self, n_fft=512, hidden=384, num_layers=4):
        super().__init__()
        freq_bins = n_fft // 2 + 1

        # Input projection: real + imag concatenated
        self.input_proj = nn.Sequential(
            nn.Conv1d(2 * freq_bins, hidden, 1),
            nn.BatchNorm1d(hidden),
            nn.PReLU(),
        )

        # Dilated residual blocks with exponentially increasing dilation
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            dilation = 2 ** (i % 5)  # cycle: 1,2,4,8,16,1,2,4,8,16,...
            self.blocks.append(DilatedResBlock(hidden, kernel_size=3, dilation=dilation))

        # Output projection
        self.output_proj = nn.Conv1d(hidden, 2 * freq_bins, 1)

        self.stft_proc = STFTProcessor(n_fft=n_fft)
        self._freq_bins = freq_bins

    def forward(self, noisy_wav):
        T = noisy_wav.shape[-1]
        X = self.stft_proc.stft(noisy_wav)

        x_ri = torch.cat([X.real, X.imag], dim=1)
        h = self.input_proj(x_ri)

        for block in self.blocks:
            h = block(h)

        mask_ri = torch.tanh(self.output_proj(h))

        mask_real = mask_ri[:, :self._freq_bins, :]
        mask_imag = mask_ri[:, self._freq_bins:, :]

        enhanced_real = X.real * mask_real - X.imag * mask_imag
        enhanced_imag = X.real * mask_imag + X.imag * mask_real
        enhanced = _complex(enhanced_real, enhanced_imag)

        return self.stft_proc.istft(enhanced, length=T)


# ---------------------------------------------------------------------------
# Model 3: AttentionMask — Transformer mask estimator
# ---------------------------------------------------------------------------

class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding for transformer input."""
    def __init__(self, d_model, max_len=2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        """x: (B, N, d_model)"""
        return x + self.pe[:, :x.size(1), :]


class AttentionMask(nn.Module):
    """
    Transformer-based mask estimation using multi-head self-attention
    over time frames. Inspired by ASPIRE's SWIM model (2024).

    Small: ~3.3M params | Large: ~80M params
    """
    name = "AttentionMask"
    description = "Transformer mask (SWIM-inspired)"

    def __init__(self, n_fft=512, d_model=256, nhead=4, num_layers=4):
        super().__init__()
        freq_bins = n_fft // 2 + 1

        # Multi-layer input projection (deeper for larger models)
        self.input_proj = nn.Sequential(
            nn.Conv1d(freq_bins, d_model, 3, padding=1),
            nn.PReLU(),
            nn.Conv1d(d_model, d_model, 1),
            nn.PReLU(),
        )

        self.pos_enc = SinusoidalPositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1, batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.PReLU(),
            nn.Linear(d_model, freq_bins),
            nn.Sigmoid(),
        )
        self.stft_proc = STFTProcessor(n_fft=n_fft)

    def forward(self, noisy_wav):
        T = noisy_wav.shape[-1]
        X = self.stft_proc.stft(noisy_wav)
        mag = X.abs()
        phase = X.angle()

        h = self.input_proj(mag)           # (B, d_model, N)
        h = h.transpose(1, 2)             # (B, N, d_model)
        h = self.pos_enc(h)
        h = self.transformer(h)            # (B, N, d_model)
        mask = self.output_proj(h)         # (B, N, F)
        mask = mask.transpose(1, 2)        # (B, F, N)

        enhanced = _polar(mag * mask, phase)
        return self.stft_proc.istft(enhanced, length=T)


# ---------------------------------------------------------------------------
# Model 4: GatedRecurrent — Deep GRU-based sequential mask estimator
# ---------------------------------------------------------------------------

class GatedRecurrent(nn.Module):
    """
    Deep Bidirectional GRU with Conv pre/post-nets for mask estimation.
    Inspired by Nayem & Williamson (ICASSP 2020, IEEE TASLP 2024).

    Small: ~2.1M params | Large: ~35M params
    """
    name = "GatedRecurrent"
    description = "Deep BiGRU mask (Nayem & Williamson)"

    def __init__(self, n_fft=512, hidden=256, num_layers=2):
        super().__init__()
        freq_bins = n_fft // 2 + 1

        # Conv pre-net: frequency feature extraction
        self.pre_net = nn.Sequential(
            nn.Conv1d(freq_bins, hidden, 3, padding=1),
            nn.BatchNorm1d(hidden),
            nn.PReLU(),
            nn.Conv1d(hidden, hidden, 3, padding=1),
            nn.BatchNorm1d(hidden),
            nn.PReLU(),
        )

        gru_kwargs = dict(
            input_size=hidden, hidden_size=hidden,
            num_layers=num_layers, batch_first=True,
            dropout=0.1 if num_layers > 1 else 0,
        )
        self.gru_fwd = nn.GRU(bidirectional=False, **gru_kwargs)
        self.gru_rev = nn.GRU(bidirectional=False, **gru_kwargs)

        # Post-net: refine GRU output
        self.post_net = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.PReLU(),
            nn.Linear(hidden, freq_bins),
            nn.Sigmoid(),
        )
        self.stft_proc = STFTProcessor(n_fft=n_fft)

    def forward(self, noisy_wav):
        T = noisy_wav.shape[-1]
        X = self.stft_proc.stft(noisy_wav)
        mag = X.abs()
        phase = X.angle()

        h = self.pre_net(mag)             # (B, hidden, N)
        h = h.transpose(1, 2)            # (B, N, hidden)
        h_fwd, _ = self.gru_fwd(h)
        h_rev, _ = self.gru_rev(torch.flip(h, dims=[1]))
        h = torch.cat([h_fwd, torch.flip(h_rev, dims=[1])], dim=-1)
        mask = self.post_net(h)           # (B, N, F)
        mask = mask.transpose(1, 2)       # (B, F, N)

        enhanced = _polar(mag * mask, phase)
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

def get_model(name, scale="small", n_fft=None, **kwargs):
    """Get a model by name with optional scale preset.

    Args:
        name: model key from MODEL_REGISTRY
        scale: "small" (1-4M params) or "large" (25-80M params)
        n_fft: override n_fft from scale config
        **kwargs: additional overrides passed to model constructor
    """
    config = SCALE_CONFIGS.get(scale, SCALE_CONFIGS["small"]).get(name, {}).copy()
    if n_fft is not None:
        config["n_fft"] = n_fft
    config.update(kwargs)
    return MODEL_REGISTRY[name](**config)

def list_models():
    return [(k, v.name, v.description) for k, v in MODEL_REGISTRY.items()]


# ---------------------------------------------------------------------------
# Parameter count utility
# ---------------------------------------------------------------------------

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("Speech Enhancement Arena — Model Architectures\n")
    for scale in ["small", "large"]:
        print(f"  === {scale.upper()} ===")
        for key in MODEL_REGISTRY:
            m = get_model(key, scale=scale)
            n = count_params(m)
            cfg = SCALE_CONFIGS[scale][key]
            n_fft = cfg.get("n_fft", 512)
            duration = 1.0 if scale == "small" else 3.0
            x = torch.randn(2, int(16000 * duration))
            y = m(x)
            print(f"    {m.name:20s} | {n/1e6:5.1f}M params | "
                  f"in={x.shape} -> out={y.shape} | n_fft={n_fft}")
        print()
    print("All models OK.")

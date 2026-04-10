"""
Speech Enhancement Arena — Live Inference Demo

FastAPI server that:
1. Loads all 4 trained models (or just the arena winner)
2. Accepts audio via upload or WebSocket (for real-time mic input)
3. Returns enhanced audio + spectrograms from all models
4. Serves the web UI for the live A/B comparison

This is the "hear it" moment. Participants speak into the mic,
noise is added, and they hear each model's enhancement in real-time.

Usage:
    python serve.py --checkpoint-dir checkpoints --port 8000

    # Then open http://localhost:8000 in browser
"""

import argparse
import base64
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).parent))

from models import get_model, MODEL_REGISTRY, STFTProcessor
from utils.data import white_noise, pink_noise, babble_noise

try:
    from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    import uvicorn
except ImportError:
    print("Install dependencies: pip install fastapi uvicorn python-multipart")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------

app = FastAPI(title="Speech Enhancement Arena")
loaded_models = {}
device = torch.device("cpu")
stft_proc = STFTProcessor(n_fft=512, hop_length=128)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_models(checkpoint_dir, device_str="cpu"):
    global device, loaded_models
    device = torch.device(device_str if device_str != "xla" else "cpu")

    ckpt_dir = Path(checkpoint_dir)

    for model_key in MODEL_REGISTRY:
        # Try arena checkpoint first, then generic
        for pattern in [f"arena_{model_key}_best.pt", f"*{model_key}*best.pt"]:
            matches = list(ckpt_dir.glob(pattern))
            if matches:
                ckpt_path = matches[0]
                try:
                    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
                    model = get_model(model_key).to(device)
                    model.load_state_dict(ckpt["model_state_dict"])
                    model.eval()
                    loaded_models[model_key] = {
                        "model": model,
                        "loss": ckpt.get("loss", None),
                        "si_sdr": ckpt.get("si_sdr", None),
                        "epoch": ckpt.get("epoch", None),
                    }
                    print(f"  [loaded] {model_key} from {ckpt_path.name} "
                          f"(loss={ckpt.get('loss', '?'):.4f})")
                except Exception as e:
                    print(f"  [error] Failed to load {model_key}: {e}")
                break
        else:
            # No checkpoint — load with random weights (still useful for demo)
            model = get_model(model_key).to(device)
            model.eval()
            loaded_models[model_key] = {
                "model": model,
                "loss": None,
                "si_sdr": None,
                "epoch": None,
            }
            print(f"  [random] {model_key} (no checkpoint found, using random weights)")

    print(f"\n  {len(loaded_models)} models ready for inference.")


# ---------------------------------------------------------------------------
# Audio processing
# ---------------------------------------------------------------------------

def add_noise(audio, noise_type="white", snr_db=5):
    """Add noise to clean audio at specified SNR."""
    length = audio.shape[-1]
    if noise_type == "white":
        noise = white_noise(length, device=audio.device)
    elif noise_type == "pink":
        noise = pink_noise(length, device=audio.device)
    elif noise_type == "babble":
        noise = babble_noise(length, device=audio.device)
    else:
        noise = white_noise(length, device=audio.device)

    snr_linear = 10 ** (snr_db / 20)
    audio_rms = audio.pow(2).mean().sqrt()
    noise_rms = noise.pow(2).mean().sqrt()
    noise = noise * (audio_rms / (noise_rms * snr_linear + 1e-8))

    return audio + noise


def compute_spectrogram(audio, sr=16000):
    """Compute log-magnitude spectrogram for visualization."""
    X = stft_proc.stft(audio.unsqueeze(0) if audio.dim() == 1 else audio)
    mag = X.abs().squeeze(0).cpu().numpy()
    log_mag = np.log10(mag + 1e-8)
    # Normalize to [0, 1] for visualization
    log_mag = (log_mag - log_mag.min()) / (log_mag.max() - log_mag.min() + 1e-8)
    return log_mag


def audio_to_base64_wav(audio_tensor, sr=16000):
    """Convert audio tensor to base64-encoded WAV."""
    audio_tensor = audio_tensor.unsqueeze(0) if audio_tensor.dim() == 1 else audio_tensor
    audio_tensor = audio_tensor.cpu().float()
    # Normalize
    peak = audio_tensor.abs().max()
    if peak > 0:
        audio_tensor = audio_tensor / peak * 0.95

    buf = io.BytesIO()
    torchaudio.save(buf, audio_tensor, sr, format="wav")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def spectrogram_to_image_data(spec):
    """Convert spectrogram array to base64 PNG for display."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(6, 2.5), dpi=100)
        ax.imshow(spec, aspect="auto", origin="lower", cmap="magma",
                  vmin=0, vmax=1)
        ax.set_xlabel("Time")
        ax.set_ylabel("Freq")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout(pad=0.5)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    html_path = Path(__file__).parent / "static" / "index.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text())
    return HTMLResponse("<h1>Place index.html in static/</h1>")


@app.get("/api/models")
async def get_models():
    return JSONResponse({
        model_key: {
            "name": info["model"].name,
            "description": info["model"].description,
            "loss": info["loss"],
            "si_sdr": info["si_sdr"],
            "trained": info["epoch"] is not None,
        }
        for model_key, info in loaded_models.items()
    })


@app.post("/api/enhance")
async def enhance_audio(
    file: UploadFile = File(...),
    noise_type: str = "white",
    snr_db: float = 5.0,
):
    """
    Upload a WAV file, get back enhanced versions from all models.
    Optionally adds noise first (for demo purposes with clean recordings).
    """
    # Load audio
    audio_bytes = await file.read()
    buf = io.BytesIO(audio_bytes)
    audio, sr = torchaudio.load(buf)
    audio = audio[0]  # mono

    if sr != 16000:
        audio = torchaudio.functional.resample(audio, sr, 16000)
        sr = 16000

    audio = audio.to(device)
    audio = audio / (audio.abs().max() + 1e-8) * 0.9

    # Add noise if requested
    if snr_db < 100:  # snr_db=100 means "don't add noise"
        noisy = add_noise(audio, noise_type=noise_type, snr_db=snr_db)
    else:
        noisy = audio.clone()

    noisy = noisy / (noisy.abs().max() + 1e-8) * 0.9

    # Run all models
    results = {}
    for model_key, info in loaded_models.items():
        model = info["model"]
        t0 = time.time()
        with torch.no_grad():
            enhanced = model(noisy.unsqueeze(0)).squeeze(0)
        inference_time = time.time() - t0

        # Compute spectrograms
        spec = compute_spectrogram(enhanced)

        results[model_key] = {
            "name": model.name,
            "audio_b64": audio_to_base64_wav(enhanced, sr),
            "spectrogram_b64": spectrogram_to_image_data(spec),
            "inference_ms": round(inference_time * 1000, 1),
            "trained": info["epoch"] is not None,
        }

    # Also return original and noisy
    return JSONResponse({
        "original": {
            "audio_b64": audio_to_base64_wav(audio, sr),
            "spectrogram_b64": spectrogram_to_image_data(compute_spectrogram(audio)),
        },
        "noisy": {
            "audio_b64": audio_to_base64_wav(noisy, sr),
            "spectrogram_b64": spectrogram_to_image_data(compute_spectrogram(noisy)),
        },
        "enhanced": results,
        "sr": sr,
        "duration_sec": round(audio.shape[0] / sr, 2),
    })


@app.post("/api/enhance_raw")
async def enhance_raw(data: dict):
    """
    Accept raw float32 audio samples (from WebSocket/mic capture).
    Expects: { "samples": [...], "sr": 16000, "noise_type": "white", "snr_db": 5 }
    """
    samples = torch.tensor(data["samples"], dtype=torch.float32, device=device)
    sr = data.get("sr", 16000)
    noise_type = data.get("noise_type", "white")
    snr_db = data.get("snr_db", 5.0)

    if sr != 16000:
        samples = torchaudio.functional.resample(samples, sr, 16000)

    samples = samples / (samples.abs().max() + 1e-8) * 0.9
    noisy = add_noise(samples, noise_type=noise_type, snr_db=snr_db)
    noisy = noisy / (noisy.abs().max() + 1e-8) * 0.9

    results = {}
    for model_key, info in loaded_models.items():
        model = info["model"]
        t0 = time.time()
        with torch.no_grad():
            enhanced = model(noisy.unsqueeze(0)).squeeze(0)
        inference_time = time.time() - t0

        results[model_key] = {
            "name": model.name,
            "audio_b64": audio_to_base64_wav(enhanced, 16000),
            "inference_ms": round(inference_time * 1000, 1),
        }

    return JSONResponse({
        "noisy": {"audio_b64": audio_to_base64_wav(noisy, 16000)},
        "enhanced": results,
    })


# ---------------------------------------------------------------------------
# Training log streaming (for live dashboard during arena)
# ---------------------------------------------------------------------------

@app.get("/api/training_logs")
async def get_training_logs():
    """Read all arena training logs for the live dashboard."""
    log_dir = Path("logs")
    all_runs = {}

    for log_file in sorted(log_dir.glob("arena_*.jsonl")):
        run_data = {"epochs": [], "header": None, "final": None}
        try:
            with open(log_file) as f:
                for line in f:
                    entry = json.loads(line.strip())
                    if entry["type"] == "header":
                        run_data["header"] = entry
                    elif entry["type"] == "epoch":
                        run_data["epochs"].append(entry)
                    elif entry["type"] == "final":
                        run_data["final"] = entry
        except (json.JSONDecodeError, IOError):
            continue

        if run_data["header"]:
            all_runs[run_data["header"]["run_id"]] = run_data

    return JSONResponse(all_runs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Speech Enhancement Arena — Live Demo")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda", "neuron", "xla"])
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  SPEECH ENHANCEMENT ARENA — Live Demo Server")
    print("=" * 60 + "\n")

    load_models(args.checkpoint_dir, args.device)

    print(f"\n  Server starting at http://{args.host}:{args.port}")
    print(f"  Open in browser to start the demo.\n")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

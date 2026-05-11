"""Streaming arena — Day 2: live model inference.

WebSocket server that accepts 48 kHz int16 PCM frames from the browser,
runs the selected speech-enhancement model on the incoming stream, and
replies with enhanced PCM at the same rate. Wire format is byte-for-byte
compatible with the Day 1 echo server, so the existing client
(`stream/client/index.html`) connects without modification.

All models in the checkpoint dir are loaded at startup. The client
selects a model via WebSocket query param: `ws://host/ws?model=conv_mask`.

MVP streaming strategy: accumulate-and-batch. Every `--window` seconds
of input we run a single forward pass and emit the full processed
window. Latency = `--window` + forward + 2×resample (≈ 1 s at default
settings). Day 3+ will move to a sliding-window strategy for lower
latency once we have GPU/MPS in the loop.

Run:
    uv run python stream/server/inference.py \\
        --checkpoint-dir checkpoints_libri \\
        --device cpu \\
        --port 8765
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from models import get_model, SCALE_CONFIGS  # noqa: E402

log = logging.getLogger("stream.infer")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

app = FastAPI(title="Streaming Arena — Inference")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)
CLIENT_DIR = Path(__file__).resolve().parent.parent / "client"

# Globals populated at startup
MODELS: dict[str, torch.nn.Module] = {}   # key → loaded eval model
MODEL_META: dict[str, dict] = {}          # key → {scale, params, loss, n_fft}
DEFAULT_MODEL: str = "conv_mask"
DEVICE: Optional[torch.device] = None
CONTEXT_SECS: float = 0.5   # how much history the model sees each forward pass
HOP_SECS: float = 0.1       # how often we emit (= latency floor)
INPUT_BOOST: float = 6.0    # pre-model gain to reach training-distribution level
CAPTURE_RATE: int = 48000
MODEL_RATE: int = 16000
RESAMPLE_DOWN: Optional[torchaudio.transforms.Resample] = None
RESAMPLE_UP: Optional[torchaudio.transforms.Resample] = None


_MODEL_KEYS = ["conv_mask", "crm", "attention", "gru"]


def load_all_models(checkpoint_dir: Path, device: torch.device) -> None:
    """Load every recognised checkpoint in *checkpoint_dir* into MODELS."""
    global MODELS, MODEL_META, DEFAULT_MODEL

    for model_key in _MODEL_KEYS:
        patterns = [f"arena_{model_key}_best.pt", f"*{model_key}*best.pt"]
        ckpt_path = None
        for pat in patterns:
            matches = list(checkpoint_dir.glob(pat))
            if matches:
                ckpt_path = matches[0]
                break
        if ckpt_path is None:
            log.info("skip %s: no checkpoint found in %s", model_key, checkpoint_dir)
            continue

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        scale = ckpt.get("scale", "small")
        n_params = ckpt.get("params", "?")
        loss = ckpt.get("loss", None)
        n_fft = SCALE_CONFIGS[scale][model_key].get("n_fft", 512)

        model = get_model(model_key, scale=scale).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        MODELS[model_key] = model
        MODEL_META[model_key] = {
            "scale": scale,
            "params": n_params,
            "loss": loss,
            "n_fft": n_fft,
        }
        log.info(
            "loaded %s (scale=%s, n_fft=%d, params=%s, loss=%s)",
            model_key, scale, n_fft, n_params,
            f"{loss:.4f}" if isinstance(loss, (int, float)) else loss,
        )

    if not MODELS:
        raise FileNotFoundError(f"No recognised checkpoints in {checkpoint_dir}")

    # Pick a sensible default: prefer conv_mask, else first loaded.
    DEFAULT_MODEL = "conv_mask" if "conv_mask" in MODELS else next(iter(MODELS))
    log.info("default model: %s  |  available: %s", DEFAULT_MODEL, list(MODELS))

    # Warm up every model with a dummy forward so the first real WebSocket
    # connection doesn't pay the CUDA kernel-compile cost (~5 s on L40S)
    # or XLA compilation cost (2-8 hours, but should be cached).
    if device.type in ("cuda", "mps", "xla"):
        dummy = torch.zeros(1, int(CONTEXT_SECS * MODEL_RATE), device=device)
        for key, model in MODELS.items():
            with torch.no_grad():
                _ = model(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elif device.type == "xla":
                import torch_xla.core.xla_model as xm
                xm.mark_step()
            log.info("warmed up %s", key)
        log.info("all models warmed up")


def build_resamplers(device: torch.device) -> None:
    """Precompute the FIR kernels for 48↔16 kHz so resample is cheap."""
    global RESAMPLE_DOWN, RESAMPLE_UP
    RESAMPLE_DOWN = torchaudio.transforms.Resample(
        orig_freq=CAPTURE_RATE, new_freq=MODEL_RATE,
        lowpass_filter_width=6, rolloff=0.99,
    ).to(device)
    RESAMPLE_UP = torchaudio.transforms.Resample(
        orig_freq=MODEL_RATE, new_freq=CAPTURE_RATE,
        lowpass_filter_width=6, rolloff=0.99,
    ).to(device)


# ---------------------------------------------------------------------------
# HTTP endpoints
# ---------------------------------------------------------------------------

@app.get("/")
async def root() -> HTMLResponse:
    return HTMLResponse((CLIENT_DIR / "index.html").read_text(encoding="utf-8"))


@app.get("/api/models")
async def list_models() -> JSONResponse:
    """Return available models and their metadata."""
    payload = {
        "default": DEFAULT_MODEL,
        "models": {
            key: {
                "label": _MODEL_LABELS.get(key, key),
                **meta,
                "loss_str": f"{meta['loss']:.4f}" if isinstance(meta.get("loss"), (int, float)) else str(meta.get("loss")),
            }
            for key, meta in MODEL_META.items()
        },
    }
    return JSONResponse(payload)


_MODEL_LABELS = {
    "conv_mask": "ConvMask",
    "crm": "CRM",
    "attention": "Attention",
    "gru": "GRU",
}

app.mount("/static", StaticFiles(directory=CLIENT_DIR), name="static")


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------

@app.websocket("/ws")
async def ws_infer(ws: WebSocket, model: str = "") -> None:
    await ws.accept()
    peer = f"{ws.client.host}:{ws.client.port}"

    # Resolve requested model, falling back to default.
    model_key = model if model in MODELS else DEFAULT_MODEL
    if model and model not in MODELS:
        log.warning("unknown model %r requested by %s, using %s", model, peer, model_key)

    active_model = MODELS[model_key]

    assert DEVICE is not None
    assert RESAMPLE_DOWN is not None and RESAMPLE_UP is not None

    context_samples = int(CONTEXT_SECS * MODEL_RATE)
    hop_samples = int(HOP_SECS * MODEL_RATE)
    # Shift the emit window slightly inward so the model has a little lookahead
    # at the boundary — reduces edge transients without amplitude modulation.
    guard_samples = int(0.010 * MODEL_RATE)  # 10 ms

    log.info(
        "ws-open  %s model=%s context=%.2fs hop=%.2fs capture=%d model_rate=%d",
        peer, model_key, CONTEXT_SECS, HOP_SECS, CAPTURE_RATE, MODEL_RATE,
    )

    # Sliding-window inference:
    #   context  — fixed-length circular history buffer fed to the model each hop.
    #              Pre-filled with silence so the first hop fires immediately.
    #   pending  — new 16 kHz samples since the last forward pass.
    context = torch.zeros(context_samples, dtype=torch.float32, device=DEVICE)
    pending = 0

    n_frames = 0
    n_forwards = 0
    t_connect = time.perf_counter()

    try:
        while True:
            frame = await ws.receive_bytes()
            n_frames += 1

            # int16 LE PCM at CAPTURE_RATE → float32 [-1,1] → resample to MODEL_RATE
            samples_48k = (
                torch.from_numpy(np.frombuffer(frame, dtype=np.int16).copy())
                .to(DEVICE, dtype=torch.float32)
                / 32768.0
            )
            new_16k = RESAMPLE_DOWN(samples_48k)
            n = new_16k.shape[0]

            # Slide context left by n, append new samples at the right.
            context = torch.cat([context[n:], new_16k])
            pending += n

            if pending < hop_samples:
                continue

            pending -= hop_samples

            # Boost input to model's training distribution before forward pass.
            # LibriSpeech typical peak is ~0.3-0.5; mic input here is ~0.04 (4%).
            # A 6× boost (~15 dB) brings it into range without clipping.
            # This is preferable to boosting the output because the model's noise
            # gate behaviour is nonlinear — louder input → less proportional static.
            model_input = (context * INPUT_BOOST).clamp(-1.0, 1.0)

            t0 = time.perf_counter()
            with torch.no_grad():
                enhanced = active_model(model_input.unsqueeze(0)).squeeze(0)

            # XLA synchronization point (for Trainium/TPU)
            if DEVICE.type == "xla":
                import torch_xla.core.xla_model as xm
                xm.mark_step()

            t_forward = (time.perf_counter() - t0) * 1000.0

            # Take hop_samples from slightly inside the output tail.
            end = len(enhanced) - guard_samples
            hop_out = enhanced[end - hop_samples : end]

            out_48k = RESAMPLE_UP(hop_out)
            out_clipped = torch.clamp(out_48k, -1.0, 1.0)

            # Log in/out peak to distinguish silent mic from model issues.
            inp_peak = context.abs().max().item()
            peak = out_clipped.abs().max().item()
            payload = (out_clipped * 32767.0).to(torch.int16).cpu().numpy().tobytes()

            hop_ms = HOP_SECS * 1000
            if t_forward > hop_ms:
                log.warning(
                    "forward #%d too slow: %.1f ms > hop %.1f ms — emitting silence",
                    n_forwards, t_forward, hop_ms,
                )
                payload = b"\x00" * len(payload)

            await ws.send_bytes(payload)
            n_forwards += 1
            if n_forwards <= 5 or n_forwards % 10 == 0:
                log.info(
                    "forward #%d [%s]: forward=%.1f ms, inp=%.4f, out=%.4f%s",
                    n_forwards, model_key, t_forward, inp_peak, peak,
                    " CLIP" if peak > 0.98 else "",
                )

    except WebSocketDisconnect:
        dur = time.perf_counter() - t_connect
        log.info(
            "ws-close %s model=%s frames=%d forwards=%d duration=%.1fs",
            peer, model_key, n_frames, n_forwards, dur,
        )
    except Exception as exc:
        log.exception("ws-error %s: %s", peer, exc)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu",
                        choices=["cpu", "mps", "cuda", "xla", "neuron"])
    parser.add_argument("--context", type=float, default=0.5,
                        help="Seconds of history fed to the model each forward pass")
    parser.add_argument("--hop", type=float, default=0.1,
                        help="Seconds of new audio between forwards (= latency floor)")
    parser.add_argument("--input-boost", type=float, default=6.0,
                        help="Pre-model gain factor (normalises quiet mic to training level)")
    parser.add_argument("--capture-rate", type=int, default=48000,
                        help="Sample rate the browser client ships (ctx.sampleRate)")
    parser.add_argument("--model-rate", type=int, default=16000,
                        help="Model's native sample rate (16 kHz for all arena models)")
    parser.add_argument("--demo-noise", default=None,
                        choices=["cafeteria", "traffic", "white"],
                        help="Add synthetic noise for dramatic demo (makes enhancement obvious)")
    parser.add_argument("--demo-bypass", action="store_true",
                        help="Bypass mode: echo raw input without enhancement")
    args = parser.parse_args()

    global DEVICE, CONTEXT_SECS, HOP_SECS, INPUT_BOOST, CAPTURE_RATE, MODEL_RATE

    # Handle XLA/Neuron device creation
    if args.device in ("xla", "neuron"):
        if args.device == "xla":
            import torch_xla.core.xla_model as xm
            DEVICE = xm.xla_device()
            log.info("Using XLA device (Trainium/TPU)")
        else:
            DEVICE = torch.device("neuron")
            log.info("Using Neuron device (Trainium native)")
    else:
        DEVICE = torch.device(args.device)
    CONTEXT_SECS = args.context
    HOP_SECS = args.hop
    INPUT_BOOST = args.input_boost
    CAPTURE_RATE = args.capture_rate
    MODEL_RATE = args.model_rate

    log.info("device=%s context=%.2fs hop=%.2fs boost=%.1fx %d→%d Hz",
             DEVICE, CONTEXT_SECS, HOP_SECS, INPUT_BOOST, CAPTURE_RATE, MODEL_RATE)
    load_all_models(args.checkpoint_dir, DEVICE)
    build_resamplers(DEVICE)

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()

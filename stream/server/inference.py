"""Streaming arena — Day 2: live model inference.

WebSocket server that accepts 48 kHz int16 PCM frames from the browser,
runs ConvMask (or any other trained model) on the incoming stream, and
replies with enhanced PCM at the same rate. Wire format is byte-for-byte
compatible with the Day 1 echo server, so the existing client
(`stream/client/index.html`) connects without modification.

MVP streaming strategy: accumulate-and-batch. Every `--window` seconds
of input we run a single forward pass and emit the full processed
window. Latency = `--window` + forward + 2×resample (≈ 1 s at default
settings). Day 3+ will move to a sliding-window strategy for lower
latency once we have GPU/MPS in the loop.

Run:
    uv run python stream/server/inference.py \\
        --checkpoint-dir checkpoints_synthetic \\
        --model conv_mask \\
        --device cpu \\
        --port 8765
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from models import get_model, SCALE_CONFIGS  # noqa: E402

log = logging.getLogger("stream.infer")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

app = FastAPI(title="Streaming Arena — Inference")
CLIENT_DIR = Path(__file__).resolve().parent.parent / "client"

# Globals populated at startup
MODEL: torch.nn.Module | None = None
MODEL_NAME: str = ""
MODEL_SCALE: str = ""
DEVICE: torch.device | None = None
WINDOW_SECS: float = 1.0
CAPTURE_RATE: int = 48000
MODEL_RATE: int = 16000
RESAMPLE_DOWN: torchaudio.transforms.Resample | None = None
RESAMPLE_UP: torchaudio.transforms.Resample | None = None


def load_model(checkpoint_dir: Path, model_key: str, device: torch.device) -> None:
    """Load a trained checkpoint and set MODEL / MODEL_SCALE globals.

    Mirrors the scale-inference pattern from `serve.py:60-100` — the
    checkpoint records its own scale so we instantiate the right size.
    """
    global MODEL, MODEL_NAME, MODEL_SCALE
    patterns = [f"arena_{model_key}_best.pt", f"*{model_key}*best.pt"]
    for pat in patterns:
        matches = list(checkpoint_dir.glob(pat))
        if matches:
            ckpt_path = matches[0]
            break
    else:
        raise FileNotFoundError(
            f"No checkpoint matching {patterns} in {checkpoint_dir}"
        )

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    scale = ckpt.get("scale", "small")
    n_params = ckpt.get("params", "?")
    loss = ckpt.get("loss", None)

    model = get_model(model_key, scale=scale).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    MODEL = model
    MODEL_NAME = model_key
    MODEL_SCALE = scale

    n_fft = SCALE_CONFIGS[scale][model_key].get("n_fft", 512)
    log.info(
        "loaded %s from %s (scale=%s, n_fft=%d, params=%s, loss=%s)",
        model_key, ckpt_path.name, scale, n_fft, n_params,
        f"{loss:.4f}" if isinstance(loss, (int, float)) else loss,
    )


def build_resamplers(device: torch.device) -> None:
    """Precompute the FIR kernels for 48↔16 kHz so resample is cheap."""
    global RESAMPLE_DOWN, RESAMPLE_UP
    # lowpass_filter_width=6 is the fast-but-adequate setting documented in
    # the plan — default (64) is a quality-over-speed kaiser_best, too slow
    # for per-chunk use. rolloff=0.99 preserves the useful speech band.
    RESAMPLE_DOWN = torchaudio.transforms.Resample(
        orig_freq=CAPTURE_RATE, new_freq=MODEL_RATE,
        lowpass_filter_width=6, rolloff=0.99,
    ).to(device)
    RESAMPLE_UP = torchaudio.transforms.Resample(
        orig_freq=MODEL_RATE, new_freq=CAPTURE_RATE,
        lowpass_filter_width=6, rolloff=0.99,
    ).to(device)


@app.get("/")
async def root() -> HTMLResponse:
    return HTMLResponse((CLIENT_DIR / "index.html").read_text())


app.mount("/static", StaticFiles(directory=CLIENT_DIR), name="static")


@app.websocket("/ws")
async def ws_infer(ws: WebSocket) -> None:
    await ws.accept()
    peer = f"{ws.client.host}:{ws.client.port}"
    log.info("ws-open  %s window=%.2fs capture=%d model=%d",
             peer, WINDOW_SECS, CAPTURE_RATE, MODEL_RATE)

    assert MODEL is not None and DEVICE is not None
    assert RESAMPLE_DOWN is not None and RESAMPLE_UP is not None

    window_samples_model = int(WINDOW_SECS * MODEL_RATE)
    # Ring buffer at model rate (16 kHz). Each WebSocket session owns its own.
    ring = torch.zeros(0, dtype=torch.float32, device=DEVICE)

    n_frames = 0
    n_forwards = 0
    t_connect = time.perf_counter()

    try:
        while True:
            frame = await ws.receive_bytes()
            n_frames += 1

            # int16 LE PCM at CAPTURE_RATE → float32 tensor in [-1, 1] on DEVICE.
            samples_48k = (
                torch.from_numpy(np.frombuffer(frame, dtype=np.int16).copy())
                .to(DEVICE, dtype=torch.float32)
                / 32768.0
            )
            # Resample incoming chunk 48 → 16 kHz. Small chunks resample cleanly
            # because the kernel width is small (6 taps). For 768-sample input
            # we get ~256 samples out (3× downsample).
            samples_16k = RESAMPLE_DOWN(samples_48k)
            ring = torch.cat([ring, samples_16k])

            # Emit a processed window every time we've accumulated enough.
            if ring.shape[0] >= window_samples_model:
                window = ring[:window_samples_model]
                ring = ring[window_samples_model:]

                t0 = time.perf_counter()
                with torch.no_grad():
                    enhanced = MODEL(window.unsqueeze(0)).squeeze(0)
                t_forward = (time.perf_counter() - t0) * 1000.0

                # Resample 16 → 48 kHz for the client's playback rate.
                out_48k = RESAMPLE_UP(enhanced)
                # float32 [-1, 1] → int16 LE.
                out_clipped = torch.clamp(out_48k, -1.0, 1.0)
                out_int16 = (out_clipped * 32767.0).to(torch.int16).cpu().numpy()
                payload = out_int16.tobytes()

                if t_forward > WINDOW_SECS * 1000:
                    # Inference slower than real-time; emitting silence
                    # instead of stalling keeps the client responsive. Log
                    # so we notice during the demo.
                    log.warning(
                        "forward #%d too slow: %.1f ms > window %.1f ms — emitting silence",
                        n_forwards, t_forward, WINDOW_SECS * 1000,
                    )
                    payload = b"\x00" * len(payload)

                await ws.send_bytes(payload)
                n_forwards += 1
                if n_forwards <= 3 or n_forwards % 5 == 0:
                    log.info(
                        "forward #%d: in=%d samples @ %d Hz, forward=%.1f ms, out=%d bytes",
                        n_forwards, window_samples_model, MODEL_RATE,
                        t_forward, len(payload),
                    )

    except WebSocketDisconnect:
        dur = time.perf_counter() - t_connect
        log.info(
            "ws-close %s frames=%d forwards=%d duration=%.1fs",
            peer, n_frames, n_forwards, dur,
        )
    except Exception as exc:
        log.exception("ws-error %s: %s", peer, exc)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--model", default="conv_mask",
                        choices=["conv_mask", "crm", "attention", "gru"])
    parser.add_argument("--device", default="cpu",
                        choices=["cpu", "mps", "cuda"])
    parser.add_argument("--window", type=float, default=1.0,
                        help="Seconds of audio per forward pass (latency floor)")
    parser.add_argument("--capture-rate", type=int, default=48000,
                        help="Sample rate the browser client ships (ctx.sampleRate)")
    parser.add_argument("--model-rate", type=int, default=16000,
                        help="Model's native sample rate (16 kHz for all arena models)")
    args = parser.parse_args()

    global DEVICE, WINDOW_SECS, CAPTURE_RATE, MODEL_RATE
    DEVICE = torch.device(args.device)
    WINDOW_SECS = args.window
    CAPTURE_RATE = args.capture_rate
    MODEL_RATE = args.model_rate

    log.info("device=%s window=%.2fs %d→%d Hz",
             DEVICE, WINDOW_SECS, CAPTURE_RATE, MODEL_RATE)
    load_model(args.checkpoint_dir, args.model, DEVICE)
    build_resamplers(DEVICE)

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()

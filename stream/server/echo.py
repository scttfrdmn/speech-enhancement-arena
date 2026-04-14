"""Streaming arena — Day 1 plumbing.

WebSocket echo server: accepts binary PCM frames from the browser, echoes
them straight back. Exists only to verify the audio capture → network →
playback loop works end-to-end before we layer models on top.

Wire format (both directions):
    Binary frame of little-endian int16 PCM samples at 16 kHz, mono.
    Typical chunk: 256 samples = 512 bytes.

Run:
    uv run python stream/server/echo.py --port 8765
    # then open stream/client/index.html in a browser
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

log = logging.getLogger("stream.echo")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

app = FastAPI(title="Streaming Arena — Echo")

CLIENT_DIR = Path(__file__).resolve().parent.parent / "client"


@app.get("/")
async def root() -> HTMLResponse:
    html = (CLIENT_DIR / "index.html").read_text()
    return HTMLResponse(html)


app.mount("/static", StaticFiles(directory=CLIENT_DIR), name="static")


@app.websocket("/ws")
async def ws_echo(ws: WebSocket) -> None:
    await ws.accept()
    peer = f"{ws.client.host}:{ws.client.port}"
    log.info("ws-open  %s", peer)
    n_frames = 0
    n_bytes = 0
    try:
        while True:
            frame = await ws.receive_bytes()
            n_frames += 1
            n_bytes += len(frame)
            # Echo. The client will play this back through its output device.
            await ws.send_bytes(frame)
    except WebSocketDisconnect:
        log.info("ws-close %s frames=%d bytes=%d", peer, n_frames, n_bytes)
    except Exception as exc:
        log.exception("ws-error %s: %s", peer, exc)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()

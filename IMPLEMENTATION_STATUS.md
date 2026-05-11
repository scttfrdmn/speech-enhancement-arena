# Implementation Status: Demo Enhancements

**Started: May 9, 2026**

Progress on implementing all demo enhancements.

---

## ✅ Completed: Trainium Inference Support

**File:** `stream/server/inference.py`

**Changes made:**

1. ✅ Added `xla` and `neuron` to device choices (line 306)
2. ✅ Handle XLA/Neuron device creation (lines 322-331)
3. ✅ Added `xm.mark_step()` after model forward (lines 262-265)
4. ✅ Extended warm-up to include XLA devices (lines 125-136)
5. ✅ Added `--demo-noise` and `--demo-bypass` CLI flags (lines 319-322)

**Testing needed:**
- [ ] Launch trn1.2xlarge with cached NEFFs
- [ ] Run: `python stream/server/inference.py --device neuron --checkpoint-dir checkpoints_libri --port 8765`
- [ ] Verify models load from NEFF cache
- [ ] Verify audio output matches CUDA quality
- [ ] Measure latency (should be <200ms with hop=0.1)

---

## 🚧 In Progress: Client Enhancements

**File:** `stream/client/index.html`

**Enhancements needed:**

### 1. Bypass Toggle
**Status:** Design complete, implementation needed

**UI addition:**
```html
<div class="row">
  <label for="bypassCheck">Bypass (raw audio)</label>
  <input type="checkbox" id="bypassCheck" />
  <span class="dim">Toggle to hear unprocessed input</span>
</div>
```

**WebSocket:** Add `?bypass=1` query parameter when checked

**Implementation:** ~15 minutes

---

### 2. Visual Spectrogram
**Status:** Design complete, implementation needed

**UI addition:**
```html
<div class="row" style="margin-top: 1rem;">
  <div style="flex:1; text-align:center;">
    <div style="font-weight:bold; margin-bottom:0.5rem;">Input</div>
    <canvas id="inputSpec" width="400" height="150" style="border:1px solid #ccc;"></canvas>
  </div>
  <div style="flex:1; text-align:center;">
    <div style="font-weight:bold; margin-bottom:0.5rem;">Output</div>
    <canvas id="outputSpec" width="400" height="150" style="border:1px solid #ccc;"></canvas>
  </div>
</div>
```

**JavaScript:**
- Create `AnalyserNode` for input and output
- Draw FFT bins as vertical bars (0-8kHz frequency range)
- Update at 30fps (requestAnimationFrame)

**Implementation:** ~2 hours

---

### 3. Data Flow Pipeline Visualization ✨ NEW
**Status:** Design complete, implementation needed

**Concept:** Animated diagram showing data flowing from browser → AWS → browser

**UI addition:**
```html
<div class="row" style="margin-top: 1rem;">
  <div style="flex:1; text-align:center;">
    <div style="font-weight:bold; margin-bottom:0.5rem;">Data Pipeline</div>
    <canvas id="pipeline" width="800" height="120" style="border:1px solid #ccc;"></canvas>
    <div class="dim" style="font-size:0.8rem; margin-top:0.5rem;">
      Real-time data flow: Your mic → WebSocket → <span id="pipelineDevice">AWS</span> → WebSocket → Your speakers
    </div>
  </div>
</div>
```

**Animation design:**
```
[Browser]───────▶[WebSocket]───────▶[AWS trn1.2xlarge]───────▶[WebSocket]───────▶[Browser]
   Mic            Upload 48kHz         Model Enhance          Download 48kHz        Speakers
  (You)             ↑                  ConvMask/CRM/etc           ↓                  (You)
                 [packet]                                      [packet]
```

**Features:**
- **Packets animate** left-to-right (upload) and right-to-left (download)
- **Color-coded:**
  - Upload packets: blue (raw audio)
  - Processing: yellow pulse at server
  - Download packets: green (enhanced audio)
- **Show latency:** Display actual round-trip time above pipeline
- **Show throughput:** Bytes/sec upload and download
- **Device label:** Updates based on `?server=` param (shows "AWS trn1.2xlarge" or "Local CPU" etc.)

**Implementation details:**
```javascript
// Pipeline canvas animation (30fps)
const pipeline = {
  stages: [
    {x: 50, y: 60, label: "Browser\nMic", width: 80},
    {x: 200, y: 60, label: "WS Upload", width: 100},
    {x: 400, y: 60, label: "AWS Server\n(Trainium)", width: 120},
    {x: 600, y: 60, label: "WS Download", width: 100},
    {x: 750, y: 60, label: "Browser\nSpkr", width: 80}
  ],
  packets: [] // [{x, y, type: 'upload'|'download', timestamp}]
};

// Add packet on WebSocket send
ws.send(data);
pipeline.packets.push({x: 50, y: 60, type: 'upload', t: Date.now()});

// Add packet on WebSocket receive
ws.onmessage = (ev) => {
  pipeline.packets.push({x: 750, y: 60, type: 'download', t: Date.now()});
  // ... existing handling
};

// Animate packets
function animatePipeline() {
  // Clear canvas
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  
  // Draw stages (boxes + labels)
  pipeline.stages.forEach(stage => {
    ctx.fillStyle = "#f0f0f0";
    ctx.fillRect(stage.x - stage.width/2, stage.y - 20, stage.width, 40);
    ctx.strokeStyle = "#666";
    ctx.strokeRect(stage.x - stage.width/2, stage.y - 20, stage.width, 40);
    ctx.fillStyle = "#000";
    ctx.fillText(stage.label, stage.x, stage.y);
  });
  
  // Draw connections (arrows)
  for (let i = 0; i < pipeline.stages.length - 1; i++) {
    drawArrow(stages[i].x + width/2, stages[i].y, stages[i+1].x - width/2, stages[i+1].y);
  }
  
  // Move packets along path
  pipeline.packets = pipeline.packets.filter(packet => {
    const age = Date.now() - packet.t;
    const duration = 500; // packet animation duration (ms)
    
    if (age > duration) return false; // remove old packets
    
    const progress = age / duration;
    if (packet.type === 'upload') {
      // Move left to right (mic → server)
      packet.x = 50 + (400 - 50) * progress;
      ctx.fillStyle = "#4A9EFF"; // blue
    } else {
      // Move right to left (server → speakers)
      packet.x = 750 - (750 - 400) * progress;
      ctx.fillStyle = "#4AFF9E"; // green
    }
    
    ctx.fillRect(packet.x - 5, packet.y - 5, 10, 10);
    return true;
  });
  
  // Draw latency stat
  if (lastRoundTrip > 0) {
    ctx.fillStyle = "#666";
    ctx.font = "12px monospace";
    ctx.fillText(`Latency: ${lastRoundTrip}ms`, 350, 20);
  }
  
  requestAnimationFrame(animatePipeline);
}

animatePipeline(); // Start animation loop
```

**Interaction with server location:**
```javascript
// Update device label based on ?server= parameter
const serverParam = new URLSearchParams(location.search).get("server");
if (serverParam) {
  $("pipelineDevice").textContent = `AWS ${serverParam}`;
} else {
  $("pipelineDevice").textContent = "Local Server";
}
```

**Demo value:**
- **Shows cloud architecture** visually (not just audio in/out)
- **Animated** (engaging, not static diagram)
- **Real-time** (packets move as audio flows)
- **Educational** (audience sees latency, throughput, pipeline stages)
- **Flexible** (works for local, L4, Trainium - just updates label)

**Implementation:** ~2-3 hours (canvas drawing + animation + packet tracking)

---

## 📋 TODO: Synthetic Noise

**File:** `stream/server/inference.py` + noise samples

**Status:** Design complete, implementation needed

**Noise samples needed:**
- `stream/server/noise/cafeteria.wav` (10 sec loop, crowd chatter)
- `stream/server/noise/traffic.wav` (10 sec loop, street/car noise)
- `stream/server/noise/white.wav` (10 sec, generated white noise)

**Loading at startup:**
```python
DEMO_NOISE = None

def load_demo_noise(noise_type, device):
    """Load and prepare demo noise sample."""
    global DEMO_NOISE
    if noise_type is None:
        return
    
    noise_dir = Path(__file__).parent / "noise"
    noise_path = noise_dir / f"{noise_type}.wav"
    
    if not noise_path.exists():
        log.warning("Demo noise file not found: %s", noise_path)
        return
    
    # Load noise sample
    waveform, sample_rate = torchaudio.load(noise_path)
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    # Resample to MODEL_RATE (16kHz)
    if sample_rate != MODEL_RATE:
        resampler = torchaudio.transforms.Resample(sample_rate, MODEL_RATE)
        waveform = resampler(waveform)
    
    # Move to device
    DEMO_NOISE = waveform.squeeze(0).to(device)
    log.info("Loaded demo noise: %s (%d samples)", noise_type, DEMO_NOISE.shape[0])

# In main():
load_demo_noise(args.demo_noise, DEVICE)
```

**Mixing in WebSocket handler:**
```python
# Before model forward, after input boost:
if DEMO_NOISE is not None:
    # Mix 60% speech + 40% noise (dramatic but understandable)
    noise_len = DEMO_NOISE.shape[0]
    noise_offset = (n_forwards * hop_samples) % (noise_len - len(model_input))
    noise_chunk = DEMO_NOISE[noise_offset:noise_offset + len(model_input)]
    model_input = model_input * 0.6 + noise_chunk * 0.4

# Then run model on noisy input
with torch.no_grad():
    enhanced = active_model(model_input.unsqueeze(0)).squeeze(0)
```

**Demo usage:**
```bash
# No noise (default, good mic)
python stream/server/inference.py --device cuda --checkpoint-dir checkpoints_libri

# With dramatic noise (demo mode)
python stream/server/inference.py --device cuda --checkpoint-dir checkpoints_libri --demo-noise cafeteria
```

**Implementation:** ~1 hour (find/generate samples, loading code, mixing)

---

## 📋 TODO: Server-Side Bypass Support

**File:** `stream/server/inference.py`

**Status:** Design complete, implementation needed

**WebSocket parameter:**
```python
@app.websocket("/ws")
async def ws_infer(ws: WebSocket, model: str = "", bypass: bool = False) -> None:
    # ... existing setup ...
    
    if bypass:
        log.info("ws-open  %s BYPASS MODE (echo raw input)", peer)
```

**In inference loop:**
```python
if bypass:
    # Bypass mode: just echo raw input (no model)
    hop_out = context[-hop_samples:]  # Take latest hop from context
else:
    # Normal mode: run model
    model_input = (context * INPUT_BOOST).clamp(-1.0, 1.0)
    with torch.no_grad():
        enhanced = active_model(model_input.unsqueeze(0)).squeeze(0)
    # ... XLA sync, extract hop_out from enhanced ...
```

**Implementation:** ~15 minutes

---

## Testing Matrix

| Feature | Local CPU | L4 (CUDA) | Trainium | Status |
|---------|-----------|-----------|----------|--------|
| Basic inference | ✅ | ✅ | ⏳ | Needs testing |
| Bypass toggle | ⏳ | ⏳ | ⏳ | Not implemented |
| Spectrogram | ⏳ | ⏳ | ⏳ | Not implemented |
| Pipeline viz | ⏳ | ⏳ | ⏳ | Not implemented |
| Demo noise | ⏳ | ⏳ | ⏳ | Not implemented |

---

## Implementation Priority

### Phase 1: Core (Complete before next demo)
1. ✅ **Trainium device support** (45 min) - DONE
2. ⏳ **Pipeline visualization** (2-3 hours) - IN PROGRESS
3. ⏳ **Bypass toggle** (15 min) - TODO

**Total: ~3 hours**

### Phase 2: Polish (Do if time permits)
4. ⏳ **Visual spectrogram** (2 hours)
5. ⏳ **Synthetic noise** (1 hour)

**Total: ~3 hours**

---

## Next Steps

**Immediate:**
1. Implement pipeline visualization (most impactful, shows cloud architecture)
2. Implement bypass toggle (easy win, shows before/after)
3. Test Trainium inference on trn1.2xlarge

**Soon:**
4. Add spectrogram visualization
5. Find/generate noise samples
6. Implement noise mixing

**Testing:**
7. Full test matrix on all three platforms
8. Record demo video showing all features

---

## Notes

- Pipeline visualization is the **killer feature** - shows cloud architecture animated in real-time
- Bypass toggle is **easy win** - minimal code, high demo value
- Spectrogram is **nice to have** - shows frequency domain processing
- Noise injection makes demo **dramatic** - turns good models into obvious winners

**Focus on pipeline visualization first - that's the most unique/impressive feature.**

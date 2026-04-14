// AudioWorklet: capture mono at ctx.sampleRate, emit fixed-size frames of
// int16 PCM to the main thread. Runs on the audio rendering thread, so no
// network / DOM work here — just buffer and ship.

class CaptureProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const { frameSamples = 256 } = options.processorOptions ?? {};
    this.frameSamples = frameSamples;
    this.buffer = new Float32Array(frameSamples);
    this.write = 0;
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || input.length === 0) return true;
    const ch = input[0];  // mono — first channel
    if (!ch) return true;

    let peak = 0;
    for (let i = 0; i < ch.length; i++) {
      const v = ch[i];
      this.buffer[this.write++] = v;
      if (Math.abs(v) > peak) peak = Math.abs(v);

      if (this.write === this.frameSamples) {
        const pcm = new Int16Array(this.frameSamples);
        for (let k = 0; k < this.frameSamples; k++) {
          const s = Math.max(-1, Math.min(1, this.buffer[k]));
          pcm[k] = s < 0 ? s * 0x8000 : s * 0x7fff;
        }
        this.port.postMessage({ pcm, level: peak }, [pcm.buffer]);
        this.buffer = new Float32Array(this.frameSamples);
        this.write = 0;
        peak = 0;
      }
    }
    return true;
  }
}

registerProcessor("capture-processor", CaptureProcessor);

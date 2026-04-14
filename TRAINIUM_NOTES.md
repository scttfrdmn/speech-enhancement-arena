# Trainium porting notes — hard-won lessons

Running this arena on `trn1.32xlarge` (torch-neuronx 2.9.0.2.13, neuronx-cc 2.24.5133.0, Neuron SDK 2.28) surfaced a set of non-obvious compile gotchas. This is a pre-flight checklist for porting any complex-valued / FFT / recurrent training workload to Neuron.

## Pre-flight checklist

### 1. Host RAM is a hard prerequisite for first compile

- `trn1.2xlarge` (32 GB host RAM) **cannot compile** moderately complex models — `neuronx-cc` gets OOM-killed mid-Tensorizer.
- Minimum safe host for first compile: `trn1.32xlarge` (512 GB) or `trn2.48xlarge` (384 GB).
- Once NEFFs are cached (to disk or S3 via `NEURON_COMPILE_CACHE_URL`), `trn1.2xlarge` is fine for inference/serving.

### 2. `neuronx-cc` is single-threaded

See [aws-neuron-sdk #646](https://github.com/aws-neuron/aws-neuron-sdk/issues/646). Each NEFF compile runs one `walrus_driver` on one core, even on 128-core instances. 98 % CPU on one thread is **expected**, not a hung process. `neuron_parallel_compile` only parallelizes *across independent modules*.

### 3. Use `--optlevel 1` for FFT / unrolled-loop workloads

Pure-PyTorch FFT implementations express each Cooley-Tukey butterfly as explicit slicing + arithmetic. For a 512-point FFT × many frames, the resulting HLO graph has hundreds of thousands of ops. At `--optlevel 3` (default) the compiler attempts aggressive unrolling and loop-fusion — Tensorizer passes scale superlinearly (we measured `ParAxesAnnotation` 38 min, `PGTiling` 107 min, `AGOrderingAnalysisPass` 68 min, `NeuronLoopFusion` 40 min × N iterations, still not done at 14 h).

```bash
export NEURON_CC_FLAGS="--optlevel 1"
```

Documented mitigation for "Too many instructions after unroll for function sg0000". Accept some runtime throughput cost for a compile that actually terminates in bounded time.

### 4. Bump parameter-tupling threshold for FFT graphs

Neuron raises `DevicePlacement: Passing arguments as a tuple is currently unsupported` when the graph exceeds 3200 parameters. Many unrolled FFT constants trip this easily.

```bash
export XLA_PARAMETER_WRAPPING_THREADSHOLD=50000   # sic — "THREADSHOLD" is the real name
```

### 5. XLA has no `aten::unfold` implementation

`torch.Tensor.unfold` is a view op without an XLA backend. Any framing code that calls it fails with `RuntimeError: operator aten::unfold ... has no implementation for the backend "xla:0"`.

**Fix:** build frame indices with `torch.arange` explicitly. See [trnsci/trnfft PR #44](https://github.com/trnsci/trnfft/pull/44).

### 6. `torch.zeros(...)` defaults to CPU

Device-agnostic library code with `torch.zeros(shape, dtype=...)` (no `device=`) returns a CPU tensor. On XLA this triggers `torch.cat(CPU, XLA)` errors or silently pushes the graph over the parameter-tupling threshold. Same for `torch.hann_window`, `torch.arange`, `torch.ones`.

**Fix:** pass `device=<input>.device` on tensor constructors inside any library called from an XLA code path.

### 7. `nn.GRU(bidirectional=True)` is pathological on XLA

`torch-neuronx`'s scan-based GRU optimization only supports unidirectional. Bidirectional falls back to default `nn.GRU` whose recurrence trips dynamic-shape graph explosion. We saw first-epoch compile not finish in 50 min.

**Fix:** split into two stacked unidirectional GRUs plus `torch.flip` on the reverse pass. Adds ~20 lines; compile drops to a few NEFFs. Implemented in `models/architectures.py` (commit b97dff5).

### 8. `@nki.jit` kernels break autograd

NKI kernel outputs carry no `grad_fn`. Forward works, `loss.backward()` raises `element 0 of tensors does not require grad`.

**Workaround:** `trnfft.set_backend("pytorch")` disables NKI dispatch. Filed as [trnsci/trnfft #56](https://github.com/trnsci/trnfft/issues/56).

**Long-term fix:** wrap each `@nki.jit` kernel in a `torch.autograd.Function` with analytic backward.

### 9a. Cross-compile on cheap x86, not Trn

`neuronx-cc` is a cross-compiler — it needs the Neuron SDK installed but **no Neuron accelerator** on the host. Works on any x86_64 with the Neuron DLAMI. Verified on `r7i.24xlarge` (768 GB RAM, $6.36/hr): neuronx-cc ran at the same speed as on `trn1.32xlarge` ($21.50/hr, same single-threaded `walrus_driver`). Use `--target=trn1` (or trn1n/inf2) to select the ISA.

Canonical cache-seeding pipeline:

1. `trn1.2xlarge` ($1.34/hr) — run `train.py` briefly to emit HLO `.pb` files into `/tmp/ubuntu/neuroncc_compile_workdir/*/`. Kill once the `.pb` files appear (seconds).
2. scp HLOs off.
3. `r7i.24xlarge` — bulk-compile with `neuronx-cc compile --framework=XLA model.hlo_module.pb --output model.neff --target=trn1 --optlevel 1`.
4. Upload NEFFs into the S3 cache path (`neuronxcc-<ver>/MODULE_<hash>/model.neff`).
5. Downstream consumers hit cache, zero compile.

This collapses the "$300 of trn1.32xlarge compile" to "$30 of r7i.24xlarge compile" for the same output.

### 9. NEFF cache is the single biggest win

- First compile: minutes to hours per unique graph.
- S3-cached rerun: entire 4-model arena in **123 s** (24× speedup vs 3006 s first run).

```bash
export NEURON_COMPILE_CACHE_URL=s3://your-bucket/trn1-xla/
```

Arena's seed cache: `s3://aws-arena-neuron-cache-scttfrdmn/trn1-xla/` (509 MB, 43 NEFFs covering all 4 models). Downstream users hit S3 and get compile-free runs.

### 10. trn2.* is Capacity-Blocks-for-ML only

All trn2 instance sizes (`trn2.3xlarge`, `trn2.48xlarge`, etc.) are **reservation-only** via [Capacity Blocks for ML](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-capacity-blocks.html). `aws ec2 run-instances` without a CBML reservation will always return `InsufficientInstanceCapacity`, and `describe-instance-types` hides the sizes you don't have a reservation for (looks like they don't exist).

- Reserve a Capacity Block for a future window; `aws ec2 create-capacity-reservation-fleet` or Console.
- Launch during the reserved window with `--market-type capacity-block --capacity-reservation-specification CapacityReservationTarget=...`.
- Plan any trn2 benchmark **days in advance**. You can't just launch one ad hoc like with trn1.

### 11. If compile looks stuck, check the latest compile log

```bash
ls -t /tmp/ubuntu/neuroncc_compile_workdir/*/log-neuron-cc.txt | head -1 | xargs tail
```

If timestamps advance → still alive (walrus runs silently between log lines). If `write_bytes` from `/proc/<pid>/io` is frozen for 30+ min *and* CPU is at 0 % → hung.

## Canonical launch command

```bash
source /opt/aws_neuronx_venv_pytorch_2_9/bin/activate
export NEURON_COMPILE_CACHE_URL=s3://your-bucket/trn1-xla/
export NEURON_CC_FLAGS="--optlevel 1"
export XLA_PARAMETER_WRAPPING_THREADSHOLD=50000

python arena.py --device xla --scale small --epochs 5 --num-samples 256 --batch-size 16
```

## References

- [aws-neuron-sdk #646: single-threaded compile](https://github.com/aws-neuron/aws-neuron-sdk/issues/646)
- [Neuron Compiler FAQ](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/compiler/neuronx-cc/faq.html)
- [Neuron Compiler CLI Reference](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/compiler/neuronx-cc/api-reference-guide/neuron-compiler-cli-reference-guide.html)
- [trnsci/trnfft PR #44: XLA device correctness fixes](https://github.com/trnsci/trnfft/pull/44)
- [trnsci/trnfft #56: NKI breaks autograd](https://github.com/trnsci/trnfft/issues/56)

# Spot Resilience: Designing Experiments That Survive Interruption

AWS spot instances are 60–70% cheaper than on-demand, but they're reclaimable: AWS can take your instance back when capacity tightens. Coming from an HPC cluster where "submit batch job, walk away, get email when done" is the default, this is a new failure mode you have to design around.

This doc covers:
- How the interruption protocol actually works
- The minimum-viable pattern: SIGTERM handler + S3 checkpoint sync
- When spot is the wrong choice
- How this repo's `train.py` implements it

---

## The Interruption Protocol

When AWS decides to reclaim your spot instance, three things happen in sequence:

1. **A "spot interruption notice" appears in instance metadata**, queryable at `http://169.254.169.254/latest/meta-data/spot/instance-action`. This returns a 404 normally, and a JSON blob like `{"action": "terminate", "time": "2026-05-10T17:30:00Z"}` once interruption is pending.
2. **Two minutes later, your processes receive `SIGTERM`**. You have this window to save state.
3. **After SIGTERM**, the instance is terminated (default) or stopped/hibernated (if you configured it). No further notice.

The protocol is well-defined and predictable. Designing around it isn't hard — it's just not the same as cluster jobs.

---

## Pattern 1 — SIGTERM Handler (Minimum Viable)

The simplest pattern: install a SIGTERM handler that writes a checkpoint and exits cleanly. On the next launch, your training script checks for that checkpoint and resumes.

```python
import signal
import sys

class GracefulShutdown:
    """Sets self.interrupted=True on SIGTERM. Check it between training steps."""
    def __init__(self):
        self.interrupted = False
        signal.signal(signal.SIGTERM, self._handler)
        signal.signal(signal.SIGINT, self._handler)

    def _handler(self, signum, frame):
        print(f"[spot] signal {signum} received — finishing this step, then saving", flush=True)
        self.interrupted = True

shutdown = GracefulShutdown()

for epoch in range(start_epoch, args.epochs + 1):
    train_one_epoch(model, loader, optimizer)
    save_checkpoint(model, optimizer, epoch)
    if shutdown.interrupted:
        sync_to_s3(checkpoint_path)   # if you use S3 for cross-instance resume
        sys.exit(0)
```

Two minutes is plenty of time to finish the current epoch, save state, and sync to S3 if needed. The key invariant: **whenever the process exits, the checkpoint on disk reflects the most recently completed step.** Anything mid-step is acceptable to lose.

---

## Pattern 2 — Metadata Polling (More Robust)

SIGTERM is delivered ~2 minutes before reclaim, but if you're already mid-epoch when it arrives, you might not finish in time. For long-epoch workloads, poll the metadata endpoint yourself and bail early:

```python
import urllib.request
import urllib.error
from threading import Thread, Event

def watch_spot_termination(stop_event: Event, poll_secs: int = 5):
    """Set stop_event when AWS posts an interruption notice."""
    url = "http://169.254.169.254/latest/meta-data/spot/instance-action"
    while not stop_event.is_set():
        try:
            resp = urllib.request.urlopen(url, timeout=2)
            if resp.status == 200:
                print("[spot] interruption notice posted — stopping after current step", flush=True)
                stop_event.set()
                return
        except urllib.error.HTTPError as e:
            if e.code != 404:
                pass  # 404 = no notice (normal); other codes = transient
        except urllib.error.URLError:
            pass  # not on EC2; ignore
        stop_event.wait(poll_secs)

stop = Event()
Thread(target=watch_spot_termination, args=(stop,), daemon=True).start()

for epoch in range(start_epoch, args.epochs + 1):
    train_one_epoch(model, loader, optimizer, stop_event=stop)  # train loop checks stop
    save_checkpoint(...)
    if stop.is_set():
        sys.exit(0)
```

The training loop can check `stop.is_set()` between batches and break out early, giving you the full 2-minute window to checkpoint instead of finishing the current epoch first.

---

## S3 As The Resume Boundary

A single spot instance can disappear. To resume on a *different* instance, the checkpoint has to live somewhere durable. S3 is the obvious choice:

```bash
# On the running instance, after each checkpoint:
aws s3 sync ./checkpoints/ s3://your-bucket/run-id/checkpoints/

# On a new instance starting up:
aws s3 sync s3://your-bucket/run-id/checkpoints/ ./checkpoints/
python train.py --run-id <same-id> --resume
```

For minimum-viable resume, sync only the `*_best.pt` files (one per model). For exact-step resume, sync the most recent intermediate checkpoint too. The arena's `train.py` saves a `_best.pt` after each loss improvement (`train.py:483`), which is enough for "pick up where you left off" in most cases.

A more elaborate pattern: an `aws s3 sync` sidecar process running in `--watch` mode, syncing every N seconds. Avoids the in-process boto3 dependency.

---

## Resume Logic in `train.py`

When `train.py` is invoked with `--resume`, it looks for `{checkpoint-dir}/{run-id}_best.pt` and loads model + optimizer state from it. The SIGTERM handler is always installed, regardless of whether `--resume` was passed. So the workflow is:

```bash
# First launch
python train.py --model conv_mask --run-id my_run --epochs 100

# Spot interrupted after, say, epoch 47 — checkpoint sits in ./checkpoints/my_run_best.pt
# (and also in S3 if your sidecar synced it)

# New instance, second launch — picks up at epoch 48
python train.py --model conv_mask --run-id my_run --epochs 100 --resume
```

The `arena.py` orchestrator passes `--run-id arena_<model>` automatically, so the resume path works for arena runs too — just add `--resume` to the arena command line.

---

## When Spot Is The Wrong Choice

Spot resilience adds complexity. Don't bother with spot when:

- **The job is shorter than the interruption window.** A 10-minute training run is unlikely to get reclaimed; the resilience scaffolding is overhead. Use spot anyway, but don't bother with the metadata-polling pattern.
- **You're doing interactive work.** Debugging a model in a Jupyter notebook, exploring data, prototyping — interruption breaks your flow more than the savings justify. Use on-demand for interactive sessions.
- **Sub-2-minute jobs.** The notice gives you 2 minutes, so if your job runs in 90 seconds, you wouldn't finish even one save cycle. Run them on on-demand or batch-schedule them.
- **You can't tolerate any reclamation latency.** Some spot reclaims happen back-to-back across regions; if you absolutely need a result by 5pm, on-demand or a small reserved instance gives you predictability.

For everything else — training runs longer than ~10 minutes — spot is almost always the right call.

---

## The spore.host angle

[**spore.host**](https://spore.host) handles the *instance lifecycle* side of spot — capacity search (`truffle spot ...`), launch with auto-termination (`spawn launch --ttl ... --on-complete terminate`), and clean idle-time shutdowns. It doesn't replace the application-level resilience above (SIGTERM handler, S3 sync, resume logic) — those still live in your training code — but it removes the surrounding bookkeeping so you can focus on the parts that touch your model.

A typical spore.host + spot workflow:

```bash
# truffle picks the cheapest g6.xlarge spot AZ across regions
truffle spot g6.xlarge --sort-by-price --active-only

# spawn launches with a TTL that's longer than your expected job duration;
# spored on the instance auto-terminates when idle (and the TTL is a hard cap)
spawn launch --name run42 --instance-type g6.xlarge --ttl 8h \
             --on-complete terminate

spawn connect run42
# inside the instance:
#   python train.py --model conv_mask --run-id run42 --resume
#   (SIGTERM handler activates if AWS reclaims; checkpoint syncs to S3)
```

If reclamation hits, the checkpoint is in S3, and you re-launch via `spawn launch` and `--resume` the run on a new instance. spore.host doesn't try to be the resume mechanism — that's an application concern — but it makes the launch-side rote work disappear.

---

## See also

- [`AWS_GETTING_STARTED.md`](AWS_GETTING_STARTED.md) — how to get to the point where you can launch a spot instance at all
- [`HYBRID_CLOUD_WORKFLOW.md`](HYBRID_CLOUD_WORKFLOW.md) — when to use spot/cloud vs your institutional cluster

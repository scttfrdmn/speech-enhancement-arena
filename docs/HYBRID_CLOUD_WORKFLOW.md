# Hybrid Cloud Workflow: Cluster + Cloud, Not Cluster vs Cloud

The framing "use the cluster" or "use the cloud" is a false dichotomy. Most research labs have access to *both* — an institutional cluster (free, queued, generally bigger hardware per node) and AWS (paid, instant, smaller-per-node but elastically parallel). The right pattern combines them: cluster for the things clusters are good at, cloud for the things cloud is good at.

This doc covers when each wins, what the boundary between them looks like, and three concrete recipes that work for most academic ML workflows.

---

## What each is actually good at

**Institutional clusters** are good at:
- **Steady, predictable, long-running compute.** Once you're at the front of the queue and your job starts, you have it for the full allocation. No reclamation, no spot interruption, no per-hour cost.
- **Big shared datasets.** Data already lives on cluster filesystems; egress is free; no S3 transfer time.
- **High per-node hardware.** Most clusters have H100, A100, or H200 nodes that would cost you $30+/hr on-demand to rent.
- **Multi-node distributed training.** Cluster interconnects (InfiniBand, Slingshot) are far better than what you get in cloud at academic-budget tier.

**AWS (or any cloud)** is good at:
- **Bursting past the queue.** When the queue says "4 days," cloud says "30 seconds." For exploratory work, that turns a week of one-experiment-at-a-time into a morning of many-experiments-in-parallel.
- **Elastic parallelism.** Need to launch 50 L4 spot instances at once for a hyperparameter sweep? Cluster allocations cap you at your fairshare; cloud caps you at your billing limit.
- **Right-sized hardware for small models.** A 3M-param model on a shared H200 is 0.1% utilization. The same model on an L4 spot is right-sized and ~20× cheaper per real-work-hour.
- **Heterogeneous hardware on demand.** Trainium for a specific architecture experiment, Inferentia for inference benchmarks, Graviton for CPU baselines — all without buying anything.

Neither dominates the other. The point of hybrid is to *not pick one*.

---

## The boundary: S3 as the bridge

In a hybrid workflow, S3 (or any object store) is the seam between cluster and cloud. Data flows look like this:

```
                        ┌──────────────┐
   cluster home/        │     S3       │        cloud spot/
   scratch FS    ◀────▶ │  (the seam)  │ ◀────▶ on-demand instances
                        └──────────────┘
   long-lived data       cheap durable      ephemeral compute
   slow to mutate        intermediate       lives hours, not days
```

The cluster's filesystem is the system of record for big datasets. The cloud's compute is the burst capacity. S3 is the cheap, durable buffer between them — `aws s3 sync` is the only tool you have to learn.

Concrete commands:

```bash
# Cluster → S3 (push a dataset for cloud-side use)
aws s3 sync /path/on/cluster/dataset/ s3://your-bucket/dataset/

# S3 → Cloud (pull on the spot instance at job start)
aws s3 sync s3://your-bucket/dataset/ /local/dataset/

# Cloud → S3 (push results before the spot instance is reclaimed)
aws s3 sync ./checkpoints/ s3://your-bucket/run-id/checkpoints/

# S3 → Cluster (pull results back for archival or further analysis)
aws s3 sync s3://your-bucket/run-id/ ~/results/run-id/
```

For very large datasets (>1 TB), Globus endpoints are faster than `aws s3 sync` and most institutions have a Globus subscription. For smaller workflows, `aws s3 sync` is fine and avoids another tool.

---

## Recipe 1 — Cluster baseline, cloud sweeps

The most common shape: the cluster has the canonical implementation and the final-paper-results training runs. Cloud does the architecture sweep that gets you to the point where the cluster run is worth doing.

```
Week 1 (cloud):
  Cloud:  Launch 20× L4 spot instances in parallel
          Each runs train.py with one architecture variant
          Each writes losses + best checkpoint to S3
          Total wall-clock: ~1 day. Total cost: ~$50.
  Cluster: idle for this phase

Week 2 (cluster):
  Cloud:  pick winning architecture from the sweep
          push winning config to git
  Cluster: queue one big training run with the winning config
          (3 days in queue + 2 days running = OK because exploration is done)
  Cloud:  idle
```

The cluster's predictable-long-run advantage matches the "final paper run" workload. The cloud's elastic-parallel advantage matches the "20 variants in a day" exploration workload. Trying to do the sweep on the cluster: you'd be waiting in 20 separate queues. Trying to do the final run on cloud spot: doable, but you'd be eating the resilience overhead for a 2-day stable run.

---

## Recipe 2 — Cluster for storage, cloud for compute

The cluster is treated as a data lake; the cloud is treated as a compute layer that touches the data through S3.

```
Cluster:  Long-term home for datasets (raw, processed, archives)
          Pushed to S3 via nightly aws s3 sync (one-way mirror or selective)
          Cluster compute optional — cluster doesn't have to run jobs in this pattern

Cloud:    All training runs happen here
          Each run: aws s3 sync the relevant subset, train, sync results back
          Spot is the default; on-demand for interactive
          Total cost scales with active compute, not data size
```

This works well when your cluster's compute is overcommitted but its filesystem is fine. You keep the institutional storage advantage (lots of space, good backup policy, IRB-approved location) without paying the queue tax.

Watch out for: **data egress costs**. AWS charges to *download* from S3 but not to upload, so a one-way cluster→S3 mirror plus cloud-side compute is the right direction. Trying to do the inverse (cloud-side data, cluster pulls back nightly) racks up egress fees fast.

---

## Recipe 3 — Cluster for long stable training, cloud for short experiments

Inversion of Recipe 1: the cluster is doing one continuous training run for weeks (e.g., a foundation model pretraining), while the cloud is used for *side* experiments that need to finish today.

```
Cluster (main thread):
  Pretraining a 200M-param model for 3 weeks
  Submitted to a long-allocation queue (one job, runs to completion)
  Don't touch it; it's running

Cloud (side thread):
  "Does this new tokenizer behave better on a small dataset?"
  Launch g6.xlarge spot, run 1-hour experiment, terminate.
  Throughput: multiple side experiments per day, independent of the cluster's progress.
```

This is the "I don't want to stop the cluster job to test something" pattern. It also avoids the "let me submit a 1-hour job and wait 3 days for it" anti-pattern, which is the worst way to use a queued cluster.

---

## When *not* to do hybrid

Hybrid has overhead. Avoid it when:

- **Your project is small enough to fit entirely on one or the other.** A semester project, a single paper with no follow-up — pick one side, finish, move on. The S3-as-seam discipline isn't worth it for one-off work.
- **Your data has residency / compliance constraints.** PHI, FERPA-bound student records, classified data — talk to your institution's compliance office before moving anything to a public-cloud bucket. The cluster's HIPAA / FedRAMP posture is usually established; cloud's is a separate (solvable, but real) project.
- **You're early-PhD and still learning ML.** The cognitive overhead of "where does my data live, where am I running this experiment, where do checkpoints go" is a tax. Use whichever side you understand first. Add hybrid later when it solves a real bottleneck.
- **Your cluster has no queue.** If your cluster delivers compute the same day you ask, the cloud's instant-availability advantage disappears. Hybrid is mostly about beating the queue; if there isn't one, the cluster wins on cost and you can ignore the cloud.

---

## Tooling, briefly

- `aws s3 sync` — the only command you need to learn for the data-seam. Idempotent, resumable, handles concurrent uploads.
- `aws s3 cp --recursive` — for one-shot transfers.
- Globus endpoints — preferred over `aws s3 sync` for very large (>1 TB) transfers. Most universities have an institutional Globus subscription; ask your research-computing office.
- [`spore.host`](https://spore.host) (`truffle` + `spawn`) — wraps the cloud-side launch + auto-terminate lifecycle so the burst-to-cloud step is one command per recipe.
- Whatever you already use on the cluster (Slurm, PBS, LSF) — no change there. The hybrid pattern doesn't replace your cluster workflow, it adds a parallel cloud workflow with S3 as the bridge.

---

## See also

- [`AWS_GETTING_STARTED.md`](AWS_GETTING_STARTED.md) — first cloud spin-up, end-to-end
- [`SPOT_RESILIENCE.md`](SPOT_RESILIENCE.md) — designing cloud-side jobs that survive spot interruption
- [`HARDWARE_SELECTION.md`](HARDWARE_SELECTION.md) — which AWS instance for which workload

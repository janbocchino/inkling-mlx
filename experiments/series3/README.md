# Series 3: quantization, recurrence and reasoning completion

Status on 23 November 2026: **blocked before reference execution by Leonardo maintenance**.

The primary question is whether lower precision changes reasoning completion,
and whether repeating the same layers magnifies that change. The intended core
comparison has one versus three passes of the variable-loop research checkpoint,
each with its own released-source, four-bit FFN and two-bit FFN conditions.
Mac deployment is optional; no model job runs automatically on the Mac.

## What exists

- `protocol.py` generates 72 small, authored diagnostic cases across calibration,
  development, evaluation and confirmation splits. Every six-case prefix is
  category-balanced. These are mechanistic diagnostics, not a broad benchmark.
- `metrics.py` separates completed answers from outputs cut off at a token limit.
  A trace cut at 192 tokens provides no observation at a 512-token budget.
- `reference_gate.py` prepares an independent tiny-model comparison between a
  pinned Transformers reference and the pinned MLX implementation. It covers
  one, three and five passes and cached/full-sequence decoding. It rejects any
  unmatched parameter. **It has not run on Leonardo and is not validated.**
- `readiness.sbatch` and `setup_reference.sh` create an isolated, run-local CPU
  reference environment on Leonardo. They never update the previous experiment's
  environment. They request no GPU and download no full checkpoint.
- `test_protocol.py` tests case balance, completion/censoring semantics, strict
  JSON scoring and representative parameter-name mappings without model imports.

The frozen protocol and source revision/weight hashes are in the accompanying
preparation bundle. The initial submitted gate job was 65188325. It remained
pending for maintenance and was cancelled with elapsed time 00:00:00.

## Run the small local checks

```sh
python3 experiments/series3/test_protocol.py
```

This imports no MLX, PyTorch or models. The reference script itself refuses to
run outside Linux. Syntax checks and these unit tests do not establish model
parity or performance.

## Resume on Leonardo

1. Verify a valid allocation and the scheduler's current maintenance reservation.
   The recorded allocation ends on 24 November; GPU maintenance was scheduled to end
   on 25 November at 08:00 Europe/Rome. Renewing SSH does not resolve this mismatch.
2. Establish a new run directory and record fresh resource deadlines. Preserve
   this cancelled attempt and all earlier results. Do not reuse partially built
   environments or incompatible numerical caches.
3. Copy this kit plus the frozen protocol to scratch. Supply `SERIES_PROJECT`
   (the existing project containing the container) and `SERIES_RUN` (the new
   scratch run) when submitting `readiness.sbatch`, and provide a verified
   authorized account with `sbatch --account`. Keep full model work off login nodes.
4. Require the independent reference gate to pass. If dependencies or numerical
   mappings need fixes, record a new code hash and rerun the gate. This is a
   development gate, not a guarantee the released checkpoint is correct.
5. Validate source reconstruction, full/cached generation and measured throughput
   on allocated compute before freezing the final prompt count and output cap.
   Only then implement/freeze and submit the scientific comparison controller.

## Remaining work before the comparison

The full comparison controller, released-checkpoint validation, source-history
transition probes, per-pass activation/routing measurements, matched calibration
sampling, dense-control adapter and optional Mac budget guard are **not yet
implemented**. The GPU execution alternative is an untested proposal. This kit
must not be described as an already running or completed scientific experiment.

See `docs/RESEARCH_PLAN.md` and `configs/series3-plan.json` for the broader design
and resource limits. Existing source models, baseline exports and old results
remain unchanged. No new full model has been downloaded.

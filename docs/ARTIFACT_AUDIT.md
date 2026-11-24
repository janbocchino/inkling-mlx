# Artifact and code audit

Reviewed 22 November 2026 against the complete experiment workspace. `manifests/artifact-inventory.json` lists 494 authored/evidence/supporting files with sizes and, for small artifacts, original checksums. Installed virtual-environment dependencies and the downloaded upstream checkout and staged third-party packages are not counted as project-authored code. Large model/tokenizer payloads were inspected through their existing manifests/configuration/header evidence rather than rehashed during the inventory pass.

## Evidence retained

- Original baseline and series-2 numerical Python and Slurm scripts; converter/series tests; matrix RPC implementation; historical local probe/download/recovery helpers.
- Baseline conversion, streaming tensor verification, smoke/context/tuning records, and launcher check.
- Series-2 plan, source/data/code fingerprints, calibration coverage, packing/hybrid/full-model pilots, numerical screens, six export reports, per-case NLL, finalists, logs, heldout outputs, context checks, timeouts and recoveries.
- Original/published file hashes in `results/PROVENANCE.json`, plus a checksum manifest for the published experiment snapshots.

Numerical Python and `.sbatch` files remain byte-identical. Descriptive `PLAN.json` and `README.txt`, and published result/log copies, have machine/account locations replaced with placeholders. Their public hashes consequently differ from original hashes. Original embedded code/data hashes continue to describe the historical inputs; they must not be interpreted as checksums of a redacted manifest itself.

## Findings and corrections

| Finding | Treatment |
|---|---|
| Early prose says all expert scales/offsets are BF16 | Corrected in the report: native NVFP4 global scaling promotes reconstructed values to FP32; metadata preserves that precision. Historical source files remain unchanged. |
| Baseline conversion time conflated with entire batch-job duration | Report distinguishes 465.8 s conversion from 8 min 33 s job. |
| 24 recipes could be read as 24 model evaluations | Report distinguishes numerical screening from six complete new exports plus baseline evaluation. |
| Reference aggregate includes 45 cases, candidate comparison 30 | Report recomputes source NLL on the exact common 30 records. |
| Baseline/S4 heldout retries overwrite run-level metadata while retaining saved cases | Completion counts, original logs and recovery status are reported; retry elapsed fields are not treated as total original evaluation duration. |
| Ambiguous shipment/order extraction prompts | Frozen data and primary scores retained; explicit post-hoc sensitivity, not an edited benchmark. |
| Short reasoning/output caps yield missing final answers | Counted as failures; do not substitute partial reasoning for a final answer. |
| Early chat helper considers a content tag sufficient | Cleaned launcher requires an end-message tag too; historical evidence is not rescored. |
| Busy Mac, large initial swap, warning pressure, Metal timeouts | Reported as measured constraints; no causal diagnosis or comfortable-fit claim. |
| Context repetitions/prefill-256 plan incomplete | Report specifies the one recovered sweep per R64/E4 and missing S4/repeated sweep. |
| Historical watcher assumes fixed paths and fails on transport EOF | Kept as archive; portable launcher/analysis are separate. No stale monitor is restarted. |
| Cache identity does not fully bind all external numerical payloads | Reproduction guide requires fresh run/cache directories after any input change. No old cache is silently reused. |
| Controller archive copies and RPC service are research-specific | Documented as a staged, trusted-cluster procedure, not a production framework. |
| Exact container digest and complete billed compute totals missing | Explicit reproducibility/cost limitations; the container definition is labeled a reconstruction recipe. |

## Cleanup performed

The publication repository separates frozen experiment snapshots from portable tools. The new command-line analysis has no MLX dependency, rejects duplicate/mismatched case definitions, and preserves paired denominators. The new streaming verifier raises explicit validation errors rather than relying on Python assertions, validates tensor offsets and lengths, and accepts an explicit model path. The new chat launcher accepts any local converted checkpoint and reports the actual shard size instead of assuming the baseline path/size.

Original numerical routines were not reformatted or “fixed” after the fact: changing them would break their connection to recorded hashes. Known limitations are documented rather than retroactively changing measured results. Small new tests cover comparison alignment, incomplete results, corrupted/truncated tensor data, and completed-answer parsing. Historical tiny-model tests are rerun separately where the pinned environment is available.

## Publication scope

Weights, source images, WikiText payloads, raw frozen dataset JSON, binary activation statistics, large search caches, virtual environments, container images, SSH sockets/certificates and session authentication tokens are excluded from Git. The exact local dataset and three checkpoint links are kept in ignored folders for convenience. Existing original artifacts and models remain in the original workspace. Historical Word/PDF documents are inventoried but not duplicated into the repository; their original builders (with prose-only path redactions) are in `archive/report-builders/`; the Markdown report supersedes their interpretation.

No GitHub repository was created or pushed, no model uploaded, and no new inference/cluster experiment was started during packaging.

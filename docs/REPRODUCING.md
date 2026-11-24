# Reproducing and extending the experiments

## 1. Reproduce the reported arithmetic

From the repository root, using Python 3.11+:

```bash
python3 scripts/reproduce_results.py verify-frozen .
python3 scripts/reproduce_results.py summarize results/series2 --output local/recomputed.json
python3 -m unittest discover -s tests -v
```

The first command verifies the archived source snapshots against the repository manifest. `results/PROVENANCE.json` also records their original workspace hashes. The second recomputes matched-case differences, fixed-seed bootstrap intervals, peak MLX allocations, descriptive generation medians, and the separately labeled extraction sensitivity. It reads existing JSON and does not load model weights or contact Leonardo. `results/derived/summary.json` contains the checked output from repository assembly.

The historical `MAC_COMPARISON.json` remains unchanged except for path redaction elsewhere in the archive. The new analysis validates duplicate IDs and paired answer/category/modality definitions instead of silently trusting them.

## 2. Check separately supplied model weights

```bash
python3 scripts/reproduce_results.py verify-model /path/to/converted/model
```

This reads tensor bytes in chunks and compares every shape, dtype, and SHA-256 against `CONVERSION_REPORT.json`. It does not allocate the model in MLX/Metal. It checks consistency with the supplied report, not authenticity of an independently untrusted report. For finalists, the original transfer also checked full file hashes against the archived finalist metadata.

Use the README's pinned Mac environment and `scripts/ask_inkling.py --model ...` for text/image inference. No model server is started. Quit the process to release its allocations. The cleaned launcher requires a completed final-answer tag before treating output as final, unlike the early interactive helper. It does not re-score historical results.

## 3. Original tests without the real checkpoint

In an environment containing the pinned MLX/MLX-LM runtime:

```bash
python experiments/series1/test_converter.py
python experiments/series2/test_converter.py
python experiments/series2/test_series.py
```

These construct tiny synthetic models and explicitly use the CPU backend. They do not need real weights or download a dataset. `test_rpc.py` additionally needs a running internal CUDA matrix service and is not part of this lightweight command set. Historical test stdout/stderr is under `results/*/tests/`.

## 4. Exact local data versus a new data freeze

`data/DATA_MANIFEST.json` records hashes of the original calibration/development/heldout JSON and image arrays. On the original workstation, their exact bytes have been copied into `experiments/series2/data/`, an ignored directory. The public `data/case-index.json` records split membership and provenance metadata without copying full questions, WikiText passages, or source images.

A public clone does not contain the payloads. To reproduce inference exactly, restore the original frozen dataset through an appropriate dataset-sharing channel and validate every manifest hash. For a new study, install `requirements/data.txt` and run the original preparation script in a separate working copy. Its WikiText viewer query and dataset revision discovery are live, and generated OCR uses an environment-dependent font. New bytes require a new dataset manifest, run ID, and evaluation context. Do not relabel a regenerated dataset as the original one or reuse old clipping/statistics caches.

## 5. Leonardo: recorded architecture and prerequisites

The historical Python controller and batch scripts are in `experiments/series2/`. Their account/partition defaults are a record of the 2026 run, not a promise of current access. Supply your own valid account and consult the site's current module/partition documentation before submitting a new run.

The original remote project layout was:

```text
PROJECT/
  source/                         original checkpoint, treated read-only
  converted/BASELINE/              validated R64 checkpoint
  containers/python311-bookworm.sif
  env-cpu/                        Python venv created inside the container
  series2-kit/                    staged copy of experiments/series2 plus frozen data
  experiments/RUN_ID/              durable archive written by the controller
SCRATCH/RUN_ID/                    runtime records, statistics, queues and exports
```

The exact old container image digest was not captured. `containers/python311-bookworm.def` is a **reconstruction recipe**, not a claim of a bit-identical historical image. The recorded CPU environment used Python 3.11, `requirements/leonardo-cpu.txt`, and MLX 0.32.4. The separate host GPU service used `profile/deeplrn`, `cineca-ai/4.3.0`, CUDA 12.1 and PyTorch `2.2.0a0+git6c8c5ad`. Do not replace that GPU environment with the Mac requirements or assume a container changes the host driver.

Before the main series, the original procedure did the following:

1. Stage the source, validated baseline, container, pinned CPU environment, code and exact frozen dataset into a fresh project directory. Make `logs/` and runtime subdirectories.
2. Run the small converter/series CPU tests inside that environment. Generate parity fixtures with `clipping.parity_fixtures(RUN_PATH)`.
3. Submit `matrix_service.sbatch` with `SERIES_KIT` and `SERIES_RUN`. Its private `GPU_ENDPOINT.json` carries a session token; never commit it. This is a trusted internal-cluster research service, not a public network endpoint.
4. Run the packing parity worker in parity-only mode. Run `test_rpc.py` and `cpu_phases.py pilot` through `full_pilot.sbatch`, which binds project/runtime paths into the container. Review `HYBRID_PARITY.json` and `PILOT.json`; both must pass.
5. Create a new runtime `RUN_MANIFEST.json` from the planned recipes and **current** code/data/source fingerprints, including the new run ID, baseline relative path, and matrix-service job ID. The archived manifest has historical paths and original metadata hashes; do not use it unchanged as an executable new-run manifest.
6. Submit `overnight.sbatch` with `SERIES_PROJECT`, `SERIES_KIT`, and `SERIES_RUN`, overriding account/partition/QoS if necessary. Preserve or lower the recorded resource ceilings. Inspect status/logs and keep only the new run's artifacts eligible for cleanup.
7. Transfer verified finalists and test them sequentially on the deployment Mac with a new results directory. Preserve failures and incomplete-case denominators.

This is a reviewed execution checklist, not a one-command cluster installer. The original code assumes the staged `/work/series2-kit` layout. Some historical watch/recovery scripts assume the old workspace layout and should not be launched from a clone; they are included for provenance. For a new run, supply an explicit portable handoff rather than starting the archived watcher unchanged.

## 6. Known numerical/recovery boundaries

The original converter's resume identity includes code, source config, recipe and search location, but does not fully hash all source weights/search-map bytes into every cache entry. Search completion checks can trust existing JSON records. They were only used with the fixed inputs of this run. For a new series, use **fresh run/search/checkpoint directories**; do not rely on these checks as general cross-run cache validation.

The controller requests a bounded set of workers and can prune only its own candidate folders. Its archival-copy resume behavior and the local watcher are experiment-specific; the download was recovered manually after EOF. The Mac recovery preserves the original four-hour deadline but uses an originating-filesystem creation timestamp. These helpers need further engineering before unattended reuse elsewhere.

## 7. Repository validation versus experiment replication

The assembly validation report under `results/VALIDATION.json` records what was actually checked after cleanup. It is separate from the historical experimental tests. No full-model inference, new cluster submission, calibration, or benchmark is silently rerun while packaging this repository.

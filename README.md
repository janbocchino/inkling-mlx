# Inkling 2 Mini → MLX: experiments for a 16 GiB Mac

An independent, AI-assisted research project exploring further quantization of Inkling 2 Mini for local **text and image** inference. Experiments ran on CINECA Leonardo and an Apple M5 Mac with 16 GiB unified memory on 20–22 November 2026.

**Result:** verified weight storage fell from **10.26 GiB to 8.48 GiB (−17.37%)**. The smallest model completed all 180 custom diagnostic questions. Other evaluations hit Metal GPU timeouts, and memory-pressure warnings persisted. **This is an exploratory result, not a quality-equivalent or comfortably fitting 16 GiB release.** “Two-bit” describes most expert codes; these are mixed-precision checkpoints.

Read the **[full experiment report](docs/REPORT.md)** for the abstract, methods, results, negative findings, corrections, and limitations.

## Start here

Analyze the committed results without a model, MLX, SSH, or an allocation:

```bash
python3 scripts/reproduce_results.py summarize results/series2
python3 scripts/reproduce_results.py verify-frozen .
python3 -m unittest discover -s tests -v
```

Python 3.11 or newer is sufficient for these commands. Optionally install the small dependency-free analysis package with `python3 -m pip install -e .` to use the `inkling-mlx` command.

## What happened

| Stage | Recorded outcome |
|---|---|
| First conversion | All experts, vision, and the loop schedule retained; 1,567 saved/downloaded tensor signatures matched |
| Baseline Mac smoke tests | Four simple text/image answers and supplied-identifier context checks worked; significant memory pressure |
| Leonardo series | 24 precision recipes screened, six new exports evaluated, three finalists archived |
| Local finalists | W128-E4: 8.480 GiB; W128-S4: 9.386 GiB, versus R64: 10.262 GiB |
| Heldout completion | E4: 180/180; R64: 90/180; S4: 117/180; the latter two hit GPU timeouts, including on retry |
| Paired quality | On 90 matched cases, E4 −2.22 pp; S4 −1.11 pp against R64; incomplete custom diagnostics |

E4 compresses **embeddings/head to four bits**. S4 protects **two selected expert layers at four bits**. Neither is an all-two-bit model. The report explains the ambiguous extraction category and its separately labeled post-hoc analysis.

## Try a local checkpoint

This step uses Apple silicon and loads several GiB of model weights. Result analysis above does not.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/mac.txt
python scripts/ask_inkling.py --model /path/to/converted/model --prompt "What is 17 times 23?"
python scripts/ask_inkling.py --model /path/to/converted/model --prompt "Read this image." --image /path/to/image.png
```

Omit `--prompt` for a simple interactive session. Each question starts fresh. The launcher is offline, uses greedy sampling, resizes images to at most 512 pixels, and caps prompt plus output at 4,096 tokens as a resource limit. Reasoning consumes the output budget even with effort set to `none`.

On the original workstation only, `models/R64`, `models/W128-E4`, and `models/W128-S4` are ignored local links to the existing checkpoints. A GitHub clone will not contain those links or any weights. Converted weights have not been uploaded. Obtain/convert weights separately; see [reproduction instructions](docs/REPRODUCING.md).

## Repository map

```text
docs/                    Paper-style report, reproduction guide, artifact/code audit
src/inkling_mlx/          Dependency-free comparison and streaming verification tools
scripts/                 Portable chat launcher and analysis entry point
experiments/series1/     Original baseline converter, tests, probes, and historical helpers
experiments/series2/     Original clipping, calibration, controller, CUDA service, and tests
experiments/series3/     Prepared research protocol; model checks not yet run
results/series1/         Baseline conversion/inference/context records
results/series2/         Remote evaluations, Mac outputs, logs, and recovery evidence
results/PROVENANCE.json  Original and published file hashes; path-redaction record
manifests/               Frozen-code checksums and reviewed artifact inventory
data/                    Dataset manifest and case metadata, without dataset payloads
requirements/            Observed Mac/container versions and data dependencies
licenses/                Upstream numerical-code notice
archive/report-builders/ Historical Word-report generation scripts
models/                  Git-ignored local links; absent from a public clone
```

## Reproducibility boundaries

The numerical experiment code is preserved byte-for-byte. It is an **archive of what ran**, including site-specific batch defaults and known recovery weaknesses. It is not a ready-to-submit generic cluster service. The portable launcher and analysis tools are separate, cleaned implementations; their tests do not retroactively change old scores.

Only reanalysis of saved results is immediately reproducible from a public clone. Inference requires the pinned model runtime and separately supplied weights. Exact calibration requires the frozen data; the local copy is ignored by Git because it contains third-party payloads. Running the original live-data preparation script creates a new freeze, potentially with different bytes. A new cluster series must receive a new run ID and manifests; old numerical/calibration caches must not be reused after input changes.

The project did **not** train a model, recover training master weights, implement full GPTQ/AWQ, prove benchmark equivalence, or establish comfortable 16 GiB inference. Read [the audit](docs/ARTIFACT_AUDIT.md) before extending it.

## Sharing on GitHub

A local Git repository is initialized on `main`; no remote is configured and nothing has been uploaded. The normal publication set includes source, docs, manifests, and compact evidence. Weights, raw dataset payloads, environments, containers, authentication files, and private local paths are ignored or redacted. `git status --short` shows the prepared files.

When ready, create your GitHub repository, review the local changes, commit them, then add its URL as `origin` and push. There is no automatic publication step.

## License and attribution

Project code is MIT; see [LICENSE](LICENSE). MLX-derived numerical logic retains the Apple MIT notice in [licenses/MLX-MIT.txt](licenses/MLX-MIT.txt). Model weights/runtime and dataset content retain their own terms; this repository's license does not replace them. See [NOTICE.md](NOTICE.md) and the report's references. This is not an official Thinking Machines, Apple, or CINECA release.

# Inkling 2 Mini follow up experiment plan

**Research objective updated:** [Understanding quantization across models and repeated computation](RESEARCH_PLAN.md) is now the primary agenda. This document retains the earlier data audit and optional deployment procedure; comfortable 16 GiB deployment is no longer the criterion for scientific success.

Prepared 23 November 2026, Europe/Vienna. This is the second research round after the completed overnight search; use a new `series3` run ID so the historical `series2` records remain immutable.

The evidence does not show that Mini is generally unsuitable for quantization. It shows that our earlier test mixed compression error with an unusually short reasoning budget, imperfect scoring, and memory/runtime failures. The follow-up should identify which mechanism matters, then test a small number of changes on Leonardo. Intensive Mac work has a **single cumulative one-hour ceiling**, including loading, compilation, transfer verification, tests, and recovery. After that, only small monitoring and analysis jobs may run locally.

## What the existing data reveals

### Most recorded failures are unfinished answers

| Model | Questions completed | Zero-scored cases | No final answer | All missing finals reached 192 tokens |
|---|---:|---:|---:|---|
| R64 | 90 | 20 | 18 | Yes |
| W128-E4 | 180 | 44 | 40 | Yes |
| W128-S4 | 117 | 30 | 27 | Yes |

For E4, **40 of 44 zero-scored cases exhausted the output allowance**. Twenty of those belong to the already-identified ambiguous shipment/order category; the other twenty are image questions. We have not generated longer answers, so this does not tell us how many would eventually be correct.

On the 90 questions shared with R64, E4 has two additional failures and S4 one. Every additional failure is a missing final answer at the cap, while R64 finished. There are no differing completed-answer scores on those shared cases. This is an observation about the matched subset, not proof that quantization only changes answer length.

### Some completed answers were penalized for wording

Three of E4's four completed zero-scored responses contain the expected answer with additional wording:

| Expected | E4 response | Issue to resolve |
|---|---|---|
| `macbook` | `Apple MacBook laptop.` | Exact normalized string comparison rejects a plausible correct description |
| `yes` | `Yes, it’s a Denny’s sign.` | A yes/no answer with an explanation is rejected |
| `13` | `13 years (2007–2019).` | Numeric parsing rejects a number with units/explanation |

The fourth is a genuine arithmetic error: `31 × 45 + 29` should be 1424; E4 answered 1324. R64/S4 also lose image points on extra wording, such as a more descriptive green-line answer. Preserve the original scores. Publish a separately versioned scorer with validated answer handling and manual adjudication for residual ambiguity; do not silently award substring matches, which can accept contradictory answers.

### The old likelihood comparison is too small for broad conclusions

All candidates used the same 30 development questions, but only **105 gold answer tokens** were scored. Three easy JSON questions contribute **57 tokens, or 54.3%** of that total; all ten image questions contribute only 15. Source and candidate aggregates are comparable on these same inputs, but the result is a narrow measurement.

The likelihood harness also prefills the assistant's final-answer segment and scores only the short gold continuation. Native generation starts from the user prompt and produces reasoning first. These are different conditional tasks; the old likelihood is not a prediction of the model's complete reasoning behavior.

The layer-protection signal remains interesting. At the same group size, W128-S4 protects stored layers 16/17 and changes text NLL from **0.9812 to 0.5363**, while image NLL worsens from **1.8382 to 2.0057**. It adds 0.475 GiB. The improved text signal persists when extraction is excluded and under category averaging, but it has not translated into a completed heldout win. Test text and vision separately rather than calling S4 universally better.

### Metadata precision offers a cheap experiment

Each checkpoint has 80 FP32 tensors holding expert affine scales/offsets. Changing only those tensors to BF16 would reduce their payload size as follows:

| Checkpoint | Current weight GiB | Calculated saving | Estimated weight GiB after metadata casting |
|---|---:|---:|---:|
| R64 | 10.262 | 1.187 GiB | 9.076 |
| W128-E4 | 8.480 | 0.593 GiB | **7.887** |
| W128-S4 | 9.386 | 0.593 GiB | 8.793 |

These are arithmetic estimates from tensor signatures, excluding small header changes. No such checkpoint has been exported or tested. Packed codes would remain identical, but metadata rounding changes reconstructed values and might change behavior. Routers, gates, norms, vision, and other protected parameters must remain untouched. The VisualInference card also reports a kernel penalty when affine metadata and activations differ in dtype; treat a speed improvement as a hypothesis to measure, not an established result for our machine.

## Why additional quantization may be difficult

The released model already uses NVFP4 for the first twenty expert layers and MXFP8 for the last four. Its authors trained with those formats. Our two-bit conversion compresses that representation further; casting it to a larger floating dtype does not recover training-master weights. [The authors describe this training and deployment policy](https://thinkingmachines.ai/news/inkling-2-mini/).

The middle layers are reused across three passes, with learned cross-pass messages. A weight perturbation can therefore affect several visits and subsequent routing decisions. Top-8 routing can change at close ties. These are plausible mechanisms for sensitivity, but routing changes alone are not evidence of quality loss, and we have not measured per-pass divergence ourselves.

Some apparent problems have other explanations. The [official card](https://huggingface.co/thinkingmachines/Inkling-2-Mini) documents repetitive reasoning even without further quantization. The earlier 192-token cap is particularly restrictive for that behavior. New runtimes also matter: a [merged llama.cpp fix](https://github.com/ggml-org/llama.cpp/pull/32704) restores missing expert scales in later loop passes. That is a concrete GGUF runtime defect, not a diagnosis of our MLX run. Identify the exact model, runtime revision, prompt, and symptom before attributing a failed Hugging Face model to quantization.

Moderate quantization remains credible. [VisualInference's card](https://huggingface.co/VisualInference/Inkling-2-Mini-MLX-2bit) reports substantially less likelihood degradation at three bits than two, although those are the author's results on different data/hardware. The testable question is where Mini's compression limit lies for our workloads and memory budget.

## Questions the next round should answer

1. Does the released checkpoint fail the same prompts under the same runtime and output cap?
2. How much apparent degradation comes from failure to finish reasoning, versus wrong completed answers?
3. Does BF16 expert metadata improve storage and deployment throughput without material quality loss?
4. Does protecting looped layers help more than protecting equally sized tail layers, and does the answer differ for text and images?
5. Can one candidate meet a declared quality/completion target and remain usable on the 16 GiB Mac?

## Practical admission criteria for published models

The target is useful text-and-image inference while the owner continues using the 16 GiB Mac. Published file size, a successful toy prompt, or a benchmark on a large Mac is insufficient. Independently assemble reproducible short chat, arithmetic/code, chart, scene-text, and repeated-turn cases; do not depend on the owner supplying a failure prompt. Treat any later example as an additional diagnostic, with evaluation membership fixed before selection.

A fresh card check on 23 November identified these controls:

| Published build | Practical evidence and limitations | Treatment |
|---|---|---|
| VisualInference 2-bit | 9.1 GiB weights; measurements on an M4 Pro with 48 GB, with 16 GB guidance inferred; only two image checks | Conventional low-bit quality comparator on Leonardo; not a proven 16 GiB deployment |
| NetProjects 2bit-mixed | 9.5 decimal GB weights plus optional draft; reported 4K peak 16.14 GB at default prefill versus 10.60 GB at prefill 128 without draft; measured on M5 Max with 128 GB; vision is two-bit and its shape description was imperfect | Useful resource/runtime stress comparator; transfer to Mac only if it wins remote testing and its exact decoder is validated |
| LookAtMyQuant 5.4 GB | 5.38 decimal GB text model, separate 637 MB vision; custom runtime and training; only four image examples in the card | Investigate as a small-format comparator if Linux execution is validated; its size is not proof of a complete dependable multimodal package |

Pin each card, checkpoint, and runtime because these projects are changing quickly. The NetProjects two-bit card checked here was revision `54eb23b731d6f8feaa02f602f75d020e0b6c33d6`. Do not compare earlier card numbers with revised weights under the same repository name.

For the proposed finalist, prefer weights at or below 8 GiB and target tracked MLX peak at or below 10 GiB on a 2K prompt and the fixed image checks. These are engineering admission targets, not guarantees of total memory fit. Verify actual system pressure and swap with the owner's ordinary applications present; do not close unrelated apps or raise the wired-memory limit automatically. A baseline already under memory pressure makes the fit verdict inconclusive. A candidate producing sustained new pressure, repeated timeouts, or materially growing swap does not qualify as comfortable deployment merely because it loaded.

Require complete, correct final answers and useful end-to-end latency under the fixed rubric. Evaluate repeated turns as well as fresh prompts. A candidate with no more than two points of loss against a poor source still needs a useful absolute result on the declared workload. If no candidate satisfies both behavior and memory requirements, the round's result should explicitly be that a satisfactory 16 GiB Mini configuration was not found.

Sources: [VisualInference](https://huggingface.co/VisualInference/Inkling-2-Mini-MLX-2bit), [NetProjects two-bit mixed](https://huggingface.co/netprojects/Inkling-2-Mini-MLX-2bit-mixed), [LookAtMyQuant](https://huggingface.co/shinmegamiboson/Inkling-2-Mini-LookAtMyQuant-5.4GB). These remain author-reported measurements until reproduced.

## Leonardo workflow

### Stage 1 Establish a valid reference and freeze the tests

Budget the first 60–90 minutes for environment and throughput pilots. Pin the source and every third-party checkpoint/runtime to immutable revisions. Reproduce source-tokenizer IDs, image preprocessing, all 48 layer visits, per-pass caches, and native expert global scales. Check full-sequence versus cached decoding on text, an image, and a context crossing the sliding-window boundary. Compare the same source checkpoint against a trusted implementation before judging compression. Use FP32 tiny-model parity checks and measure the full-source backend's own BF16/FP32 variation; do not reject a valid port solely because near-tie routing prevents bit-identical logits.

Start with the validated Linux CPU-MLX/CUDA bridge. It previously generated only about **1.1 tokens/second**, so a large generative benchmark cannot be promised within one night. A faster supported CUDA path may be piloted in an isolated environment, but is not assumed available. Do not modify the host driver or spend the run developing a new model framework.

The old CUDA bridge decodes affine codes with `32 // bits`, which is not sufficient for MLX's special three/six-bit packing. Its old parity fixtures covered two and four bits only. A three-bit published control is admitted only after exact packing/dequantization parity fixtures pass; otherwise use a four-bit control or report it as deferred. Keep native NVFP4/MXFP8 and all per-expert global scales covered by tests.

Create a new dataset/scorer version. Remove or explicitly repair ambiguous extraction tasks before freezing. Keep historical questions as a separate retrospective stress set. Use distinct calibration, development and locked evaluation IDs, with balanced categories, answer-length-aware summaries, and source-image separation. Report text and image metrics independently. Keep semantic adjudication blind to model identity.

### Stage 2 Diagnose before converting more models

Compare three core controls: the released source, existing E4, and a pinned published two-bit model such as VisualInference. Candidate weight values must be preserved by any Linux import adapter; template/runtime differences receive a separate label. If a common validated backend is unavailable, do not attribute cross-runtime differences solely to weight precision.

Use a fixed balanced diagnostic panel, including text, charts, scene text, and examples of the user's reported failure when available. On a validated faster backend, target 24 cases per core model with up to 1,536 output tokens. Store emitted token IDs and raw control tags so completion can be measured at 192, 512, and 1,536 tokens from the same generation trace. Prefix replay is valid only if the output limit does not change prompting, sampling, or earlier tokens; check that property first. Otherwise run the cap conditions separately.

Use greedy decoding as the primary paired condition. On a small separate panel, compare the source's shipped sampling settings, including temperature 1.0, using three fixed seeds for source and candidate. This tests whether greedy looping is being mistaken for a compression failure. Keep this secondary and do not select favorable seeds.

Before candidate outcomes are examined, use pilot throughput to freeze one of two scopes:

- **Full diagnostic scope:** the 24-case panel plus a larger locked set, only if measured throughput predicts completion within the reserved phase budget.
- **Slow-backend fallback:** six fixed balanced cases per core model at up to 768 tokens, plus a small predeclared long-budget panel if time permits. Teacher-forced diagnostics carry most of the comparison. Label the result reduced-scope rather than an accuracy validation.

Give this stage a firm time/token quota. No candidate gets a selectively easier subset. Report intended and completed counts, raw correctness, final-answer completion rate, truncation rate, repeated reasoning, and runtime exceptions separately. Never grade text copied out of unfinished reasoning as the official final answer.

### Stage 3 Evaluate targeted changes

Run these interventions in priority order; do not repeat the 24-recipe sweep:

| Priority | Intervention | What it isolates |
|---|---|---|
| 1 | E4 with identical codes and only FP32 expert scale/offset tensors cast to BF16 | Metadata dtype, storage and numerical perturbation |
| 2 | Restore eight-bit embeddings/head relative to the best metadata setting | Whether the 0.431 GiB head/embedding saving particularly hurts image answers |
| 3 | Protect layers 16/17 at four bits versus an equally sized two-layer tail pair | Looped versus tail sensitivity at the same memory budget and metadata policy |
| Optional | Block-diagonal Hessian error compensation on the vulnerable expert projections | Whether compensating correlated errors beats diagonal clipping at the same stored precision |

Run small module-level counterfactuals before materializing full exports. For protected-layer comparisons, equalize group size and metadata dtype and verify actual bytes. The optional method is new implementation work, not an existing switch: label it blockwise error compensation rather than full GPTQ. Gate it on numerical tests, memory, and time. Full per-expert Hessians can be expensive; use bounded blocks and a declared rare-expert fallback. Never reuse old clipping maps as if they encoded compensated rounding.

If collecting new statistics, target 32,000 actual processed input tokens with at least 48 diverse images. Record text and expanded visual token counts separately and accept a smaller explicitly reported calibration if the pilot cannot meet the budget. Training/calibration examples and teacher traces must exclude locked evaluation items. Source statistics are collected from the reference, not the already-compressed candidate.

For a fixed probe set, record hidden-state error by stored layer and loop visit, routing margins and top-8 overlap. Compare candidate divergence with the source backend's own precision variation. These are diagnostic measurements, not substitute accuracy scores. Do not change the fixed three-pass schedule to gain speed; variable looping belongs to a separately trained research model.

### Stage 4 Select using independent behavior and resource results

Score heldout teacher continuations long enough to cover reasoning/control transitions as well as final answers. Keep gold final-answer NLL as a separate diagnostic. Use equal-case and equal-category summaries alongside token-weighted NLL so a few easy JSON strings cannot dominate.

Selection uses development data only. Evaluate the chosen candidate against the released source and existing E4 on the locked set; do not tune after inspecting it. A provisional engineering pass requires at most two percentage points overall loss and five per category, at most two points of additional missing-final rate, and no hidden runtime failures. Report paired intervals and denominators; small samples cannot establish equivalence. Incomplete evaluation never receives a pass. If the released source itself fails the user's workload, report that as a source/workload limitation rather than inventing a quantization rescue.

Export at most four new variants in total, keeping a rolling maximum of two full new checkpoints. Transfer at most one finalist to the Mac. Verify serialized tensors, unchanged packed codes for the metadata-only intervention, protected parameters, and file hashes. No training-master recovery, expert pruning, distillation, or large QAT run is part of this round.

## Remote resource limits and timing

Plan for a **9–12 hour remote run**, excluding queue delay. Cap the controller at 12 hours, one A100 matrix service at 12 GPU-hours, and an optional second A100 worker at four GPU-hours. Keep all main GPU usage, including recovery, below **18 GPU-hours**. Use approximately 16 CPU cores and 192 GiB host memory for the existing bridge, with measured pilot checks before any larger request. Keep new artifacts below **60 GiB** and retain a 60 GiB work-space reserve. These are request ceilings, not estimates of billed use.

| Remote stage | Indicative elapsed window | Gate |
|---|---|---|
| Runtime/data/throughput checks | 0–1.5 h | Stop model comparisons if the source path is wrong |
| Source/E4/published diagnostics | 1.5–4 h | Freeze full or reduced scope from pilot timing |
| Metadata and targeted ablations | 4–8 h | Skip optional Hessian work if time is tight |
| Locked evaluation and export | 8–11 h | Choose using development only; disclose incomplete tests |
| Archive, verify and release jobs | 11–12 h | No new phase after the reserve begins |

Parallel work is limited to the optional second worker and must fit the same GPU-hour ceiling. Heavy generation receives an explicit quota from measured throughput. The existing slow bridge cannot support hundreds of long generations within these bounds; reduce scope instead of exceeding the budget.

The recorded allocation ends **24 November 2026**. The SSH control socket is gone and batch authentication failed during this planning check. Before launch, renew SSH, verify `saldo -b`, and establish the actual scheduler/account cutoff. The controller deadline must be the earlier of the twelve-hour limit and the confirmed expiry minus a finalization buffer. If a full run no longer fits, run only the source/metadata diagnostic. Do not switch accounts silently or assume a different allocation is available.

## One intensive Mac hour

The Mac is used only for native Metal deployment checks after Leonardo selects the candidate. Start the hour in a user-chosen window, not automatically when a download finishes. Set one durable wall-clock deadline before any intensive local work, at most 3,600 seconds away. Restarts and retries keep that same deadline; another heartbeat cannot grant a fresh hour.

| Window | Work |
|---|---|
| 0–8 min | Transfer/verify the single finalist if necessary; abort or reduce testing if transfer exceeds this allocation |
| 8–30 min | Existing E4: load, one warmup, up to 24 fixed cases (16 text/8 image), and brief 512/2,048-token context checks |
| 30–52 min | New finalist: same cases/settings/context checks, sequentially, with no concurrent model |
| 52–57 min | Remaining paired checks only if they fit the existing deadline; no open-ended debugging |
| 57–60 min | Stop all owned inference descendants, release Metal allocations, save reports and verify no owned GPU job remains |

The 24-case count is a target, not a promise: use per-request deadlines and record missing cases. Keep output budgets large enough to test completion, with budget curves obtained from stored traces; per-request timeouts remain separately reported. The clock includes imports, loading, compilation, checksums, stalled generation, and recovery. Stop earlier after a repeatable GPU timeout or sustained critical memory pressure. Do not raise the system GPU wired-memory limit just to force a pass.

Record time to a complete final answer, first-token delay, generated tokens, tracked MLX peak, pressure and system-wide swap before/after. Warning pressure or growing swap prevents a claim of comfortable fit. No baseline with 18 GiB weights needs to be loaded on this Mac; source-quality comparisons stay on Leonardo. If the finalist uses a different runtime, its loader/parser must be prepared and validated remotely before the window; failed local integration consumes the same hour and is reported, not retried all day.

After the hour, local work is limited to short SSH/status checks and JSON/report processing, targeting one CPU core, under 256 MiB working memory and short bursts. No MLX/Metal initialization, model loading, large model downloads/checksums, conversion, warmups, or background benchmark retries. Continuing generation or adjudication runs on Leonardo, subject to its account deadline.

## Deliverables and stopping decisions

The round should produce a causal diagnosis, a table comparing the reference/old/new models under the same conditions, per-case failure categories, completion-versus-budget curves, per-pass numerical probes, measured sizes, and the one-hour Mac report. A negative result is useful: identifying intrinsic reasoning loops, a runtime defect, an inadequate scorer, or an unacceptable two-bit trade-off is preferable to another unsupported “fits 16 GB” claim.

Stop further compression if validated two-bit candidates repeatedly fail while the released source succeeds, and the better settings exceed the Mac's usable memory. In that case, report Mini as a poor choice for this particular 16 GiB workload—not universally unquantizable. If metadata rounding gives similar quality and lower pressure, retain the simpler change before investing in a larger calibration algorithm.

This plan does not launch jobs. The immediate next operational step is SSH renewal and the short source/runtime pilot; the Mac hour waits until a finalist is ready and the user has a suitable window.

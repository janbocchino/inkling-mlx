# Requantizing Inkling 2 Mini for 16 GiB Apple Silicon

## An exploratory study of mixed precision, clipping, and practical multimodal inference

**Experiment dates:** 20–22 November 2026, Europe/Vienna

**Status:** Independent experimental report; not peer reviewed

**Scope:** One released model, one 16 GiB Apple M5 Mac, and a bounded CINECA Leonardo experiment

**Implementation and analysis:** AI-assisted; findings checked against the archived code and measurements

## Abstract

Can a released multimodal mixture-of-experts model be compressed further for useful inference on a 16 GiB Mac? We investigate Inkling 2 Mini using a structurally verified affine MLX conversion, followed by a bounded search over 24 mixed-precision recipes on CINECA Leonardo. The initial group-64 baseline occupied 10.262 GiB. Six new exports were evaluated on the same 30 development questions; three finalists were retained and two transferred to an Apple M5 Mac. The smallest finalist occupied 8.480 GiB, a 17.37% reduction relative to the baseline, and completed a 180-question text-and-image diagnostic evaluation. The baseline and the other finalist completed only 90 and 117 questions before Metal GPU timeouts, and bounded retries also failed. On the 90 shared completed questions, the smaller finalist scored 2.22 percentage points below the baseline. Memory-pressure warnings and substantial pre-existing system swap prevent a claim of comfortable 16 GiB deployment. The study demonstrates a verified reduction in stored weights and identifies scale precision, incomplete evaluation, output-budget failures, and system memory pressure as important constraints. It does not establish quality equivalence or a reliable deployment configuration.

## 1. Motivation and research questions

Local text-and-image inference can avoid per-request server dependence and put capable models on comparatively inexpensive hardware. A 16 GiB unified-memory Mac is an attractive target, but model files must share memory with macOS, applications, attention state, image activations, and temporary buffers. Reducing weight size is therefore necessary for this target, but insufficient evidence of usability.

This project began as a first quantization experiment. It developed into a reproducible systems study with three questions:

1. Can the released checkpoint be repacked into MLX affine weights without losing experts, image support, or the model's loop schedule?
2. Can group size, clipping, and selective higher precision improve the storage–quality trade-off relative to a simple two-bit-expert baseline?
3. Do the resulting models run reliably, with acceptable memory pressure and latency, on the actual 16 GiB target?

The answer is positive for structural conversion and weight reduction. The evidence for reliable deployment remains incomplete.

## 2. Model and background

The archived source card describes Inkling 2 Mini as a 26.4-billion-parameter multimodal model. Its sparse feed-forward layers route tokens to 8 of 128 experts. Twenty-four stored decoder layers produce 48 layer applications because the middle 12 layers execute three times with shared weights. This reduces stored parameters relative to an unshared network of the same execution depth, while each loop visit still needs its own inference state. The released experts use NVFP4 and MXFP8 representations. These properties and the source's recommended hardware are described in the [model card][model].

Quantization assigns a small number of codes to each group of weights. For affine quantization, reconstruction can be written as `w_hat = scale * code + offset`. A two-bit code has four possible values, but each group also stores a scale and offset. Other model components can use different precisions. Consequently, “two-bit” here refers mainly to expert codes, not to every stored parameter or the average checkpoint bit rate.

We use [MLX 0.32.4][mlx] and a pinned Mini-specific [MLX-LM implementation][port]. Our clipping objectives are simple reconstruction proxies. They are neither full Hessian-based GPTQ nor an implementation of AWQ; the study does not compare against those methods.

## 3. Experimental design

### 3.1 Source and invariants

The requested source revision was `1fb4134b9a74f9681054f4ef82e4bf1f458a2337` of `thinkingmachines/Inkling-2-Mini`. The MLX-LM runtime was pinned to `0c4c99192230f1943dd468d297f0b8a64ae7b7c2` in `jundot/mlx-lm`. The original checkpoint was already quantized: reconstructing its represented values does not recover full-precision training weights.

Every conversion retained all 24 stored layers, 48 execution visits, 48 expert projection stacks, and 128 experts per stack. The vision encoder, routing scores, loop controls, norms, and convolutions retained their source tensors. Compatible attention and other dense matrices were requantized according to the recipe. Tokenizer and image-processing assets were retained. The optional speculative drafter was excluded.

Conversions checked protected tensors before and after repacking, saved weights, loaded the result strictly, and compared every serialized tensor's shape, dtype, and SHA-256 signature. These are checks of structure and serialization, not evidence of retained task accuracy.

### 3.2 Series 1: establish a baseline

The first recipe, **R64**, used affine round-to-nearest with group size 64: routed experts at two bits, other eligible linear matrices at four bits, and token embeddings/output head at eight bits. One expert or dense-matrix slice was reconstructed at a time, avoiding a full dequantized model in host memory.

The Python conversion phase took **465.8 seconds**; the complete successful Slurm job, including tests and overhead, took **8 minutes 33 seconds**. Earlier reports sometimes used the full job time as the conversion time. The recorded peak host memory was approximately **15.9 GiB**, based on the historical Slurm-accounting summary rather than a newly repeated measurement.

The baseline's 1,567 tensor signatures matched after reload and after transfer to the Mac. Four simple arithmetic, extraction, code, and image probes were correct. Separate identifier-retrieval probes succeeded at 503, 2,048, and 4,088 prompt tokens. These were smoke tests with known answers, not a retained-accuracy benchmark against the released source.

### 3.3 Series 2: clipping and mixed precision

We defined three expert quantization methods:

| Method | Per-group objective | Calibration required |
|---|---|---|
| R | Existing affine round-to-nearest | No |
| C | Choose clipping to minimize weight reconstruction squared error | No |
| W | Choose clipping to minimize reconstruction error weighted by source input second moments | Yes |

For each group, clipping shrinks the interval around the midpoint of its minimum and maximum. C minimizes `sum_j (w_j - w_hat_j)^2`; W minimizes `sum_j E[x_j^2] * (w_j - w_hat_j)^2`. W uses only diagonal input statistics: it omits correlations and interactions between quantized layers.

Statistics were accumulated for each expert projection across all observed loop visits. Projection/expert pairs with fewer than 32 routed calibration tokens fell back to C. Of 6,144 projection/expert pairs, **5,942 (96.71%)** met the threshold; 202 used the fallback. These counts describe expert matrices, not 6,144 distinct routed expert networks.

The 24 recipes combined groups 32/64/128, R/C/W methods, attention protection, embedding/head compression, and selected four-bit expert layers. S4 protected stored layers 16 and 17, selected by a reconstruction-benefit heuristic with an additional loop-visit multiplier. This is a sensitivity heuristic, not a measured causal ranking of layer importance. E4 means **four-bit embeddings and output head**, not four-bit experts. A8 means eight-bit attention; ordinary eligible attention uses four bits.

The full clipping plan had 11 factors. Two auxiliary GPU search workers did not start within their 30-minute wait allowance and were intentionally cancelled. The exact MLX CPU fallback searched factors **1.00, 0.95, and 0.90**, completing all **6,144 expert matrices** in approximately **7,735 seconds**. Thus 24 recipes were screened numerically, but only six were fully exported and evaluated; this was not 24 complete model benchmarks.

### 3.4 A material correction: reconstructed precision

Early documentation stated that reconstructed experts and all affine metadata were BF16. That description was incorrect. In the pinned implementation, NVFP4 reconstruction with a FP32 global scale promotes the resulting weights to **FP32**, even when the requested dequantization dtype is BF16. The MXFP8 tail reconstructs to BF16. The baseline's serialized tensor signatures confirm FP32 metadata for the former experts and BF16 metadata for the tail.

A preliminary hybrid backend that rounded the promoted source to BF16 failed a numerical check. It was rejected before the main series. The corrected backend preserved the promoted FP32 values, and R/C/W retained the same metadata policy. Some historical JSON description fields still say “through BF16”; the numerical code, tensor dtypes, precision finding, and this correction take precedence over those stale strings.

This helps explain the importance of group size. Two FP32 affine metadata values add **64/group_size bits per weight**: one bit at group 64 and half a bit at group 128, in addition to a two-bit code. BF16 metadata adds half as much. Increasing group size reduces this overhead while making each group harder to approximate.

## 4. Systems and evaluation protocol

### 4.1 Leonardo and hybrid execution

The initial environment job confirmed an A100-SXM GPU with 64 GiB memory, driver 535.274.02, host glibc 2.28, and a working CUDA 12.1/PyTorch calculation. This was an infrastructure preflight, not model inference. A site Slurm plugin printed permission errors alongside successful runs; those messages were recorded without modifying the site installation.

The selected MLX environment needed newer user-space libraries than the host. CPU MLX ran in a Python 3.11 Debian Bookworm Singularity container, following CINECA's supported [container workflow][cineca]. A container does not update the host GPU driver. The chosen MLX CUDA wheel was not used.

A CPU-only source pilot took 287.7 seconds for 49 tokens. The final hybrid setup kept model control flow, attention, vision processing, and statistics in MLX, while an authenticated intra-cluster service executed selected matrix operators using the site's PyTorch CUDA environment. Exact FP32 experts resided in host memory with a bounded 48 GiB A100 cache. The corrected 49-token pilot took 28.44 seconds; this illustrative comparison excludes loading and is not a controlled end-to-end speedup benchmark.

The full source pilot took about 1,441 seconds, including approximately 1,234 seconds of loading/registration, and reached **99.58 GiB** peak host RSS. This source-reconstruction cost is different from the streaming conversion's approximately 15.9 GiB peak.

The main controller completed in **7.99 hours**, within a 12-hour controller deadline, an 18 main GPU-hour request ceiling, and a 60 GiB new-artifact cap. The GPU-hour ceiling was a resource bound, not measured consumption. Supervised pilots were separate; complete billed GPU/node-hour totals were not recovered for this report.

### 4.2 Numerical checks

Four initial converter tests and six series tests covered source preservation, finite constants, expert global scales, text/image forwards, loop statistics, unsorted top-k routing, clipping objectives, answer-boundary masking, mixed precision, and interrupted export recovery. A separate controller check covered shortlist retention and cleanup.

The final packing pilot tested **192 BF16/FP32 fixtures** against MLX's affine quantizer, with exact packed-code matches and zero reconstruction difference. Tiny complete reference/candidate forwards, each with text and image input, matched CPU and hybrid results with maximum absolute error **1.19e-6**. This establishes parity for those fixtures and tiny networks, not exhaustive equivalence of all full-model computations or the Metal backend.

### 4.3 Frozen data and selection

| Split | Text cases | Image cases | Purpose |
|---|---:|---:|---|
| Calibration | 96 | 24 | Collect source activation statistics |
| Development | 48 | 24 | Candidate selection; a fixed subset was used |
| Heldout | 120 | 60 | Native Mac generation after candidate selection |

Calibration accepted all **120 cases**, totaling **11,329 input tokens** under the implementation's accounting. Inputs combined WikiText training passages, authored short-answer tasks, generated image tasks, and TextVQA training images. Development and heldout used disjoint selected ChartQA test images and TextVQA validation image IDs, plus authored and generated tasks. The [data manifest](../data/DATA_MANIFEST.json) records revisions, file hashes, and limitations.

The 372 complete inputs were checked for exact duplication before freezing. This does not imply article-level independence, unseen reasoning templates, or absence of pretraining contamination. Broad authored task operations are shared across splits. Image tasks are small, selected subsets, not official full benchmarks.

The source reference initially scored 45 development questions. Timing then fixed a **30-question subset (20 text, 10 image; 105 answer tokens)** before candidate comparisons. All seven affine models and the source rows reported below use these same 30 questions. Reporting the source's aggregate over all 45 against candidates' 30 would be misleading.

Development scoring measured teacher-forced negative log likelihood (NLL) on gold answer tokens, excluding prompt/image tokens. It is not generated-answer accuracy. Text/image NLL were also averaged with equal modality weight for finalist selection. A rolling shortlist retained the smallest export, the best balanced NLL, and a compromise. The two Mac downloads were the smallest and best-NLL retained candidates.

### 4.4 Native Mac protocol

The target was an Apple M5 with **16 GiB unified memory**, macOS 26.6.2, Python 3.13.13, and MLX 0.32.4. The pinned Mini port and supplied template were used with greedy decoding, reasoning effort `none`, images resized to at most 512 pixels on the longer side, prefill chunks of 64, and **192 output tokens including reasoning**. A final content tag and end-message tag were required; reaching the cap without a completed final answer counted as failure.

Text tasks used exact normalized answers, with JSON parsing for structured output. Chart answers used a 5% numeric tolerance. Scene-text scoring used approximate VQA normalization and consensus scoring. This is explicitly a **custom diagnostic suite**, not official ChartQA/TextVQA accuracy. Code questions in series 2 evaluated numeric Python expressions; they were not broad executable code-generation tests.

A watchdog bounded the Mac stage to four hours and stopped prolonged critical memory pressure. Warning-level pressure was recorded but did not automatically stop a run. The experiment was sequential on a busy Mac, without a controlled clean-memory baseline or randomized run order.

## 5. Results

### 5.1 Storage and development likelihood

GiB means bytes divided by 2^30. Sizes below include the safetensors shard files, not all tokenizer/configuration assets or total inference memory.

| Model | Weight GiB | Text NLL | Image NLL | All-token NLL |
|---|---:|---:|---:|---:|
| Reconstructed released source | — | 0.2788 | 2.9404 | 0.6591 |
| R64, original baseline | 10.2624 | 0.9196 | 2.4972 | 1.1450 |
| R128 | 8.9115 | 1.3123 | 2.3363 | 1.4586 |
| C128 | 8.9115 | 1.0206 | 2.0150 | 1.1626 |
| W128 | 8.9115 | 0.9812 | 1.8382 | 1.1036 |
| W128-A8 | 9.2123 | 0.9743 | 1.8296 | 1.0965 |
| **W128-E4** | **8.4801** | **0.9550** | **2.0837** | **1.1163** |
| **W128-S4** | **9.3861** | **0.5363** | **2.0057** | **0.7462** |

Evidence: [remote results](../results/series2/leonardo/RESULTS.json) and [evaluation records](../results/series2/leonardo/evaluation).

Increasing the group size without clipping hurt text likelihood in this small sample. At group 128, clipping improved it, and activation weighting improved both modality scores relative to C128. Protecting two expert layers produced the strongest text result among exported affine candidates. Eight-bit attention produced a small NLL change at additional storage cost.

The source scored worse on the selected image answers than several requantized models. That is an observed result on **only 15 gold image-answer tokens**, not evidence that quantization generally improves vision. The source remained much better on selected text answers. The very small token denominators and imperfect prompts constrain interpretation.

The smallest finalist's weights total **9,105,445,257 bytes**, versus **11,019,161,398 bytes** for R64: **17.37% smaller**. S4 totals **10,078,212,345 bytes**, **8.54% smaller**. Their complete downloaded directories total 9,135,827,451 and 10,108,594,578 bytes, respectively. File checksums were verified after transfer.

### 5.2 Heldout completion and paired quality

| Model | Completed / intended | Raw score on available cases | Missing final answers | Termination |
|---|---:|---:|---:|---|
| R64 | 90 / 180 | 77.78% | 18 | Metal GPU timeout |
| W128-E4 | 180 / 180 | 75.56% | 40 | Completed |
| W128-S4 | 117 / 180 | 74.36% | 27 | Metal GPU timeout |

The raw percentages have different denominators and must not be read as a fair ranking. One bounded retry of each incomplete model failed during warmup with the same Metal command-buffer timeout; saved cases were retained without adding new ones.

| Candidate vs R64 | Shared cases | Paired score change | Paired bootstrap 95% interval |
|---|---:|---:|---:|
| W128-E4 | 90 | −2.22 percentage points | −5.56 to 0.00 pp |
| W128-S4 | 90 | −1.11 percentage points | −3.33 to 0.00 pp |

These correspond to two and one fewer correct answers on the matched subset. Intervals use 2,000 paired bootstrap resamples with a fixed seed. They do not account for all dataset-selection effects or prove equivalence. The completed prefix is not a random sample of all intended questions, although inputs were deterministically interleaved and shuffled before execution.

Neither candidate passed the frozen engineering filter, which required 180 completed pairs, no more than two percentage points overall loss, and no category worse by more than five points. On these 90 pairs, E4 lost one chart and one scene-text answer; S4 lost one chart answer. Category denominators were small (11 chart and 10 scene-text cases).

### 5.3 A dataset flaw and secondary analysis

The authored extraction category asks for a shipment code but names only an order code and a warehouse code. The answer key nevertheless expects the order code. All three models scored zero on the available extraction items, commonly exhausting the reasoning/output cap. This is an ambiguous task definition, not a clean measure of extraction ability.

We did not edit the frozen questions, answers, or primary scores. A **post-hoc** analysis excluding extraction leaves 78 matched cases: E4 changes by **−2.56 pp**, S4 by **−1.28 pp**. E4's complete non-extraction subset scores 85.0% on 160 cases, but there is no complete matched baseline for those 160. This secondary analysis cannot rescue the primary acceptance claim. Development selection also included this task category and should be treated as affected by the flaw.

### 5.4 Memory and latency

| Model | Peak tracked MLX allocation, completed heldout cases | Median generation rate on available cases | Cases recording warning pressure |
|---|---:|---:|---:|
| R64 | 11.03 GiB | 26.7 tokens/s | 90 / 90 |
| W128-E4 | 9.25 GiB | 22.2 tokens/s | 179 / 180 |
| W128-S4 | 10.15 GiB | 18.6 tokens/s | 116 / 117 |

These are not total system memory requirements or controlled speed comparisons. Case sets, generated lengths, compilation state, run order, and background memory pressure differ. Smaller stored weights did not produce a consistent measured speed advantage.

Before series 1, system swap already contained about 13.55 GiB. It rose to about 19.51 GiB after the first four probes. System-wide swap includes other applications, so this increase cannot be assigned exclusively to model weights. Recovered series-2 context tests also showed roughly 14.4–15.7 GiB of swap used and warning-level pressure. Process RSS alone would understate the Metal/unified-memory footprint.

The planned repeated context/prefill sweep was reduced after timeouts to one recovered sweep each for R64 and E4. Both executed all three requested context lengths; S4 did not receive this recovered sweep.

| Actual prompt tokens | R64 first token | E4 first token | R64 generation | E4 generation |
|---:|---:|---:|---:|---:|
| 503 | 42.7 s | 40.7 s | 26.13 tokens/s | 15.05 tokens/s |
| 2,048 | 28.0 s | 31.0 s | 10.47 tokens/s | 15.04 tokens/s |
| 3,968 | 129.8 s | 66.5 s | 9.15 tokens/s | 11.81 tokens/s |

The context output budget was 128 tokens. Completed execution alone does not guarantee a correct, finished final answer. Series-1 identifier retrieval was checked separately; the recovered series-2 files preserve the raw context outputs for inspection. There are too few repetitions to infer a robust latency advantage. The launcher’s roughly 4,096-token limit is an experimental resource cap, not a claim about the model's architectural context limit.

## 6. Discussion

Three findings are useful beyond this particular checkpoint. First, metadata dtype matters: a nominal two-bit representation can pay substantial per-group FP32 overhead, so group-size changes can reduce storage materially. A converter must inspect actual reconstructed and serialized dtypes rather than assume a requested cast controls subsequent promotion.

Second, a supercomputer and a cheap Mac answer different questions. Leonardo supplied enough host memory and GPU throughput for source reconstruction, calibration, and candidate selection. It could not establish the target Mac's latency, pressure, or reliability. Native deployment tests exposed failures absent from structural verification.

Third, evaluation completion is an outcome. Reporting only file size, successful toy prompts, or the best partial accuracy would hide the main practical constraint. E4's full run is encouraging, but it does not imply repeated-run reliability; the baseline and S4 timeouts remain unexplained. We did not isolate whether pressure, backend compilation, workload, or another Metal/runtime issue caused them.

## 7. Limitations and deviations from the plan

- One source checkpoint and one busy Mac were tested. There is no multi-device replication, clean-memory experiment, power measurement, or controlled performance ranking.
- The reference is the released quantized representation, not training master weights. No native released-source Metal accuracy comparison was performed.
- Calibration and diagnostics are small, partly authored, and not representative of general multilingual, coding, agentic, or visual performance.
- The development set actually compared 30 questions and 105 answer tokens, rather than all 72 available questions. The source's extra 15 scored cases were excluded from the comparison table.
- Only six of 24 screened recipes received full-model development evaluation. Selection is adaptive, with no exhaustive optimality claim.
- The clipping fallback used three factors instead of eleven; auxiliary search GPUs never performed the main search.
- An A100 matrix service and a higher 18 GPU-hour ceiling replaced an earlier six-GPU-hour plan. No host driver was changed.
- The extraction task is ambiguous; its sensitivity exclusion was post-hoc. The TextVQA normalization is approximate.
- Output caps include reasoning and can turn usable reasoning without a finished final answer into a scored failure.
- Two heldout runs remain incomplete despite retries. Context repeats and prefill-256 comparisons were curtailed. Missing tests are not treated as passes.
- Historical prose contains corrected precision statements and machine-specific paths. Published copies redact locations; original and published hashes are tracked separately.

## 8. Conclusion and next steps

We produced a structurally verified multimodal MLX model whose expert codes are predominantly two-bit, then reduced its weight storage from **10.262 to 8.480 GiB** while retaining one complete 180-question local run. This establishes a practical compression result, not a quality-equivalent or comfortably fitting 16 GiB release.

A follow-up should first isolate the Metal timeout under a stable Mac workload, complete identical paired question sets, and repeat latency/memory tests in randomized order. A newly versioned diagnostic set should repair the extraction ambiguity before further selection. Broader benchmark coverage and larger image-answer denominators are needed before recommending a deployment model.

## 9. Reproducibility and artifact availability

The repository contains original numerical Python/batch scripts, dependency pins, machine-redacted evidence, per-case outputs, conversion signatures, source/data manifests, and tools to recompute comparisons without loading weights. [Reproduction instructions](REPRODUCING.md) distinguish reanalysis, local inference, and an expensive new cluster run. [The artifact audit](ARTIFACT_AUDIT.md) records corrections and cleanup decisions.

Model weights, virtual environments, containers, credentials, and third-party dataset payloads are excluded from Git. On the originating machine, the frozen dataset is copied to a Git-ignored local directory for exact reruns. A fresh public clone can reproduce the reported arithmetic from committed evidence, but cannot reproduce inference without separately obtaining weights and the frozen data. The original preparation script queries live dataset metadata and uses an environment-dependent font; rerunning it is a **new data freeze**, not guaranteed exact recreation.

All numerical conclusions in this report come from local archived measurements. External references establish model/runtime/dataset context, not independent validation of our results. Earlier Word/PDF reports are historical snapshots; this report corrects and supersedes their interpretation where stated.

## References

1. Thinking Machines Lab. [Inkling 2 Mini source card at the requested revision][model]. Context checked against the locally archived card; the revision URL was unavailable to the web reader during the final review.
2. MLX-LM contributors and jundot. [Mini-specific implementation proposal][port]; [pinned runtime commit][runtime].
3. Apple MLX contributors. [MLX v0.32.4 affine CPU quantization implementation][mlx].
4. CINECA. [Singularity and Apptainer containers][cineca].
5. Masry et al. [ChartQA project][chartqa]; [dataset mirror/revision][chartmirror]. Only selected disjoint images were used.
6. Singh et al. [TextVQA project][textvqa]; [dataset mirror/revision][textmirror]. Only selected train/validation images were used.
7. Merity et al. [WikiText dataset][wiki]. Frozen training passages were used for calibration.

## Appendix A. Execution history

| Stage | Observation and consequence |
|---|---|
| Access and preflight, 20 Nov | SSH authentication and changed-host-key issues were separate setup problems. The chat confirms subsequent login, not independent fingerprint verification. GPU preflight passed. |
| Initial conversion launch | A host-side check of a container interpreter symlink failed before conversion; the launcher was corrected to execute the interpreter inside the container. |
| Baseline conversion, 20–21 Nov | Corrected job completed; 1,567 saved/downloaded tensor signatures matched. |
| Baseline Mac probes, 21 Nov | Simple text/image and supplied-identifier checks worked; system swap and latency exposed the memory constraint. |
| Series-2 pilots, 21–22 Nov | CPU throughput proved inadequate; early backend/precision pilots were cancelled or rejected, then exact packing and hybrid parity checks passed. |
| Main controller, 22 Nov | Started 00:16 CET, finished about 08:16 CET; 24 recipes screened, six new exports evaluated, three archived. |
| Local transfer | A transport EOF interrupted the second download; the same handoff was restarted, resumed the transfer, and checked the downloaded hashes. |
| Native evaluation/recovery | E4 completed; R64/S4 timed out, including on bounded retry. One recovered context sweep per R64/E4 completed. |
| Completion | The bounded local run ended at 10:22 CET on 22 Nov, explicitly marked complete with incomplete Mac tests. The monitor was stopped. |

[model]: https://huggingface.co/thinkingmachines/Inkling-2-Mini/blob/1fb4134b9a74f9681054f4ef82e4bf1f458a2337/README.md
[port]: https://github.com/ml-explore/mlx-lm/pull/2086
[runtime]: https://github.com/jundot/mlx-lm/commit/0c4c99192230f1943dd468d297f0b8a64ae7b7c2
[mlx]: https://github.com/ml-explore/mlx/blob/v0.32.4/mlx/backend/cpu/quantized.cpp
[cineca]: https://docs.hpc.cineca.it/services/singularity.html
[chartqa]: https://github.com/vis-nlp/ChartQA
[chartmirror]: https://huggingface.co/datasets/lmms-lab/ChartQA/tree/9e63b7df1592a1c2158e735cc1725454aef0d6d9
[textvqa]: https://textvqa.org/
[textmirror]: https://huggingface.co/datasets/lmms-lab/textvqa/tree/9c0699cd19768ac5ab97568f6b3cbac4c0062884
[wiki]: https://huggingface.co/datasets/Salesforce/wikitext

# Understanding quantization across models and repeated computation

Prepared 23 November 2026. This replaces deployment on a 16 GiB Mac as the primary objective of the proposed next research round. The earlier plan's retrospective findings remain useful. Heavy computation stays on Leonardo; the Mac receives at most one cumulative intensive hour, and native Mac validation is optional.

## Research objective

Identify mechanisms that explain when compression preserves useful behavior and when it changes computation. A successful round produces controlled comparisons, numerical traces and a defensible explanation, even if it produces no better downloadable model.

Our existing result suggests a concrete starting hypothesis: quantization may alter reasoning completion before it causes an obvious increase in wrong completed answers. On the 90 shared questions, E4's two additional failures were unfinished outputs at the token cap. That motivates a test; it does not prove a general effect or show that more tokens would repair those answers.

The work so far has been conversion, post-training quantization, calibration and evaluation. It has not trained a new language model. Actual quantization-aware training would be a separate intervention, with its own training data and budget.

## Distinguish three different questions

| Question | Meaning | Correct comparison |
|---|---|---|
| Does a model convert correctly to MLX | The architecture, represented weights, tokenizer, images and caches are implemented correctly | Same checkpoint before additional quantization, with backend precision differences measured |
| Does a model tolerate quantization | Lower precision changes weights without excessive behavioral loss | Each quantized variant against its own source under matched settings |
| Does the compressed representation run efficiently | The runtime has suitable packed kernels and metadata/activation dtypes | Same quantized values under packed versus reconstructed execution, with hardware measured separately |

A model can convert easily but quantize poorly. A numerically strong quantizer can also use a custom format without an efficient stock MLX kernel. Model rankings on these axes need not agree.

## What is known and what could be learned

Activation importance and weight geometry already have substantial literature. [AWQ](https://arxiv.org/abs/2306.00978) protects important channels using activation information; [GPTQ](https://arxiv.org/abs/2210.17323) compensates rounding errors using approximate second-order information; [QuIP](https://arxiv.org/abs/2307.13304) changes the representation using incoherence processing. Simply comparing these ideas is a replication study, not an invention.

The reasoning-length evidence is conditional. [Quantization Hurts Reasoning](https://arxiv.org/abs/2504.04823) reports no increased output length in its studied setting. [Quantization Inflates Reasoning](https://arxiv.org/abs/2606.25519) reports longer reasoning and repetition in other settings. Different models, tasks and methods can produce different effects. We should ask when each pattern occurs, rather than assume all quantized models overthink.

Looped quantization is also an existing topic. [Feedback Exposure and Calibration Blindness](https://arxiv.org/abs/2609.30820) distinguishes sensitive feedback entry points from calibration that misses later recurrent states. [LoopQuant](https://arxiv.org/abs/2610.16368) studies loop-aware calibration and precision. A plausible contribution here is a carefully controlled test on a newly released native-low-precision looped MoE, connecting internal error and routing to reasoning termination, and reproducing the key comparisons with an MLX-compatible representation. Novelty remains a hypothesis until the full related work and our results are assessed.

## Questions and experiments

### Does compression change when reasoning ends

Compare each source model and its quantized variants on identical prompts, recording complete emitted token traces. Measure correct final answers, probability and timing of the transition out of reasoning, final-answer completion, repetitions and runtime failures. A model that answers correctly after 900 tokens is different from one that never produces the correct answer.

Compute accuracy and completion curves at fixed output budgets, and separately compare under a generous common cap. Preserve censored cases that never finish; do not compare reasoning length only among successful responses. Conditional summaries among cases both variants solve may be shown as secondary, with their selection bias stated.

Probe transition-token probabilities on the same source histories, then compare free-running generation. This separates a local change in stopping preference from divergence caused by the model reading its own increasingly different history. A late control token can have a larger effect on final-answer usability than a larger perturbation of an ordinary word.

Use one fixed greedy condition and a small, separately reported sampled condition with paired seeds. Token-budget changes must not silently change the prompt, template or sampler. Assess censored lengths with completion-by-budget and restricted mean emitted-token summaries; do not pretend the cap is an observed natural completion length.

### Do later loop states need different calibration

Use [TML-d24-varloop-research](https://huggingface.co/thinkingmachines/TML-d24-varloop-research), which was explicitly trained with one to five loop passes. Change its loop count while holding its weights fixed. For each count, compare the quantized version with that same checkpoint's native result at that count. Ordinary Inkling Mini stays at its trained fixed three passes.

The first mechanistic panel is:

| Factor | Core settings |
|---|---|
| Loop passes | 1 and 3; add 5 if throughput permits |
| Representation | Released native precision, affine 4-bit FFN, affine 2-bit FFN |
| Calibration | Round-to-nearest control; diagonal weighting from first-pass versus all-pass inputs on selected two-bit cells |
| Held fixed | Weights before intervention, group size 128, metadata policy, unquantized controls, tokenizer, prompts and sampler |

The core loop/precision grid has six cells; the five-pass extension adds three. Do not take the Cartesian product with every quantizer, model, group size and dtype. Stage the calibration ablations after the core grid establishes an interpretable signal.

For the first-pass/all-pass calibration contrast, use the same prompts and an equal number of sampled token-visits. For example, choose one deterministic random loop visit per eligible input token in the mixed-pass condition, rather than giving it three times as many observations. Record expert coverage and rare-expert fallbacks. A separate full pooled calibration condition can show the benefit of additional observations, but it must not be conflated with the distribution contrast.

Record source and candidate hidden-state error, activation distributions, top-8 routing overlap, routing margins and local layer-output error by loop visit. Shared weights remain shared; creating separate weights per pass would change the model. The existing statistics retain pooled squared inputs and per-visit counts, not the per-visit activations needed for this experiment, so new measurement is necessary.

For a task metric S, calculate the quantization penalty at depth L as `S_quant(L) - S_native(L)`. Compare the penalty at three passes with the penalty at one. This avoids confusing a lower-quality one-pass native model with quantization robustness. Training favored three passes and post-training used three, so results at other depths still reflect that model's training history.

This research checkpoint has different later training from production Mini. Its outcomes cannot be relabeled as production Mini outcomes. Port correctness and explicit variable-loop support must pass before this panel begins.

### Which measurable properties predict sensitivity

Use a small cross-model replication panel: [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) as a dense control, production Inkling Mini as the existing looped MoE case, and optionally [OLMoE-1B-7B-0924-Instruct](https://huggingface.co/allenai/OLMoE-1B-7B-0924-Instruct) as a non-looped sparse control.

These models differ in size, training, tokenizers, reasoning policy and source precision. A three-model comparison can reveal associations; it cannot establish that architecture alone causes sensitivity. Mechanistic claims come from within-model interventions and matched native baselines. The first round prioritizes the variable-loop experiment and one dense control; OLMoE is an extension if time/storage permit, not an additional guaranteed full sweep.

Measure layer/expert weight ranges, outlier concentration, source activation second moments, local reconstruction error and activation-weighted output error. Ask whether those measures predict behavioral loss on unseen cases. Do not choose predictors on heldout results or count every token/expert as an independent model.

Use text tasks common to all models for the cross-model panel. Vision is a separate Inkling-only extension, since the dense text control cannot answer image questions. Primary controlled quantization affects FFN/expert matrices only; attention, routing, norms, heads, vision and caches retain their declared source policy. Report the actual fraction of weights affected and stored bytes. The same nominal bits can represent very different whole-model compression ratios.

The source Mini is already NVFP4/MXFP8, while the dense control has a different starting representation. Normalize each result to its own released source; do not claim a matched BF16-training-master experiment.

### Does the method matter more than the model

Begin with the implemented round-to-nearest and activation-weighted clipping methods at fixed group size and precision. Our W method is not AWQ. A real GPTQ/AWQ implementation is admitted only if it supports the architecture correctly and can be validated without consuming the run developing a new port. A small module-level error-compensation comparison is an acceptable clearly labeled extension.

Use identical calibration inputs for method comparisons. Test calibration from prose versus reasoning/code at an equal token budget, evaluating on distinct domains. This asks whether apparent robustness is actually calibration coverage. A small orthogonal-rotation probe on isolated linear operations can illustrate geometry sensitivity, as motivated by QuIP; it is a numerical diagnostic unless the corresponding transforms and kernels are implemented throughout the model.

Test FP32 versus BF16 affine expert metadata on one existing Mini two-bit profile as a separate ablation with identical integer codes. Measure the numerical changes rather than interpreting smaller files as a quality result. This representation-level experiment connects the previous size calculation to actual behavior and possible kernel effects without making Mac fit the primary objective.

## Practical execution

### Phase 1 Reference and throughput gates

Use Leonardo for downloads, conversion, activation collection and model execution. Validate the variable-loop source against its intended implementation at both loop counts, including full-sequence versus cached decoding. Validate the dense control independently. No model family is accepted solely because its configuration loads.

Preserve the site's host driver and isolated environments. The old bridge's approximately 1.1 generated tokens/second is a real constraint. Measure runtime before freezing the executable scope; choose a fixed balanced prompt set and token quota that all core cells can finish. If the variable-loop port does not validate promptly, execute the stopping-behavior experiment on production Mini and the dense control, and explicitly defer the loop-depth causal test.

### Phase 2 Measurement before large exports

Use small fixed calibration and heldout sets, with new hashes and corrected scoring. Run six loop/precision cells first, with a declared reduced set if the throughput gate requires it. Analyze layer-level counterfactuals without serializing dozens of full checkpoints. Retain aggregate statistics and a bounded diagnostic reservoir, capped around 2 GiB, rather than all activations or every output logit.

Compare global behavior with local weight/output error and transition-token changes. If no meaningful loop interaction appears, report that result rather than expanding the grid until one does. Use paired resampling at the prompt level for completion/accuracy and report the full cell matrix. Group repeated seeds under their originating prompt, and treat short teacher-forced scores as diagnostics rather than final reasoning quality.

### Phase 3 Replication and optional MLX validation

Replicate the primary stopping/precision result on Qwen3-8B at native/four/two-bit FFN settings. Add a non-looped MoE only if the frozen core comparisons are complete and the remaining budget permits. Hold out a separate confirmation set before inspecting exploratory rankings.

After a validated Linux comparison, optionally export one or two standard MLX-compatible profiles and run a short native check during a user-chosen Mac window. This checks portability and packed-kernel behavior; failure to fit on the Mac does not invalidate the scientific numerical comparison. Custom codebooks or rotations require their own runtime support and are not automatically ordinary MLX checkpoints.

## Resource and integrity limits

- Preserve the prior 12-hour controller and 18 main GPU-hour ceilings for the first executable round; split the broader agenda into later rounds rather than silently exceeding them.
- Keep new artifacts within 60 GiB and maintain the 60 GiB work-space reserve. Download only the research checkpoint plus one additional control initially; inspect actual source sizes first. Additional models replace an extension rather than automatically accumulating on disk.
- Keep original sources, completed exports and historical scores immutable. Freeze algorithm/code/data/source/dtype/caches separately for every changed intervention.
- Mac intensive work is optional and remains capped at one cumulative hour including loading, compilation, transfer checks and cleanup. Afterward, only short SSH/status and small JSON/report jobs run locally.
- SSH renewal and a fresh allocation check are needed before launch. The recorded account ends 24 November; the executable deadline must precede the verified expiry. No new training or quantization job has been launched by this plan.

## What would count as a useful result

Examples of defensible findings include: quantization leaves final-answer accuracy similar under a generous cap but reduces the probability of finishing within a short cap; later-loop calibration reduces compression loss at matched observation count; layer-local reconstruction error predicts some failure categories but not stopping behavior; or a pattern appears in Mini but not in the dense control.

None of these outcomes is guaranteed. A clean null result, an invalidated metric, or a documented failure of a calibration method is also useful. The deliverable is a research report and reusable measurement dataset explaining the conditions of a result. It does not need to be the smallest or fastest Hugging Face model.

The initial experiments should be described as a pilot and replication with a targeted extension. A claim of new science requires a clear distinction from the existing reasoning-inflation and loop-calibration literature, adequate controls and independent confirmation.

## Execution attempt on 23 November

SSH renewal succeeded. Reference gate job 65188325 was submitted, remained pending because all Booster nodes were reserved for maintenance until 25 November at 08:00 CET, and was cancelled with elapsed time 00:00:00. The recorded allocation ends 24 November. No model inference or quantization comparison ran, no new full checkpoint was downloaded, and zero GPU time was consumed. Eight small, model-free protocol/scoring tests pass; independent model parity remains untested. A valid allocation after maintenance is required to continue. See `experiments/series3/README.md` for implemented preparation and remaining work.

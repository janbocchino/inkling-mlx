# Attribution and scope of the license

The original experiment toolkit carries the MIT notice in `LICENSE`. Newly organized project code and documentation use that same project license. The affine CUDA search formulation follows Apple's MLX v0.32.4 CPU quantizer; retain `licenses/MLX-MIT.txt` and its copyright notice.

The pinned MLX-LM runtime is an external dependency at `jundot/mlx-lm@0c4c99192230f1943dd468d297f0b8a64ae7b7c2`, associated with [Mini support PR 2086](https://github.com/ml-explore/mlx-lm/pull/2086). It is installed separately rather than vendored here.

The archived model card labels Inkling 2 Mini Apache 2.0 and links separate model policies. Weights and the full card are not redistributed in this Git repository. Consult the [upstream model repository](https://huggingface.co/thinkingmachines/Inkling-2-Mini) for its terms and notices before distributing converted weights.

Calibration/evaluation used selected [WikiText](https://huggingface.co/datasets/Salesforce/wikitext), [ChartQA](https://github.com/vis-nlp/ChartQA), and [TextVQA](https://textvqa.org/) content through revision-recorded mirrors. Dataset rights are separate from this code license. Raw dataset payloads and source images are excluded from Git. Saved model responses and expected short answers are retained as experimental evidence, with provenance; no claim of MIT ownership over third-party content is made.

The work was performed by the project owner with substantial AI assistance for code, operation, and reporting. External model/runtime/dataset authors retain their contributions; this project does not claim affiliation with them.

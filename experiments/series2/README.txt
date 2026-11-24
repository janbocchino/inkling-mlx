Inkling 2 Mini overnight experiment series 2

This kit searches 24 precision recipes for text and image inference on 16 GiB
Apple silicon. The original source and the validated R64 baseline are read only.
See PLAN.json for all recipes, limits, source/runtime revisions and deviations.

Execution

1. prepare_data.py freezes custom calibration, development and heldout subsets.
2. test_converter.py and test_series.py check reconstruction, images, loop
   coverage, clipping, mixed precision, source preservation and interrupted resume.
3. GPU packing parity uses 192 fixtures, covering BF16 and FP32 scale metadata,
   including zero ranges and constants.
4. matrix_service.sbatch hosts numeric matrix operations on an A100. Model
   control flow, attention, image processing and statistics remain in pinned MLX.
   It accepts authenticated numeric arrays on the internal cluster network.
5. full_pilot.sbatch compares tiny CPU and hybrid CUDA text/image forwards, then
   checks the reconstructed full source and answer likelihoods.
6. overnight.sbatch starts controller.py. It collects source activations, starts
   at most two three-hour A100 clipping workers, exports up to six candidates
   sequentially, scores a fixed subset, and retains up to three verified finalists.
7. watch_overnight.py waits locally, downloads at most two finalists with full
   file checksum verification, and runs native Metal heldout and context checks.

Limits

Main controller: 12 hours, 16 CPU cores, 192 GiB host RAM.
Matrix service: one A100, 16 CPU cores, 192 GiB host RAM, at most 12 hours.
The exact FP32 experts stay in host RAM with a bounded 48 GiB A100 cache.
Search: at most two A100 workers, each eight CPU cores and 32 GiB RAM, three hours.
Main GPU request ceiling: 18 GPU-hours, plus the small supervised pilots.
New experiment artifacts: 60 GiB across work and scratch; 60 GiB work reserve.
Local Mac inference stage: four hours maximum, with a critical-memory watchdog.

CPU-only source inference took 287.7 seconds for a 49-token pilot. Therefore this
implementation uses an A100 for matrix operators rather than relying on CPU
throughput. This is an implementation change from the original six GPU-hour plan.
The installed MLX CUDA wheel is not used; the host driver remains untouched.
MLX promotes the reconstructed NVFP4 source to FP32 when it applies global
scaling. This precision is preserved. R, C and W use the same scale precision:
FP32 for these experts and BF16 for the MXFP8 tail. Clipping is compared without
quietly changing the metadata precision.

Methods

R is the original affine round-to-nearest recipe.
C chooses an expert clipping range by reconstruction mean squared error.
W uses diagonal source input-activation statistics to weight that error. Experts
with fewer than 32 routed calibration tokens fall back to the C objective.
This diagonal method is neither full GPTQ nor AWQ. Shared layers aggregate
statistics across loop visits. Routers, vision, convolutions and loop controls
retain their source tensors in every exported checkpoint.

Quality claims

CPU/hybrid likelihoods use FP32 arithmetic. They are screening measurements;
native Metal generation is checked separately. The 180 heldout questions form
a custom diagnostic subset, not an official full ChartQA or TextVQA benchmark.
TextVQA normalization in the Mac scorer is approximate. Report the subset IDs,
case counts and uncertainty rather than claiming official benchmark accuracy.
Mac tests include reasoning in the 192-token output cap; missing final answers
count as failures. The original 2-bit baseline is the paired control.

Recovery

Logs, atomic JSON records and per-module export checkpoints persist in the run
directory. Rerunning a failed conversion with the same recipe/context reuses
hash-verified module files. The controller retries an export once and otherwise
continues with valid recipes. Candidate files outside the rolling shortlist are
expendable; their recipes, hashes and scores remain. Only this experiment's files
are eligible for its cleanup. The source and other account jobs are untouched.

Important files

RUN_MANIFEST.json freezes code hashes, data hashes, source hashes and limits.
PILOT.json and HYBRID_PARITY.json record the full and tiny reference checks.
SCREENED_RECIPES.json and PROMOTED_RECIPES.json record the search and promotions.
FINALISTS.json lists final folders and file checksums.
RESULTS.json, results.csv and MORNING_SUMMARY.txt describe Leonardo results.
LOCAL_STATUS.json, MAC_COMPARISON.json and EXPERIMENT_SUMMARY.txt describe the
local handoff and real Mac validation.

References and licenses

The numerical affine formulation follows Apple's MLX v0.32.4 CPU implementation:
https://github.com/ml-explore/mlx/blob/v0.32.4/mlx/backend/cpu/quantized.cpp
MLX is Copyright Apple Inc., licensed under MIT; see MLX_LICENSE.txt.
This toolkit's code is MIT, see LICENSE.txt. Dataset/image rights are separate.
Model source: https://huggingface.co/thinkingmachines/Inkling-2-Mini
Dataset and source revisions plus source links are recorded in DATA_MANIFEST.json.

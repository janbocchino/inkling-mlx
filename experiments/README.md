# Original execution snapshots

These scripts document what actually ran. Numerical Python and batch files are byte-identical to the experiment artifacts; `../manifests/frozen-code.json` checks the published snapshots, and `../results/PROVENANCE.json` links them to original hashes.

`series1/` contains the initial converter and Mac probes. `series2/` contains the clipping/calibration/controller/CUDA implementation and original tests. `series2/recovery/` contains the actual one-off Mac recovery and controller check; they assume historical paths and are not generic entry points.

Historical README/PLAN descriptions have path redactions and known stale BF16 statements; consult `../docs/REPORT.md` and `../docs/ARTIFACT_AUDIT.md` for corrections. Do not launch historical watchers or batch scripts blindly. The portable launcher and no-inference analysis are under `../scripts/`.

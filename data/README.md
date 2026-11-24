# Data provenance

`DATA_MANIFEST.json` is the original frozen manifest; its content hash is `32b35424191b266bbd309818ec745d3e93d8a5c1aa339c85a837e2e11bbfb613`. It records all original JSON/image-array hashes and source revisions. `case-index.json` publishes split IDs and origin metadata without the third-party payloads.

The original machine has an exact, Git-ignored dataset copy in `experiments/series2/data/`. A public clone does not contain that directory. See [reproduction boundaries](../docs/REPRODUCING.md) before regenerating data. Frozen extraction questions are known to be ambiguous and must not be edited while reusing old results.

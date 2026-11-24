import hashlib
import json
from pathlib import Path
import struct
import time

root = Path(__file__).resolve().parents[1] / "outputs"
model = root / "Inkling-2-Mini-expert-2bit-g64"
report = json.loads((model / "CONVERSION_REPORT.json").read_text())
expected = report["tensor_signatures"]
dtype_names = {"BF16": "mlx.core.bfloat16", "F32": "mlx.core.float32",
               "F16": "mlx.core.float16", "U32": "mlx.core.uint32",
               "U8": "mlx.core.uint8", "I32": "mlx.core.int32"}
seen = set()
started = time.monotonic()
files = []
for path in sorted(model.glob("model*.safetensors")):
    with path.open("rb") as file:
        header_len = struct.unpack("<Q", file.read(8))[0]
        header = json.loads(file.read(header_len))
        base = 8 + header_len
        for key, metadata in header.items():
            if key == "__metadata__":
                continue
            assert key not in seen, f"Duplicate tensor: {key}"
            assert key in expected, f"Unexpected tensor: {key}"
            want = expected[key]
            assert metadata["shape"] == want["shape"], key
            assert dtype_names[metadata["dtype"]] == want["dtype"], key
            begin, end = metadata["data_offsets"]
            file.seek(base + begin)
            remaining = end - begin
            digest = hashlib.sha256()
            while remaining:
                data = file.read(min(8 * 1024**2, remaining))
                assert data, f"Truncated tensor: {key}"
                digest.update(data)
                remaining -= len(data)
            assert digest.hexdigest() == want["sha256"], key
            seen.add(key)
    files.append({"file": path.name, "bytes": path.stat().st_size})
assert seen == set(expected), "Some tensors are missing"
result = {"status": "passed", "verified_tensor_count": len(seen),
          "files": files, "weights_bytes": sum(x["bytes"] for x in files),
          "elapsed_seconds": time.monotonic() - started,
          "method": "Streamed SHA256 of each local safetensors tensor compared with the post-conversion tensor signatures on Leonardo"}
(root / "DOWNLOAD_VERIFICATION.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))

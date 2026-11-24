"""Stream safetensors signatures and verify frozen source files without MLX."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct

DTYPES = {"BF16": ("mlx.core.bfloat16", 2), "F32": ("mlx.core.float32", 4),
          "F16": ("mlx.core.float16", 2), "U32": ("mlx.core.uint32", 4),
          "U8": ("mlx.core.uint8", 1), "I32": ("mlx.core.int32", 4),
          "I64": ("mlx.core.int64", 8), "BOOL": ("mlx.core.bool_", 1)}


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_frozen(root: Path) -> dict:
    expected = json.loads((root / "manifests/frozen-code.json").read_text())
    for name, digest in expected.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid manifest path")
        if sha_file(root / relative) != digest:
            raise ValueError(f"Frozen file changed: {name}")
    return {"status": "passed", "verified_files": len(expected)}


def verify_model(model: Path) -> dict:
    """Check all stored tensor shapes, dtypes and bytes against its conversion report."""
    expected = json.loads((model / "CONVERSION_REPORT.json").read_text())["tensor_signatures"]
    seen = set()
    files = sorted(model.glob("model*.safetensors"))
    if not files:
        raise ValueError("No model safetensors shards found")
    for path in files:
        size = path.stat().st_size
        with path.open("rb") as stream:
            raw = stream.read(8)
            if len(raw) != 8:
                raise ValueError(f"Truncated header: {path.name}")
            length = struct.unpack("<Q", raw)[0]
            if length > min(size - 8, 100 * 1024**2):
                raise ValueError(f"Invalid header length: {path.name}")
            header = json.loads(stream.read(length))
            start = 8 + length
            for name, entry in header.items():
                if name == "__metadata__":
                    continue
                if name in seen or name not in expected:
                    raise ValueError(f"Duplicate or unexpected tensor: {name}")
                want = expected[name]
                dtype, width = DTYPES[entry["dtype"]]
                if entry["shape"] != want["shape"] or dtype != want["dtype"]:
                    raise ValueError(f"Shape/dtype mismatch: {name}")
                elements = 1
                for dimension in entry["shape"]:
                    if not isinstance(dimension, int) or dimension < 0:
                        raise ValueError(f"Invalid shape: {name}")
                    elements *= dimension
                begin, end = entry["data_offsets"]
                if not 0 <= begin <= end <= size - start or end - begin != elements * width:
                    raise ValueError(f"Invalid tensor range: {name}")
                stream.seek(start + begin)
                remaining = end - begin
                digest = hashlib.sha256()
                while remaining:
                    block = stream.read(min(remaining, 8 * 1024**2))
                    if not block:
                        raise ValueError(f"Truncated tensor: {name}")
                    digest.update(block)
                    remaining -= len(block)
                if digest.hexdigest() != want["sha256"]:
                    raise ValueError(f"Tensor hash mismatch: {name}")
                seen.add(name)
    if seen != set(expected):
        raise ValueError(f"Missing {len(set(expected) - seen)} tensors")
    return {"status": "passed", "verified_tensors": len(seen),
            "weight_bytes": sum(path.stat().st_size for path in files)}

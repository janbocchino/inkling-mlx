"""Freeze balanced authored tasks and immutable source revisions; no model loading."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import urllib.request
from common import atomic_json, digest, sha_file

SEED = 20261123
CATEGORIES = ["arithmetic", "lookup", "code", "logic", "structured", "instruction"]
SOURCES = ["thinkingmachines/TML-d24-varloop-research", "Qwen/Qwen3-8B"]
TRANSFORMERS_COMMIT = "430050f6f71195b76004b5d61bbb446ec0508fb3"
MLX_LM_COMMIT = "0c4c99192230f1943dd468d297f0b8a64ae7b7c2"


def task(split, category, index):
    rng = random.Random(f"{SEED}/{split}/{category}/{index}")
    a, b, c = [rng.randrange(7, 80) for _ in range(3)]
    identifier = f"{split}_{category}_{index:02d}"
    if category == "arithmetic":
        question = f"Compute {a} times {b}, then add {c}. Reply with only the integer."
        answer = str(a * b + c)
    elif category == "lookup":
        names = [f"node_{split}_{index}_{i}" for i in range(5)]
        rng.shuffle(names)
        pairs = list(zip(names, names[1:] + ["END"]))
        rng.shuffle(pairs)
        question = ("Links: " + "; ".join(f"{x} -> {y}" for x, y in pairs)
                    + f". Starting at {names[0]}, follow exactly three links. Return only the resulting node name.")
        answer = names[3]
    elif category == "code":
        question = (f"In Python, what is sum(x * x for x in [{a}, {b}, {c}] if x % 2 == 0)? "
                    "Reply with only the integer.")
        answer = str(sum(x * x for x in [a, b, c] if x % 2 == 0))
    elif category == "logic":
        question = (f"A store starts with {a+b} boxes, receives {c}, and sells {b}. "
                    "How many boxes remain? Reply with only the integer.")
        answer = str(a + c)
    elif category == "structured":
        question = (f'Return JSON only, with exactly these fields: "code" is "{identifier}" '
                    f'and "count" is the integer {a}.')
        answer = json.dumps({"code": identifier, "count": a}, separators=(",", ":"))
    else:
        question = f"Return exactly this identifier in uppercase, without explanation: {identifier}"
        answer = identifier.upper()
    return {"id": identifier, "category": category, "modality": "text",
            "question": question, "answer": answer, "origin": "authored",
            "difficulty_scope": "small mechanistic diagnostic, not a general intelligence benchmark"}


def make_splits():
    result = {}
    # Prefixes remain category-balanced if throughput requires a reduced scope.
    for split, repeats in [("calibration", 4), ("development", 2), ("evaluation", 4), ("confirmation", 2)]:
        result[split] = [task(split, category, index)
                         for index in range(repeats) for category in CATEGORIES]
    identities = [(row["question"], row["answer"]) for rows in result.values() for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("Duplicate complete input across splits")
    return result


def pin_source(name):
    with urllib.request.urlopen(f"https://huggingface.co/api/models/{name}?blobs=true", timeout=30) as stream:
        metadata = json.load(stream)
    if metadata.get("gated"):
        raise ValueError("Gated source requires explicit access setup")
    weights = [{"file": item["rfilename"], "bytes": item.get("size", item.get("lfs", {}).get("size")),
                "sha256": item.get("lfs", {}).get("sha256")}
               for item in metadata["siblings"] if item["rfilename"].endswith(".safetensors")
               and "mtp" not in item["rfilename"].lower()]
    if not weights or any(not item["bytes"] or not item["sha256"] for item in weights):
        raise ValueError("Missing full weight size/hash metadata")
    return {"repository": name, "revision": metadata["sha"], "weights": weights,
            "weight_bytes": sum(item["bytes"] for item in weights)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pin-sources", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new protocol directory; never overwrite a freeze")
    splits = make_splits()
    args.output.mkdir(parents=True)
    for name, rows in splits.items():
        atomic_json(args.output / f"{name}.json", rows)
    manifest = {"seed": SEED, "dataset_version": "authored-diagnostics-v1",
                "counts": {name: dict(Counter(row["category"] for row in rows)) for name, rows in splits.items()},
                "files": {f"{name}.json": sha_file(args.output / f"{name}.json") for name in splits},
                "broad_task_templates_overlap": True,
                "scope": "Controlled small diagnostic; not an official benchmark or proof of generalization",
                "source_revisions": [pin_source(name) for name in SOURCES] if args.pin_sources else [],
                "reference_transformers_commit": TRANSFORMERS_COMMIT,
                "mlx_lm_commit": MLX_LM_COMMIT,
                "original_data_and_scores_unchanged": True}
    manifest["hash"] = digest(manifest)
    atomic_json(args.output / "PROTOCOL.json", manifest)
    print(json.dumps({"protocol_hash": manifest["hash"], "counts": manifest["counts"]}))


if __name__ == "__main__":
    main()


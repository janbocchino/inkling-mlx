#!/usr/bin/env python3
"""Repack Inkling 2 Mini's trained NVFP4/MXFP8 experts into affine MLX.

This is an experimental round-to-nearest baseline, not calibrated GPTQ/AWQ.
No model inference, network access, upload, or source-file mutation is performed.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import time

os.environ.setdefault("MLX_ENABLE_TF32", "0")

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm.layers.moe import QuantizedSwitchLinear, SwitchLinear
from mlx_lm.utils import load_model, save_config, save_model

RUNTIME_COMMIT = "0c4c99192230f1943dd468d297f0b8a64ae7b7c2"
SOURCE_REVISION = "1fb4134b9a74f9681054f4ef82e4bf1f458a2337"
MATRIX_CLASSES = (nn.Linear, nn.Embedding, SwitchLinear, QuantizedSwitchLinear)
ASSETS = (
    "generation_config.json", "tokenizer.json", "tokenizer_config.json",
    "special_tokens_map.json", "processor_config.json", "chat_template.jinja",
)


def log(event, **fields):
    print(json.dumps({"event": event, **fields}), flush=True)


def check_runtime():
    if importlib.metadata.version("mlx") != "0.32.4":
        raise RuntimeError("This converter was tested with mlx==0.32.4.")
    distribution = importlib.metadata.distribution("mlx-lm")
    direct = json.loads(distribution.read_text("direct_url.json") or "{}")
    commit = direct.get("vcs_info", {}).get("commit_id")
    if commit != RUNTIME_COMMIT:
        raise RuntimeError(f"Install the pinned MLX-LM commit {RUNTIME_COMMIT} first.")


def choose_bits(path, module):
    # Router scores, loop controllers, norms and convolutions retain their dtype.
    if path.startswith("model.vision_tower.") or "_message_gate." in path:
        return None
    if path.endswith(".mlp.gate"):
        return None
    if ".mlp.experts." in path:
        return 2
    if path == "lm_head" or path.endswith(".embed_tokens"):
        return 8
    return 4


def matrix_dims(module):
    if isinstance(module, QuantizedSwitchLinear):
        return module.num_experts, module.output_dims, module.input_dims
    return tuple(module.weight.shape)


def array_signature(array):
    # Byte views preserve BF16 and integer bit patterns; no float32 conversion.
    mx.eval(array)
    raw = np.asarray(array.reshape(-1).view(mx.uint8))
    return {
        "shape": list(array.shape), "dtype": str(array.dtype),
        "sha256": hashlib.sha256(raw.tobytes()).hexdigest(),
    }


def fingerprints(model, exclude=()):
    return {
        key: array_signature(value)
        for key, value in tree_flatten(model.parameters())
        if not any(key.startswith(path + ".") for path in exclude)
    }


def reconstruct_expert(module, index):
    if isinstance(module, QuantizedSwitchLinear):
        scale = module.get("global_scale")
        kwargs = {} if scale is None else {"global_scale": scale[index]}
        return mx.dequantize(
            module.weight[index], module.scales[index], module.get("biases")[index]
            if module.get("biases") is not None else None,
            group_size=module.group_size, bits=module.bits, mode=module.mode,
            dtype=mx.bfloat16, **kwargs,
        )
    return module.weight[index].astype(mx.bfloat16)


def repack(module, bits, group_size, tile_rows=1024):
    """Reconstruct at most one expert or a slice of a dense matrix at a time."""
    expert = isinstance(module, (SwitchLinear, QuantizedSwitchLinear))
    shape = matrix_dims(module)
    parts, scales, biases = [], [], []
    sample_error = sample_signal = 0.0

    if expert:
        chunks = ((i, reconstruct_expert(module, i)) for i in range(shape[0]))
    else:
        chunks = (
            (i, module.weight[i:i + tile_rows].astype(mx.bfloat16))
            for i in range(0, shape[0], tile_rows)
        )

    for index, weights in chunks:
        packed, scale, bias = mx.quantize(weights, group_size=group_size, bits=bits)
        mx.eval(packed, scale, bias)
        if not mx.all(mx.isfinite(scale) & mx.isfinite(bias)).item():
            raise ValueError("Non-finite quantization metadata")
        # A small reconstruction diagnostic, not a model quality measurement.
        if index == 0:
            original = weights[:min(16, weights.shape[0])].astype(mx.float32)
            reconstructed = mx.dequantize(
                packed[:original.shape[0]], scale[:original.shape[0]],
                bias[:original.shape[0]], group_size=group_size, bits=bits,
            ).astype(mx.float32)
            sample_error = mx.mean(mx.square(original - reconstructed)).item()
            sample_signal = mx.mean(mx.square(original)).item()
        parts.append(packed[None] if expert else packed)
        scales.append(scale[None] if expert else scale)
        biases.append(bias[None] if expert else bias)

    packed = mx.concatenate(parts, axis=0)
    scale = mx.concatenate(scales, axis=0)
    bias = mx.concatenate(biases, axis=0)
    mx.eval(packed, scale, bias)
    has_bias = "bias" in module
    if expert:
        new = QuantizedSwitchLinear(
            shape[2], shape[1], shape[0], bias=has_bias,
            group_size=group_size, bits=bits, mode="affine",
        )
    elif isinstance(module, nn.Embedding):
        new = nn.QuantizedEmbedding(shape[0], shape[1], group_size=group_size, bits=bits)
    else:
        new = nn.QuantizedLinear(
            shape[1], shape[0], bias=has_bias, group_size=group_size, bits=bits,
        )
    new.weight, new.scales, new.biases = packed, scale, bias
    if has_bias:
        new.bias = module.bias
    return new, {
        "shape": list(shape), "bits": bits, "group_size": group_size,
        "mode": "affine", "source_mode": getattr(module, "mode", "floating"),
        "sample_weight_mse": sample_error,
        "sample_relative_mse": sample_error / max(sample_signal, 1e-30),
    }


def audit_layout(model, config, expert_bits=2):
    text = config["text_config"]
    loop = text["loop_config"]
    expected_visits = text["num_hidden_layers"] + (
        loop["core_end"] - loop["core_start"]
    ) * (loop["num_passes"] - 1)
    language = model.model.language_model
    if len(language.layer_visits) != expected_visits:
        raise ValueError("The loop schedule changed")
    if len(model.make_cache()) != expected_visits:
        raise ValueError("The per-pass cache layout changed")
    experts = [(p, m) for p, m in model.named_modules() if ".mlp.experts." in p
               and isinstance(m, (SwitchLinear, QuantizedSwitchLinear))]
    if len(experts) != 2 * text["num_hidden_layers"]:
        raise ValueError(f"Unexpected expert stack count: {len(experts)}")
    for path, module in experts:
        if not isinstance(module, QuantizedSwitchLinear):
            raise ValueError(f"Expert was not quantized: {path}")
        if module.bits != expert_bits or module.mode != "affine":
            raise ValueError(f"Expert precision was not repacked: {path}")
        if module.num_experts != text["n_routed_experts"]:
            raise ValueError(f"An expert was lost: {path}")
    if not config.get("vision_config") or "vision_tower" not in model.model:
        raise ValueError("Vision encoder is missing")
    for path, module in model.named_modules():
        if path.startswith("model.vision_tower.") and (
            isinstance(module, (nn.QuantizedLinear, nn.QuantizedEmbedding))
        ):
            raise ValueError("Vision encoder was unexpectedly quantized")
    return {"stored_layers": len(language.layers), "layer_visits": expected_visits,
            "expert_stacks": len(experts), "experts_per_stack": text["n_routed_experts"]}


def copy_assets(source, destination):
    copied = []
    for name in ASSETS:
        if (source / name).is_file():
            shutil.copy2(source / name, destination / name)
            copied.append(name)
    if (source / "tiktoken").is_dir():
        shutil.copytree(source / "tiktoken", destination / "tiktoken")
        copied.extend(str(p.relative_to(source)) for p in (source / "tiktoken").rglob("*") if p.is_file())
    for path in source.glob("LICENSE*"):
        if path.is_file():
            shutil.copy2(path, destination / path.name)
            copied.append(path.name)
    if (source / "README.md").is_file():
        shutil.copy2(source / "README.md", destination / "SOURCE_MODEL_CARD.md")
    for name in copied:
        if (source / name).read_bytes() != (destination / name).read_bytes():
            raise ValueError(f"Metadata copy differs: {name}")
    return copied


def convert(source, destination, group_size=64, *, check_assets=True):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f"Output already exists; choose a new path: {destination}")
    if destination == source or source in destination.parents:
        raise ValueError("Keep the output outside the original source directory")
    config = json.loads((source / "config.json").read_text())
    if config.get("model_type") != "inkling2_loop" or not config.get("vision_config"):
        raise ValueError("Expected the text-and-image Inkling 2 Mini checkpoint")
    if check_assets:
        for name in ("tokenizer.json", "tokenizer_config.json", "processor_config.json", "chat_template.jinja"):
            if not (source / name).is_file():
                raise FileNotFoundError(f"Missing source asset: {name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".incomplete")
    partial.mkdir()  # Existing incomplete jobs are never overwritten.
    started = time.monotonic()
    report = {
        "status": "running", "profile": "expert-2_attention-4_embedding-8",
        "quantization": "affine round-to-nearest", "activation_calibrated": False,
        "source": str(source), "source_revision_requested": SOURCE_REVISION,
        "source_config_sha256": hashlib.sha256((source / "config.json").read_bytes()).hexdigest(),
        "runtime_commit": RUNTIME_COMMIT, "group_size": group_size, "modules": [],
        "packages": {name: importlib.metadata.version(name)
                     for name in ("mlx", "mlx-lm", "numpy", "transformers")},
        "quality_validated": False, "mac_16gb_fit_validated": False,
    }
    def write_report():
        (partial / "CONVERSION_REPORT.json").write_text(json.dumps(report, indent=2) + "\n")
    write_report()
    try:
        log("loading_source")
        model, config = load_model(source, lazy=True, strict=True)
        paths = [p for p, m in model.named_modules() if isinstance(m, MATRIX_CLASSES)]
        decisions = {}
        for path in paths:
            module = dict(model.named_modules())[path]
            bits = choose_bits(path, module)
            if bits is not None and matrix_dims(module)[-1] % group_size:
                if ".mlp.experts." in path:
                    raise ValueError(f"Expert dimension not divisible by group size: {path}")
                bits = None
            decisions[path] = bits
        changed = [p for p, bits in decisions.items() if bits is not None]
        unchanged_hashes = fingerprints(model, exclude=changed)
        qconfig = {"bits": 2, "group_size": group_size, "mode": "affine"}
        for index, path in enumerate(paths):
            bits = decisions[path]
            if bits is None:
                qconfig[path] = False
                continue
            module = dict(model.named_modules())[path]
            replacement, details = repack(module, bits, group_size)
            model.update_modules(tree_unflatten([(path, replacement)]))
            qconfig[path] = {"bits": bits, "group_size": group_size, "mode": "affine"}
            report["modules"].append({"path": path, **details})
            log("repacked", index=index + 1, total=len(paths), path=path, bits=bits,
                source_mode=details["source_mode"], elapsed_s=round(time.monotonic() - started, 1))
            del module, replacement
            gc.collect()
            mx.clear_cache()
        config = copy.deepcopy(config)
        config["quantization"] = qconfig
        config["quantization_config"] = qconfig
        config.pop("quantize_activations", None)
        report["layout"] = audit_layout(model, config)
        if fingerprints(model, exclude=changed) != unchanged_hashes:
            raise ValueError("An unquantized tensor changed")
        log("hashing_converted_tensors")
        expected = fingerprints(model)
        report["tensor_signatures"] = expected
        write_report()
        log("saving")
        save_model(partial, model, donate_model=True)
        save_config(config, partial / "config.json", keep_vision_config=True)
        del model
        gc.collect()
        mx.clear_cache()
        report["copied_assets"] = copy_assets(source, partial)
        log("strict_reload_and_hash_check")
        reloaded, reloaded_config = load_model(partial, lazy=True, strict=True)
        if audit_layout(reloaded, reloaded_config) != report["layout"]:
            raise ValueError("Saved architecture differs")
        if fingerprints(reloaded) != expected:
            raise ValueError("Saved/reloaded tensors differ")
        del reloaded
        gc.collect()
        mx.clear_cache()
        report["status"] = "converted_and_structurally_verified"
        report["elapsed_s"] = round(time.monotonic() - started, 1)
        report["weight_bytes"] = sum(p.stat().st_size for p in partial.glob("model*.safetensors"))
        report["weight_gib"] = round(report["weight_bytes"] / 2**30, 3)
        write_report()
        (partial / "README.md").write_text(
            "# Inkling 2 Mini experimental MLX expert-2bit baseline\n\n"
            "Experts: 2-bit affine; other eligible linear matrices: 4-bit; "
            "embeddings and output head: 8-bit. Group size: " + str(group_size) + ".\n\n"
            "Vision, routing, loop controls, norms and convolutions retain source precision. "
            "All layers, experts and loop passes are retained.\n\n"
            "This is round-to-nearest requantization of the trained NVFP4/MXFP8 release. "
            "It is not activation-calibrated and does not recover original training weights. "
            "Quality, speed and fit on a 16 GB Mac have not been validated.\n\n"
            "Requires MLX 0.32.4 and MLX-LM commit `" + RUNTIME_COMMIT + "`.\n\n"
            "Original model: https://huggingface.co/thinkingmachines/Inkling-2-Mini\n"
            "Original license: Apache-2.0; see SOURCE_MODEL_CARD.md and the upstream policies.\n"
        )
        partial.rename(destination)
        log("complete", output=str(destination), weight_gib=report["weight_gib"],
            elapsed_s=report["elapsed_s"], quality_validated=False)
        return report
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        write_report()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--group-size", type=int, choices=(32, 64, 128), default=64)
    args = parser.parse_args()
    check_runtime()
    mx.set_default_device(mx.cpu)
    convert(args.source, args.output, args.group_size)


if __name__ == "__main__":
    main()

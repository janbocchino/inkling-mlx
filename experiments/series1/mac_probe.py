#!/usr/bin/env python3
"""Bounded text/image smoke tests for the local Inkling checkpoint."""
import os
os.environ.setdefault("MLX_ENABLE_TF32", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse
import gc
import json
from pathlib import Path
import platform
import resource
import subprocess
import time
import traceback

import mlx.core as mx
import numpy as np
from PIL import Image, ImageDraw
from mlx_lm import load, stream_generate
from mlx_lm.processors import load_processor
from mlx_lm.sample_utils import make_sampler


def system_memory():
    result = {}
    for key in ("vm.swapusage", "kern.memorystatus_vm_pressure_level"):
        completed = subprocess.run(["sysctl", "-n", key], capture_output=True, text=True)
        result[key] = completed.stdout.strip() if completed.returncode == 0 else None
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=192)
    parser.add_argument("--cases", default="arithmetic,extraction,code,vision")
    parser.add_argument("--context-lengths", default="")
    parser.add_argument("--prefill-step", type=int, default=64)
    args = parser.parse_args()
    report = {
        "state": "starting", "hardware": mx.device_info(),
        "platform": platform.platform(), "system_before": system_memory(),
        "precision_flag": os.environ["MLX_ENABLE_TF32"],
        "sampling": "greedy", "reasoning_effort": "none", "cases": [],
        "prefill_step_tokens": args.prefill_step,
        "scope": "Smoke tests, not a standard benchmark or a native-source quality comparison",
    }
    def save(event, **fields):
        report.update(fields)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"event": event, **fields}), flush=True)
    try:
        limit = int(mx.device_info()["max_recommended_working_set_size"] * 0.98)
        mx.set_memory_limit(limit)
        mx.set_cache_limit(128 * 1024**2)
        report["mlx_memory_limit_bytes"] = limit
        save("loading", state="loading")
        started = time.monotonic()
        model, tokenizer, config = load(str(args.model), return_config=True)
        report["load_seconds"] = time.monotonic() - started
        report["loaded_active_bytes"] = mx.get_active_memory()
        save("loaded", state="loaded")
        definitions = {
            "arithmetic": ("What is 17 multiplied by 23? Reply with only the number.", "391"),
            "extraction": ('From this sentence extract only the order number: "Order AX-728 was shipped on Tuesday." Reply with the order number only.', "AX-728"),
            "code": ("Write a Python function named larger(a, b) that returns the larger value. Give only the function.", "manual review"),
            "vision": ("Describe the two shapes and their colors and positions in this image in one sentence.", "red square on the left and blue circle on the right"),
        }
        for target in filter(None, args.context_lengths.split(",")):
            target = int(target)
            prefix = "The project access code is LANTERN-42.\n"
            filler = "Background information: the office has two meeting rooms and a shared calendar.\n"
            suffix = "\nWhat is the project access code? Reply with the code only."
            base = tokenizer.apply_chat_template(
                [{"role": "user", "content": prefix + suffix}],
                tokenize=False, add_generation_prompt=True, reasoning_effort="none",
            )
            repeats = max(0, (target - len(tokenizer.encode(base, add_special_tokens=False)))
                          // len(tokenizer.encode(filler, add_special_tokens=False)))
            definitions[f"context_{target}"] = (prefix + filler * repeats + suffix, "LANTERN-42")
        image = Image.new("RGB", (240, 160), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((20, 45, 90, 115), fill="red")
        draw.ellipse((150, 45, 220, 115), fill="blue")
        image_path = args.report.with_name("vision_test.png")
        image.save(image_path)
        report["test_image"] = str(image_path)
        names = list(filter(None, args.cases.split(",")))
        names += [f"context_{target}" for target in filter(None, args.context_lengths.split(","))]
        for name in names:
            question, expected = definitions[name]
            content = [{"type": "text", "text": question}]
            if name == "vision":
                content.append({"type": "image"})
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": content}], tokenize=False,
                add_generation_prompt=True, reasoning_effort="none",
            )
            media = {}
            if name == "vision":
                processor = load_processor(config)
                prompt, media = processor(prompt, images=[np.asarray(image)])
            tokens = mx.array(tokenizer.encode(prompt, add_special_tokens=False))
            mx.clear_cache()
            mx.reset_peak_memory()
            started = time.monotonic()
            embeddings = model.embed_inputs(tokens, **media) if media else None
            if embeddings is not None:
                mx.eval(embeddings)
            text = []
            first_token_seconds = None
            final = None
            save("case_started", state="generating", current_case=name)
            for response in stream_generate(
                model, tokenizer, tokens, max_tokens=args.max_tokens,
                sampler=make_sampler(temp=0), prefill_step_size=args.prefill_step,
                input_embeddings=embeddings,
            ):
                if first_token_seconds is None:
                    first_token_seconds = time.monotonic() - started
                text.append(response.text)
                final = response
            elapsed = time.monotonic() - started
            record = {
                "case": name, "prompt": question, "expected": expected,
                "output": "".join(text), "elapsed_seconds": elapsed,
                "first_token_seconds": first_token_seconds,
                "peak_mlx_bytes": mx.get_peak_memory(),
                "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "system_after": system_memory(),
            }
            if final:
                record.update(prompt_tokens=final.prompt_tokens,
                              generated_tokens=final.generation_tokens,
                              prompt_tokens_per_second=final.prompt_tps,
                              generation_tokens_per_second=final.generation_tps)
            report["cases"].append(record)
            save("case_complete", state="case_complete", current_case=name)
            print(json.dumps(record), flush=True)
            del embeddings, tokens, media
            gc.collect()
            mx.clear_cache()
        save("complete", state="complete", system_after=system_memory())
    except BaseException as error:
        save("failed", state="failed", error=f"{type(error).__name__}: {error}",
             traceback=traceback.format_exc(), system_after=system_memory())
        raise


if __name__ == "__main__":
    main()

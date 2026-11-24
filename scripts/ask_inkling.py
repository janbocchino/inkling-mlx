#!/usr/bin/env python3
"""Run a local Inkling checkpoint; each question starts with a fresh context."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import time


def final_answer(text: str) -> tuple[str, bool]:
    """A final answer requires both Inkling's content and end-message tags."""
    if "<|content_text|>" not in text:
        return "", False
    answer = text.rsplit("<|content_text|>", 1)[1]
    if "<|end_message|>" not in answer:
        return answer.strip(), False
    return answer.split("<|end_message|>", 1)[0].strip(), True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Local converted checkpoint folder")
    parser.add_argument("--prompt", help="Omit for a simple interactive session")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--prefill-step", type=int, choices=[64, 256], default=64)
    args = parser.parse_args()
    if not 1 <= args.max_tokens <= 1024:
        parser.error("--max-tokens must be between 1 and 1024")
    model_path = args.model.expanduser().resolve()
    if not (model_path / "model.safetensors.index.json").is_file():
        parser.error("--model must point to a local converted checkpoint")
    for name, value in {"MLX_ENABLE_TF32": "0", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}.items():
        os.environ.setdefault(name, value)
    import mlx.core as mx
    import numpy as np
    from PIL import Image
    from mlx_lm import load, stream_generate
    from mlx_lm.processors import load_processor
    from mlx_lm.sample_utils import make_sampler

    mx.set_memory_limit(int(mx.device_info()["max_recommended_working_set_size"] * .98))
    mx.set_cache_limit(128 * 1024**2)
    gib = sum(path.stat().st_size for path in model_path.glob("model*.safetensors")) / 2**30
    print(f"Loading local weights ({gib:.2f} GiB).", flush=True)
    model, tokenizer, config = load(str(model_path), return_config=True)
    processor = load_processor(config)
    print("Ready. Each question starts fresh; enter /quit to finish.", flush=True)
    while True:
        try:
            question = args.prompt if args.prompt is not None else input("\nQuestion: ").strip()
            if question.lower() in {"/quit", "/exit"}:
                break
            if not question:
                continue
            image_path = args.image
            if args.prompt is None:
                name = input("Image path, or Return for text only: ").strip().strip("\"'")
                image_path = Path(name).expanduser() if name else None
            content = [{"type": "text", "text": question}]
            if image_path:
                content.append({"type": "image"})
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": content}], tokenize=False,
                add_generation_prompt=True, reasoning_effort="none")
            media = {}
            if image_path:
                with Image.open(image_path.expanduser()) as image:
                    if image.has_transparency_data:
                        from mlx_lm.models.inkling2.processor import _composite_alpha
                        pixels = Image.fromarray(_composite_alpha(np.asarray(image.convert("RGBA"))))
                    else:
                        pixels = image.convert("RGB")
                    pixels.thumbnail((512, 512), Image.Resampling.LANCZOS)
                    prompt, media = processor(prompt, images=[np.asarray(pixels)])
            tokens = mx.array(tokenizer.encode(prompt, add_special_tokens=False))
            if tokens.size + args.max_tokens > 4096:
                raise ValueError("This launcher caps prompt plus output at 4096 tokens")
            started = time.monotonic()
            embeddings = model.embed_inputs(tokens, **media) if media else None
            if embeddings is not None:
                mx.eval(embeddings)
            fragments, last = [], None
            for response in stream_generate(
                model, tokenizer, tokens, max_tokens=args.max_tokens,
                sampler=make_sampler(temp=0), prefill_step_size=args.prefill_step,
                input_embeddings=embeddings):
                fragments.append(response.text)
                last = response
            answer, complete = final_answer("".join(fragments))
            print("\n" + answer if complete else "\nNo complete tagged final answer before the output limit.")
            if last:
                print(f"{time.monotonic()-started:.1f}s total; {last.generation_tps:.1f} generated tokens/s, including reasoning.")
            del embeddings, tokens, media
            mx.clear_cache()
        except (EOFError, KeyboardInterrupt):
            break
        except (ValueError, OSError) as error:
            print(f"Cannot process that request: {error}")
            if args.prompt is not None:
                raise SystemExit(1) from error
        if args.prompt is not None:
            break


if __name__ == "__main__":
    main()

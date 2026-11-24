#!/usr/bin/env python3
"""Local text/image chat with the converted checkpoint. No network service."""
import os
os.environ.setdefault("MLX_ENABLE_TF32", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse
from pathlib import Path
import time

import mlx.core as mx
import numpy as np
from PIL import Image
from mlx_lm import load, stream_generate
from mlx_lm.processors import load_processor
from mlx_lm.sample_utils import make_sampler


def final_answer(text):
    if "<|content_text|>" in text:
        return text.rsplit("<|content_text|>", 1)[1].split("<|end_message|>", 1)[0].strip(), True
    return text.strip(), False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--prefill-step", type=int, choices=(64, 256), default=64)
    args = parser.parse_args()
    if not 1 <= args.max_tokens <= 1024:
        parser.error("--max-tokens must be between 1 and 1024")
    model_path = Path(__file__).resolve().parent / "Inkling-2-Mini-expert-2bit-g64"
    if not (model_path / "model.safetensors.index.json").is_file():
        raise SystemExit("The local model folder is missing.")
    mx.set_memory_limit(int(mx.device_info()["max_recommended_working_set_size"] * .98))
    mx.set_cache_limit(128 * 1024**2)
    print("Loading Inkling locally. Its weights occupy about 11 GB of memory.", flush=True)
    model, tokenizer, config = load(str(model_path), return_config=True)
    processor = load_processor(config)
    print("Ready. Nothing you enter is sent to Leonardo or an online service.", flush=True)
    if args.prompt is None:
        print("Each question starts fresh. Enter /quit to close and release the model memory.")
    while True:
        if args.prompt is not None:
            question = args.prompt
            image_path = args.image
        else:
            try:
                question = input("\nQuestion: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if question.lower() in ("/quit", "/exit"):
                break
            if not question:
                continue
            try:
                name = input("Image path, or press Return for text only: ").strip().strip('"\'')
            except (EOFError, KeyboardInterrupt):
                print()
                break
            image_path = Path(name).expanduser() if name else None
        try:
            content = [{"type": "text", "text": question}]
            if image_path:
                content.append({"type": "image"})
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": content}], tokenize=False,
                add_generation_prompt=True, reasoning_effort="none",
            )
            media = {}
            if image_path:
                with Image.open(image_path) as original:
                    if original.has_transparency_data:
                        from mlx_lm.models.inkling2.processor import _composite_alpha
                        pixels = Image.fromarray(_composite_alpha(np.asarray(original.convert("RGBA"))))
                    else:
                        pixels = original.convert("RGB")
                    pixels.thumbnail((512, 512))
                    prompt, media = processor(prompt, images=[np.asarray(pixels)])
            tokens = mx.array(tokenizer.encode(prompt, add_special_tokens=False))
            if tokens.size + args.max_tokens > 4096:
                raise ValueError("Use a shorter prompt or fewer output tokens; this launcher caps the total at 4096 tokens.")
            started = time.monotonic()
            embeddings = model.embed_inputs(tokens, **media) if media else None
            if embeddings is not None:
                mx.eval(embeddings)
            print("Generating...", flush=True)
            fragments = []
            last = None
            for response in stream_generate(
                model, tokenizer, tokens, max_tokens=args.max_tokens,
                sampler=make_sampler(temp=0), prefill_step_size=args.prefill_step,
                input_embeddings=embeddings,
            ):
                fragments.append(response.text)
                last = response
            answer, has_final = final_answer("".join(fragments))
            print("\n" + answer)
            if not has_final:
                print("\nThe model did not produce a tagged final answer before the output limit.")
            if last:
                print(f"\n{time.monotonic()-started:.1f}s total; {last.generation_tps:.1f} generated tokens/s, including reasoning.")
            del embeddings, tokens, media
            mx.clear_cache()
        except (ValueError, OSError) as error:
            print(f"Cannot process that request: {error}")
        if args.prompt is not None:
            break


if __name__ == "__main__":
    main()

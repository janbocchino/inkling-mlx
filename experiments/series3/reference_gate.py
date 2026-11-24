"""Independent tiny-model CPU check. Run on Leonardo, never a full checkpoint.

The Transformers implementation supplies the random weights and logits. Every
parameter must map exactly once; missing mappings stop the experiment. Passing
this gate validates this tiny configuration, not the released checkpoint.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
import sys
import time
from common import atomic_json, check_deadline


def mlx_name(name):
    if name == "lm_head.weight":
        return name
    if not name.startswith("model."):
        raise ValueError(f"Unexpected reference parameter: {name}")
    name = name.replace("model.", "model.language_model.", 1)
    name = name.replace("embed_tokens.embed_norm.", "embed_norm.")
    name = re.sub(r"(\.mlp\.experts\.(?:gate_up|down)_proj)$", r"\1.weight", name)
    name = name.replace(".self_attn.q_proj.linear.", ".self_attn.q_proj.")
    name = name.replace(".self_attn.q_proj.norm.scale", ".self_attn.q_l2_clip.scale")
    for kind in ("k", "v"):
        name = name.replace(f".self_attn.{kind}_proj.linear.", f".self_attn.{kind}_proj.")
        name = name.replace(f".self_attn.{kind}_proj.sconv.", f".self_attn.{kind}_sconv.")
    name = name.replace(".self_attn.o_proj.linear.", ".self_attn.o_proj.")
    name = name.replace(".self_attn.o_proj.gate_proj.", ".self_attn.gate_proj.")
    name = name.replace(".self_attn.relative_position_bias.linear.", ".self_attn.r_proj.")
    name = name.replace(".self_attn.relative_position_bias.norm.scale", ".self_attn.rel_l2_clip.scale")
    return name.replace(".self_attn.relative_position_bias.proj", ".self_attn.rel_logits_proj.proj")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deadline-unix", type=float, required=True)
    args = parser.parse_args()
    if sys.platform != "linux":
        raise RuntimeError("This model job is remote-only; run on Leonardo")
    check_deadline(args.deadline_unix)
    import numpy as np
    import torch
    import transformers
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_lm.models.inkling2 import Model, ModelArgs
    from transformers.models.inkling2_loop.configuration_inkling2_loop import Inkling2LoopTextConfig
    from transformers.models.inkling2_loop.modeling_inkling2_loop import Inkling2LoopForCausalLM

    mx.set_default_device(mx.cpu)
    torch.set_num_threads(4)
    start = time.time()
    records = []
    for seed in (0, 1):
        # Covers local/full attention, block boundaries and a looped full layer.
        text = dict(hidden_size=64, num_hidden_layers=8, vocab_size=128,
                    unpadded_vocab_size=120, num_attention_heads=4,
                    num_key_value_heads=2, head_dim=16, swa_num_attention_heads=4,
                    swa_num_key_value_heads=2, swa_head_dim=16, sliding_window_size=8,
                    local_layer_ids=[0, 1, 2, 3, 4, 6], d_rel=4, rel_extent=8,
                    log_scaling_n_floor=4, log_scaling_alpha=0.1, rms_norm_eps=1e-6,
                    qk_l2_clip_threshold=3.0, rel_l2_clip_threshold=2.0,
                    sconv_kernel_size=4, n_layers_per_attn_res_block=3,
                    n_routed_experts=4, num_experts_per_tok=2, intermediate_size=64,
                    mlp_gate_softcap=4.0, mlp_up_softcap=25.0, route_scale=2.0,
                    final_logit_softcapping=30.0,
                    loop_config=dict(core_start=3, core_end=6, num_passes=3,
                                     gate_rank=8, is_varloop=True))
        hf_text = {key: value for key, value in text.items()
                   if key not in ("local_layer_ids", "loop_config", "intermediate_size", "sconv_kernel_size")}
        hf_text.update(moe_intermediate_size=64, conv_kernel_size=4, eos_token_id=119,
                       layer_types=["hybrid_sliding" if i in text["local_layer_ids"] else "hybrid"
                                    for i in range(8)],
                       loop_layer_index_start=3, loop_layer_index_end=6,
                       loop_num_passes=3, loop_gate_rank=8)
        torch.manual_seed(seed)
        config = Inkling2LoopTextConfig(**hf_text)
        config._attn_implementation = "eager"
        reference = Inkling2LoopForCausalLM(config).float().eval()
        model = Model(ModelArgs.from_dict(dict(model_type="inkling2_loop", text_config=text)))
        expected = dict(tree_flatten(model.parameters()))
        weights = {}
        for name, tensor in reference.state_dict().items():
            mapped = mlx_name(name)
            if mapped in weights or mapped not in expected:
                raise ValueError(f"Unmapped or duplicate reference tensor: {name} -> {mapped}")
            value = tensor.detach().float().cpu().numpy()
            if name.endswith("conv1d.weight"):
                value = value.swapaxes(1, 2)
            if name == "lm_head.weight":
                value = np.pad(value, ((0, text["vocab_size"] - value.shape[0]), (0, 0)))
            if tuple(value.shape) != tuple(expected[mapped].shape):
                raise ValueError(f"Shape mismatch: {name}: {value.shape} versus {expected[mapped].shape}")
            weights[mapped] = mx.array(value)
        if weights.keys() != expected.keys():
            raise ValueError(f"Unfilled MLX tensors: {sorted(expected.keys() - weights.keys())}")
        model.load_weights(list(weights.items()), strict=True)
        mx.eval(model.parameters())
        tokens = np.random.default_rng(seed).integers(0, 120, size=(1, 20), dtype=np.int64)
        for passes in (1, 3, 5):
            check_deadline(args.deadline_unix)
            config.loop_num_passes = passes
            model.model.language_model.args.loop_config["num_passes"] = passes
            cache = model.make_cache()
            visits = 8 + (passes - 1) * 3
            if len(cache) != visits:
                raise ValueError("Logical visit count and cache count differ")
            with torch.no_grad():
                hf = reference(torch.tensor(tokens), use_cache=False).logits.numpy()
                hf_chunks, hf_cache = [], None
                for left, right in [(0, 6), (6, 12)] + [(i, i + 1) for i in range(12, 20)]:
                    out = reference(torch.tensor(tokens[:, left:right]), cache=hf_cache, use_cache=True)
                    hf_cache = out.cache
                    hf_chunks.append(out.logits.numpy())
            full = np.asarray(model(mx.array(tokens)))
            chunks = [np.asarray(model(mx.array(tokens[:, left:right]), cache=cache))
                      for left, right in [(0, 6), (6, 12)] + [(i, i + 1) for i in range(12, 20)]]
            comparisons = {"independent_reference": (full, hf),
                           "mlx_cached_full": (np.concatenate(chunks, 1), full),
                           "reference_cached_full": (np.concatenate(hf_chunks, 1), hf)}
            for label, (actual, wanted) in comparisons.items():
                passed = bool(np.allclose(actual, wanted, atol=1e-4, rtol=1e-4))
                records.append(dict(seed=seed, passes=passes, check=label, passed=passed,
                                    max_absolute_error=float(np.max(np.abs(actual - wanted)))))
                atomic_json(args.output, dict(status="checking", records=records))
                if not passed:
                    raise ValueError(f"Reference gate failed: {records[-1]}")
    atomic_json(args.output, dict(status="passed", records=records, seconds=time.time() - start,
                                 torch=torch.__version__, transformers=transformers.__version__,
                                 limits="Tiny FP32 text configuration only; released native checkpoint still needs validation"))


if __name__ == "__main__":
    main()

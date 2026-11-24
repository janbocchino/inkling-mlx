#!/usr/bin/env python3
"""Small CPU-only tests; no real checkpoint or model download required."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm.layers.moe import SwitchLinear
from mlx_lm.models.inkling2 import Model, ModelArgs
from mlx_lm.utils import load_model, save_config, save_model

from convert_inkling import array_signature, audit_layout, check_runtime, convert, repack


def tiny_config():
    return {
        "model_type": "inkling2_loop", "image_token_id": 127,
        "text_config": {
            "hidden_size": 64, "num_hidden_layers": 4, "vocab_size": 128,
            "unpadded_vocab_size": 120, "num_attention_heads": 4,
            "num_key_value_heads": 2, "head_dim": 16,
            "swa_num_attention_heads": 4, "swa_num_key_value_heads": 2,
            "swa_head_dim": 16, "sliding_window_size": 8,
            "local_layer_ids": [0, 1, 3], "d_rel": 4, "rel_extent": 8,
            "log_scaling_n_floor": 4, "log_scaling_alpha": 0.1,
            "rms_norm_eps": 1e-6, "qk_l2_clip_threshold": 3.0,
            "rel_l2_clip_threshold": 2.0, "sconv_kernel_size": 4,
            "n_layers_per_attn_res_block": 1,
            "n_routed_experts": 2, "num_experts_per_tok": 1,
            "intermediate_size": 64, "mlp_gate_softcap": 4.0,
            "mlp_up_softcap": 25.0, "route_scale": 2.0,
            "final_logit_softcapping": 30.0,
            "loop_config": {"core_start": 1, "core_end": 3,
                            "num_passes": 3, "gate_rank": 8, "is_varloop": False},
        },
        "vision_config": {
            "hidden_size": 32, "num_hidden_layers": 1, "num_attention_heads": 2,
            "native_patch_size": 2, "adapter_merge_size": 2,
            "posemb_grid_size": 4, "projection_hidden_dim": 48,
        },
    }


def make_source(path):
    config = tiny_config()
    model = Model(ModelArgs.from_dict(config))
    # Use the same cast policy as the port; the router/controller gains stay FP32.
    model.update(tree_unflatten([
        (key, value.astype(mx.bfloat16) if model.cast_predicate(key) else value)
        for key, value in tree_flatten(model.parameters())
    ]))
    replacements = []
    for name, module in model.named_modules():
        if isinstance(module, SwitchLinear):
            if ".layers.3." in name:
                quantized = module.to_quantized(32, 8, mode="mxfp8")
            else:
                quantized = module.to_quantized(16, 4, mode="nvfp4", global_scale=True)
                quantized.global_scale = quantized.global_scale * mx.array([0.5, 1.5])
            replacements.append((name, quantized))
    model.update_modules(tree_unflatten(replacements))
    mx.eval(model.parameters())
    save_model(path, model)
    config["quantization_config"] = {"quant_method": "modelopt", "quant_algo": "NVFP4"}
    save_config(copy.deepcopy(config), path / "config.json", keep_vision_config=True)
    (path / "processor_config.json").write_text('{"fixture": true}')
    (path / "chat_template.jinja").write_text("image-and-text fixture")
    (path / "tokenizer.json").write_text('{"fixture": true}')
    (path / "tokenizer_config.json").write_text('{"fixture": true}')
    (path / "tiktoken").mkdir()
    (path / "tiktoken/tokenizer.model").write_bytes(b"fixture-tokenizer")
    return config


class ConverterTests(unittest.TestCase):
    def setUp(self):
        mx.set_default_device(mx.cpu)
        mx.random.seed(12)

    def test_repack_uses_each_expert_global_scale(self):
        module = SwitchLinear(64, 64, 2, bias=False)
        module.weight = mx.random.normal((2, 64, 64)).astype(mx.bfloat16)
        for mode, group, bits, global_scale in (
            ("nvfp4", 16, 4, True), ("mxfp8", 32, 8, False)
        ):
            native = module.to_quantized(group, bits, mode=mode, global_scale=global_scale)
            if global_scale:
                native.global_scale = native.global_scale * mx.array([0.25, 2.0])
            converted, _ = repack(native, 2, 64)
            for expert in range(2):
                kwargs = {"global_scale": native.global_scale[expert]} if global_scale else {}
                reconstructed = mx.dequantize(
                    native.weight[expert], native.scales[expert],
                    group_size=group, bits=bits, mode=mode, dtype=mx.bfloat16, **kwargs,
                )
                expected = mx.quantize(reconstructed, group_size=64, bits=2)
                for actual, want in zip(
                    (converted.weight[expert], converted.scales[expert], converted.biases[expert]),
                    expected,
                ):
                    self.assertTrue(mx.array_equal(actual, want).item(), mode)
            self.assertIsNone(converted.global_scale)

    def test_zero_and_constant_rows_are_finite(self):
        for number in (0, 1.5, -0.75):
            module = nn.Linear(64, 64, bias=False)
            module.weight = mx.full((64, 64), number, dtype=mx.bfloat16)
            converted, _ = repack(module, 2, 64)
            decoded = mx.dequantize(converted.weight, converted.scales, converted.biases,
                                    group_size=64, bits=2)
            self.assertTrue(mx.all(mx.isfinite(decoded)).item())
            self.assertTrue(mx.allclose(decoded, module.weight, atol=0.02).item())

    def test_signatures_handle_scalars_and_transposed_arrays(self):
        scalar = mx.array(1.25, dtype=mx.bfloat16)
        self.assertEqual(array_signature(scalar)["shape"], [])
        transposed = mx.arange(12, dtype=mx.float32).reshape(3, 4).T
        contiguous = mx.contiguous(transposed)
        self.assertEqual(array_signature(transposed), array_signature(contiguous))

    def test_full_loop_and_vision_conversion_reload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "source", root / "output"
            source.mkdir()
            make_source(source)
            # Source bytes must remain identical after conversion.
            original = {str(p.relative_to(source)): p.read_bytes()
                        for p in source.rglob("*") if p.is_file()}
            report = convert(source, output, group_size=64)
            self.assertEqual(report["layout"]["layer_visits"], 8)
            self.assertEqual(report["layout"]["expert_stacks"], 8)
            self.assertFalse(report["quality_validated"])
            self.assertEqual(original, {str(p.relative_to(source)): p.read_bytes()
                                       for p in source.rglob("*") if p.is_file()})
            loaded, config = load_model(output, lazy=True, strict=True)
            self.assertEqual(audit_layout(loaded, config)["layer_visits"], 8)
            self.assertEqual(loaded.lm_head.bits, 8)
            self.assertEqual(loaded.model.language_model.embed_tokens.bits, 8)
            tokens = mx.array([[1, 2, 3]])
            result = loaded(tokens)
            self.assertEqual(result.shape, (1, 3, 120))
            self.assertTrue(mx.all(mx.isfinite(result)).item())
            # Also exercise the saved image encoder and image-token insertion.
            image = mx.ones((3, 4, 4))
            image_tokens = mx.array([[1, 127, 2]])
            visual_result = loaded(image_tokens, pixel_values=[image])
            self.assertEqual(visual_result.shape, (1, 3, 120))
            self.assertTrue(mx.all(mx.isfinite(visual_result)).item())
            self.assertEqual((output / "tiktoken/tokenizer.model").read_bytes(), b"fixture-tokenizer")
            with self.assertRaises(FileExistsError):
                convert(source, output)


if __name__ == "__main__":
    check_runtime()
    unittest.main(verbosity=2)

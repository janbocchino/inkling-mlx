"""Pinned MLX helpers for source reconstruction, calibration and scoring."""
from __future__ import annotations
import gc, json, re, time
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_unflatten
from mlx_lm.layers.moe import QuantizedSwitchLinear, SwitchLinear
from mlx_lm.utils import load_model, load_tokenizer
from mlx_lm.processors import load_processor
from convert_inkling import reconstruct_expert
from common import atomic_json, event

def expert_modules(model):
    return [(p, m) for p, m in model.named_modules()
            if '.mlp.experts.' in p and isinstance(m, (SwitchLinear, QuantizedSwitchLinear))]

def layer_number(path):
    return int(re.search(r'\.layers\.(\d+)\.', path).group(1))

class Statistics:
    def __init__(self):
        self.sums = {}; self.counts = {}; self.visits = {}; self.calls = {}; self.enabled=True
    def begin_case(self): self.calls = {}
    def observe(self, name, x, indices, expert_count):
        if not self.enabled:return
        mx.eval(x, indices)
        values = np.asarray(x.astype(mx.float32)).reshape(-1, x.shape[-1])
        idx = np.asarray(indices).reshape(-1).astype(np.int64)
        if len(values) != len(idx):
            if len(idx) % len(values): raise ValueError('Cannot align routed activations')
            values = np.repeat(values, len(idx) // len(values), axis=0)
        if name not in self.sums:
            self.sums[name] = np.zeros((expert_count, values.shape[-1]), np.float32)
            self.counts[name] = np.zeros(expert_count, np.int64)
            self.visits[name] = {}
        np.add.at(self.sums[name], idx, np.square(values))
        counts = np.bincount(idx, minlength=expert_count)
        self.counts[name] += counts
        visit = self.calls.get(name, 0); self.calls[name] = visit + 1
        key = str(visit)
        self.visits[name].setdefault(key, np.zeros(expert_count, np.int64))
        self.visits[name][key] += counts
    def save(self, directory, accepted):
        directory = Path(directory); directory.mkdir(exist_ok=True, parents=True)
        report = {'accepted_cases': accepted, 'modules': {}, 'source_activations': True}
        for i, name in enumerate(sorted(self.sums)):
            filename = f'stat-{i:03d}.npz'
            temp = directory / (filename + '.tmp')
            with temp.open('wb') as f:
                np.savez_compressed(f, sum2=self.sums[name], count=self.counts[name])
            temp.replace(directory / filename)
            report['modules'][name] = {'file': filename, 'counts': self.counts[name].tolist(),
                'at_least_32_routed_tokens': int((self.counts[name] >= 32).sum()),
                'total_experts': len(self.counts[name]),
                'loop_visit_counts': {k:v.tolist() for k,v in self.visits[name].items()}}
        atomic_json(directory / 'coverage.json', report)
        return report

class ObservedSwitch(SwitchLinear):
    def __call__(self, x, indices, sorted_indices=False):
        self._collector.observe(self._path, x, indices, self.num_experts)
        return super().__call__(x, indices, sorted_indices)

def load_reference(source, collector=None):
    model, config = load_model(Path(source), lazy=True, strict=True)
    modules = expert_modules(model)
    for i, (path, old) in enumerate(modules):
        shape = (old.num_experts, old.output_dims, old.input_dims)
        cls = ObservedSwitch if collector is not None else SwitchLinear
        new = cls(shape[2], shape[1], shape[0], bias='bias' in old)
        new.weight = mx.stack([reconstruct_expert(old, e).astype(mx.float32) for e in range(shape[0])])
        if 'bias' in old: new.bias = old.bias.astype(mx.float32)
        if collector is not None:
            new._collector, new._path = collector, path
        mx.eval(new.weight)
        model.update_modules(tree_unflatten([(path, new)]))
        modules[i] = (path, None)  # Release native modules rather than retaining the whole source.
        del old, new
        gc.collect(); mx.clear_cache()
        event('reference_module_ready', module=path, index=i+1, total=len(modules))
    model.set_dtype(mx.float32)
    mx.eval(model.parameters())
    from matrix_client import attach
    attach(model,source_reference=True)
    return model, config

def load_candidate(path):
    model, config = load_model(Path(path), lazy=True, strict=True)
    model.set_dtype(mx.float32)
    mx.eval(model.parameters())
    from matrix_client import attach
    attach(model,source_reference=False)
    return model, config

def case_arrays(case, data_root, tokenizer, config, with_answer=True):
    content = [{'type': 'text', 'text': case['question']}]
    if case.get('image'): content.append({'type': 'image'})
    messages = [{'role': 'user', 'content': content}]
    if with_answer:
        marker='INKLING_ANSWER_BOUNDARY_7F2E9183'
        full = tokenizer.apply_chat_template(messages + [{'role': 'assistant', 'content': marker}],
            tokenize=False, add_generation_prompt=False, reasoning_effort='none')
        if full.count(marker)!=1:raise ValueError('Ambiguous assistant answer boundary')
        position = full.index(marker)
        prompt = full[:position]
    else:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False,
            add_generation_prompt=True, reasoning_effort='none')
    media = {}
    if case.get('image'):
        image = np.load(Path(data_root) / case['image'], allow_pickle=False)
        prompt, media = load_processor(config)(prompt, images=[image])
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    answer_ids = tokenizer.encode(case['answer'], add_special_tokens=False) if with_answer else []
    ids = prompt_ids + answer_ids
    if not prompt_ids or (with_answer and not answer_ids): raise ValueError('Empty token sequence')
    return mx.array([ids], dtype=mx.int32), media, len(prompt_ids), len(answer_ids)

def score_case(model, config, tokenizer, case, data_root):
    inputs, media, start, count = case_arrays(case, data_root, tokenizer, config)
    if inputs.shape[1] > 512: raise ValueError('Evaluation case exceeds 512 token cap')
    logits = model(inputs, **media)
    wanted = inputs[:, start: start + count]
    selected = logits[:, start - 1:start + count - 1, :]
    losses = mx.logsumexp(selected, axis=-1) - mx.take_along_axis(selected, wanted[..., None], axis=-1).squeeze(-1)
    loss = mx.sum(losses)
    mx.eval(loss)
    value = float(loss.item())
    if not np.isfinite(value): raise ValueError('Nonfinite answer loss')
    return {'id': case['id'], 'modality': case['modality'], 'category': case['category'],
        'answer_tokens': count, 'prompt_tokens': start, 'nll_sum': value, 'nll_per_token': value / count}

def aggregate(records):
    result = {}
    for group in ['all', 'text', 'image']:
        rows = [r for r in records if group == 'all' or r['modality'] == group]
        tokens = sum(r['answer_tokens'] for r in rows)
        result[group] = {'cases':len(rows), 'answer_tokens':tokens,
            'nll_per_token': sum(r['nll_sum'] for r in rows) / tokens if tokens else None}
    return result

"""Weight quantization specs for the DiT and the text encoder (MLX `nn.quantize`).

A spec is a dict: {"bits": 8, "group_size": 64, "mode": "affine", "overrides": {"<path substring>": {...}},
"skip": ["<path substring>", ...]}. Overrides change bits/group/mode for matching layers; skips keep them in bf16.
The same spec is stored in the exported config so loading re-applies identical structure.

DiT default skips: the small, sensitive conditioning paths (patch embedder, timestep MLP, the shared modulation and
the head modulation, the final projection) stay in bf16; they are ~1% of the weights.
"""

from __future__ import annotations

import copy

import mlx.core as mx
import mlx.nn as nn

DIT_SKIP = ["video_embedder", "time_embedding", "shared_block_modulation", "video_head_modulation",
            "final_linear_video", "text_embedder.proj"]


def _predicate(spec: dict, default_skip: list[str]):
    skip = list(spec.get("skip", default_skip))
    overrides = spec.get("overrides", {})
    base = {"bits": spec["bits"], "group_size": spec["group_size"], "mode": spec.get("mode", "affine")}

    def pred(path: str, module):
        if not hasattr(module, "to_quantized") or any(s in path for s in skip):
            return False
        if module.weight.shape[-1] % base["group_size"]:
            return False
        params = dict(base)
        for key, extra in overrides.items():
            if key in path:
                params.update(extra)
        if params.get("bits") == 16:  # an override can keep a layer in bf16
            return False
        return params  # non-affine modes need matching values, e.g. nvfp4: bits 4, group 16

    return pred


def quantize_dit(model: nn.Module, spec: dict | None) -> nn.Module:
    if spec:
        nn.quantize(model, class_predicate=_predicate(spec, DIT_SKIP))
    return model


def quantize_text_encoder(model: nn.Module, spec: dict | None) -> nn.Module:
    if spec:
        nn.quantize(model, class_predicate=_predicate(spec, spec.get("skip", [])))
    return model


def describe(spec: dict | None) -> str:
    if not spec:
        return "bf16"
    s = f"{spec.get('mode', 'affine')}{spec['bits'] if spec.get('mode', 'affine') == 'affine' else ''}g{spec['group_size']}"
    if spec.get("overrides"):
        s += "+" + ",".join(f"{k}:{v.get('bits', '')}" for k, v in spec["overrides"].items())
    return s


def weight_bytes(model: nn.Module) -> int:
    from mlx.utils import tree_flatten

    return sum(v.nbytes for _, v in tree_flatten(model.parameters()))


def spec_copy(spec):
    return copy.deepcopy(spec)

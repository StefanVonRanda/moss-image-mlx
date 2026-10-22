"""Unmerged LoRA adapters on the DiT (kohya keys, the format of the published MOSS-Image adapters).

y = W x + strength * (alpha / rank) * up(down(x)). The low-rank path runs in the adapter's dtype, and the base weights
stay untouched: merging an adapter into bf16 or quantized weights would round most of a distillation delta away.
Works over bf16, fp32 and quantized linears. Adapters can be stacked; each belongs to a named group whose strength
can be changed per step (set_strength) without touching the other groups.
"""

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

PREFIX = "diffusion_model."


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Module, down: mx.array, up: mx.array, scale: float, group: str = "default"):
        super().__init__()
        self.base = base
        self.down, self.up = down, up  # (rank, in), (out, rank)
        self.scale = scale  # strength * alpha / rank
        self.base_scale = scale
        self.group = group

    def __call__(self, x):
        y = self.base(x)
        if self.scale == 0:  # adapter switched off for this call: exactly the base layer, at base cost
            return y
        return y + ((x.astype(self.down.dtype) @ self.down.T) @ self.up.T).astype(y.dtype) * self.scale


def _resolve(model, name: str):
    """(parent, key) for a dotted module path through attributes, lists and dicts."""
    parts = name.split(".")
    node = model
    for part in parts[:-1]:
        node = node[int(part)] if isinstance(node, list) else node[part] if isinstance(node, dict) else getattr(node, part)
    return node, parts[-1]


def _get(parent, key):
    return parent[int(key)] if isinstance(parent, list) else parent[key] if isinstance(parent, dict) else getattr(parent, key)


def _set(parent, key, value):
    if isinstance(parent, list):
        parent[int(key)] = value
    elif isinstance(parent, dict):
        parent[key] = value
    else:
        setattr(parent, key, value)


def apply_lora(model, path: str | Path, strength: float = 1.0, dtype=mx.bfloat16, group: str = "default") -> int:
    """Wrap every linear the adapter file targets (around any adapter already there); returns the number wrapped."""
    weights = mx.load(str(path))
    suffix = ".lora_down.weight"
    names = sorted(k[len(PREFIX) : -len(suffix)] for k in weights if k.startswith(PREFIX) and k.endswith(suffix))
    if not names:
        raise ValueError(f"{path}: no kohya LoRA keys ({PREFIX}<module>{suffix})")
    for name in names:
        parent, key = _resolve(model, name)
        base = _get(parent, key)
        # Cast one module at a time and drop the file's copy, so the full file never sits in memory next to the DiT.
        down = weights.pop(f"{PREFIX}{name}.lora_down.weight").astype(dtype)
        up = weights.pop(f"{PREFIX}{name}.lora_up.weight").astype(dtype)
        alpha = weights.pop(f"{PREFIX}{name}.alpha", None)
        mx.eval(down, up)
        rank = down.shape[0]
        scale = strength * (float(alpha.item()) if alpha is not None else rank) / rank
        _set(parent, key, LoRALinear(base, down, up, scale, group))
    del weights
    mx.clear_cache()
    return len(names)


def set_strength(model, strength: float, group: str | None = None):
    """Scale the adapters of `group` (all groups if None) relative to the strength they were applied with;
    0 = switched off."""
    for _, module in model.named_modules():
        if isinstance(module, LoRALinear) and (group is None or module.group == group):
            module.scale = module.base_scale * strength

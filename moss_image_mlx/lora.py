"""Unmerged LoRA adapters on the DiT (kohya keys, the format of the published MOSS-Image adapters, and the others
in use: diffusers/PEFT, MOSS Image Edit's bare `layers.`, sd-scripts' legacy `lora_unet_`, and ai-toolkit's fused qkv).

y = W x + strength * (alpha / rank) * up(down(x)). The low-rank path runs in the adapter's dtype, and the base weights
stay untouched: merging an adapter into bf16 or quantized weights would round most of a distillation delta away.
Works over bf16, fp32 and quantized linears. Adapters can be stacked; each belongs to a named group whose strength
can be changed per step (set_strength) without touching the other groups.
"""

import re
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

# kohya: diffusion_model.layers.3.proj.q.{lora_down.weight,lora_up.weight,alpha}
# diffusers/PEFT: transformer.layers.3.proj.q.{lora_A.weight,lora_B.weight}
# MOSS Image Edit: layers.3.proj.q.{lora_A,lora_B}, no prefix (the colour/depth control adapter also has bare
# text_embedder.blocks.0.{qkv,proj,ffn_gate,ffn_up,ffn_down})
_KEY = re.compile(r"^(?:(?:diffusion_model|transformer)\.|(?=layers\.|text_embedder\.))(.+?)\.(lora_down|lora_up|lora_A|lora_B|alpha)(?:\.weight)?$")
# sd-scripts' older export (many ComfyUI LoRAs): the dotted path with dots as underscores, so leaf names keep theirs:
# lora_unet_layers_3_proj_ffn_up_gate.{lora_down.weight,lora_up.weight,alpha}
_LEGACY_KEY = re.compile(
    r"^lora_unet_layers_(\d+)_proj_(q|k|v|o|qkv|ffn_up_gate|ffn_down)\.(lora_down|lora_up|alpha)(?:\.weight)?$"
)
_SLOT = {"lora_down": "down", "lora_A": "down", "lora_up": "up", "lora_B": "up", "alpha": "alpha"}


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


def _split_key(key: str) -> tuple[str, str]:
    """Module path and slot (down/up/alpha) for one LoRA tensor key."""
    m = _KEY.match(key)
    if m:
        return m.group(1), _SLOT[m.group(2)]
    m = _LEGACY_KEY.match(key)
    if m:
        return f"layers.{m.group(1)}.proj.{m.group(2)}", _SLOT[m.group(3)]
    raise ValueError(f"unrecognized LoRA key: {key}")


def load_lora(path: str | Path, control: bool = False) -> dict[str, dict]:
    """Group a LoRA file by module path: {"layers.3.proj.q": {"down": ..., "up": ..., "alpha": ...}}. A control
    adapter's video_embedder tensors are skipped with control=True (apply_control sets them), refused otherwise."""
    mods: dict[str, dict] = {}
    for key, value in mx.load(str(path)).items():
        if key.startswith("video_embedder."):
            if not control:
                raise ValueError(f"{path} replaces video_embedder: a control adapter, load it with --control")
            continue
        name, slot = _split_key(key)
        mods.setdefault(name, {})[slot] = value
    return mods


def apply_lora(model, path: str | Path, strength: float = 1.0, dtype=mx.bfloat16, group: str = "default",
               control: bool = False) -> int:
    """Wrap every linear the adapter file targets (around any adapter already there); returns the number of adapter
    modules applied. A fused qkv adapter on a block with separate q, k, v wraps each with the shared down and its
    third of up."""
    mods = load_lora(path, control)
    if not mods:
        raise ValueError(f"{path}: no LoRA keys")
    count = len(mods)
    for name in sorted(mods):
        tensors = mods.pop(name)  # one module at a time, so the full file never sits in memory next to the DiT
        parent, key = _resolve(model, name)
        rank = tensors["down"].shape[0]
        alpha = tensors.get("alpha")
        scale = strength * (float(alpha.item()) if alpha is not None else rank) / rank
        down = tensors["down"].astype(dtype)
        ups = {key: tensors["up"]}
        if key == "qkv" and isinstance(parent, dict) and "qkv" not in parent:
            ups = dict(zip("qkv", mx.split(tensors["up"], 3, axis=0)))
        for leaf, up in ups.items():
            up = up.astype(dtype)
            mx.eval(down, up)
            _set(parent, leaf, LoRALinear(_get(parent, leaf), down, up, scale, group))
    mx.clear_cache()
    return count


def apply_control(model, path: str | Path, strength: float = 1.0, dtype=mx.float32) -> int:
    """A colour/depth control adapter (darask0/MOSS-Image-color-depth-control): its video_embedder [dim, 3 x 128]
    replaces the DiT's, so the input becomes [latent | colour-map latent | depth-map latent] (dit.embed), and its
    LoRA wraps the block and text-embedder linears in group "control". strength scales the LoRA only, never the
    embedder (as Hayate). Returns the number of LoRA modules."""
    weights = mx.load(str(path))
    if "video_embedder.weight" not in weights:
        raise ValueError(f"{path}: no video_embedder, not a control adapter")
    w, b = weights["video_embedder.weight"], weights["video_embedder.bias"]
    old = model.video_embedder.weight
    if w.shape[0] != old.shape[0] or w.shape[1] <= old.shape[1] or w.shape[1] % old.shape[1]:
        raise ValueError(f"{path}: video_embedder {tuple(w.shape)} does not extend {tuple(old.shape)}")
    layer = nn.Linear(w.shape[1], w.shape[0])
    layer.weight, layer.bias = w.astype(old.dtype), b.astype(old.dtype)
    model.video_embedder = layer
    del weights
    return apply_lora(model, path, strength, dtype, "control", control=True)


def set_strength(model, strength: float, group: str | None = None):
    """Scale the adapters of `group` (all groups if None) relative to the strength they were applied with;
    0 = switched off."""
    for _, module in model.named_modules():
        if isinstance(module, LoRALinear) and (group is None or module.group == group):
            module.scale = module.base_scale * strength

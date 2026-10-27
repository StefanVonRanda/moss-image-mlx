"""MOSS-Image diffusion transformer in MLX.

Mirrors MOSSImageTransformer in the official pipeline_moss_image.py (OpenMOSS, Apache-2.0) operation by operation:
bf16 weights and residual stream, RMSNorm reduced in fp32 then cast back before the weight multiply, RoPE phases in
fp32 (computed on the CPU, cast to bf16), the timestep sinusoid in float64, and a single shared adaLN modulation
for every block. Parameter names equal the reference state-dict keys (except time_embedding, see sanitize).
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np


class RMSNorm(nn.Module):
    """torch: (x.float() * rsqrt(mean(x.float()**2) + 1e-6)).to(x.dtype) * weight"""

    def __init__(self, dim: int):
        super().__init__()
        self.weight = mx.ones((dim,))

    def __call__(self, x):
        return mx.fast.rms_norm(x.astype(mx.float32), None, 1e-6).astype(x.dtype) * self.weight


@mx.compile
def situ(hidden, gate):
    return (4.0 * mx.tanh(gate / 4.0) * mx.sigmoid(gate)) * (25.0 * mx.tanh(hidden / 25.0))


@mx.compile
def _norm_modulate(x, weight, scale, shift):
    """RMSNorm (fp32 reduction, cast, weight) then adaLN modulation, each op rounding like the reference."""
    return (mx.fast.rms_norm(x.astype(mx.float32), None, 1e-6).astype(x.dtype) * weight) * (1 + scale) + shift


@mx.compile
def _gated_add(x, y, gate):
    return x + y * gate


def silu_like_torch(x):
    """torch's fused silu on bf16 computes in fp32 and rounds once."""
    x32 = x.astype(mx.float32)
    return (x32 * mx.sigmoid(x32)).astype(x.dtype)


@mx.compile
def rotate(x, sin, cos):
    half = x.shape[-1] // 2
    return x * cos + mx.concatenate([-x[..., half:], x[..., :half]], axis=-1) * sin


def _sdpa(q, k, v, extra, hd: int):
    """Attention over this sequence's keys plus `extra` (k, v) of reference tokens, (B, heads, M, hd) each. Returns
    the output and this sequence's own (k, v), which a reference pass keeps for the edit."""
    kv = (k, v)
    if extra is not None:
        k, v = mx.concatenate([k, extra[0]], axis=2), mx.concatenate([v, extra[1]], axis=2)
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=hd**-0.5), kv


def attention(q, k, v, rope, heads: int, extra=None):
    """q, k, v: (B, N, dim); q and k are already per-head RMS-normalised. rope: (sin, cos), each (1, N, 1, hd).
    Returns the output and the rotated (k, v), see _sdpa."""
    b, n, dim = q.shape
    hd = dim // heads
    sin, cos = rope
    q = rotate(q.reshape(b, n, heads, hd), sin, cos).transpose(0, 2, 1, 3)
    k = rotate(k.reshape(b, n, heads, hd), sin, cos).transpose(0, 2, 1, 3)
    v = v.reshape(b, n, heads, hd).transpose(0, 2, 1, 3)
    out, kv = _sdpa(q, k, v, extra, hd)
    return out.transpose(0, 2, 1, 3).reshape(b, n, dim), kv


def _head_norm(norm: RMSNorm, x, heads: int):
    b, n, dim = x.shape
    return norm(x.reshape(b, n, heads, dim // heads)).reshape(b, n, dim)


# "Precise" mode: matmuls and attention keep bf16 inputs (speed), everything elementwise runs in fp32 and the
# residual stream stays fp32, so bf16 rounding enters only at matmul/attention inputs.
F32, BF16 = mx.float32, mx.bfloat16


def _norm32(norm: RMSNorm, x):
    return mx.fast.rms_norm(x.astype(F32), norm.weight.astype(F32), 1e-6)


def _head_norm32(norm: RMSNorm, x, heads: int):
    b, n, dim = x.shape
    return _norm32(norm, x.reshape(b, n, heads, dim // heads)).reshape(b, n, dim)


def _attention32(q, k, v, rope, heads: int, extra=None):
    """q, k fp32 (normalised), v bf16; RoPE in fp32, attention in bf16."""
    b, n, dim = q.shape
    hd = dim // heads
    sin, cos = rope
    q = rotate(q.reshape(b, n, heads, hd), sin, cos).astype(BF16).transpose(0, 2, 1, 3)
    k = rotate(k.reshape(b, n, heads, hd), sin, cos).astype(BF16).transpose(0, 2, 1, 3)
    v = v.reshape(b, n, heads, hd).transpose(0, 2, 1, 3)
    out, kv = _sdpa(q, k, v, extra, hd)
    return out.transpose(0, 2, 1, 3).reshape(b, n, dim), kv


class TextBlock(nn.Module):
    def __init__(self, dim: int, ffn_dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = RMSNorm(dim), RMSNorm(dim)
        self.q_norm, self.k_norm = RMSNorm(dim // heads), RMSNorm(dim // heads)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.ffn_gate, self.ffn_up = nn.Linear(dim, ffn_dim), nn.Linear(dim, ffn_dim)
        self.ffn_down = nn.Linear(ffn_dim, dim)

    def __call__(self, x, rope, precise: bool = False):
        if precise:
            q, k, v = mx.split(self.qkv(_norm32(self.norm1, x).astype(BF16)), 3, axis=-1)
            q, k = _head_norm32(self.q_norm, q, self.heads), _head_norm32(self.k_norm, k, self.heads)
            x = x + self.proj(_attention32(q, k, v, rope, self.heads)[0]).astype(F32)
            y = _norm32(self.norm2, x).astype(BF16)
            h = situ(self.ffn_up(y).astype(F32), self.ffn_gate(y).astype(F32))
            return x + self.ffn_down(h.astype(BF16)).astype(F32)
        q, k, v = mx.split(self.qkv(self.norm1(x)), 3, axis=-1)
        q, k = _head_norm(self.q_norm, q, self.heads), _head_norm(self.k_norm, k, self.heads)
        x = x + self.proj(attention(q, k, v, rope, self.heads)[0])
        y = self.norm2(x)
        return x + self.ffn_down(situ(self.ffn_up(y), self.ffn_gate(y)))


class TextProjection(nn.Module):
    def __init__(self, text_dim: int, dim: int, ffn_dim: int, heads: int):
        super().__init__()
        self.proj = nn.Linear(text_dim, dim)
        self.blocks = [TextBlock(dim, ffn_dim, heads) for _ in range(2)]

    def __call__(self, x, rope, precise: bool = False):
        x = self.proj(x.astype(BF16 if precise else self.proj.weight.dtype))
        if precise:
            x = x.astype(F32)
        for block in self.blocks:
            x = block(x, rope, precise)
        return x


class Block(nn.Module):
    def __init__(self, dim: int, ffn_dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = RMSNorm(dim), RMSNorm(dim)
        self.q_norm, self.k_norm = RMSNorm(dim // heads), RMSNorm(dim // heads)
        self.proj = {name: nn.Linear(dim, dim, bias=False) for name in ("q", "k", "v", "o")}
        self.proj["ffn_up_gate"] = nn.Linear(dim, 2 * ffn_dim, bias=False)
        self.proj["ffn_down"] = nn.Linear(ffn_dim, dim, bias=False)

    def __call__(self, x, rope, mod, precise: bool = False, extra=None, keep: bool = False):
        """extra: reference (k, v) this block's queries also attend to. keep: also return this block's own (k, v)."""
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = (mod[:, i : i + 1] for i in range(6))
        p = self.proj
        if precise:  # x and mod are fp32 here
            y = (_norm32(self.norm1, x) * (1 + scale_a) + shift_a).astype(BF16)
            q = _head_norm32(self.q_norm, p["q"](y), self.heads)
            k = _head_norm32(self.k_norm, p["k"](y), self.heads)
            out, kv = _attention32(q, k, p["v"](y), rope, self.heads, extra)
            x = x + p["o"](out).astype(F32) * gate_a
            y = (_norm32(self.norm2, x) * (1 + scale_f) + shift_f).astype(BF16)
            hidden, gate = mx.split(p["ffn_up_gate"](y).astype(F32), 2, axis=-1)
            x = x + p["ffn_down"](situ(hidden, gate).astype(BF16)).astype(F32) * gate_f
            return (x, kv) if keep else x
        y = _norm_modulate(x, self.norm1.weight, scale_a, shift_a)
        q = _head_norm(self.q_norm, p["q"](y), self.heads)
        k = _head_norm(self.k_norm, p["k"](y), self.heads)
        out, kv = attention(q, k, p["v"](y), rope, self.heads, extra)
        x = _gated_add(x, p["o"](out), gate_a)
        y = _norm_modulate(x, self.norm2.weight, scale_f, shift_f)
        hidden, gate = mx.split(p["ffn_up_gate"](y), 2, axis=-1)
        x = _gated_add(x, p["ffn_down"](situ(hidden, gate)), gate_f)
        return (x, kv) if keep else x


def rope_table(height: int, width: int, text_length: int, head_dim: int, base: float, dtype, f_offset: int = 0) -> tuple:
    """(sin, cos), each (1, N, 1, head_dim), for image tokens then text tokens; coordinates (r, t, h, w), every t
    moved by f_offset (see EDIT_FORMATS)."""
    h, w = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    image = np.stack([np.zeros_like(h), np.full_like(h, text_length), h, w], -1).reshape(-1, 4).astype(np.float32)
    image[:, 2] -= np.float32((height - 1) / 2)
    image[:, 3] -= np.float32((width - 1) / 2)
    text = np.zeros((text_length, 4), dtype=np.float32)
    text[:, 1] = np.arange(text_length, dtype=np.float32)
    coords = np.concatenate([image, text], 0)
    coords[:, 1] += np.float32(f_offset)
    return _rope(coords, head_dim, base, dtype)


# Where the reference latents sit on the t ("frame") axis of the 4-D RoPE, per edit LoRA format. Both put reference i
# at r = i + 1 with centred rows and columns, at timestep 0, attending only to references; target and text attend to
# them too. With L = text length:
# - "chinoll" (chinoll/Moss-Image-Edit, ComfyUI's index_timestep_zero_text_start): references at t = 0, the text's
#   start; target at t = L, text at t = 0 .. L-1 (the stock layout).
# - "ai-toolkit" (MOSS-Image-Edit-HQ, CharRef; index_timestep_zero_isolated): references at t = L, the target's frame
#   (ai-toolkit/ostris: _image_ids(i + 1, text_len, ..)), target at t = L, text at 0 .. L-1. RoPE scores depend only on
#   coordinate differences, so subtracting L from every t gives the same attention: target t = 0, text t = -L .. -1,
#   references t = 0 (ComfyUI-Moss-Image-Edit-Native's body_rope f_offset = -text_len). That form is used here: the
#   references' keys stay independent of the prompt, so one reference_cache serves both CFG branches, whose L differ.
EDIT_FORMATS = ("chinoll", "ai-toolkit")


def reference_rope(height: int, width: int, index: int, head_dim: int, base: float, dtype, frame: int = 0) -> tuple:
    """(sin, cos) for the latents of reference image `index` (0-based): coordinates (index + 1, frame, h, w), centred
    like the target's. The target keeps r = 0. frame stays 0 for both formats (see EDIT_FORMATS)."""
    h, w = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    coords = np.stack([np.full_like(h, index + 1), np.full_like(h, frame), h, w], -1).reshape(-1, 4).astype(np.float32)
    coords[:, 2] -= np.float32((height - 1) / 2)
    coords[:, 3] -= np.float32((width - 1) / 2)
    return _rope(coords, head_dim, base, dtype)


def _rope(coords: np.ndarray, head_dim: int, base: float, dtype) -> tuple:
    bands_n = head_dim // 8
    exponent = np.arange(bands_n, dtype=np.float32) / np.float32(bands_n)
    bands = (np.float32(1) / (np.float32(base) ** exponent)).astype(np.float32)
    phase = coords[..., None] * bands  # (N, 4, bands)
    n = phase.shape[0]
    sin = np.sin(phase).reshape(n, -1)  # (N, head_dim / 2): axes r, t, h, w
    cos = np.cos(phase).reshape(n, -1)
    sin = mx.array(np.concatenate([sin, sin], -1)).astype(dtype)[None, :, None, :]
    cos = mx.array(np.concatenate([cos, cos], -1)).astype(dtype)[None, :, None, :]
    return sin, cos


def timestep_embedding(timestep: float, dim: int) -> np.ndarray:
    """float64 sinusoid as in the reference (cos first), returned as float32."""
    t = np.float64(np.float32(timestep))
    half = dim // 2
    freq = 10000.0 ** (-np.arange(half, dtype=np.float64) / half)
    phase = t * freq
    return np.concatenate([np.cos(phase), np.sin(phase)]).astype(np.float32)[None]


class MOSSImageTransformer(nn.Module):
    def __init__(self, dim=3072, ffn_dim=8192, num_heads=24, num_layers=42, text_dim=4096, freq_dim=256,
                 latent_channels=128, rope_base=10000.0):
        super().__init__()
        self.config = dict(dim=dim, ffn_dim=ffn_dim, num_heads=num_heads, num_layers=num_layers, text_dim=text_dim,
                           freq_dim=freq_dim, latent_channels=latent_channels, rope_base=rope_base)
        self.video_embedder = nn.Linear(latent_channels, dim)
        self.text_embedder = TextProjection(text_dim, dim, ffn_dim, num_heads)
        self.time_embedding = [nn.Linear(freq_dim, dim), nn.Linear(dim, dim)]
        self.shared_block_modulation = nn.Linear(dim, 6 * dim)
        self.layers = [Block(dim, ffn_dim, num_heads) for _ in range(num_layers)]
        self.final_norm_video = RMSNorm(dim)
        self.final_linear_video = nn.Linear(dim, latent_channels, bias=False)
        self.video_head_modulation = nn.Linear(dim, 2 * dim)
        self._rope_cache = {}

    def modulation(self, timestep: float):
        """Shared block modulation (1, 6, dim) and head modulation (1, 2, dim), both in the weight dtype."""
        dtype = self.time_embedding[0].weight.dtype
        emb = mx.array(timestep_embedding(timestep, self.config["freq_dim"])).astype(dtype)
        time = self.time_embedding[1](silu_like_torch(self.time_embedding[0](emb)))
        dim = self.config["dim"]
        mod = self.shared_block_modulation(time).reshape(1, 6, dim)
        head = self.video_head_modulation(time).reshape(1, 2, dim)
        return mod, head

    def rope(self, height: int, width: int, text_length: int, dtype, f_offset: int = 0):
        key = (height, width, text_length, dtype, f_offset)
        if key not in self._rope_cache:
            self._rope_cache = {key: rope_table(height, width, text_length, self.config["dim"] // self.config["num_heads"],
                                                self.config["rope_base"], dtype, f_offset)}
        return self._rope_cache[key]

    def reference_cache(self, references: list) -> list:
        """Per block, the (k, v) of the reference latents (each (1, 128, h, w)), for an edit with MOSS Image Edit.

        References are clean (timestep 0) and attend only to each other, so their keys and values depend on nothing
        else: computed once here, they serve every step and both CFG branches. Memory: 2 x tokens x dim per block,
        2.1 GB for one 1024^2 reference in bf16."""
        dtype = self.video_embedder.weight.dtype
        precise = bool(getattr(self, "precise", False)) and dtype != F32
        act = F32 if precise else dtype
        mod, _ = self.modulation(0.0)
        hd = self.config["dim"] // self.config["num_heads"]
        tokens, sins, coss = [], [], []
        for index, z in enumerate(references):
            _, c, height, width = z.shape
            tokens.append(self.embed(z))  # a control embedder sees zero control: references carry no map
            sin, cos = reference_rope(height, width, index, hd, self.config["rope_base"], act)
            sins.append(sin); coss.append(cos)
        x, rope = mx.concatenate(tokens, axis=1), (mx.concatenate(sins, axis=1), mx.concatenate(coss, axis=1))
        if precise:
            x, mod = x.astype(F32), mod.astype(F32)
        cache = []
        for layer in self.layers:
            x, kv = layer(x, rope, mod, precise, keep=True)
            mx.eval(x, kv)  # one block at a time: the graph of all 42 would hold every activation at once
            cache.append(kv)
        return cache

    def embed(self, z, control=None):
        """Image tokens (1, h * w, dim) for latents z (1, 128, h, w). A control embedder (lora.apply_control: input
        wider than 128) reads the channel concat [z | control], control (1, extra, h, w) = colour-map and depth-map
        latents; None means zeros, i.e. no control, which is also what references get."""
        _, c, height, width = z.shape
        extra = self.video_embedder.weight.shape[1] - c
        if control is not None and not extra:
            raise ValueError("control latents need a control adapter (--control)")
        if extra:
            control = mx.zeros((1, extra, height, width), z.dtype) if control is None else control.astype(z.dtype)
            z = mx.concatenate([z, control], axis=1)
        x = z.reshape(1, z.shape[1], height * width).transpose(0, 2, 1)
        return self.video_embedder(x.astype(self.video_embedder.weight.dtype))

    def __call__(self, latents, timestep: float, context, skip_layers=(), references=None, control=None,
                 edit_format="chinoll"):
        """latents (1, 128, H, W) fp32; context (1, L, text_dim) with no padding. references: reference_cache(), laid
        out as edit_format (EDIT_FORMATS). control: (1, 256, H, W) colour and depth latents for a control adapter, see
        embed. Returns velocity (1, 128, H, W)."""
        b, c, height, width = latents.shape
        if b != 1 or context.shape[0] != 1:
            raise ValueError("The reference evaluates each sample separately; pass batch 1.")
        dtype = self.video_embedder.weight.dtype
        precise = bool(getattr(self, "precise", False)) and dtype != F32
        mod, head = self.modulation(timestep)
        video = self.embed(latents, control)
        act = F32 if precise else dtype
        if precise:
            mod, head, video = mod.astype(F32), head.astype(F32), video.astype(F32)
        f_offset = -context.shape[1] if references and edit_format == "ai-toolkit" else 0
        sin, cos = self.rope(height, width, context.shape[1], act, f_offset)
        n_img = height * width
        text = self.text_embedder(context, (sin[:, n_img:], cos[:, n_img:]), precise)
        x = mx.concatenate([video, text], axis=1)
        skip = set(skip_layers or ())
        for index, layer in enumerate(self.layers):
            if index not in skip:
                x = layer(x, (sin, cos), mod, precise, references[index] if references else None)
        if precise:
            x = _norm32(self.final_norm_video, x[:, :n_img]) * (1 + head[:, 1:2]) + head[:, 0:1]
            x = self.final_linear_video(x.astype(BF16))
        else:
            x = self.final_norm_video(x[:, :n_img])
            x = x * (1 + head[:, 1:2]) + head[:, 0:1]
            x = self.final_linear_video(x)
        return x.transpose(0, 2, 1).reshape(1, c, height, width)

    @staticmethod
    def sanitize(weights: dict) -> dict:
        out = {}
        for key, value in weights.items():
            if key == "rope.bands":  # derived constant, recomputed in fp32
                continue
            out[key.replace("time_embedding.2.", "time_embedding.1.")] = value
        return out


def load_transformer(path: str | Path, dtype=None) -> MOSSImageTransformer:
    """Load a `transformer/` directory: the original diffusers layout, or an exported MLX one (possibly quantized,
    marked by "moss_image_mlx" in config.json)."""
    path = Path(path)
    config = json.loads((path / "config.json").read_text())
    keys = ("dim", "ffn_dim", "num_heads", "num_layers", "text_dim", "freq_dim", "latent_channels", "rope_base")
    model = MOSSImageTransformer(**{k: config[k] for k in keys if k in config})
    exported = config.get("moss_image_mlx")
    if exported and exported.get("quantization"):
        from .quantize import quantize_dit

        quantize_dit(model, exported["quantization"])  # build QuantizedLinear modules before loading their weights
    weights = {}
    for file in sorted(path.glob("*.safetensors")):
        weights.update(mx.load(str(file)))
    if not exported:
        weights = model.sanitize(weights)
    if dtype is not None:
        weights = {k: v.astype(dtype) if mx.issubdtype(v.dtype, mx.floating) else v for k, v in weights.items()}
    model.load_weights(list(weights.items()), strict=True)
    return model

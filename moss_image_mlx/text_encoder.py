"""MOSS-Image text encoder: the Qwen3.5-9B language backbone (mlx-lm `qwen3_5`), final normalised hidden states.

The reference runs transformers' Qwen3_5Model on the chat-templated prompt (one prompt at a time, no padding,
`enable_thinking=True`, generation prompt appended) and takes `last_hidden_state`. Only the language path is
needed; the vision tower in the checkpoint is skipped.

Fidelity detail: Qwen3.5's RMSNorms multiply by (1 + weight) in fp32. mlx-lm folds the 1 into a bf16 weight,
which rounds (1 + w) to 8 bits of mantissa (up to 0.4% per-channel scale error at every norm). Here those norms
keep an fp32 (1 + w) and compute in fp32 like the reference, then cast back.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.gated_delta import normalize_qk
from mlx_lm.models.qwen3_5 import GatedDeltaNet, Qwen3_5TextModel, TextModelArgs

MAX_TOKENS = 4096


def _silu_like_torch(x):
    x32 = x.astype(mx.float32)
    return (x32 * mx.sigmoid(x32)).astype(x.dtype)


class ReferenceGatedDeltaNet(GatedDeltaNet):
    """Gated DeltaNet with the reference's precision (transformers' torch_chunk_gated_delta_rule): q/k L2-norm,
    decay, beta and the recurrence in fp32 from the bf16 projections; output cast back to bf16; gated RMSNorm as
    Qwen3_5RMSNormGated. mlx-lm's default path normalises q/k and adds `a + dt_bias` in bf16, which drifted up to
    0.9 cosine on single tokens of a 650-token prompt. Prefill only (no cache)."""

    def __call__(self, inputs, mask=None, cache=None):
        if cache is not None:
            raise NotImplementedError("the diffusion text encoder never generates tokens")
        f32 = mx.float32
        B, S, _ = inputs.shape
        qkv = self.in_proj_qkv(inputs)
        z = self.in_proj_z(inputs).reshape(B, S, self.num_v_heads, self.head_v_dim)
        b, a = self.in_proj_b(inputs), self.in_proj_a(inputs)
        conv_in = mx.concatenate([mx.zeros((B, self.conv_kernel_size - 1, self.conv_dim), dtype=inputs.dtype), qkv], 1)
        conv_out = _silu_like_torch(self.conv1d(conv_in))
        q, k, v = [
            t.reshape(B, S, h, d).astype(f32)
            for t, h, d in zip(mx.split(conv_out, [self.key_dim, 2 * self.key_dim], -1),
                               [self.num_k_heads, self.num_k_heads, self.num_v_heads],
                               [self.head_k_dim, self.head_k_dim, self.head_v_dim])
        ]
        q, k = normalize_qk(q, k, inv_scale=self.head_k_dim**-0.5, eps=1e-6)
        beta = mx.sigmoid(b).astype(f32)  # torch: b.sigmoid() in bf16, cast to fp32 in the kernel
        log_decay = -mx.exp(self.A_log.astype(f32)) * nn.softplus(a.astype(f32) + self.dt_bias.astype(f32))
        out, _ = _gated_delta(q, k, v, mx.exp(log_decay), beta)
        out = out.astype(inputs.dtype)
        # Qwen3_5RMSNormGated: normalise in fp32, cast, * weight (bf16), * silu(gate in fp32), cast.
        h = mx.fast.rms_norm(out.astype(f32), None, self.norm.eps).astype(inputs.dtype) * self.norm.weight
        h = (h.astype(f32) * nn.silu(z.astype(f32))).astype(inputs.dtype)
        return self.out_proj(h.reshape(B, S, -1))


class ReferenceAttention(nn.Module):
    """Qwen3_5Attention with transformers' rounding points: per-head (1 + w) RMSNorm on q/k, RoPE on the first
    `rot_dim` dims with cos/sin computed in fp32 and cast to bf16 then applied with bf16 multiplies, causal SDPA,
    output * sigmoid(gate) in bf16. Replaces mlx-lm's fused RoPE (fp32 internally) via __class__ swap."""

    def __call__(self, x, mask=None, cache=None):
        if cache is not None:
            raise NotImplementedError
        B, L, _ = x.shape
        q, gate = mx.split(self.q_proj(x).reshape(B, L, self.num_attention_heads, -1), 2, axis=-1)
        gate = gate.reshape(B, L, -1)
        q = self.q_norm(q).transpose(0, 2, 1, 3)
        k = self.k_norm(self.k_proj(x).reshape(B, L, self.num_key_value_heads, -1)).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, self.num_key_value_heads, -1).transpose(0, 2, 1, 3)
        cos, sin = _rope_cos_sin(L, self.rot_dim, self.rope_theta, x.dtype)
        q, k = _apply_rope(q, cos, sin, self.rot_dim), _apply_rope(k, cos, sin, self.rot_dim)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask="causal")
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1) * mx.sigmoid(gate)
        return self.o_proj(out)


class ReferenceMLP(nn.Module):
    """Qwen3_5MLP: down(silu(gate(x)) * up(x)), with silu rounded to bf16 before the multiply (as torch does)."""

    def __call__(self, x):
        return self.down_proj(_silu_like_torch(self.gate_proj(x)) * self.up_proj(x))


_ROPE_CACHE: dict = {}


def _rope_cos_sin(length: int, rot_dim: int, theta: float, dtype):
    key = (length, rot_dim, theta, dtype)
    if key not in _ROPE_CACHE:
        import numpy as np

        inv_freq = (1.0 / (np.float32(theta) ** (np.arange(0, rot_dim, 2).astype(np.float32) / np.float32(rot_dim))))
        freqs = np.arange(length, dtype=np.float32)[:, None] * inv_freq.astype(np.float32)[None, :]
        emb = np.concatenate([freqs, freqs], -1)
        _ROPE_CACHE.clear()
        _ROPE_CACHE[key] = (mx.array(np.cos(emb)).astype(dtype), mx.array(np.sin(emb)).astype(dtype))
    return _ROPE_CACHE[key]


def _apply_rope(x, cos, sin, rot_dim: int):
    rot, rest = x[..., :rot_dim], x[..., rot_dim:]
    half = rot_dim // 2
    rotated = mx.concatenate([-rot[..., half:], rot[..., :half]], axis=-1)
    return mx.concatenate([rot * cos + rotated * sin, rest], axis=-1)


def _gated_delta(q, k, v, decay, beta):
    """Run mlx-lm's recurrence (Metal kernel when available) on precomputed fp32 decay and beta."""
    from mlx_lm.models import gated_delta as gd

    B, _, _, Dk = q.shape
    Hv, Dv = v.shape[-2:]
    state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)
    if mx.default_device() == mx.gpu and mx.metal.is_available() and Dk % 32 == 0:
        return gd.gated_delta_kernel(q, k, v, decay, beta, state, None)
    return gd.gated_delta_ops(q, k, v, decay, beta, state, None)


class OffsetRMSNorm(nn.Module):
    """Qwen3.5 RMSNorm: (x.float() normalised * (1 + w).float()).to(x.dtype). Stores (1 + w) in fp32."""

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((dim,), dtype=mx.float32)

    def __call__(self, x):
        return mx.fast.rms_norm(x.astype(mx.float32), self.weight, self.eps).astype(x.dtype)


def _use_reference_layers(model: Qwen3_5TextModel, args: TextModelArgs):
    """Swap mlx-lm's fused layers for transformers-faithful ones (same parameters, different rounding points)."""
    for layer in model.layers:
        layer.mlp.__class__ = ReferenceMLP
        if layer.is_linear:
            layer.linear_attn.__class__ = ReferenceGatedDeltaNet
        else:
            attn = layer.self_attn
            attn.__class__ = ReferenceAttention
            attn.rot_dim = int(attn.head_dim * args.partial_rotary_factor)
            attn.rope_theta = float(args.rope_theta)
    _replace_offset_norms(model, args.rms_norm_eps)


def _replace_offset_norms(model: Qwen3_5TextModel, eps: float):
    for layer in model.layers:
        layer.input_layernorm = OffsetRMSNorm(layer.input_layernorm.weight.shape[0], eps)
        layer.post_attention_layernorm = OffsetRMSNorm(layer.post_attention_layernorm.weight.shape[0], eps)
        if not layer.is_linear:
            attn = layer.self_attn
            attn.q_norm = OffsetRMSNorm(attn.q_norm.weight.shape[0], eps)
            attn.k_norm = OffsetRMSNorm(attn.k_norm.weight.shape[0], eps)
    model.norm = OffsetRMSNorm(model.norm.weight.shape[0], eps)


OFFSET_SUFFIXES = (".input_layernorm.weight", ".post_attention_layernorm.weight", ".q_norm.weight", ".k_norm.weight")


def sanitize(weights: dict) -> dict:
    """Reference Qwen3_5Model keys -> Qwen3_5TextModel keys, with conv1d moved to MLX layout and (1 + w) norms."""
    out = {}
    for key, value in weights.items():
        if not key.startswith("language_model."):
            continue  # vision tower
        key = key.removeprefix("language_model.")
        if key.endswith("conv1d.weight") and value.shape[-1] != 1:
            value = value.moveaxis(2, 1)
        if key == "norm.weight" or key.endswith(OFFSET_SUFFIXES):
            value = value.astype(mx.float32) + 1.0
        out[key] = value
    return out


class TextEncoder:
    def __init__(self, text_encoder_dir: str | Path, tokenizer_dir: str | Path):
        from transformers import AutoTokenizer

        config = json.loads((Path(text_encoder_dir) / "config.json").read_text())
        args = TextModelArgs.from_dict(config["text_config"])
        self.model = Qwen3_5TextModel(args)
        _use_reference_layers(self.model, args)
        exported = config.get("moss_image_mlx")  # exported MLX layout: already sanitised, maybe quantized
        if exported and exported.get("quantization"):
            from .quantize import quantize_text_encoder

            quantize_text_encoder(self.model, exported["quantization"])
        weights = {}
        for file in sorted(Path(text_encoder_dir).glob("*.safetensors")):
            weights.update(mx.load(str(file)))
        self.model.load_weights(list((weights if exported else sanitize(weights)).items()), strict=True)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))

    def token_ids(self, prompt: str, max_tokens: int = MAX_TOKENS) -> list[int]:
        text = self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                                  add_generation_prompt=True, enable_thinking=True)
        return self.tokenizer(text, truncation=True, max_length=max_tokens)["input_ids"]

    def encode(self, prompt: str, max_tokens: int = MAX_TOKENS) -> mx.array:
        """(1, L, 4096) bf16 final hidden states for one prompt, unpadded."""
        ids = mx.array(self.token_ids(prompt, max_tokens))[None]
        hidden = self.model(ids)
        mx.eval(hidden)
        return hidden

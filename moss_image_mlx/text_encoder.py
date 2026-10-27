"""MOSS-Image text encoder: the Qwen3.5-9B language backbone (mlx-lm `qwen3_5`), final normalised hidden states.

The reference runs transformers' Qwen3_5Model on the chat-templated prompt (one prompt at a time, no padding,
`enable_thinking=True`, generation prompt appended) and takes `last_hidden_state`. Plain prompts need only the
language path. The ai-toolkit edit format (MOSS-Image-Edit-HQ) also shows the encoder its reference images, as
"Picture N: <|vision_start|><|image_pad|><|vision_end|>" before the prompt: those run through Qwen3.5's vision tower
(VisionTower, ported from mlx-vlm's qwen3_5; weights from the official text_encoder, the MLX tiers ship none), whose
outputs replace the expanded <|image_pad|> tokens, with Qwen3.5's 3-D (t, h, w) M-RoPE positions.

Fidelity detail: Qwen3.5's RMSNorms multiply by (1 + weight) in fp32. mlx-lm folds the 1 into a bf16 weight,
which rounds (1 + w) to 8 bits of mantissa (up to 0.4% per-channel scale error at every norm). Here those norms
keep an fp32 (1 + w) and compute in fp32 like the reference, then cast back.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

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
        positions = getattr(self, "positions", None)  # (3, L) M-RoPE positions of a prompt with images
        if positions is None:
            cos, sin = _rope_cos_sin(L, self.rot_dim, self.rope_theta, x.dtype)
        else:
            cos, sin = _mrope_cos_sin(positions, self.rot_dim, self.rope_theta, self.mrope_section, x.dtype)
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


def _mrope_cos_sin(positions: np.ndarray, rot_dim: int, theta: float, section, dtype):
    """Qwen3.5's interleaved M-RoPE (Qwen3_5TextRotaryEmbedding): per frequency slot i, the t position, except
    slots i = 1, 4, .. < 3 * section[1] take h and i = 2, 5, .. < 3 * section[2] take w. Equal rows = plain RoPE."""
    inv_freq = 1.0 / (np.float32(theta) ** (np.arange(0, rot_dim, 2).astype(np.float32) / np.float32(rot_dim)))
    freqs = np.asarray(positions, dtype=np.float32)[..., None] * inv_freq.astype(np.float32)  # (3, L, rot_dim / 2)
    f = freqs[0].copy()
    for axis in (1, 2):
        f[:, axis : section[axis] * 3 : 3] = freqs[axis][:, axis : section[axis] * 3 : 3]
    emb = np.concatenate([f, f], -1)
    return mx.array(np.cos(emb)).astype(dtype), mx.array(np.sin(emb)).astype(dtype)


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
            attn.mrope_section = (getattr(args, "rope_parameters", None) or {}).get("mrope_section", [11, 11, 10])
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


# Vision (ai-toolkit edit format) ------------------------------------------------------------------------------------
VISION_BLOCK = "<|vision_start|><|image_pad|><|vision_end|>"
IMAGE_PAD = "<|image_pad|>"
# Qwen3.5's preprocessor_config.json (Qwen2VLImageProcessorFast): 16 px patches, 2x2 merged into one token, two
# identical frames per image, mean = std = 0.5, the size rounded to 32 px within these pixel bounds.
PATCH, MERGE, FRAMES = 16, 2, 2
VISION_MIN_PIXELS, VISION_MAX_PIXELS = 65536, 16777216
VISION_REPO, VISION_SHARD = "OpenMOSS-Team/MOSS-Image-SFT", "text_encoder/model-00004-of-00004.safetensors"


def vision_prompt(prompt: str, images: int) -> str:
    """ai-toolkit's edit prompt: "Picture N: <vision block>" for each image, then the prompt (inside the chat
    template's user turn)."""
    return "".join(f"Picture {i + 1}: {VISION_BLOCK}" for i in range(images)) + prompt


def smart_resize(height: int, width: int, factor: int = PATCH * MERGE, min_pixels: int = VISION_MIN_PIXELS,
                 max_pixels: int = VISION_MAX_PIXELS) -> tuple[int, int]:
    """transformers' qwen2_vl smart_resize: sides rounded to `factor`, area kept within [min_pixels, max_pixels]."""
    h, w = round(height / factor) * factor, round(width / factor) * factor
    if h * w > max_pixels:
        beta = math.sqrt(height * width / max_pixels)
        h, w = max(factor, math.floor(height / beta / factor) * factor), max(factor, math.floor(width / beta / factor) * factor)
    elif h * w < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h, w = math.ceil(height * beta / factor) * factor, math.ceil(width * beta / factor) * factor
    return h, w


def vision_patches(pixels: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    """uint8 RGB (H, W, 3) -> (patches (gh * gw, 3 * 2 * 16 * 16) float32, (gh, gw) in 16 px patches), as the
    Qwen2-VL image processor lays them out: the frame doubled, each 2x2 merge group contiguous, (C, T, P, P) per row."""
    from PIL import Image

    h, w = smart_resize(*pixels.shape[:2])
    if (h, w) != pixels.shape[:2]:
        pixels = np.asarray(Image.fromarray(pixels).resize((w, h), Image.BICUBIC))
    x = (pixels.astype(np.float32) / 255 - 0.5) / 0.5
    gh, gw = h // PATCH, w // PATCH
    x = np.broadcast_to(x.transpose(2, 0, 1)[None], (FRAMES, 3, h, w))
    x = x.reshape(FRAMES, 3, gh // MERGE, MERGE, PATCH, gw // MERGE, MERGE, PATCH).transpose(2, 5, 3, 6, 1, 0, 4, 7)
    return np.ascontiguousarray(x.reshape(gh * gw, 3 * FRAMES * PATCH * PATCH)), (gh, gw)


def mrope_positions(ids, image_token_id: int, grids) -> np.ndarray:
    """(3, L) Qwen3.5 position ids (Qwen3_5Model.get_rope_index): text tokens count up on all three axes; each run
    of image tokens (merged grid gh x gw, in `grids` order) sits at t = p, h = p + row, w = p + column, where p is
    the next text position, and the text after it resumes at p + max(gh, gw)."""
    ids, out, pos, i, grids = list(ids), [], 0, 0, iter(grids)
    while i < len(ids):
        if ids[i] != image_token_id:
            out.append(np.full((3, 1), pos))
            pos, i = pos + 1, i + 1
            continue
        gh, gw = next(grids)
        rows, cols = np.repeat(np.arange(gh), gw), np.tile(np.arange(gw), gh)
        out.append(np.stack([np.zeros(gh * gw, int), rows, cols]) + pos)
        pos, i = pos + max(gh, gw), i + gh * gw
    return np.concatenate(out, 1)


class _VisionAttention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def __call__(self, x, cos, sin):
        n, dim = x.shape
        qkv = self.qkv(x).reshape(n, 3, self.heads, -1).transpose(1, 2, 0, 3)  # (3, heads, n, hd)

        def rope(t):  # fp32, cast back (apply_rotary_pos_emb_vision)
            t32 = t.astype(mx.float32)
            half = t.shape[-1] // 2
            return (t32 * cos + mx.concatenate([-t32[..., half:], t32[..., :half]], -1) * sin).astype(t.dtype)

        q, k, v = rope(qkv[0])[None], rope(qkv[1])[None], qkv[2][None]
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=(dim // self.heads) ** -0.5)  # full, one image
        return self.proj(out[0].transpose(1, 0, 2).reshape(n, dim))


class _VisionMLP(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.linear_fc1, self.linear_fc2 = nn.Linear(dim, hidden), nn.Linear(hidden, dim)

    def __call__(self, x):
        return self.linear_fc2(nn.gelu_approx(self.linear_fc1(x)))  # gelu_pytorch_tanh


class _VisionBlock(nn.Module):
    def __init__(self, dim: int, hidden: int, heads: int):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(dim, eps=1e-6), nn.LayerNorm(dim, eps=1e-6)
        self.attn, self.mlp = _VisionAttention(dim, heads), _VisionMLP(dim, hidden)

    def __call__(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        return x + self.mlp(self.norm2(x))


class _VisionMerger(nn.Module):
    def __init__(self, dim: int, out: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.linear_fc1 = nn.Linear(dim * MERGE**2, dim * MERGE**2)
        self.linear_fc2 = nn.Linear(dim * MERGE**2, out)

    def __call__(self, x):
        x = self.norm(x).reshape(-1, self.linear_fc1.weight.shape[1])
        return self.linear_fc2(nn.gelu(self.linear_fc1(x)))


class VisionTower(nn.Module):
    """Qwen3.5's vision encoder (mlx-vlm qwen3_vl.VisionModel, as qwen3_5 uses it; no deepstack): one image of
    patches -> one 4096-d embedding per merged 2x2 group. The Conv3d patch embedding (kernel = stride = its input) is
    the equivalent Linear over the (C, T, P, P) patch rows."""

    def __init__(self, config: dict):
        super().__init__()
        dim, heads = config["hidden_size"], config["num_heads"]
        self.patch_proj = nn.Linear(config["in_channels"] * config["temporal_patch_size"] * config["patch_size"] ** 2, dim)
        self.pos_embed = nn.Embedding(config["num_position_embeddings"], dim)
        self.side = int(config["num_position_embeddings"] ** 0.5)
        self.blocks = [_VisionBlock(dim, config["intermediate_size"], heads) for _ in range(config["depth"])]
        self.merger = _VisionMerger(dim, config["out_hidden_size"])
        self.rot_dim = dim // heads // 2

    def _position_embedding(self, gh: int, gw: int):
        """The learned 48x48 table resampled bilinearly (align_corners) to gh x gw, fp32 weights, in merge order."""
        n = self.side
        hs, ws = np.linspace(0, n - 1, gh, dtype=np.float32), np.linspace(0, n - 1, gw, dtype=np.float32)
        h0, w0 = hs.astype(np.int32), ws.astype(np.int32)
        h1, w1 = np.minimum(h0 + 1, n - 1), np.minimum(w0 + 1, n - 1)
        dh, dw = (hs - h0)[:, None], (ws - w0)[None, :]
        idx = [(a[:, None] * n + b[None, :]).reshape(-1) for a in (h0, h1) for b in (w0, w1)]
        wts = [((1 - dh) * (1 - dw)), ((1 - dh) * dw), (dh * (1 - dw)), (dh * dw)]
        emb = sum(self.pos_embed(mx.array(i)).astype(mx.float32) * mx.array(wt.reshape(-1, 1)) for i, wt in zip(idx, wts))
        emb = emb.reshape(gh // MERGE, MERGE, gw // MERGE, MERGE, -1).transpose(0, 2, 1, 3, 4)
        return emb.reshape(gh * gw, -1)

    def _rotary(self, gh: int, gw: int):
        """2-D RoPE (cos, sin) per patch in merge order: half the dims rotate with the row, half with the column."""
        inv_freq = 1.0 / (10000.0 ** (np.arange(0, self.rot_dim, 2, dtype=np.float32) / np.float32(self.rot_dim)))
        rows = (np.arange(gh // MERGE)[:, None, None, None] * MERGE + np.arange(MERGE)[None, None, :, None])
        cols = (np.arange(gw // MERGE)[None, :, None, None] * MERGE + np.arange(MERGE)[None, None, None, :])
        rows, cols = (np.broadcast_to(a, (gh // MERGE, gw // MERGE, MERGE, MERGE)).reshape(-1) for a in (rows, cols))
        emb = np.concatenate([rows[:, None] * inv_freq, cols[:, None] * inv_freq], -1).astype(np.float32)
        emb = np.concatenate([emb, emb], -1)
        return mx.array(np.cos(emb)), mx.array(np.sin(emb))

    def __call__(self, patches: mx.array, grid: tuple[int, int]) -> mx.array:
        gh, gw = grid
        x = self.patch_proj(patches.astype(self.patch_proj.weight.dtype))
        x = x + self._position_embedding(gh, gw).astype(x.dtype)
        cos, sin = self._rotary(gh, gw)
        for block in self.blocks:
            x = block(x, cos, sin)
        return self.merger(x)

    @staticmethod
    def sanitize(weights: dict) -> dict:
        """Official checkpoint keys (visual.*) -> VisionTower keys; the Conv3d kernel flattened to the Linear's."""
        out = {}
        for key, value in weights.items():
            if not key.startswith("visual."):
                continue
            key = key.removeprefix("visual.")
            if key.startswith("patch_embed.proj."):
                key = key.replace("patch_embed.proj.", "patch_proj.")
                value = value.reshape(value.shape[0], -1) if key.endswith("weight") else value
            out[key] = value
        return out


def load_vision_tower() -> VisionTower:
    """The vision tower from the official text_encoder (its last shard holds every visual.* tensor; downloaded once,
    1.9 GB), run in fp32 (1.7 GB): in bf16 its 27 blocks amplify rounding to 4% relative error against an fp32
    reference (transformers' Qwen3_5VisionModel), and mlx-vlm's bf16 run to 5% (worst token cosine 0.71); in fp32
    it is 0.3%. It runs once per image and prompt, and the result is cached."""
    from huggingface_hub import hf_hub_download

    config = json.loads(Path(hf_hub_download(VISION_REPO, "text_encoder/config.json")).read_text())["vision_config"]
    tower = VisionTower(config)
    weights = VisionTower.sanitize(mx.load(hf_hub_download(VISION_REPO, VISION_SHARD)))
    weights = {k: v.astype(mx.float32) for k, v in weights.items()}
    tower.load_weights(list(weights.items()), strict=True)
    mx.eval(tower.parameters())
    tower.eval()
    return tower


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
        self.image_token_id = config.get("image_token_id", 248056)
        self.vision = None  # loaded on the first prompt with images

    def token_ids(self, prompt: str, max_tokens: int = MAX_TOKENS, image_tokens=()) -> list[int]:
        """image_tokens: per <|image_pad|> in the prompt, how many tokens it expands to (as the processor does)."""
        text = self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                                  add_generation_prompt=True, enable_thinking=True)
        parts = text.split(IMAGE_PAD)
        if len(parts) - 1 != len(image_tokens):
            raise ValueError(f"the prompt has {len(parts) - 1} {IMAGE_PAD} placeholders for {len(image_tokens)} images")
        text = parts[0] + "".join(IMAGE_PAD * n + part for n, part in zip(image_tokens, parts[1:]))
        ids = self.tokenizer(text, truncation=True, max_length=max_tokens)["input_ids"]
        if ids.count(self.image_token_id) != sum(image_tokens):
            raise ValueError(f"the prompt and its images exceed {max_tokens} tokens")
        return ids

    def encode(self, prompt: str, max_tokens: int = MAX_TOKENS, images=()) -> mx.array:
        """(1, L, 4096) bf16 final hidden states for one prompt, unpadded. images: uint8 RGB arrays, one per
        <|image_pad|> placeholder in the prompt, in order (see vision_prompt)."""
        if not images:
            hidden = self.model(mx.array(self.token_ids(prompt, max_tokens))[None])
            mx.eval(hidden)
            return hidden
        if self.vision is None:
            self.vision = load_vision_tower()
        patches = [vision_patches(np.asarray(image)) for image in images]
        ids = self.token_ids(prompt, max_tokens, [gh * gw // MERGE**2 for _, (gh, gw) in patches])
        features = mx.concatenate([self.vision(mx.array(p), grid) for p, grid in patches])
        is_image = np.array(ids) == self.image_token_id
        source = mx.array(np.maximum(np.cumsum(is_image) - 1, 0))
        tokens = mx.array(ids)[None]
        embeds = self.model.embed_tokens(tokens)[0]
        embeds = mx.where(mx.array(is_image)[:, None], features[source].astype(embeds.dtype), embeds)
        positions = mrope_positions(ids, self.image_token_id, [(gh // MERGE, gw // MERGE) for _, (gh, gw) in patches])
        attention = [layer.self_attn for layer in self.model.layers if not layer.is_linear]
        for attn in attention:
            attn.positions = positions
        try:
            hidden = self.model(tokens, input_embeddings=embeds[None])
            mx.eval(hidden)
        finally:
            for attn in attention:
                attn.positions = None
        return hidden

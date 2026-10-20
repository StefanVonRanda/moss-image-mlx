"""FLUX.2 VAE decoder (diffusers AutoencoderKLFlux2 layout) in MLX, fp32, decoded in row strips.

MOSS-Image ships FLUX.2's encoder with the distilled small decoder (widths 96/192/384/384); the widths are read
from vae/config.json. Text-to-image needs only the decoder.

Memory: every 3x3 convolution runs in horizontal strips with a one-row halo, GroupNorm statistics are reduced over
the whole tensor strip by strip, and the single-head mid-block attention runs in query chunks. The result equals
the whole-tensor computation (up to reduction order) with no tile seams, while the peak stays near a few
full-resolution activations. Adapted from felix-ab/flux3-mlx's autoencoder (Apache-2.0).
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

STRIP_BYTES = 256 * 1024 * 1024  # budget for one strip's conv input (the im2col temporary is ~9x this)


def _strip_rows(x: mx.array) -> int:
    _, h, w, c = x.shape
    return max(8, min(h, STRIP_BYTES // (w * c * 4)))


def group_norm(groups: int, channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(groups, channels, eps=1e-6, affine=True, pytorch_compatible=True)


def group_stats(norm: nn.GroupNorm, x: mx.array):
    """Two-pass GroupNorm mean and 1/std over the whole tensor, reduced strip by strip. Returns (B, groups)."""
    b, h, w, c = x.shape
    g, rows = norm.num_groups, _strip_rows(x)
    n = h * w * (c // g)
    total = mx.zeros((b, g), dtype=mx.float32)
    for r in range(0, h, rows):
        total = total + x[:, r:r + rows].astype(mx.float32).reshape(b, -1, g, c // g).sum(axis=(1, 3))
        mx.eval(total)
    mean = total / n
    sq = mx.zeros((b, g), dtype=mx.float32)
    for r in range(0, h, rows):
        d = x[:, r:r + rows].astype(mx.float32).reshape(b, -1, g, c // g) - mean[:, None, :, None]
        sq = sq + (d * d).sum(axis=(1, 3))
        mx.eval(sq)
    return mean, mx.rsqrt(sq / n + norm.eps)


def group_apply(norm: nn.GroupNorm, x: mx.array, stats) -> mx.array:
    mean, rstd = stats
    b, h, w, c = x.shape
    g = norm.num_groups
    y = (x.reshape(b, h, w, g, c // g) - mean[:, None, None, :, None]) * rstd[:, None, None, :, None]
    return y.reshape(b, h, w, c) * norm.weight + norm.bias


def conv3_strips(conv: nn.Conv2d, x: mx.array, pre=None, post=None) -> mx.array:
    """3x3, stride 1, padding 1 conv over row strips with a one-row halo: equal to conv(pre(x)) (+ post)."""
    _, h, _, _ = x.shape
    rows, outs = _strip_rows(x), []
    for r in range(0, h, rows):
        a, e = max(r - 1, 0), min(r + rows + 1, h)
        xs = x[:, a:e]
        y = conv(pre(xs) if pre else xs)[:, r - a: r - a + min(rows, h - r)]
        if post is not None:
            y = post(y, r, r + y.shape[1])
        mx.eval(y)
        outs.append(y)
    return outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=1)


class ResnetBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, groups: int):
        super().__init__()
        self.norm1, self.conv1 = group_norm(groups, in_ch), nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2, self.conv2 = group_norm(groups, out_ch), nn.Conv2d(out_ch, out_ch, 3, padding=1)
        if in_ch != out_ch:
            self.conv_shortcut = nn.Conv2d(in_ch, out_ch, 1)

    def __call__(self, x):
        s1 = group_stats(self.norm1, x)
        h = conv3_strips(self.conv1, x, pre=lambda t: nn.silu(group_apply(self.norm1, t, s1)))
        s2 = group_stats(self.norm2, h)
        shortcut = self.conv_shortcut if "conv_shortcut" in self else (lambda t: t)
        return conv3_strips(self.conv2, h, pre=lambda t: nn.silu(group_apply(self.norm2, t, s2)),
                            post=lambda y, a, e: shortcut(x[:, a:e]) + y)


class Attention(nn.Module):
    """diffusers' VAE mid-block attention: GroupNorm, one head of width C, Linear q/k/v/out, residual."""

    def __init__(self, channels: int, groups: int):
        super().__init__()
        self.group_norm = group_norm(groups, channels)
        self.to_q, self.to_k, self.to_v = (nn.Linear(channels, channels) for _ in range(3))
        self.to_out = [nn.Linear(channels, channels)]

    def __call__(self, x):
        b, height, width, c = x.shape
        h = self.group_norm(x).reshape(b, 1, height * width, c)
        q, k, v = self.to_q(h), self.to_k(h), self.to_v(h)
        # One head of width C materialises (HW)^2 scores; query chunks keep that to chunk x HW (softmax is per row).
        chunk = max(1024, (1 << 30) // (4 * height * width))
        parts = []
        for i in range(0, height * width, chunk):
            parts.append(mx.fast.scaled_dot_product_attention(q[:, :, i:i + chunk], k, v, scale=c**-0.5))
            mx.eval(parts[-1])
        h = parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=2)
        return x + self.to_out[0](h).reshape(b, height, width, c)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def __call__(self, x):
        """Nearest 2x then conv, strip by strip: input rows [a, e) cover output rows [2r, 2r + 2s) plus halo."""
        b, h, w, c = x.shape
        rows, outs = max(4, _strip_rows(x) // 4), []
        for r in range(0, h, rows):
            s = min(rows, h - r)
            a, e = max(r - 1, 0), min(r + s + 1, h)
            xs = x[:, a:e]
            up = mx.broadcast_to(xs[:, :, None, :, None, :], (b, e - a, 2, w, 2, c)).reshape(b, 2 * (e - a), 2 * w, c)
            y = self.conv(up)[:, 2 * (r - a): 2 * (r - a) + 2 * s]
            mx.eval(y)
            outs.append(y)
        return outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=1)


class _Mid(nn.Module):
    def __init__(self, ch: int, groups: int):
        super().__init__()
        self.resnets = [ResnetBlock(ch, ch, groups), ResnetBlock(ch, ch, groups)]
        self.attentions = [Attention(ch, groups)]

    def __call__(self, x):
        return self.resnets[1](self.attentions[0](self.resnets[0](x)))


class _Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, layers: int, groups: int, upsample: bool):
        super().__init__()
        self.resnets = [ResnetBlock(in_ch if i == 0 else out_ch, out_ch, groups) for i in range(layers + 1)]
        if upsample:
            self.upsamplers = [Upsample(out_ch)]

    def __call__(self, x):
        for resnet in self.resnets:
            x = resnet(x)
        if "upsamplers" in self:
            x = self.upsamplers[0](x)
        return x


class Downsample(nn.Module):
    """diffusers Downsample2D(use_conv=True, padding=0): zero-pad one row/col at the bottom/right, 3x3 stride-2 conv."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=0)

    def __call__(self, x):
        b, h, w, c = x.shape
        x = mx.pad(x, [(0, 0), (0, 1), (0, 1), (0, 0)])
        rows, outs = max(4, _strip_rows(x) // 4), []
        for r in range(0, h // 2, rows):  # output rows [r, r + s) read padded input rows [2r, 2r + 2s + 1)
            s = min(rows, h // 2 - r)
            y = self.conv(x[:, 2 * r: 2 * r + 2 * s + 1])
            mx.eval(y)
            outs.append(y)
        return outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=1)


class _Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, layers: int, groups: int, downsample: bool):
        super().__init__()
        self.resnets = [ResnetBlock(in_ch if i == 0 else out_ch, out_ch, groups) for i in range(layers)]
        if downsample:
            self.downsamplers = [Downsample(out_ch)]

    def __call__(self, x):
        for resnet in self.resnets:
            x = resnet(x)
        if "downsamplers" in self:
            x = self.downsamplers[0](x)
        return x


class Encoder(nn.Module):
    def __init__(self, latent_channels: int, widths: list[int], layers_per_block: int, groups: int):
        super().__init__()
        self.conv_in = nn.Conv2d(3, widths[0], 3, padding=1)
        self.down_blocks = [_Down(widths[max(i - 1, 0)], widths[i], layers_per_block, groups, downsample=i < len(widths) - 1)
                            for i in range(len(widths))]
        self.mid_block = _Mid(widths[-1], groups)
        self.conv_norm_out = group_norm(groups, widths[-1])
        self.conv_out = nn.Conv2d(widths[-1], 2 * latent_channels, 3, padding=1)  # mean and logvar

    def __call__(self, x):
        h = conv3_strips(self.conv_in, x)
        for block in self.down_blocks:
            h = block(h)
        h = self.mid_block(h)
        stats = group_stats(self.conv_norm_out, h)
        return conv3_strips(self.conv_out, h, pre=lambda t: nn.silu(group_apply(self.conv_norm_out, t, stats)))


class Decoder(nn.Module):
    def __init__(self, latent_channels: int, widths: list[int], layers_per_block: int, groups: int):
        super().__init__()
        rev = list(reversed(widths))
        self.conv_in = nn.Conv2d(latent_channels, rev[0], 3, padding=1)
        self.mid_block = _Mid(rev[0], groups)
        self.up_blocks = [_Up(rev[max(i - 1, 0)], rev[i], layers_per_block, groups, upsample=i < len(rev) - 1)
                          for i in range(len(rev))]
        self.conv_norm_out = group_norm(groups, rev[-1])
        self.conv_out = nn.Conv2d(rev[-1], 3, 3, padding=1)

    def __call__(self, z):
        h = self.mid_block(conv3_strips(self.conv_in, z))
        for block in self.up_blocks:
            h = block(h)
        stats = group_stats(self.conv_norm_out, h)
        return conv3_strips(self.conv_out, h, pre=lambda t: nn.silu(group_apply(self.conv_norm_out, t, stats)))


class VAEDecoder(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        latent = config["latent_channels"]
        widths = config.get("decoder_block_out_channels") or config["block_out_channels"]
        self.decoder = Decoder(latent, widths, config["layers_per_block"], config["norm_num_groups"])
        self.post_quant_conv = nn.Conv2d(latent, latent, 1)
        self.bn_eps = config.get("batch_norm_eps", 1e-4)
        self._config = config
        self.running_mean = mx.zeros((4 * latent,))
        self.running_var = mx.ones((4 * latent,))

    def load_encoder(self, path: str | Path):
        """Attach the encoder from the official diffusers VAE file (encoder.* and quant_conv.* keys, OIHW)."""
        if "encoder" in self:
            return
        config = self._config
        self.encoder = Encoder(config["latent_channels"], config["block_out_channels"], config["layers_per_block"],
                               config["norm_num_groups"])
        self.quant_conv = nn.Conv2d(2 * config["latent_channels"], 2 * config["latent_channels"], 1)
        weights = {}
        for key, value in mx.load(str(path)).items():
            if key.startswith(("encoder.", "quant_conv.")):
                weights[key] = (value.transpose(0, 2, 3, 1) if value.ndim == 4 else value).astype(mx.float32)
        self.encoder.load_weights([(k[8:], v) for k, v in weights.items() if k.startswith("encoder.")], strict=True)
        self.quant_conv.load_weights([(k[11:], v) for k, v in weights.items() if k.startswith("quant_conv.")], strict=True)
        mx.eval(self.encoder.parameters(), self.quant_conv.parameters())

    def encode_packed(self, image: mx.array) -> mx.array:
        """Image (1, 16h, 16w, 3) in [-1, 1] -> packed, normalised DiT latents (1, 128, h, w), fp32. Inverse of
        decode_packed: encoder, quant_conv, posterior mean, patchify 2x2, batch-norm normalisation."""
        moments = self.quant_conv(self.encoder(image.astype(mx.float32)))
        z = moments[..., : moments.shape[-1] // 2]  # DiagonalGaussianDistribution.mode()
        z = pack_latents(z)
        return (z - self.running_mean[None, :, None, None]) * mx.rsqrt(self.running_var + self.bn_eps)[None, :, None, None]

    def decode_packed(self, latents: mx.array) -> mx.array:
        """Packed DiT latents (1, 128, h, w) -> image (1, 16h, 16w, 3) in [-1, 1], fp32.

        As in the reference pipeline: batch-norm de-normalisation, then unpatchify (b, 32, 2, 2, h, w) -> (b, 32, 2h, 2w).
        """
        z = latents.astype(mx.float32)
        z = z * mx.sqrt(self.running_var + self.bn_eps)[None, :, None, None] + self.running_mean[None, :, None, None]
        return mx.clip(self.decoder(self.post_quant_conv(unpack_latents(z))), -1.0, 1.0)


def unpack_latents(z: mx.array) -> mx.array:
    """(b, 4c, h, w) packed -> (b, 2h, 2w, c): channel 4k + 2i + j lands on pixel (2y + i, 2x + j)."""
    b, c, h, w = z.shape
    return z.reshape(b, c // 4, 2, 2, h, w).transpose(0, 4, 2, 5, 3, 1).reshape(b, 2 * h, 2 * w, c // 4)


def pack_latents(x: mx.array) -> mx.array:
    """(b, 2h, 2w, c) -> (b, 4c, h, w); exact inverse of unpack_latents."""
    b, hh, ww, c = x.shape
    return x.reshape(b, hh // 2, 2, ww // 2, 2, c).transpose(0, 5, 2, 4, 1, 3).reshape(b, 4 * c, hh // 2, ww // 2)


def official_vae_file() -> str:
    """The diffusers VAE weights (encoder + decoder) from the official release; the MLX tiers ship only the decoder."""
    from huggingface_hub import hf_hub_download

    return hf_hub_download("OpenMOSS-Team/MOSS-Image-SFT", "vae/diffusion_pytorch_model.safetensors")


def load_vae_decoder(path: str | Path) -> VAEDecoder:
    path = Path(path)
    config = json.loads((path / "config.json").read_text())
    vae = VAEDecoder(config)
    if config.get("moss_image_mlx"):  # exported MLX layout (decoder only, OHWI, fp32)
        weights = {}
        for file in sorted(path.glob("*.safetensors")):
            weights.update(mx.load(str(file)))
        vae.load_weights(list(weights.items()), strict=True)
        mx.eval(vae.parameters())
        return vae
    weights = {}
    for file in sorted(path.glob("*.safetensors")):
        for key, value in mx.load(str(file)).items():
            if key.startswith("bn."):
                if key.endswith(("running_mean", "running_var")):
                    weights[key[3:]] = value.astype(mx.float32)
                continue
            if not key.startswith(("decoder.", "post_quant_conv.")):
                continue
            if value.ndim == 4:  # conv OIHW -> OHWI
                value = value.transpose(0, 2, 3, 1)
            weights[key] = value.astype(mx.float32)
    vae.load_weights(list(weights.items()), strict=True)
    mx.eval(vae.parameters())
    return vae


def to_uint8(image: mx.array):
    """[-1, 1] -> uint8 exactly like diffusers' postprocess: ((x / 2 + 0.5).clamp(0, 1) * 255).round()."""
    import numpy as np

    x = np.array(image.astype(mx.float32))
    return np.round(np.clip(x / 2 + 0.5, 0, 1) * 255).astype(np.uint8)

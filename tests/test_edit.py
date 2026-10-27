"""Instruction editing with reference images (MOSS Image Edit) and the RGBA decoder. No weights needed."""

import mlx.core as mx
import numpy as np
from PIL import Image

from moss_image_mlx.dit import MOSSImageTransformer, reference_rope, rotate, situ
import pytest

from moss_image_mlx.pipeline import Request, load_image, load_reference, source_alpha
from moss_image_mlx.vae import VAEDecoder


def _tiny():
    return MOSSImageTransformer(dim=64, ffn_dim=128, num_heads=2, num_layers=2, text_dim=32, freq_dim=16)


def _joint(m, latents, t, ctx, ref, ref_frame=0):
    """The edit written out directly: one sequence [target, text, reference], references at timestep 0 seeing only
    references, target and text seeing everything (MOSS Image Edit's contract). ref_frame: the references' RoPE t
    coordinate with the stock target/text layout (0 for chinoll; the text length for ai-toolkit, as ostris trains)."""
    _, c, H, W = latents.shape
    _, _, h, w = ref.shape
    heads, hd = m.config["num_heads"], m.config["dim"] // m.config["num_heads"]
    mod_t, head = m.modulation(t)
    mod_0, _ = m.modulation(0.0)
    sin, cos = m.rope(H, W, ctx.shape[1], mx.float32)
    video = m.video_embedder(latents.reshape(1, c, H * W).transpose(0, 2, 1))
    text = m.text_embedder(ctx, (sin[:, H * W:], cos[:, H * W:]))
    refx = m.video_embedder(ref.reshape(1, c, h * w).transpose(0, 2, 1))
    rsin, rcos = reference_rope(h, w, 0, hd, m.config["rope_base"], mx.float32, ref_frame)
    x = mx.concatenate([video, text, refx], axis=1)
    nb, n = video.shape[1] + text.shape[1], x.shape[1]
    sin, cos = mx.concatenate([sin, rsin], axis=1), mx.concatenate([cos, rcos], axis=1)
    mods = mx.concatenate([mx.repeat(mod_t[:, None], nb, axis=1), mx.repeat(mod_0[:, None], n - nb, axis=1)], axis=1)
    qi, ki = mx.arange(n)[:, None], mx.arange(n)[None, :]
    mask = (qi < nb) | (ki >= nb)

    def split_heads(z):
        return z.reshape(1, n, heads, hd)

    for layer in m.layers:
        sa, ca, ga, sf, cf, gf = (mods[:, :, i] for i in range(6))
        p = layer.proj
        y = layer.norm1(x) * (1 + ca) + sa
        q = rotate(layer.q_norm(split_heads(p["q"](y))), sin, cos).transpose(0, 2, 1, 3)
        k = rotate(layer.k_norm(split_heads(p["k"](y))), sin, cos).transpose(0, 2, 1, 3)
        v = split_heads(p["v"](y)).transpose(0, 2, 1, 3)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=hd**-0.5, mask=mask)
        x = x + p["o"](out.transpose(0, 2, 1, 3).reshape(1, n, -1)) * ga
        y = layer.norm2(x) * (1 + cf) + sf
        hidden, gate = mx.split(p["ffn_up_gate"](y), 2, axis=-1)
        x = x + p["ffn_down"](situ(hidden, gate)) * gf
    x = m.final_norm_video(x[:, :H * W]) * (1 + head[:, 1:2]) + head[:, 0:1]
    return m.final_linear_video(x).transpose(0, 2, 1).reshape(1, c, H, W)


def test_cached_references_equal_the_joint_masked_sequence():
    mx.random.seed(0)
    m = _tiny()
    latents, ref = mx.random.normal((1, 128, 3, 4)), mx.random.normal((1, 128, 2, 5))
    ctx = mx.random.normal((1, 6, 32))
    cached = m(latents, 700.0, ctx, references=m.reference_cache([ref]))
    plain = m(latents, 700.0, ctx)
    expected = _joint(m, latents, 700.0, ctx, ref)
    assert mx.abs(cached - expected).max().item() < 1e-4
    assert mx.abs(cached - plain).max().item() > 1e-3  # the reference does reach the target


def test_ai_toolkit_references_sit_at_the_text_length_frame():
    mx.random.seed(1)
    m = _tiny()
    latents, ref = mx.random.normal((1, 128, 3, 4)), mx.random.normal((1, 128, 2, 5))
    refs = m.reference_cache([ref])  # prompt-independent: one cache for both CFG branches
    for length in (6, 9):
        ctx = mx.random.normal((1, length, 32))
        hq = m(latents, 700.0, ctx, references=refs, edit_format="ai-toolkit")
        # same attention, phases at shifted positions: fp32 sin/cos rounding only
        assert mx.abs(hq - _joint(m, latents, 700.0, ctx, ref, ref_frame=length)).max().item() < 5e-4
        chinoll = m(latents, 700.0, ctx, references=refs)
        assert mx.abs(chinoll - _joint(m, latents, 700.0, ctx, ref, ref_frame=0)).max().item() < 1e-4
        assert mx.abs(hq - chinoll).max().item() > 1e-3  # the frame matters
    # without references the format changes nothing
    assert mx.array_equal(m(latents, 700.0, ctx, edit_format="ai-toolkit"), m(latents, 700.0, ctx))


def test_rgba_decoder_outputs_four_channels():
    config = {"latent_channels": 4, "block_out_channels": [32, 32], "layers_per_block": 1, "norm_num_groups": 8,
              "out_channels": 4}
    image = VAEDecoder(config).decode_packed(mx.random.normal((1, 16, 2, 3)))
    assert image.shape == (1, 8, 12, 4)


def test_reference_keeps_aspect_on_the_16px_grid(tmp_path):
    Image.new("RGB", (1000, 600)).save(tmp_path / "r.png")
    assert load_reference(tmp_path / "r.png", 512).shape == (304, 512, 3)
    assert load_reference(tmp_path / "r.png", 2048).shape == (608, 1008, 3)  # never enlarged, only snapped


def _cutout(path):
    """64x32 RGBA: an opaque red left half, a fully transparent right half whose hidden colour is black."""
    a = np.zeros((32, 64, 4), np.uint8)
    a[:, :32] = (255, 0, 0, 255)
    Image.fromarray(a).save(path)


def test_transparent_sources_are_flattened_onto_grey(tmp_path):
    _cutout(tmp_path / "c.png")
    for image in (load_image(tmp_path / "c.png", 64, 32), load_reference(tmp_path / "c.png", 64)):
        assert image.shape == (32, 64, 3)
        assert (image[:, :28] == (255, 0, 0)).all()  # opaque pixels untouched
        assert (image[:, 36:] == 128).all()  # transparent = neutral grey, not the black under alpha 0


def test_keep_alpha_takes_the_source_alpha_at_output_size(tmp_path):
    _cutout(tmp_path / "c.png")
    Image.new("RGB", (64, 32), "blue").save(tmp_path / "opaque.png")
    edit = source_alpha(Request("x", width=128, height=64, references=[str(tmp_path / "c.png"), "unused.png"],
                                keep_alpha=True))
    assert edit.shape == (64, 128) and (edit[:, :56] == 255).all() and (edit[:, 72:] == 0).all()
    crop = source_alpha(Request("x", width=32, height=32, init_image=str(tmp_path / "c.png")))  # centre crop
    assert (crop[:, :14] == 255).all() and (crop[:, 18:] == 0).all()
    assert (source_alpha(Request("x", width=32, height=32, init_image=str(tmp_path / "opaque.png"))) == 255).all()
    with pytest.raises(ValueError, match="keep_alpha"):
        Request("x", keep_alpha=True).validate(24)

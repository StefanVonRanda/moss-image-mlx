"""Colour/depth control adapter (darask0/MOSS-Image-color-depth-control format). No weights needed."""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from moss_image_mlx.dit import MOSSImageTransformer
from moss_image_mlx.lora import LoRALinear, apply_control, apply_lora
from moss_image_mlx.pipeline import Request, _sample


def _tiny():
    mx.random.seed(0)
    return MOSSImageTransformer(dim=64, ffn_dim=128, num_heads=2, num_layers=2, text_dim=32, freq_dim=16)


def _control_file(tmp_path, model, lora_scale=0.1):
    """The adapter's layout: bare keys, video_embedder [dim, 384] whose first 128 columns are the model's own, and
    rank-4 LoRAs on a block linear and the text-embedder block linears."""
    rng = np.random.default_rng(0)
    old = np.array(model.video_embedder.weight)
    w = {"video_embedder.weight": mx.array(np.concatenate([old, rng.standard_normal((64, 256)).astype(np.float32)], 1)),
         "video_embedder.bias": model.video_embedder.bias}
    for name, (i, o) in {"layers.0.proj.q": (64, 64), "text_embedder.blocks.0.qkv": (64, 192),
                         "text_embedder.blocks.1.ffn_down": (128, 64)}.items():
        w[name + ".lora_A.weight"] = mx.array(rng.standard_normal((4, i)).astype(np.float32))
        w[name + ".lora_B.weight"] = mx.array(lora_scale * rng.standard_normal((o, 4)).astype(np.float32))
    path = tmp_path / "control.safetensors"
    mx.save_safetensors(str(path), w)
    return path


def _inputs():
    rng = np.random.default_rng(1)
    f = lambda *s: mx.array(rng.standard_normal(s).astype(np.float32))  # noqa: E731
    return f(1, 128, 3, 4), f(1, 6, 32), f(1, 256, 3, 4)


def test_control_adapter_loads_and_shapes(tmp_path):
    m = _tiny()
    path = _control_file(tmp_path, m)
    with pytest.raises(ValueError, match="--control"):
        apply_lora(m, path)  # a control adapter given as a plain LoRA
    assert apply_control(m, path) == 3
    assert m.video_embedder.weight.shape == (64, 384)
    assert isinstance(m.text_embedder.blocks[0].qkv, LoRALinear)
    assert isinstance(m.text_embedder.blocks[1].ffn_down, LoRALinear)
    assert isinstance(m.layers[0].proj["q"], LoRALinear) and m.layers[0].proj["q"].group == "control"
    latents, ctx, control = _inputs()
    out = m(latents, 500.0, ctx, control=control)
    assert out.shape == (1, 128, 3, 4)
    assert mx.abs(out - m(latents, 500.0, ctx)).max().item() > 1e-3  # the maps reach the output
    with pytest.raises(ValueError, match="control adapter"):
        _tiny()(latents, 500.0, ctx, control=control)


def test_no_control_is_the_original_model(tmp_path):
    """Zero-strength LoRA + control embedder whose first 128 columns are the original: without maps (and with
    references) the model is exactly the original one."""
    m = _tiny()
    latents, ctx, _ = _inputs()
    ref = mx.random.normal((1, 128, 2, 5))
    before = m(latents, 500.0, ctx)
    refs_before = m(latents, 500.0, ctx, references=m.reference_cache([ref]))
    apply_control(m, _control_file(tmp_path, m, lora_scale=0.0))
    assert mx.abs(m(latents, 500.0, ctx) - before).max().item() < 1e-5
    after = m(latents, 500.0, ctx, references=m.reference_cache([ref]))
    assert mx.abs(after - refs_before).max().item() < 1e-5  # references use the first 128 columns


class _Recorder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers, self.config, self.seen = [], {"latent_channels": 128}, []

    def __call__(self, latents, t, ctx, skip=(), references=None, control=None, edit_format="chinoll"):
        self.seen.append(control is not None)
        return mx.zeros_like(latents)


@pytest.mark.parametrize("end, on", [(0.5, 4), (1.0, 8), (0.0, 0), (0.3, 3)])
def test_control_end_zeroes_after_fraction(end, on):
    m = _Recorder()
    req = Request(prompt="x", width=64, height=48, steps=8, guidance=1.0, uncond_skip_layers=[], control_end=end)
    _sample(m, req, None, None, None, None, control=mx.zeros((1, 256, 3, 4)))
    assert m.seen == [True] * on + [False] * (8 - on)


def test_control_with_cfg_reaches_both_branches():
    m = _Recorder()
    req = Request(prompt="x", width=64, height=48, steps=4, guidance=3.0, uncond_skip_layers=[], control_end=0.5)
    _sample(m, req, mx.zeros((1, 2, 32)), mx.zeros((1, 2, 32)), None, None, control=mx.zeros((1, 256, 3, 4)))
    assert m.seen == [True, True, True, True, False, False, False, False]


def test_validate_control():
    Request(prompt="x", control_color="c.png", init_image="i.png").validate(42)  # control + image-to-image runs
    with pytest.raises(ValueError, match="turbo"):
        Request(prompt="x", control_color="c.png", turbo=True).validate(42)
    with pytest.raises(ValueError, match="control_end"):
        Request(prompt="x", control_end=1.5).validate(42)

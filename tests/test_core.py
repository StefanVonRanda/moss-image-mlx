"""Fast tests: no model weights needed. Fixtures in tests/fixtures/ were produced by torch 2.14 running the official
pipeline_moss_image.py (scheduler) and torch.randn with a CPU generator (noise)."""

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from moss_image_mlx import layout
from moss_image_mlx.cache import PromptCache
from moss_image_mlx.cli import _layers, quant_spec
from moss_image_mlx.dit import MOSSImageTransformer
from moss_image_mlx.quantize import quantize_dit
from moss_image_mlx.scheduler import sigmas_for, timesteps_for, torch_linspace
from moss_image_mlx.torch_randn import randn
from moss_image_mlx.vae import conv3_strips

FIX = json.loads((Path(__file__).parent / "fixtures" / "torch_reference.json").read_text())


@pytest.mark.parametrize("key", sorted(FIX["sigmas"]))
def test_schedule_is_bit_identical_to_official(key):
    steps, shift = key.split("_")
    want = np.array([float.fromhex(h) for h in FIX["sigmas"][key]], dtype=np.float32)
    got = sigmas_for(int(steps), float(shift))
    assert np.array_equal(got, want)
    assert np.array_equal(timesteps_for(got), (want[:-1] * np.float32(1000)).astype(np.float32))


def test_torch_linspace_endpoints():
    v = torch_linspace(1.0, 0.001, 1000)
    assert v.dtype == np.float32 and v[0] == np.float32(1.0) and v[-1] == np.float32(0.001)


@pytest.mark.parametrize("seed", sorted(FIX["noise"]))
def test_seed_noise_matches_torch_cpu_generator(seed):
    ref = FIX["noise"][seed]
    got = randn(tuple(ref["shape"]), int(seed)).reshape(-1)
    assert np.abs(got - np.array(ref["values"], dtype=np.float32)).max() < 1e-5


def _toy():
    m = MOSSImageTransformer(dim=96, ffn_dim=128, num_heads=4, num_layers=3, text_dim=64)
    m.set_dtype(mx.bfloat16)
    return m


def test_toy_dit_shapes_skip_and_precise():
    m = _toy()
    x, ctx = mx.random.normal((1, 128, 4, 6)), mx.random.normal((1, 5, 64)).astype(mx.bfloat16)
    a, b = m(x, 900.0, ctx), m(x, 900.0, ctx, skip_layers=[1])
    m.precise = True
    c = m(x, 900.0, ctx)
    mx.eval(a, b, c)
    assert a.shape == (1, 128, 4, 6)
    assert not mx.array_equal(a, b)  # skipping a block changes the prediction
    assert c.shape == a.shape


def test_quantization_keeps_conditioning_layers_in_bf16():
    m = MOSSImageTransformer(dim=128, ffn_dim=256, num_heads=4, num_layers=2, text_dim=64)
    m.set_dtype(mx.bfloat16)
    quantize_dit(m, {"bits": 8, "group_size": 64})
    assert isinstance(m.layers[0].proj["q"], nn.QuantizedLinear)
    for name in ("video_embedder", "shared_block_modulation", "video_head_modulation", "final_linear_video"):
        assert type(getattr(m, name)) is nn.Linear
    assert type(m.time_embedding[0]) is nn.Linear


def test_strip_convolution_equals_whole_tensor(monkeypatch):
    import moss_image_mlx.vae as vae

    monkeypatch.setattr(vae, "STRIP_BYTES", 8 * 9 * 4 * 3)  # force several strips on a small input
    conv = nn.Conv2d(4, 5, 3, padding=1)
    x = mx.random.normal((1, 37, 9, 4))
    assert np.allclose(np.array(conv3_strips(conv, x)), np.array(conv(x)), atol=1e-5)


def test_layout_conversion_and_checks():
    assert layout.from_top_left(128, 64, 896, 192, 1024, 1024) == [-448, -384, -320, 384]
    assert layout.from_normalized_1000([0, 0, 500, 1000], 1024, 768) == [-384, -512, 0, 512]
    good = {"summary": {"caption": "poster"}, "items": [{"label": "text", "text": "HI", "bbox": [-100, -200, 0, 200]}]}
    assert layout.check(good, 1024, 1024) == []
    bad = {"summary": {"caption": "poster"}, "items": [{"label": "text", "text": "HI", "bbox": [-600, -200, 0, 200]}]}
    assert any("leaves" in p for p in layout.check(bad, 1024, 1024))
    swapped = {"summary": {"caption": "x"}, "items": [{"label": "object", "caption": "o", "bbox": [10, 0, 0, 10]}]}
    assert any("min >= max" in p for p in layout.check(swapped, 1024, 1024))


def test_prompt_cache_round_trip(tmp_path):
    te_dir = tmp_path / "te"
    te_dir.mkdir()
    (te_dir / "config.json").write_text("{}")
    cache = PromptCache(te_dir, tmp_path / "cache")
    h = mx.random.normal((1, 7, 16)).astype(mx.bfloat16)
    assert cache.get("a prompt", 4096) is None
    cache.put("a prompt", 4096, h)
    assert mx.array_equal(cache.get("a prompt", 4096), h)
    assert cache.get("a prompt", 512) is None  # token limit is part of the key


def test_cli_parsing():
    assert quant_spec("8") == {"bits": 8, "group_size": 64}
    assert quant_spec("4g32") == {"bits": 4, "group_size": 32}
    assert quant_spec("bf16") is None
    assert quant_spec("nvfp4")["mode"] == "nvfp4"
    assert _layers("22") == [22] and _layers("") == [] and _layers("22, 24") == [22, 24]

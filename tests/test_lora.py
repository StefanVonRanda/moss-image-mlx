import mlx.core as mx
import mlx.nn as nn
import numpy as np

from moss_image_mlx.lora import LoRALinear, apply_lora


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [nn.Module()]
        self.layers[0].proj = {"q": nn.Linear(64, 32, bias=False), "ffn_down": nn.Linear(64, 32, bias=False)}


def _adapter(tmp_path, rank=8, alpha=4.0):
    rng = np.random.default_rng(0)
    w = {}
    for name in ("q", "ffn_down"):
        k = f"diffusion_model.layers.0.proj.{name}"
        w[k + ".lora_down.weight"] = mx.array(rng.standard_normal((rank, 64)).astype(np.float32))
        w[k + ".lora_up.weight"] = mx.array(rng.standard_normal((32, rank)).astype(np.float32))
        w[k + ".alpha"] = mx.array(np.float32(alpha))
    path = tmp_path / "adapter.safetensors"
    mx.save_safetensors(str(path), w)
    return path, w


def test_unmerged_lora_matches_formula(tmp_path):
    path, w = _adapter(tmp_path)
    model = Tiny()
    base_w = np.array(model.layers[0].proj["q"].weight)
    assert apply_lora(model, path, strength=0.7, dtype=mx.float32) == 2
    layer = model.layers[0].proj["q"]
    assert isinstance(layer, LoRALinear)
    x = mx.array(np.random.default_rng(1).standard_normal((3, 64)).astype(np.float32))
    down = np.array(w["diffusion_model.layers.0.proj.q.lora_down.weight"])
    up = np.array(w["diffusion_model.layers.0.proj.q.lora_up.weight"])
    expected = np.array(x) @ base_w.T + 0.7 * (4.0 / 8) * (np.array(x) @ down.T @ up.T)
    np.testing.assert_allclose(np.array(layer(x)), expected, rtol=1e-4, atol=1e-4)
    np.testing.assert_array_equal(np.array(layer.base.weight), base_w)  # base untouched (unmerged)


def test_lora_over_quantized_linear(tmp_path):
    path, _ = _adapter(tmp_path)
    model = Tiny()
    model.set_dtype(mx.bfloat16)  # as the real DiT
    nn.quantize(model, group_size=32, bits=8)
    apply_lora(model, path, strength=1.0)
    layer = model.layers[0].proj["ffn_down"]
    assert isinstance(layer.base, nn.QuantizedLinear)
    y = layer(mx.ones((2, 64), dtype=mx.bfloat16))
    assert y.dtype == mx.bfloat16 and y.shape == (2, 32)

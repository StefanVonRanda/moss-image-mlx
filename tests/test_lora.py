import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

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
    with mx.stream(mx.cpu):  # some Apple GPUs run fp32 matmuls at reduced precision (0.017 abs error here)
        y = np.array(layer(x))
    np.testing.assert_allclose(y, expected, rtol=1e-4, atol=1e-4)
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


def test_adapter_groups_switch_independently(tmp_path):
    from moss_image_mlx.lora import set_strength

    path, _ = _adapter(tmp_path)
    model = Tiny()
    apply_lora(model, path, strength=1.5, dtype=mx.float32, group="destripe")
    apply_lora(model, path, strength=1.0, dtype=mx.float32, group="student")  # wraps around the first
    outer = model.layers[0].proj["q"]
    assert outer.group == "student" and outer.base.group == "destripe"
    set_strength(model, 0.0, "student")
    assert outer.scale == 0 and outer.base.scale == outer.base.base_scale  # the other group is untouched
    set_strength(model, 1.0, "student")
    assert outer.scale == outer.base_scale


def test_turbo_schedule():
    from moss_image_mlx import turbo
    from moss_image_mlx.pipeline import Request

    prefix, finish = turbo.schedule()
    assert len(prefix) == len(turbo.ACTORS) + 1 and prefix[0] == 1.0
    np.testing.assert_allclose(prefix, [1.0, .983, .957, .919, .871, .791, .671], atol=1.5e-3)
    np.testing.assert_allclose(finish, [.671, .536, .373, .193, 0.0], atol=1.5e-3)
    assert turbo.PASSES == 15
    assert Request(prompt="x", guidance=1.0, turbo=True).uses_cfg()  # the original-model steps need the negative


def test_request_per_step_validation():
    import pytest

    from moss_image_mlx.pipeline import Request

    r = Request(prompt="x", sigmas=[1.0, 0.5, 0.0], step_guidance=[3.0, 1.0], step_lora=[0.0, 1.0])
    r.validate(42)
    assert r.n_steps == 2 and r.uses_cfg()
    assert not Request(prompt="x", guidance=1.0).uses_cfg()
    with pytest.raises(ValueError):
        Request(prompt="x", sigmas=[1.0, 0.5, 0.1]).validate(42)
    with pytest.raises(ValueError):
        Request(prompt="x", sigmas=[1.0, 0.5, 0.0], step_lora=[1.0]).validate(42)


def test_cli_lora_spec():
    from moss_image_mlx.cli import lora_spec

    assert lora_spec("a/b.safetensors") == ("a/b.safetensors", 1.0)
    assert lora_spec("a/b.safetensors:0.7") == ("a/b.safetensors", 0.7)  # the macOS app's form
    assert lora_spec("a/b.safetensors@0.7") == ("a/b.safetensors", 0.7)  # upstream's form
    assert lora_spec("/x y/c.safetensors:1") == ("/x y/c.safetensors", 1.0)
    assert lora_spec("C:/odd:name.safetensors") == ("C:/odd:name.safetensors", 1.0)
    assert lora_spec("a.safetensors:-0.5") == ("a.safetensors", -0.5)


# Key formats on the real DiT layout (tiny sizes): kohya, sd-scripts legacy, diffusers/PEFT, bare, fused qkv.
MODULES = ("q", "k", "v", "o", "ffn_up_gate", "ffn_down")
KEYS = {"kohya": ("diffusion_model.layers.{i}.proj.{n}", "lora_down.weight", "lora_up.weight", True),
        "kohya_legacy": ("lora_unet_layers_{i}_proj_{n}", "lora_down.weight", "lora_up.weight", True),
        "diffusers": ("transformer.layers.{i}.proj.{n}", "lora_A.weight", "lora_B.weight", False),
        "bare": ("layers.{i}.proj.{n}", "lora_A", "lora_B", False)}  # MOSS Image Edit: no prefix, no .weight


def _tiny_dit():
    from moss_image_mlx.dit import MOSSImageTransformer

    return MOSSImageTransformer(dim=64, ffn_dim=128, num_heads=2, num_layers=2, text_dim=32, freq_dim=16)


def _write_dit_lora(path, model, fmt, rank=4, alpha=2.0):
    rng = np.random.default_rng(0)
    name, down_key, up_key, has_alpha = KEYS[fmt]
    tensors = {}
    for i, block in enumerate(model.layers):
        for n in MODULES:
            out_dim, in_dim = block.proj[n].weight.shape
            base = name.format(i=i, n=n)
            tensors[f"{base}.{down_key}"] = mx.array(rng.standard_normal((rank, in_dim), dtype=np.float32) * 0.1)
            tensors[f"{base}.{up_key}"] = mx.array(rng.standard_normal((out_dim, rank), dtype=np.float32) * 0.1)
            if has_alpha:
                tensors[f"{base}.alpha"] = mx.array(alpha, dtype=mx.float32)
    mx.save_safetensors(str(path), tensors)
    return tensors


@pytest.mark.parametrize("fmt", list(KEYS))
def test_key_formats_load_and_apply(tmp_path, fmt):
    from moss_image_mlx.lora import load_lora

    model = _tiny_dit()
    tensors = _write_dit_lora(tmp_path / "a.safetensors", model, fmt)
    mods = load_lora(tmp_path / "a.safetensors")
    assert set(mods) == {f"layers.{i}.proj.{n}" for i in range(2) for n in MODULES}
    x = mx.random.normal((1, 3, 64)).astype(mx.bfloat16)
    before = model.layers[1].proj["o"](x)
    assert apply_lora(model, tmp_path / "a.safetensors", strength=0.5, dtype=mx.float32) == 12
    name, down_key, up_key, has_alpha = KEYS[fmt]
    base = name.format(i=1, n="o")
    down, up = tensors[f"{base}.{down_key}"], tensors[f"{base}.{up_key}"]
    alpha = 2.0 if has_alpha else 4.0  # no alpha tensor: alpha = rank
    expect = before + ((x.astype(mx.float32) @ down.T @ up.T) * (0.5 * alpha / 4)).astype(x.dtype)
    assert mx.allclose(model.layers[1].proj["o"](x), expect, atol=1e-2, rtol=1e-2).item()


def test_unknown_key_rejected(tmp_path):
    from moss_image_mlx.lora import load_lora

    mx.save_safetensors(str(tmp_path / "bad.safetensors"), {"something.else": mx.zeros((2, 2))})
    with pytest.raises(ValueError, match="unrecognized"):
        load_lora(tmp_path / "bad.safetensors")


def test_fused_qkv_adapter_splits_over_q_k_v(tmp_path):
    model = _tiny_dit()
    rng = np.random.default_rng(0)
    down = mx.array(rng.standard_normal((4, 64), dtype=np.float32))
    up = mx.array(rng.standard_normal((192, 4), dtype=np.float32))
    mx.save_safetensors(str(tmp_path / "a.safetensors"), {"diffusion_model.layers.1.proj.qkv.lora_A.weight": down,
                                                          "diffusion_model.layers.1.proj.qkv.lora_B.weight": up})
    x = mx.random.normal((1, 3, 64))
    with mx.stream(mx.cpu):  # exact fp32 matmuls (see test_unmerged_lora_matches_formula)
        before = {n: model.layers[1].proj[n](x) for n in "qkv"}
        assert apply_lora(model, tmp_path / "a.safetensors", 0.5, dtype=mx.float32) == 1
        delta = (x @ down.T) @ up.T * 0.5  # no alpha tensor: alpha = rank
        for i, n in enumerate("qkv"):
            assert isinstance(model.layers[1].proj[n], LoRALinear)
            assert mx.allclose(model.layers[1].proj[n](x), before[n] + delta[..., 64 * i:64 * (i + 1)], atol=1e-4)

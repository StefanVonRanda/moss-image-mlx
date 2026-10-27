"""The ai-toolkit edit format's text-encoder side: prompt and token layout, Qwen3.5 M-RoPE positions, the vision
patch layout, the vision tower's shapes and the prompt-cache key. No weights needed (the tokenizer test uses a cached
MOSS tokenizer when there is one)."""

import glob
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from moss_image_mlx.cache import PromptCache
from moss_image_mlx.pipeline import Request, load_vision, prompt_key
from moss_image_mlx.text_encoder import (VISION_BLOCK, TextEncoder, VisionTower, _mrope_cos_sin, _rope_cos_sin,
                                         mrope_positions, smart_resize, vision_patches, vision_prompt)

IMG = 248056


def test_vision_prompt_prefixes_pictures():
    assert vision_prompt("Make it red", 0) == "Make it red"
    assert vision_prompt("Make it red", 2) == f"Picture 1: {VISION_BLOCK}Picture 2: {VISION_BLOCK}Make it red"


def test_mrope_positions_follow_get_rope_index():
    # 3 text tokens, a 2x3 image, 2 text tokens, a 1x1 image, 1 text token
    ids = [1, 2, 3] + [IMG] * 6 + [4, 5] + [IMG] + [6]
    pos = mrope_positions(ids, IMG, [(2, 3), (1, 1)])
    assert pos.shape == (3, len(ids))
    assert (pos[:, :3] == [[0, 1, 2]] * 3).all()
    assert pos[0, 3:9].tolist() == [3] * 6                     # t: the image's start
    assert pos[1, 3:9].tolist() == [3, 3, 3, 4, 4, 4]          # h: start + row
    assert pos[2, 3:9].tolist() == [3, 4, 5, 3, 4, 5]          # w: start + column
    assert (pos[:, 9:11] == [[6, 7]] * 3).all()                # resumes at start + max(h, w)
    assert pos[:, 11].tolist() == [8, 8, 8] and pos[:, 12].tolist() == [9, 9, 9]
    assert (mrope_positions([1, 2, 3], IMG, []) == np.arange(3)).all()


def test_mrope_interleaves_sections_and_reduces_to_plain_rope():
    text = np.broadcast_to(np.arange(7), (3, 7))
    for a, b in zip(_mrope_cos_sin(text, 64, 1e7, [11, 11, 10], mx.float32), _rope_cos_sin(7, 64, 1e7, mx.float32)):
        assert mx.array_equal(a, b)
    pos = np.stack([np.zeros(2), np.ones(2) * 5, np.ones(2) * 9]).astype(int)
    cos, _ = _mrope_cos_sin(pos, 64, 1e7, [11, 11, 10], mx.float32)
    inv = 1.0 / (1e7 ** (np.arange(0, 64, 2) / 64))
    want = np.zeros(32)
    want[1:33:3] = 5 * inv[1:33:3]                             # h slots 1, 4, .., 31
    want[2:30:3] = 9 * inv[2:30:3]                             # w slots 2, 5, .., 29; t (= 0) elsewhere
    assert np.allclose(np.array(cos[0]), np.cos(np.concatenate([want, want])), atol=1e-6)


def test_smart_resize_and_vision_sizes(tmp_path):
    assert smart_resize(384, 384) == (384, 384)
    assert smart_resize(332, 443) == (320, 448)
    assert smart_resize(100, 100) == (256, 256)  # at least 65536 pixels
    from PIL import Image

    Image.new("RGB", (1024, 768)).save(tmp_path / "a.png")
    Image.new("RGB", (200, 100)).save(tmp_path / "b.png")
    assert load_vision(tmp_path / "a.png", 384 * 384).shape == (333, 443, 3)  # scaled down to <= 384^2
    assert load_vision(tmp_path / "b.png", 384 * 384).shape == (100, 200, 3)  # never up


def test_vision_patches_layout():
    rng = np.random.default_rng(0)
    pixels = rng.integers(0, 256, (64, 96, 3), dtype=np.uint8)
    pixels = np.asarray(__import__("PIL.Image", fromlist=["Image"]).fromarray(pixels).resize((256, 256)))
    patches, (gh, gw) = vision_patches(pixels)
    assert (gh, gw) == (16, 16) and patches.shape == (256, 3 * 2 * 16 * 16)
    x = (pixels.astype(np.float32) / 255 - 0.5) / 0.5

    def patch(r, c):  # (C, T, P, P), the frame doubled
        p = x[r * 16:(r + 1) * 16, c * 16:(c + 1) * 16].transpose(2, 0, 1)
        return np.stack([p, p], 1).reshape(-1)

    # each 2x2 merge group contiguous, groups in row-major order
    for row, (r, c) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1), (0, 2), (0, 3), (1, 2)]):
        assert np.allclose(patches[row], patch(r, c))
    assert np.allclose(patches[4 * 8], patch(2, 0))


def test_vision_tower_shapes_and_conv_as_linear():
    cfg = dict(hidden_size=32, num_heads=2, intermediate_size=48, depth=2, in_channels=3, temporal_patch_size=2,
               patch_size=16, num_position_embeddings=16, out_hidden_size=24, spatial_merge_size=2)
    tower = VisionTower(cfg)
    conv = mx.random.normal((32, 3, 2, 16, 16))
    flat = VisionTower.sanitize({"visual.patch_embed.proj.weight": conv, "language_model.x": conv})
    assert list(flat) == ["patch_proj.weight"] and flat["patch_proj.weight"].shape == (32, 1536)
    patches, grid = vision_patches(np.zeros((64, 96, 3), np.uint8) + 128)
    pixels = np.zeros((256, 256, 3), np.uint8)
    patches, grid = vision_patches(pixels)
    out = tower(mx.array(patches), grid)
    assert out.shape == (grid[0] * grid[1] // 4, 24)


def _tokenizer_dir():
    found = glob.glob(str(Path.home() / ".cache/huggingface/hub/models--VisualInference--MOSS-Image-SFT-MLX-*"
                                         "/snapshots/*/tokenizer/tokenizer.json"))
    return str(Path(found[0]).parent) if found else None


@pytest.mark.skipif(_tokenizer_dir() is None, reason="no cached MOSS tokenizer")
def test_token_layout_expands_image_pads():
    from transformers import AutoTokenizer

    enc = TextEncoder.__new__(TextEncoder)
    enc.tokenizer, enc.image_token_id = AutoTokenizer.from_pretrained(_tokenizer_dir()), IMG
    plain = enc.token_ids("Make it red")
    ids = enc.token_ids(vision_prompt("Make it red", 2), image_tokens=[4, 6])
    assert ids.count(IMG) == 10
    start, end = enc.tokenizer.convert_tokens_to_ids(["<|vision_start|>", "<|vision_end|>"])
    first = ids.index(IMG)
    assert ids[first - 1] == start and ids[first + 4] == end and ids[first:first + 4] == [IMG] * 4
    text = enc.tokenizer.decode(ids)
    assert text.count("Picture 1: ") == 1 and text.index("Picture 2: ") < text.index("Make it red")
    assert text.index("<|im_start|>user") < text.index("Picture 1: ")  # inside the user turn
    with pytest.raises(ValueError):
        enc.token_ids(vision_prompt("x", 1), image_tokens=[])
    with pytest.raises(ValueError):
        enc.token_ids(vision_prompt("x", 1), max_tokens=20, image_tokens=[64])


def test_prompt_cache_key_includes_images(tmp_path):
    te = tmp_path / "te"
    te.mkdir()
    (te / "config.json").write_text("{}")
    cache = PromptCache(te, tmp_path / "cache")
    h = mx.ones((1, 3, 4))
    a, b = np.zeros((32, 32, 3), np.uint8), np.ones((32, 32, 3), np.uint8)
    cache.put("p", 4096, h, [a])
    assert cache.get("p", 4096) is None and cache.get("p", 4096, [b]) is None
    assert mx.array_equal(cache.get("p", 4096, [a.copy()]), h)
    assert cache._path("p", 4096, []) == cache._path("p", 4096)  # text-only keys unchanged


def test_request_fields_and_prompt_key():
    r = Request(prompt="x", references=["a.png"], edit_format="ai-toolkit", vision_pixels=512 * 512)
    r.validate(42)
    assert prompt_key(r, "x") == ("x", 4096, (("a.png",), 262144))
    assert prompt_key(Request(prompt="x", references=["a.png"]), "x") == ("x", 4096, ())  # chinoll: text only
    with pytest.raises(ValueError):
        Request(prompt="x", edit_format="qwen").validate(42)

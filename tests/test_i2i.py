"""Image-to-image and inpainting helpers: no model weights needed."""

import mlx.core as mx
import numpy as np
import pytest

from moss_image_mlx.pipeline import Request, latent_mask, start_index
from moss_image_mlx.vae import pack_latents, unpack_latents


def test_validate_rules():
    Request(prompt="x").validate(42)
    Request(prompt="x", init_image="a.png", strength=0.0).validate(42)
    Request(prompt="x", init_image="a.png", mask="m.png").validate(42)
    with pytest.raises(ValueError, match="strength"):
        Request(prompt="x", init_image="a.png", strength=1.5).validate(42)
    with pytest.raises(ValueError, match="strength"):
        Request(prompt="x", strength=-0.1).validate(42)
    with pytest.raises(ValueError, match="mask"):
        Request(prompt="x", mask="m.png").validate(42)


def test_start_index():
    assert start_index(8, 1.0) == 0          # full schedule
    assert start_index(8, 0.0) == 8          # nothing to run
    assert start_index(8, 0.5) == 4
    assert start_index(50, 0.6) == 20
    assert 0 <= start_index(7, 0.33) <= 7


def test_latent_mask_shape_and_softness():
    full = np.zeros((64, 48), dtype=np.float32)
    full[:16, :16] = 1          # exactly one patch
    full[16:32, 16:24] = 1      # half a patch
    m = latent_mask(full, 64, 48)
    assert m.shape == (1, 1, 4, 3)
    assert m[0, 0, 0, 0] == 1.0 and m[0, 0, 1, 1] == 0.5 and m[0, 0, 3, 2] == 0.0
    assert 0 <= m.min() and m.max() <= 1


def test_pack_unpack_inverse():
    z = mx.random.normal((2, 128, 5, 7))
    assert mx.array_equal(pack_latents(unpack_latents(z)), z)
    x = mx.random.normal((1, 10, 14, 32))
    assert mx.array_equal(unpack_latents(pack_latents(x)), x)
    # the channel layout matches the reference unpatchify: channel 4k + 2i + j -> pixel (2y + i, 2x + j), channel k
    z = mx.arange(128 * 4).reshape(1, 128, 2, 2).astype(mx.float32)
    x = unpack_latents(z)
    k, i, j, y, xx = 5, 1, 0, 1, 1
    assert float(x[0, 2 * y + i, 2 * xx + j, k]) == float(z[0, 4 * k + 2 * i + j, y, xx])


def test_mask_feather(tmp_path):
    import numpy as np
    from PIL import Image

    from moss_image_mlx.pipeline import load_mask

    m = np.zeros((256, 256), np.uint8)
    m[100:140, 60:200] = 255
    Image.fromarray(m).save(tmp_path / "m.png")
    hard = load_mask(tmp_path / "m.png", 256, 256, 0)
    soft = load_mask(tmp_path / "m.png", 256, 256, 16)
    assert hard[120, 100] == 1 and hard[90, 100] == 0
    assert soft[120, 100] == 1.0, "painted pixels stay fully regenerated"
    assert soft[94, 100] > 0.8 and 0 < soft[80, 100] < 0.8 and soft[60, 100] == 0, "dilated edge then a ramp to zero"

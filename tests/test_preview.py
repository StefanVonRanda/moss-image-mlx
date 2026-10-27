"""Live preview: x0 estimate, RGB projection, fitting and the per-step file. No model weights needed."""

import mlx.core as mx
import numpy as np
from PIL import Image

from moss_image_mlx import preview
from moss_image_mlx.pipeline import Request, _sample


def test_x0_estimate_inverts_the_flow():
    rng = np.random.default_rng(0)
    x0, noise, sigma = rng.normal(size=(1, 128, 2, 3)), rng.normal(size=(1, 128, 2, 3)), 0.7
    xt, v = (1 - sigma) * x0 + sigma * noise, noise - x0          # flow matching: v = d x_t / d sigma
    assert np.allclose(preview.x0_estimate(xt, sigma, v), x0)
    assert np.allclose(xt + (0 - sigma) * v, x0)                  # _sample's Euler update, stepped to sigma 0


def test_to_rgb_shape_dtype_range():
    z = mx.array(np.random.default_rng(1).normal(scale=50, size=(1, 128, 4, 6)).astype(np.float32))
    rgb = preview.to_rgb(z)
    assert rgb.shape == (4, 6, 3) and rgb.dtype == np.uint8
    assert rgb.min() == 0 and rgb.max() == 255                    # extreme latents clip instead of wrapping
    flat = preview.to_rgb(np.zeros((1, 128, 1, 1), np.float32))[0, 0]
    assert np.array_equal(flat, np.round(preview.RGB_BIAS * 255))


def test_fit_recovers_a_linear_map():
    rng = np.random.default_rng(2)
    w, b = rng.normal(scale=0.01, size=(128, 3)), np.array([0.5, 0.4, 0.3])
    latents = [rng.normal(size=(1, 128, 8, 12)).astype(np.float32) for _ in range(3)]  # 288 rows, 129 unknowns
    images = [(np.tensordot(z[0], w, axes=(0, 0)) + b) * 255 for z in latents]
    images = [im.repeat(16, axis=0).repeat(16, axis=1) for im in images]   # area average undoes this exactly
    fw, fb = preview.fit(latents, images)
    assert np.allclose(fw, w, atol=1e-3) and np.allclose(fb, b, atol=1e-3)


class _Flat:
    """Stand-in DiT: predicts v = 0, so the estimate is the starting noise at every step."""
    layers = [None] * 24
    config = {"latent_channels": 128}

    def __call__(self, latents, t, context, skip=(), references=None, control=None):
        return mx.zeros_like(latents)

    def named_modules(self):  # no adapters to switch
        return []


def test_sample_writes_the_preview_each_step(tmp_path):
    out = tmp_path / "live.png"
    seen = []
    req = Request(prompt="x", width=96, height=32, steps=3, guidance=1.0, preview=str(out))
    final = _sample(_Flat(), req, None, None, None, lambda i, n: seen.append(np.asarray(Image.open(out))))
    assert len(seen) == 3 and seen[0].shape == (2, 6, 3)
    assert np.array_equal(seen[-1], preview.to_rgb(final))
    assert list(tmp_path.iterdir()) == [out]                      # written via a temp file and renamed


def test_turbo_writes_the_preview_each_step(tmp_path):
    from moss_image_mlx import turbo

    out = tmp_path / "live.png"
    seen = []
    req = Request(prompt="x", width=96, height=32, turbo=True, preview=str(out))
    final = turbo.sample(_Flat(), req, None, 0, None, lambda i, n: seen.append(np.asarray(Image.open(out))))
    assert len(seen) == len(turbo.ACTORS) + len(turbo.FINISH_IDX) and seen[0].shape == (2, 6, 3)
    assert np.array_equal(seen[-1], preview.to_rgb(final))

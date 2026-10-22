"""Live preview: the denoiser's current clean-image estimate as a latent-resolution RGB PNG, without the VAE.

Flow matching runs x_t = (1 - sigma) x0 + sigma noise and steps x += (sigma_next - sigma) v with v = noise - x0, so
the estimate is x0 = x_t - sigma v. A fixed linear map (+ bias) takes its 128 packed channels to RGB at H/16 x W/16.
The map is a least-squares fit of real final latents against their VAE decodes area-averaged to that size. Refit:
  python -m moss_image_mlx.preview --model compact --lora ~/moss-mlx/loras/MOSS-Image-SFT-8-steps-dmd.safetensors
and paste the printed RGB_WEIGHTS / RGB_BIAS over the ones below.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

# Fitted on 10 renders of PROMPTS (compact + 8-step DMD LoRA, 512^2, 8 steps); held-out PSNR 20.5 dB at H/16.
RGB_WEIGHTS = np.array([ 0.0161,  0.0088,  0.0041,  0.0115, -0.0005,  0.0024,  0.0042, -0.0027,  0.0062,  0.0060,  0.0005,  0.0078,
  0.0014, -0.0000,  0.0073, -0.0032,  0.0017,  0.0132, -0.0002, -0.0058,  0.0125, -0.0013, -0.0003,  0.0189,
  0.0057, -0.0053, -0.0092,  0.0010, -0.0033, -0.0048,  0.0068,  0.0010,  0.0026,  0.0036, -0.0032, -0.0044,
 -0.0122,  0.0338,  0.0829,  0.0163,  0.0405,  0.0771,  0.0139,  0.0465,  0.0776,  0.0318,  0.0516,  0.0739,
 -0.0021, -0.0056,  0.0015, -0.0040, -0.0104, -0.0061,  0.0064,  0.0038, -0.0003,  0.0027, -0.0033, -0.0058,
 -0.0017, -0.0117, -0.0109, -0.0044, -0.0126, -0.0122,  0.0007, -0.0061, -0.0058, -0.0033, -0.0075, -0.0139,
 -0.0141,  0.0121, -0.0095, -0.0147,  0.0055, -0.0036, -0.0064,  0.0114, -0.0071, -0.0076,  0.0163, -0.0051,
  0.0058,  0.0013,  0.0013,  0.0131,  0.0106,  0.0078, -0.0060, -0.0046, -0.0028,  0.0063, -0.0009, -0.0019,
 -0.0657, -0.0476, -0.0129, -0.0674, -0.0434, -0.0027, -0.0641, -0.0374,  0.0051, -0.0539, -0.0208,  0.0318,
 -0.0031, -0.0065,  0.0029, -0.0053, -0.0129, -0.0019, -0.0003, -0.0027, -0.0053, -0.0025, -0.0148, -0.0100,
 -0.0016, -0.0015, -0.0036, -0.0019, -0.0029, -0.0012, -0.0024, -0.0031, -0.0006, -0.0043, -0.0024, -0.0039,
 -0.0046, -0.0096, -0.0209, -0.0016, -0.0002, -0.0131,  0.0013,  0.0038, -0.0096,  0.0010,  0.0022, -0.0135,
  0.0035,  0.0053,  0.0047,  0.0053,  0.0098,  0.0074,  0.0042, -0.0004,  0.0005,  0.0038,  0.0009,  0.0053,
 -0.0034, -0.0038, -0.0167, -0.0012, -0.0050, -0.0017, -0.0030,  0.0007, -0.0013,  0.0050,  0.0041, -0.0053,
  0.0080,  0.0048, -0.0034,  0.0017,  0.0020, -0.0021, -0.0014, -0.0039, -0.0076,  0.0013,  0.0016, -0.0019,
 -0.0297,  0.0050,  0.0035, -0.0304,  0.0078,  0.0019, -0.0343, -0.0045, -0.0066, -0.0331,  0.0056, -0.0020,
 -0.0052, -0.0053, -0.0058, -0.0073, -0.0074, -0.0072, -0.0013,  0.0011,  0.0042,  0.0002,  0.0035,  0.0094,
  0.0009, -0.0024, -0.0046,  0.0092,  0.0036, -0.0013, -0.0059,  0.0047, -0.0037, -0.0005,  0.0005, -0.0053,
 -0.0032, -0.0003,  0.0037,  0.0069,  0.0064,  0.0058,  0.0042,  0.0111, -0.0020,  0.0026,  0.0098,  0.0003,
  0.0264, -0.0193, -0.0061,  0.0264, -0.0179,  0.0029,  0.0307, -0.0167,  0.0141,  0.0256, -0.0151,  0.0115,
 -0.0056,  0.0015,  0.0057, -0.0028,  0.0047,  0.0058, -0.0009,  0.0018,  0.0026, -0.0038, -0.0043, -0.0009,
  0.0135,  0.0013,  0.0087,  0.0082, -0.0031,  0.0006,  0.0042, -0.0066, -0.0025,  0.0056, -0.0042, -0.0028,
  0.0048,  0.0143, -0.0022,  0.0013,  0.0073,  0.0002, -0.0024,  0.0095,  0.0034, -0.0053,  0.0110,  0.0045,
 -0.0008,  0.0067, -0.0007, -0.0035,  0.0021,  0.0007, -0.0081, -0.0022, -0.0030, -0.0137, -0.0066, -0.0019,
 -0.0034, -0.0006, -0.0005,  0.0093,  0.0007, -0.0118, -0.0010,  0.0023, -0.0036,  0.0197, -0.0024, -0.0109,
 -0.0014,  0.0045,  0.0007, -0.0011,  0.0057,  0.0034,  0.0029,  0.0078,  0.0086,  0.0073,  0.0073,  0.0057,
  0.0050, -0.0119, -0.0000,  0.0088, -0.0093, -0.0041,  0.0075, -0.0102,  0.0030,  0.0106, -0.0054, -0.0011,
  0.0157,  0.0021,  0.0018,  0.0147,  0.0034,  0.0007,  0.0116, -0.0013,  0.0027,  0.0077, -0.0035,  0.0074,
 -0.0089, -0.0037, -0.0026, -0.0110, -0.0021, -0.0010, -0.0088, -0.0033, -0.0109, -0.0060, -0.0004, -0.0064,
  0.0032,  0.0089,  0.0010, -0.0042,  0.0092, -0.0003, -0.0024,  0.0090,  0.0011, -0.0027,  0.0055,  0.0016,
 -0.0048, -0.0016, -0.0033, -0.0062,  0.0018, -0.0075, -0.0008, -0.0044, -0.0055,  0.0034,  0.0063,  0.0013,
  0.0010, -0.0053,  0.0001,  0.0030, -0.0022,  0.0012, -0.0013, -0.0059, -0.0116,  0.0077, -0.0026, -0.0078], np.float32).reshape(128, 3)
RGB_BIAS = np.array([0.4403, 0.4384, 0.3896], np.float32)


def x0_estimate(latents, sigma: float, prediction):
    return latents - sigma * prediction


def to_rgb(latents) -> np.ndarray:
    """Packed latents (1, 128, h, w) -> uint8 (h, w, 3)."""
    z = np.asarray(latents[0], dtype=np.float32)
    rgb = np.tensordot(z, RGB_WEIGHTS, axes=(0, 0)) + RGB_BIAS
    return (np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8)


def write(latents, path: str | Path):
    """Atomic: a reader sees the previous preview or this one, never half a file."""
    from PIL import Image

    tmp = f"{path}.tmp"
    Image.fromarray(to_rgb(latents)).save(tmp, format="PNG")
    os.replace(tmp, path)


def fit(latents: list, images: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares (weights (128, 3), bias (3,)) from final packed latents (1, 128, h, w) and their decodes
    (uint8 (16h, 16w, 3)), each decode area-averaged over 16 x 16 pixels to latent resolution."""
    x = np.concatenate([np.asarray(z[0], np.float32).reshape(z.shape[1], -1).T for z in latents])
    y = np.concatenate([im.reshape(im.shape[0] // 16, 16, im.shape[1] // 16, 16, 3).mean(axis=(1, 3)).reshape(-1, 3)
                        / 255 for im in images])
    solution = np.linalg.lstsq(np.hstack([x, np.ones((len(x), 1), np.float32)]), y, rcond=None)[0]
    return solution[:-1].astype(np.float32), solution[-1].astype(np.float32)


PROMPTS = ["A red sports car on a sunny street", "A green forest with a river and moss-covered stones",
           "A blue ocean with a white sailboat at sunset", "Close-up portrait of a woman with freckles, soft light",
           "Oranges, grapes and lemons in a bowl on a purple tablecloth", "A city street at night with neon signs",
           "A poster with the word HELLO in yellow letters on pink", "Snowy mountains under a grey sky",
           "A brown dog on green grass", "A bright abstract painting with cyan, magenta and orange shapes"]


def main(argv=None):
    import argparse

    from .cli import _lora_spec, resolve_model
    from .pipeline import MOSSImagePipeline, Request

    p = argparse.ArgumentParser(description="Refit RGB_WEIGHTS / RGB_BIAS from real renders of PROMPTS")
    p.add_argument("--model", default="compact")
    p.add_argument("--lora", action="append", default=[])
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--size", type=int, default=512)
    args = p.parse_args(argv)
    pipe = MOSSImagePipeline(resolve_model(args.model), loras=[_lora_spec(x) for x in args.lora])
    latents, images = [], []
    embeds = pipe.encode(PROMPTS, Request(prompt="").max_sequence_length)
    for i, (prompt, pos) in enumerate(zip(PROMPTS, embeds)):
        req = Request(prompt=prompt, width=args.size, height=args.size, steps=args.steps, guidance=1.0, seed=i)
        z = pipe.denoise(req, pos, None, keep=True)
        latents.append(z)
        images.append(pipe.decode(z, keep=True))
    weights, bias = fit(latents, images)
    w, b = fit(latents[1:], images[1:])  # quality on an image the fit has not seen
    guess = np.clip(np.tensordot(np.asarray(latents[0][0], np.float32), w, axes=(0, 0)) + b, 0, 1) * 255
    small = images[0].reshape(args.size // 16, 16, args.size // 16, 16, 3).mean(axis=(1, 3))
    mse = np.mean((guess - small) ** 2)
    print(f"# held-out PSNR at latent resolution: {10 * np.log10(255 ** 2 / mse):.1f} dB")
    np.set_printoptions(precision=4, suppress=True, threshold=10_000, linewidth=116, floatmode="fixed")
    print("RGB_WEIGHTS = np.array(" + np.array2string(weights.ravel(), separator=", ") + ", np.float32).reshape(128, 3)")
    print("RGB_BIAS = np.array(" + np.array2string(bias, separator=", ") + ", np.float32)")


if __name__ == "__main__":
    main()

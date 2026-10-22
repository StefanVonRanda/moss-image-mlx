"""Turbo: about 2 minutes per ~1 MP image, 15 network passes instead of the official 100 (50 steps x CFG).

A trained CFG-free student (a rank-64 LoRA on the SFT weights) runs the first half of the original model's own
50-step trajectory in 6 steps; at the second of them (sigma .957 -> .919, where lettering is decided) the original
model takes one guided step instead. The original model then finishes with 4 DPM-Solver++(2M) evaluations
(sigma .671 -> 0). The destripe LoRA (nynxz) is applied throughout. Research notes, evaluation and decisions:
https://github.com/felix-ab/moss-image-turbo-research
"""

import mlx.core as mx
import numpy as np

from .lora import set_strength
from .scheduler import sigmas_for
from .torch_randn import randn

# The student, trained on ~2,000 prompts at ~1 MP in 9 aspect ratios against the destriped original model.
STUDENT = {"repo": "VisualInference/MOSS-Image-SFT-Turbo-MLX", "file": "turbo_student.safetensors", "revision": None}
# nynxz's SFT destripe adapter (Apache-2.0), pinned; strength 1.5 at ~1 MP.
DESTRIPE = {"repo": "nynxz/MOSS-Image", "file": "loras/moss_sft_destripe_top64_bf16.safetensors",
            "revision": "7b071d5915362f0101f0aa913315fd10a0772059", "strength": 1.5}
PREFIX_IDX = [0, 4, 9, 15, 21, 28, 35]  # student steps on the official 50-step grid (sigma 1 -> .671)
ACTORS = "SSTSSS"  # S = student (no CFG), T = original model with CFG
FINISH_IDX = [35, 40, 44, 47]  # original-model DPM-Solver++(2M) evaluations, then sigma 0
PASSES = ACTORS.count("S") + 2 * ACTORS.count("T") + 2 * len(FINISH_IDX)


def schedule():
    """(prefix sigmas as float32, finish sigmas) on the official 50-step grid."""
    g = sigmas_for(50, 5.0)
    return g[PREFIX_IDX], [float(g[i]) for i in FINISH_IDX] + [0.0]


def sample(model, req, positive, negative, latents=None, on_step=None):
    if negative is None:
        raise ValueError("turbo needs the negative-prompt embedding (CFG for the original-model steps)")
    if latents is None:
        latents = mx.array(randn((1, model.config["latent_channels"], req.height // 16, req.width // 16), req.seed))
    skip = tuple(req.uncond_skip_layers)
    prefix, finish = schedule()
    n = len(ACTORS) + len(finish) - 1
    f32 = mx.float32

    def velocity(x, sigma, student: bool):
        set_strength(model, 1.0 if student else 0.0, "student")
        t = float(np.float32(sigma) * np.float32(1000))
        cond = model(x, t, positive).astype(f32)
        if student:
            return cond
        uncond = model(x, t, negative, skip).astype(f32)
        return uncond + req.guidance * (cond - uncond)

    x = latents.astype(f32)
    for i, actor in enumerate(ACTORS):
        v = velocity(x, prefix[i], actor == "S")
        x = x + mx.array(prefix[i + 1] - prefix[i], dtype=f32) * v  # float32 difference, as the reference
        mx.eval(x)
        if on_step:
            on_step(i, n)

    lam = lambda s: float(np.log((1 - s) / s))  # noqa: E731
    prev = None  # (lambda, x0) of the previous evaluation
    for j in range(len(finish) - 1):
        s, s2 = finish[j], finish[j + 1]
        x0 = x - s * velocity(x, s, False)
        if s2 == 0:
            x = x0
        else:
            d = x0
            if prev is not None:
                r = (lam(s) - prev[0]) / (lam(s2) - lam(s))
                d = (1 + 1 / (2 * r)) * x0 - (1 / (2 * r)) * prev[1]
            x = (s2 / s) * x + ((1 - s2) - (1 - s) * s2 / s) * d
            prev = (lam(s), x0)
        mx.eval(x)
        if on_step:
            on_step(len(ACTORS) + j, n)
    set_strength(model, 1.0, "student")
    return x

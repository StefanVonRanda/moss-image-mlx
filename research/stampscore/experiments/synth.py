"""Synthetic perturbations of clean bases (destriped references + admitted real photos).

stamps: a 48-px motif from the image centre pasted on a 70-px grid over the central third (as recurrence.calibrate)
  aligned    stamp origins on the 14-px token grid
  unaligned  every origin +7 px (worst case for a single tiling)
  jitter     each origin + uniform 0..13 px independently
  soft       jitter, alpha 0.5 blend (partial stamping)
degradations: gaussian noise sd 6; gaussian blur sigma 1.5 / 3 / 6 (6 ~ posterior-mean-like)
shifts: crop (k, k) .. -(14 - k) for k in 0, 3, 7, 10 (same size; phase invariance of the score itself)
"""
import numpy as np
from PIL import Image, ImageFilter

from dataset import E_PIDS, H_PIDS, REAL, RUNS, ADMITTED_REAL
from stampscore.core import load


def bases():
    b = [(f"E_ref_d/{p}_s{s}", RUNS / "E_ref_d" / f"{p}_s{s}.png") for p in E_PIDS for s in (101, 202)]
    b += [(f"H_ref_d/{p}_s101", RUNS / "H_ref_d" / f"{p}_s101.png") for p in H_PIDS]
    b += [(f"real/{d}/{f.stem}", f) for d in ADMITTED_REAL for f in sorted((REAL / d).glob("*.jpg"))]
    return [(k, p) for k, p in b if p.exists()]


def stamp(img, mode, seed=0, size=48, step=70):
    h, w = img.shape[:2]
    rng = np.random.default_rng(seed)
    out = img.copy().astype(np.float64)
    motif = img[h // 2 - size // 2:h // 2 + size // 2, w // 2 - size // 2:w // 2 + size // 2].astype(np.float64)
    for y in range(h // 42 * 14, 2 * h // 3 - size, step):
        for x in range(w // 42 * 14, 2 * w // 3 - size, step):
            jy, jx = {"aligned": (0, 0), "unaligned": (7, 7)}.get(mode, tuple(rng.integers(0, 14, 2)))
            a = 0.5 if mode == "soft" else 1.0
            sl = np.s_[y + jy:y + jy + size, x + jx:x + jx + size]
            out[sl] = a * motif + (1 - a) * out[sl]
    return np.clip(out, 0, 255).astype(np.uint8)


def blur(img, s):
    return np.array(Image.fromarray(img).filter(ImageFilter.GaussianBlur(s)))


VARIANTS = {
    "orig": lambda a: a,
    "aligned": lambda a: stamp(a, "aligned"),
    "unaligned": lambda a: stamp(a, "unaligned"),
    "jitter": lambda a: stamp(a, "jitter"),
    "soft": lambda a: stamp(a, "soft"),
    "noise6": lambda a: np.clip(a + np.random.default_rng(0).normal(0, 6, a.shape), 0, 255).astype(np.uint8),
    "blur1.5": lambda a: blur(a, 1.5),
    "blur3": lambda a: blur(a, 3),
    "blur6": lambda a: blur(a, 6),
    **{f"shift{k}": (lambda k: lambda a: np.ascontiguousarray(a[k:a.shape[0] - 14 + k, k:a.shape[1] - 14 + k]))(k)
       for k in (0, 3, 7, 10)},
}


def synth_jobs():
    jobs = []
    for key, p in bases():
        for v, fn in VARIANTS.items():
            L = (lambda p, fn: lambda: fn(load(p, 1024)))(p, fn)
            for ph in ("grid", "p4", "p9"):
                jobs.append((f"synth/{v}/{key}", L, ph, (6, 9), (56.0,)))
    return jobs

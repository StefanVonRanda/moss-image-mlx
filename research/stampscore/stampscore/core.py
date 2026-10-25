"""stampscore core: within-image near-duplicate texture ("stamping") from DINOv2 patch tokens at 1:1.

For every textured token (local luma sd > SD_FLOOR), find its most similar token at least MIN_DIST px away; dup_frac is
the share whose nearest-neighbour cosine is >= DUP. Tokens are pooled over several tilings offset by a fraction of the
14-px patch (PHASES), so a repeated motif that is misaligned with the ViT grid in one tiling is near-aligned in another.

A detail guard rejects images that are not finished photographs: blur (mid-band luma energy far below the reference's)
and decoded posterior means, which also carry the VAE decoder's fine periodic lattice (fine/mid band ratio far above
the reference's). Both read as repetitive to DINO.

Same defaults as tq/recurrence.py (dinov2-small, layer 6, MIN_DIST 56, SD_FLOOR .02) except the phase pooling and
DUP .80 (was .85; recalibrated on Felix's blind severity ratings and checked on independent data, REPORT.md §10).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn
from PIL import Image

torch.set_grad_enabled(False)
TILE, PATCH = 224, 14
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
NETS = {"dinov2-small": "facebook/dinov2-small", "dinov2-base": "facebook/dinov2-base"}
PHASES = {  # tiling offsets (dy, dx); worst-case misalignment with the patch grid in brackets
    "grid": ((0, 0),),  # 7 px (the original recurrence.py)
    "p4": ((0, 0), (0, 7), (7, 0), (7, 7)),  # 3.5 px (novelty/rec_phase.py)
    "p9": tuple((a, b) for a in (0, 5, 9) for b in (0, 5, 9)),  # 2.5 px
}
DUP, MIN_DIST, SD_FLOOR, LAYER = 0.80, 56.0, 0.02, 6  # DUP .85 reproduces recurrence.py / rec_phase.py
DETAIL_MIN_RATIO = 0.6  # candidate / reference mid-band (1-4 px) energy below this: low_detail (REPORT.md, guard)
LATTICE_MAX_RATIO = 1.6  # candidate / reference fine-to-mid ratio above this: decoder_lattice (clean outputs <= 1.46)
DETAIL_MIN_ABS = 0.010  # no-reference floor on mid-band rms luma (finished outputs >= .024, real photos >= .013)
MIN_TOKENS = 50


def load(path, max_side=None) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    a = np.array(im)
    if max_side and max(a.shape[:2]) > max_side:  # centre crop, never resample (the score is scale-dependent)
        h, w = a.shape[:2]
        sh, sw = min(h, max_side), min(w, max_side)
        y, x = (h - sh) // 2, (w - sw) // 2
        a = a[y:y + sh, x:x + sw]
    return np.ascontiguousarray(a)


def _gauss(x: torch.Tensor, s: float) -> torch.Tensor:
    r = max(1, int(3 * s + 0.5))
    k = torch.exp(-torch.arange(-r, r + 1, dtype=torch.float64) ** 2 / (2 * s * s))
    k = (k / k.sum()).to(x.dtype)
    x = Fn.conv2d(Fn.pad(x, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1))
    return Fn.conv2d(Fn.pad(x, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1))


def bands(img: np.ndarray) -> tuple[float, float]:
    """RMS luma (0..1) of the fine band (below gaussian sigma 1 px) and the mid band (sigma 1-4 px)."""
    L = torch.from_numpy(img.astype(np.float64).mean(-1) / 255)[None, None]
    g1, g4 = _gauss(L, 1.0), _gauss(L, 4.0)
    return float(((L - g1) ** 2).mean().sqrt()), float(((g1 - g4) ** 2).mean().sqrt())


def detail(img: np.ndarray) -> float:
    """Mid-band rms luma: blur drives it towards 0 (the fine band is unreliable: decoder lattices and JPEG fill it)."""
    return bands(img)[1]


class Extractor:
    def __init__(self, net="dinov2-small", device="cpu", batch=16):
        from transformers import AutoModel
        repo = NETS.get(net, net)
        snap = Path.home() / f".cache/huggingface/hub/models--{repo.replace('/', '--')}/snapshots"
        src = str(next(snap.iterdir())) if snap.exists() else repo
        self.model = AutoModel.from_pretrained(src).eval().to(device)
        self.net, self.device, self.batch = net, device, batch
        self.depth = self.model.config.num_hidden_layers

    def tokens(self, img: np.ndarray, layers=(LAYER,), phases="p4"):
        """Token features per layer (N, C, L2-normalised), centres (N, 2) and local luma sd (N), pooled over tilings."""
        offs = PHASES[phases] if isinstance(phases, str) else phases
        lum = img.astype(np.float64).mean(-1) / 255
        feats = {l: [] for l in layers}
        pos, sd = [], []
        g = TILE // PATCH
        for dy, dx in offs:
            sub = img[dy:, dx:]
            h, w = sub.shape[:2]
            ys = list(range(0, h - TILE + 1, TILE)) + ([h - TILE] if h % TILE else [])
            xs = list(range(0, w - TILE + 1, TILE)) + ([w - TILE] if w % TILE else [])
            corners = [(y, x) for y in ys for x in xs]
            F = {l: [] for l in layers}
            for i in range(0, len(corners), self.batch):
                t = torch.stack([torch.from_numpy(sub[y:y + TILE, x:x + TILE]).permute(2, 0, 1)
                                 for y, x in corners[i:i + self.batch]]).float() / 255
                out = self.model(pixel_values=((t - MEAN) / STD).to(self.device), output_hidden_states=True)
                for l in layers:
                    F[l].append(out.hidden_states[l][:, 1:].float().cpu())
            P = []
            for y, x in corners:
                for k in range(g * g):
                    r, c = divmod(k, g)
                    P.append((y + r * PATCH + PATCH // 2 + dy, x + c * PATCH + PATCH // 2 + dx))
            P = np.array(P, float)
            _, idx = np.unique(P, axis=0, return_index=True)  # overlapping edge tiles: one token per position
            for l in layers:
                feats[l].append(torch.cat(F[l]).reshape(-1, F[l][0].shape[-1]).numpy()[idx])
            pos.append(P[idx])
            sd.append(np.array([lum[int(cy) - 7:int(cy) + 7, int(cx) - 7:int(cx) + 7].std() for cy, cx in P[idx]]))
        pos, sd = np.concatenate(pos), np.concatenate(sd)
        out = {}
        for l in layers:
            f = np.concatenate(feats[l]).astype(np.float32)
            out[l] = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-8)
        return out, pos, sd


def nn_cos(f: np.ndarray, pos: np.ndarray, min_dists=(MIN_DIST,), chunk=2048) -> dict:
    """Nearest-neighbour cosine of every token, excluding partners closer than each min_dist (one matmul for all)."""
    ft = torch.from_numpy(f)
    pt = torch.from_numpy(pos.astype(np.float32))
    res = {m: np.empty(len(f), np.float32) for m in min_dists}
    for i in range(0, len(f), chunk):
        s = ft[i:i + chunk] @ ft.T
        d = torch.cdist(pt[i:i + chunk], pt)
        for m in min_dists:
            res[m][i:i + chunk] = s.masked_fill(d < m, -1).max(1).values.numpy()
    return res


@dataclass
class Score:
    dup_frac: float | None
    n: int
    detail: float
    status: str = "ok"  # ok | low_detail | decoder_lattice | too_few_tokens
    ref_dup_frac: float | None = None
    excess: float | None = None
    detail_ratio: float | None = None
    lattice_ratio: float | None = None
    extra: dict = field(default_factory=dict)

    def as_dict(self):
        d = {k: v for k, v in self.__dict__.items() if v is not None and k != "extra"}
        return {**d, **self.extra}


class StampScorer:
    def __init__(self, net="dinov2-small", layer=LAYER, phases="p4", dup=DUP, min_dist=MIN_DIST, sd_floor=SD_FLOOR,
                 device="cpu", threads=None):
        if threads:
            torch.set_num_threads(threads)
        self.ex = Extractor(net, device)
        self.layer, self.phases, self.dup, self.min_dist, self.sd_floor = layer, phases, dup, min_dist, sd_floor

    def raw(self, img: np.ndarray, mask: np.ndarray | None = None):
        f, p, sd = self.ex.tokens(img, (self.layer,), self.phases)
        keep = sd > self.sd_floor
        if mask is not None:
            keep &= mask[p[:, 0].astype(int), p[:, 1].astype(int)]
        f, p = f[self.layer][keep], p[keep]
        if len(f) < MIN_TOKENS:
            return None, len(f)
        nn = nn_cos(f, p, (self.min_dist,))[self.min_dist]
        return float((nn >= self.dup).mean()), len(f)

    def score(self, img: np.ndarray, ref: np.ndarray | None = None, mask=None, guard=True) -> Score:
        dup, n = self.raw(img, mask)
        fine, mid = bands(img)
        s = Score(dup, n, mid)
        if dup is None:
            s.status = "too_few_tokens"
        if ref is not None:
            rd, _ = self.raw(ref, mask)
            rf, rm = bands(ref)
            s.ref_dup_frac = rd
            s.detail_ratio = mid / (rm + 1e-9)
            s.lattice_ratio = (fine / (mid + 1e-9)) / (rf / (rm + 1e-9) + 1e-9)
            if dup is not None and rd is not None:
                s.excess = dup - rd
            if guard and s.detail_ratio < DETAIL_MIN_RATIO:
                s.status = "low_detail"
            elif guard and s.lattice_ratio > LATTICE_MAX_RATIO:
                s.status = "decoder_lattice"
        elif guard and mid < DETAIL_MIN_ABS:
            s.status = "low_detail"
        return s


def default_threads():
    return int(os.environ.get("STAMPSCORE_THREADS", min(6, os.cpu_count() or 4)))

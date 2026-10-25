"""Standard image metrics on the same data, to show what they do (not) see of stamping.

  lpips     LPIPS-alex to the same-seed destriped reference (full res)            lower = closer to the reference
  sifid     SIFID (SinGAN): Frechet distance of Inception first-pool (64-ch) patch statistics, image vs reference
  clipiqa   CLIP-IQA (pyiqa), no reference                                          higher = "better"
  musiq     MUSIQ-KonIQ (pyiqa), no reference, native resolution                    higher = "better"
  patch-FID / patch-KID per run: Inception pool3 on non-overlapping 299-px crops at 1:1, run vs the references of the
            same prompts and seeds (set-level; KID is the unbiased small-sample variant).

Usage: metrics_std.py main | synth | crops | crops_r2 | patchfid      -> out/std_<what>.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision
from scipy import linalg

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import OUT, RUNS, items  # noqa: E402
from stampscore.core import load  # noqa: E402

torch.set_grad_enabled(False)
torch.set_num_threads(4)
IMNET = (torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))


class Std:
    def __init__(self):
        import lpips
        import pyiqa
        self.lp = lpips.LPIPS(net="alex", verbose=False).eval()
        self.iqa = {n: pyiqa.create_metric(n, device="cpu") for n in ("clipiqa", "musiq")}
        inc = torchvision.models.inception_v3(weights="IMAGENET1K_V1").eval()
        inc.fc = torch.nn.Identity()
        self.inc = inc
        self.stem = torch.nn.Sequential(inc.Conv2d_1a_3x3, inc.Conv2d_2a_3x3, inc.Conv2d_2b_3x3, inc.maxpool1)

    @staticmethod
    def t(a):
        return torch.from_numpy(a).permute(2, 0, 1)[None].float() / 255

    def _incin(self, x):
        return (x - IMNET[0]) / IMNET[1]

    def sifid_stats(self, a):
        x = self.inc._transform_input(self._incin(self.t(a)))
        f = self.stem(x)[0].flatten(1).T.double().numpy()
        return f.mean(0), np.cov(f, rowvar=False)

    def pool3(self, crops):
        out = []
        for i in range(0, len(crops), 16):
            x = torch.cat([self.t(c) for c in crops[i:i + 16]])
            out.append(self.inc(self._incin(x)).double().numpy())
        return np.concatenate(out)

    def all(self, a, ref=None):
        x = self.t(a)
        r = {n: float(m(x)) for n, m in self.iqa.items()}
        if ref is not None and ref.shape == a.shape:
            r["lpips"] = float(self.lp(x * 2 - 1, self.t(ref) * 2 - 1))
            r["sifid"] = frechet(*self.sifid_stats(a), *self.sifid_stats(ref))
        return r


def frechet(m1, s1, m2, s2):
    cs = linalg.sqrtm(s1 @ s2)
    return float(((m1 - m2) ** 2).sum() + np.trace(s1 + s2 - 2 * cs.real))


def kid(x, y, sub=100, reps=50, seed=0):
    rng = np.random.default_rng(seed)
    d = x.shape[1]
    k = lambda a, b: (a @ b.T / d + 1) ** 3  # noqa: E731
    vals = []
    m = min(sub, len(x), len(y))
    for _ in range(reps):
        a, b = x[rng.choice(len(x), m, False)], y[rng.choice(len(y), m, False)]
        kxx, kyy, kxy = k(a, a), k(b, b), k(a, b)
        vals.append((kxx.sum() - np.trace(kxx)) / (m * (m - 1)) + (kyy.sum() - np.trace(kyy)) / (m * (m - 1))
                    - 2 * kxy.mean())
    return float(np.mean(vals)), float(np.std(vals))


def anchor_img(i):
    return load(RUNS / (i["anchor"] + ".png")) if i.get("anchor") else None


def main_set(S):
    res = {}
    for i in items():
        a = load(i["path"], 1024 if i["group"] == "real" else None)
        res[i["key"]] = S.all(a, anchor_img(i))
        print(i["key"], res[i["key"]], flush=True)
    return res


def synth_set(S):
    from synth import VARIANTS, bases
    res = {}
    for key, p in bases():
        ref = load(p, 1024)
        for v in ("orig", "aligned", "unaligned", "jitter", "soft", "noise6", "blur1.5", "blur3", "blur6"):
            res[f"synth/{v}/{key}"] = S.all(VARIANTS[v](ref), ref)
            print(v, key, res[f"synth/{v}/{key}"], flush=True)
    return res


def crop_set(S, r2=False):
    key = json.loads((Path(__file__).resolve().parents[1] / f"labels/{'key_r2' if r2 else 'key'}.json").read_text())
    site = Path(__file__).resolve().parents[3] / f"release/x/site/{'stamp-severity' if r2 else 'stamp-label'}"
    res = {}
    for cid, r in key.items():
        a = load(site / "img" / r["file"])
        ref = None
        if r.get("anchor"):
            ref = load(r["anchor"])[r["y"]:r["y"] + 512, r["x"]:r["x"] + 512]
        elif r["kind"] == "catch_stamp":
            ref = load(r["src"])[r["y"]:r["y"] + 512, r["x"]:r["x"] + 512]
        res[cid] = S.all(a, ref)
        print(cid, r["run"], res[cid], flush=True)
    return res


def patch_crops(a, c=299):
    h, w = a.shape[:2]
    return [a[y:y + c, x:x + c] for y in range(0, h - c + 1, c) for x in range(0, w - c + 1, c)]


def patchfid(S):
    it = [i for i in items() if i.get("anchor")]
    runs = sorted({i["run"] for i in it})
    res = {}
    for run in runs:
        sel = [i for i in it if i["run"] == run]
        fx = S.pool3([c for i in sel for c in patch_crops(load(i["path"]))])
        fy = S.pool3([c for i in sel for c in patch_crops(anchor_img(i))])
        mu1, mu2 = fx.mean(0), fy.mean(0)
        fid = frechet(mu1, np.cov(fx, rowvar=False), mu2, np.cov(fy, rowvar=False))
        res[run] = {"group": sel[0]["group"], "n_img": len(sel), "n_crops": len(fx), "patch_fid": fid,
                    "patch_kid": kid(fx, fy)}
        print(run, res[run], flush=True)
    return res


if __name__ == "__main__":
    what = sys.argv[1]
    S = Std()
    fn = {"main": main_set, "synth": synth_set, "crops": crop_set, "crops_r2": lambda S: crop_set(S, True),
          "patchfid": patchfid}[what]
    (OUT / f"std_{what}.json").write_text(json.dumps(fn(S), indent=1))

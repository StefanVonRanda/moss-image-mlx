"""stampscore on the 82 labelling crops (512 px) and the posterior-mean probe states (blur-guard test).

crops: dup_frac per phase mode on the crop and on the anchor crop at the same location (excess), detail and detail
ratio; for generated crops also the whole-image p4 excess from the cache.
probe: decoded clean estimates x0 along the quality trajectory (runs/probe_states, 768^2), teacher vs student.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cache import dup, get, path  # noqa: E402
from dataset import OUT, RUNS  # noqa: E402
from stampscore.core import DETAIL_MIN_RATIO, LATTICE_MAX_RATIO, Extractor, bands, detail, load, nn_cos  # noqa: E402

LAB = Path(__file__).resolve().parents[1] / "labels"
SITE = Path(__file__).resolve().parents[3] / "release/x/site/stamp-label"


def dups(ex, img):
    out = {}
    for ph in ("grid", "p4", "p9"):
        f, p, sd = ex.tokens(img, (6,), ph)
        k = sd > 0.02
        out[ph] = float((nn_cos(f[6][k], p[k], (56.0,))[56.0] >= 0.85).mean()) if k.sum() >= 50 else float("nan")
    return out


def crops(ex, r2=False):
    key = json.loads((LAB / ("key_r2.json" if r2 else "key.json")).read_text())
    site = SITE.parent / "stamp-severity" if r2 else SITE
    res = {}
    for cid, r in key.items():
        a = load(site / "img" / r["file"])
        ref = None
        if r.get("anchor"):
            ref = load(r["anchor"])[r["y"]:r["y"] + 512, r["x"]:r["x"] + 512]
        elif r["kind"] == "catch_stamp":
            ref = load(r["src"])[r["y"]:r["y"] + 512, r["x"]:r["x"] + 512]
        d = {"dup": dups(ex, a), "detail": detail(a)}
        if ref is not None:
            d["ref_dup"] = dups(ex, ref)
            d["excess"] = {k: d["dup"][k] - d["ref_dup"][k] for k in d["dup"]}
            d["detail_ratio"] = d["detail"] / detail(ref)
            (f, m), (rf, rm) = bands(a), bands(ref)
            d["lattice_ratio"] = (f / m) / (rf / rm)
        if r["kind"] == "gen" and r.get("anchor"):
            k = f"{r['run']}/{r['pid']}_s{r['seed']}"
            ak = f"{Path(r['anchor']).parent.name}/{r['pid']}_s{r['seed']}"
            if path("dinov2-small", "p4", k).exists() and k != ak:
                d["image_excess_p4"] = dup(get("dinov2-small", "p4", k)) - dup(get("dinov2-small", "p4", ak))
        res[cid] = d
        print(cid, r["run"], json.dumps(d), flush=True)
    return res


def probe(ex):
    res = {}
    for d in sorted((RUNS / "probe_states").iterdir()):
        final = load(d / "final_quality.png")
        ff, fm = bands(final)
        for p in sorted(d.glob("step*.png")):
            a = load(p)
            f, m = bands(a)
            r = {"dup": dups(ex, a), "detail": m, "detail_ratio_vs_final": m / fm, "lattice_ratio": (f / m) / (ff / fm)}
            r["flagged"] = r["detail_ratio_vs_final"] < DETAIL_MIN_RATIO or r["lattice_ratio"] > LATTICE_MAX_RATIO
            res[f"{d.name}/{p.stem}"] = r
            print(d.name, p.stem, json.dumps(r), flush=True)
        res[f"{d.name}/final_quality"] = {"dup": dups(ex, final), "detail": fm}
    return res


if __name__ == "__main__":
    torch.set_num_threads(4)
    ex = Extractor("dinov2-small")
    what = sys.argv[1]
    fn = {"crops": crops, "crops_r2": lambda e: crops(e, True), "probe": probe}[what]
    (OUT / f"stamp_{what}.json").write_text(json.dumps(fn(ex), indent=1))

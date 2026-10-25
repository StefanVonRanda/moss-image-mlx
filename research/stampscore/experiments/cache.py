"""Cache nearest-neighbour cosines per token so threshold / layer / distance / floor sweeps need no recompute.

out/cache/<net>/<phases>/<key>.npz holds sd (N) and nn_<layer>_<min_dist> (N), all tokens (unfiltered).

Usage: cache.py main | synth   (CPU; STAMPSCORE_THREADS, default 6)
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch  # noqa: E402

from stampscore.core import Extractor, default_threads, load, nn_cos  # noqa: E402
from dataset import OUT, items  # noqa: E402

LAYERS, DISTS, FLOOR = (3, 6, 9, 12), (28.0, 56.0, 112.0), 0.02
CACHE = OUT / "cache"


def path(net, phases, key):
    return CACHE / net / phases / (key.replace("/", "__") + ".npz")


def record(ex, img, phases, layers, dists):
    f, p, sd = ex.tokens(img, layers, phases)
    keep = sd > FLOOR  # neighbours are searched among textured tokens only, as in recurrence.py
    rec = {"sd": sd[keep].astype(np.float32), "pos": p[keep].astype(np.float32), "n_all": len(sd)}
    for l in layers:
        for m, nn in nn_cos(f[l][keep], p[keep], dists).items():
            rec[f"nn_{l}_{int(m)}"] = nn
    return rec


def run(jobs, net):
    ex = Extractor(net)
    t0 = time.time()
    for i, (key, getimg, phases, layers, dists) in enumerate(jobs):
        out = path(net, phases, key)
        if out.exists():
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, **record(ex, getimg(), phases, layers, dists))
        print(f"{net} {phases} {i + 1}/{len(jobs)} {key} {time.time() - t0:.0f}s", flush=True)


def get(net, phases, key):
    return dict(np.load(path(net, phases, key)))


def dup(rec, layer=6, md=56, t=0.85):
    nn = rec[f"nn_{layer}_{int(md)}"]
    return float((nn >= t).mean()) if len(nn) >= 50 else float("nan")


def main_jobs():
    it = items()
    L = lambda i: (lambda: load(i["path"], 1024 if i["group"] == "real" else None))  # noqa: E731  crop real photos only
    small = [(i["key"], L(i), ph, LAYERS, DISTS) for ph in ("grid", "p4") for i in it]
    small += [(i["key"], L(i), "p9", (6,), (56.0,)) for i in it]
    base = [(i["key"], L(i), "p4", LAYERS, DISTS) for i in it]
    return small, base


if __name__ == "__main__":
    torch.set_num_threads(default_threads())
    if sys.argv[1] == "main":
        small, base = main_jobs()
        run(small, "dinov2-small")
        run(base, "dinov2-base")
    elif sys.argv[1] == "synth":
        from synth import synth_jobs
        run(synth_jobs(), "dinov2-small")

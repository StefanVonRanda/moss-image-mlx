"""Blind labelling set: 512-px crops at 1:1, one location per (prompt, seed) slot shared by every method.

The location is drawn at random (fixed seed) among the most textured windows of the destriped reference, never from any
score. Adds 6 real-photo crops and 4 catch trials (2 synthetic jittered stamps, 2 heavy blurs of a reference).
Writes the shuffled JPEGs (q93) to the site folder and the sealed key to labels/key.json (outside the site).

Round 2 (`make_crops.py r2`, 0-3 severity page stamp-severity/): same slots and methods, a new location per slot that
overlaps the round-1 window by < 25% of its area, no catch trials (the pasted squares were obvious), key_r2.json.
"""
import json
import secrets
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import E_PIDS, H_PIDS, REAL, RUNS  # noqa: E402
from stampscore.core import load  # noqa: E402
from synth import blur, stamp  # noqa: E402

C = 512
SITE = Path(__file__).resolve().parents[3] / "release/x/site/stamp-label"
LAB = Path(__file__).resolve().parents[1] / "labels"
E_RUNS = ["E_ref_d", "E_quality", "E_akh4", "E_M6F2_l20", "E_M6F2_l50", "E_M6F2_ramp", "E_v2_r2k", "E_s3F4",
          "E_oracle463", "E_T6S2d"]
H_RUNS = ["H_ref_d", "H_quality", "H_s2F4"]
GROUP = {"E_ref_d": "ref", "H_ref_d": "ref", "E_quality": "dmd", "E_akh4": "dmd", "H_quality": "dmd",
         "E_M6F2_l20": "blend", "E_M6F2_l50": "blend", "E_M6F2_ramp": "blend"}


def location(img, rng, avoid=None):
    lum = img.astype(np.float64).mean(-1) / 255
    h, w = lum.shape
    b = lum[:h // 14 * 14, :w // 14 * 14].reshape(h // 14, 14, w // 14, 14).std((1, 3)) > 0.02
    cand = []
    for y in range(0, h - C + 1, 64):
        for x in range(0, w - C + 1, 64):
            cand.append((b[y // 14:(y + C) // 14, x // 14:(x + C) // 14].mean(), y, x))
    if avoid is not None:
        ay, ax = avoid
        cand = [c for c in cand if max(0, C - abs(c[1] - ay)) * max(0, C - abs(c[2] - ax)) < 0.25 * C * C]
    cand.sort(reverse=True)
    top = cand[:max(1, int(0.3 * len(cand)))]
    _, y, x = top[rng.integers(len(top))]
    return y, x


def main(rnd="r1"):
    r2 = rnd == "r2"
    site, keyfile, prefix = (SITE.parent / "stamp-severity", LAB / "key_r2.json", "s") if r2 else (SITE, LAB / "key.json", "c")
    r1 = json.loads((LAB / "key.json").read_text()) if r2 else {}
    prev = {(v["pid"], v["seed"]): (v["y"], v["x"]) for v in r1.values() if v["kind"] in ("gen", "real")}
    rng = np.random.default_rng(20261023 if r2 else 20261022)
    rows = []
    slots = [("E", p, s) for p in E_PIDS for s in (101, 202)] + [("H", p, 101) for p in H_PIDS]
    for fam, pid, seed in slots:
        anchor = RUNS / f"{fam}_ref_d" / f"{pid}_s{seed}.png"
        if not anchor.exists():
            continue
        y, x = location(load(anchor), rng, prev.get((pid, seed)))
        for run in (E_RUNS if fam == "E" else H_RUNS):
            p = RUNS / run / f"{pid}_s{seed}.png"
            if p.exists():
                rows.append(dict(kind="gen", run=run, group=GROUP.get(run, "clean"), pid=pid, seed=seed, y=y, x=x,
                                 src=str(p), anchor=str(anchor)))
    for d, files in [("lichen_stone", ["00", "02"]), ("pine_bark", ["00", "01"]), ("bread_crust", ["00", "01"])]:
        for f in files:
            img = load(REAL / d / f"{f}.jpg", 1024)
            y, x = location(img, rng, next((v["y"], v["x"]) for v in r1.values() if v["src"].endswith(f"{d}/{f}.jpg")) if r2 else None)
            rows.append(dict(kind="real", run="real", group="real", pid=d, seed=0, y=y, x=x,
                             src=str(REAL / d / f"{f}.jpg"), crop1024=True))
    for pid, seed in ([] if r2 else [("tex_bread", 101), ("u_moss_wall", 202)]):
        rows.append(dict(kind="catch_stamp", run="synth_jitter", group="catch_stamp", pid=pid, seed=seed, y=320, x=320,
                         src=str(RUNS / "E_ref_d" / f"{pid}_s{seed}.png")))
    for pid, seed in ([] if r2 else [("u_bark", 101), ("tex_bread", 202)]):
        a = RUNS / "E_ref_d" / f"{pid}_s{seed}.png"
        y, x = next((r["y"], r["x"]) for r in rows if r["pid"] == pid and r["seed"] == seed)
        rows.append(dict(kind="catch_blur", run="synth_blur6", group="catch_blur", pid=pid, seed=seed, y=y, x=x,
                         src=str(a), anchor=str(a)))
    order = rng.permutation(len(rows))
    site.joinpath("img").mkdir(parents=True, exist_ok=True)
    LAB.mkdir(exist_ok=True)
    key, items = {}, []
    for n, i in enumerate(order):
        r = rows[i]
        img = load(r["src"], 1024 if r.get("crop1024") else None)
        if r["kind"] == "catch_stamp":
            img = stamp(img, "jitter", seed=n)
        elif r["kind"] == "catch_blur":
            img = blur(img, 6)
        crop = img[r["y"]:r["y"] + C, r["x"]:r["x"] + C]
        cid = f"{prefix}{n + 1:02d}"
        fname = f"{secrets.token_hex(6)}.jpg"
        Image.fromarray(crop).save(site / "img" / fname, quality=93, subsampling=0)
        key[cid] = {**r, "file": fname}
        items.append({"id": cid, "img": f"img/{fname}"})
    keyfile.write_text(json.dumps(key, indent=1))
    (site / "items.json").write_text(json.dumps(items))
    print(len(items), "crops")


if __name__ == "__main__":
    main(*sys.argv[1:])

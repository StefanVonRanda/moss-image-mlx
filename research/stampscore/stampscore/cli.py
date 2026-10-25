"""stampscore CLI.

  stampscore IMG [IMG ...] --ref REF            excess over one reference (same prompt and seed, same size)
  stampscore IMG [IMG ...] --ref-dir DIR        reference = DIR/<same file name>
  stampscore IMG [IMG ...]                      no-reference mode (absolute dup_frac; organic materials only)
  stampscore probe IMG                          synthetic check on one clean image: stamps, noise, blur, shift

Prints one line per image (or JSON lines with --json). Exit code 0 always; read `status` and `verdict`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from . import core
from .core import StampScorer, load

EXCESS_FLAG = 0.02  # t .80 excess above this: "stamping" (Felix severity: 0 12%, 1 50%, 2 86%, 3 100%)
NOREF_FLAG = 0.20  # t .80 absolute dup above this: screening only (DMD 81%, clean 19%, refs 22%)
SEVERE_FLAG = 0.10  # t .80 excess above this: "severe stamping" (severity 3: 85%, 2: 36%, 1: 21%, 0: 3%)


def verdict(s, has_ref):
    if s.status != "ok":
        return {"low_detail": "not scored: too little fine detail (blur / posterior mean)",
                "decoder_lattice": "not scored: decoder lattice (posterior mean / intermediate state)",
                "too_few_tokens": "not scored: too little texture"}[s.status]
    if has_ref:
        return "severe stamping" if s.excess > SEVERE_FLAG else "stamping" if s.excess > EXCESS_FLAG else "no excess"
    return "likely stamped (no-ref)" if s.dup_frac > NOREF_FLAG else "no evidence (no-ref)"


def fmt(v):
    return f"{v:.4f}" if isinstance(v, float) else str(v)


def probe(sc: StampScorer, path: str, as_json: bool):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
    from synth import VARIANTS  # noqa: E402
    ref = load(path, 1024)
    base = sc.score(ref).dup_frac
    for v in ("aligned", "unaligned", "jitter", "soft", "noise6", "blur1.5", "blur6", "shift7"):
        img = VARIANTS[v](ref)
        r = sc.score(img, ref if img.shape == ref.shape else None)
        d = {"variant": v, "dup_frac": r.dup_frac, "delta": (r.dup_frac or 0) - base, "status": r.status,
             "detail_ratio": r.detail_ratio}
        print(json.dumps(d) if as_json else "  ".join(f"{k}={fmt(x)}" for k, x in d.items()))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="stampscore", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--ref", help="reference image for every input")
    g.add_argument("--ref-dir", help="directory with references of the same file names")
    ap.add_argument("--net", default="dinov2-small", choices=sorted(core.NETS))
    ap.add_argument("--layer", type=int, default=core.LAYER)
    ap.add_argument("--phases", default="p4", choices=sorted(core.PHASES))
    ap.add_argument("--threshold", type=float, default=core.DUP, help="near-duplicate cosine (default .80; .85 = recurrence.py; flags are calibrated for .80)")
    ap.add_argument("--min-dist", type=float, default=core.MIN_DIST, help="exclude partners closer than this (px)")
    ap.add_argument("--max-side", type=int, default=None, help="centre-crop larger images to this size (never resized)")
    ap.add_argument("--mask", help="optional mask image (white = score); same size as the input")
    ap.add_argument("--no-guard", action="store_true", help="score low-detail images anyway")
    ap.add_argument("--threads", type=int, default=core.default_threads())
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    sc = StampScorer(a.net, a.layer, a.phases, a.threshold, a.min_dist, threads=a.threads)
    if a.images[0] == "probe":
        return probe(sc, a.images[1], a.json)
    mask = np.array(load(a.mask).mean(-1) > 127) if a.mask else None
    for p in a.images:
        img = load(p, a.max_side)
        rp = a.ref or (str(Path(a.ref_dir) / Path(p).name) if a.ref_dir else None)
        ref = load(rp, a.max_side) if rp else None
        if ref is not None and ref.shape != img.shape:
            print(f"{p}: reference size {ref.shape[:2]} != image size {img.shape[:2]}; skipped", file=sys.stderr)
            continue
        s = s0 = sc.score(img, ref, mask=mask, guard=not a.no_guard)
        d = {"image": p, **s0.as_dict(), "verdict": verdict(s, ref is not None)}
        print(json.dumps(d) if a.json else "  ".join(f"{k}={fmt(v)}" for k, v in d.items()))


if __name__ == "__main__":
    main()

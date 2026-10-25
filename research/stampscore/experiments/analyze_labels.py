"""Score Felix's blind labels against stampscore and the standard metrics.

Usage: analyze_labels.py LABELS.json        (the JSON from the page's "Copy as JSON"; '-' reads stdin)
Writes out/labels_results.json and prints markdown tables.

Main set = generated crops (72; excess metrics need the same-location reference crop). Labels: stamped = 1,
not = 0, unsure dropped (sensitivity row counts unsure as 0.5 via a second AUC with unsure as positive and negative).
Catch trials (2 synthetic stamps, 2 heavy blurs) and the 6 real-photo crops are reported separately.
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stats import auc  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"


def load_labels(arg):
    txt = sys.stdin.read() if arg == "-" else Path(arg).read_text()
    j = json.loads(txt)
    return {k: v for k, v in j["labels"].items() if v.get("label")}, j


def main(arg):
    lab, raw = load_labels(arg)
    key = json.loads((ROOT / "labels/key.json").read_text())
    st = json.loads((OUT / "stamp_crops.json").read_text())
    sd = json.loads((OUT / "std_crops.json").read_text())
    (ROOT / "labels/felix_labels.json").write_text(json.dumps(raw, indent=1))
    R = {"n_labelled": len(lab), "counts": Counter(v["label"] for v in lab.values())}

    rows = defaultdict(Counter)
    for cid, v in lab.items():
        rows[key[cid]["group"]][v["label"]] += 1
    R["by_group"] = {g: dict(c) for g, c in rows.items()}
    print("## Labels by (hidden) group\n\n| group | stamped | not | unsure |\n|---|---|---|---|")
    for g in ("ref", "clean", "blend", "dmd", "real", "catch_stamp", "catch_blur"):
        c = rows.get(g, Counter())
        print(f"| {g} | {c['stamped']} | {c['not']} | {c['unsure']} |")
    print()

    METRICS = {
        "stampscore excess p4 (crop)": lambda c: st[c]["excess"]["p4"],
        "stampscore excess p9 (crop)": lambda c: st[c]["excess"]["p9"],
        "stampscore excess grid (crop)": lambda c: st[c]["excess"]["grid"],
        "stampscore abs p4 (crop, no-ref)": lambda c: st[c]["dup"]["p4"],
        "stampscore excess p4 (whole image)": lambda c: st[c].get("image_excess_p4", np.nan),
        "LPIPS to ref": lambda c: sd[c].get("lpips", np.nan),
        "SIFID to ref": lambda c: sd[c].get("sifid", np.nan),
        "-CLIP-IQA": lambda c: -sd[c]["clipiqa"],
        "-MUSIQ": lambda c: -sd[c]["musiq"],
        "1 - detail ratio (blur)": lambda c: 1 - st[c].get("detail_ratio", np.nan),
    }
    gen = [c for c in lab if key[c]["kind"] == "gen" and lab[c]["label"] != "unsure"]
    y = np.array([lab[c]["label"] == "stamped" for c in gen])
    gen_u = [c for c in lab if key[c]["kind"] == "gen"]
    rng = np.random.default_rng(0)
    boots = [rng.integers(0, len(gen), len(gen)) for _ in range(2000)]
    R["metrics"] = {}
    vals = {m: np.array([f(c) for c in gen], float) for m, f in METRICS.items()}
    print(f"## Agreement with labels (generated crops, n = {len(gen)}: {int(y.sum())} stamped, {int((~y).sum())} not; unsure dropped)\n")
    print("| metric | AUC [95% boot] | Spearman | AUC, unsure = stamped | AUC, unsure = not |\n|---|---|---|---|---|")
    for m, v in vals.items():
        a = auc(v[y], v[~y])
        bs = [auc(v[b][y[b]], v[b][~y[b]]) for b in boots]
        rho = spearmanr(v, y, nan_policy="omit")[0]
        vu = np.array([METRICS[m](c) for c in gen_u], float)
        lu = [lab[c]["label"] for c in gen_u]
        a_us = auc(vu[[l != "not" for l in lu]], vu[[l == "not" for l in lu]])
        a_un = auc(vu[[l == "stamped" for l in lu]], vu[[l != "stamped" for l in lu]])
        R["metrics"][m] = {"auc": a, "ci": [float(np.nanquantile(bs, .025)), float(np.nanquantile(bs, .975))],
                           "spearman": float(rho), "auc_unsure_pos": a_us, "auc_unsure_neg": a_un}
        print(f"| {m} | {a:.3f} [{R['metrics'][m]['ci'][0]:.2f}, {R['metrics'][m]['ci'][1]:.2f}] | {rho:+.2f} | {a_us:.3f} | {a_un:.3f} |")
    print()
    ref_m = "stampscore excess p4 (crop)"
    print("## Paired bootstrap: AUC(stampscore excess p4) minus AUC(other)\n\n| other | Δ AUC | 95% | P(Δ ≤ 0) |\n|---|---|---|---|")
    R["paired"] = {}
    for m in vals:
        if m == ref_m:
            continue
        d = [auc(vals[ref_m][b][y[b]], vals[ref_m][b][~y[b]]) - auc(vals[m][b][y[b]], vals[m][b][~y[b]]) for b in boots]
        d = np.array(d)[~np.isnan(d)]
        R["paired"][m] = {"delta": float(np.mean(d)), "ci": [float(np.quantile(d, .025)), float(np.quantile(d, .975))],
                          "p_le0": float((d <= 0).mean())}
        print(f"| {m} | {np.mean(d):+.3f} | [{np.quantile(d, .025):+.2f}, {np.quantile(d, .975):+.2f}] | {(d <= 0).mean():.3f} |")
    print()
    # operating point: excess p4 > .005 (the report card's practical threshold)
    for thr in (0.0, 0.005, 0.01, 0.02):
        pred = vals[ref_m] > thr
        R.setdefault("operating", {})[thr] = {"tpr": float(pred[y].mean()), "fpr": float(pred[~y].mean())}
    print("operating points (crop excess p4 > t):", R["operating"], "\n")
    print("## Catch trials and real photos\n")
    for c, v in sorted(lab.items()):
        k = key[c]
        if k["kind"] != "gen":
            s = st[c]
            print(f"- {c} {k['kind']} {k['pid']}: label **{v['label']}**; crop dup p4 {s['dup']['p4']:.3f}"
                  + (f", excess {s['excess']['p4']:+.3f}, detail ratio {s['detail_ratio']:.2f}" if "excess" in s else "")
                  + (f"; note: {v['note']}" if v.get("note") else ""))
    notes = {c: v["note"] for c, v in lab.items() if v.get("note")}
    R["notes"] = {c: {"note": n, "run": key[c]["run"], "pid": key[c]["pid"]} for c, n in notes.items()}
    (OUT / "labels_results.json").write_text(json.dumps(R, indent=1, default=float))


if __name__ == "__main__":
    main(sys.argv[1])

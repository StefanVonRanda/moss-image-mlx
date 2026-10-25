"""Round 2: Felix's 0-3 severity ratings (stamp-severity page) against stampscore and the standard metrics.

Usage: analyze_severity.py SEVERITY.json      ('-' reads stdin) -> labels/felix_severity.json, out/severity_results.json

Generated crops only for metric agreement (excess metrics need the same-location reference crop); real photos are
reported separately. Also checks round-1 / round-2 consistency on the same images (different crop locations).
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stats import auc  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT, LAB = ROOT / "out", ROOT / "labels"
LEVELS = ("0", "1", "2", "3")


def main(arg):
    raw = json.loads(sys.stdin.read() if arg == "-" else Path(arg).read_text())
    (LAB / "felix_severity.json").write_text(json.dumps(raw, indent=1))
    sev = {c: int(v["label"]) for c, v in raw["labels"].items() if v.get("label") in LEVELS}
    key = json.loads((LAB / "key_r2.json").read_text())
    st = json.loads((OUT / "stamp_crops_r2.json").read_text())
    sd = json.loads((OUT / "std_crops_r2.json").read_text())
    R = {"n_rated": len(sev), "counts": Counter(sev.values())}

    by = defaultdict(Counter)
    for c, s in sev.items():
        by[key[c]["group"]][s] += 1
    R["by_group"] = {g: dict(v) for g, v in by.items()}
    print("## Severity by (hidden) group\n\n| group | 0 | 1 | 2 | 3 | mean |\n|---|---|---|---|---|---|")
    for g in ("real", "ref", "clean", "blend", "dmd"):
        v = by.get(g, Counter())
        n = sum(v.values())
        print(f"| {g} | {v[0]} | {v[1]} | {v[2]} | {v[3]} | {sum(k * x for k, x in v.items()) / max(n, 1):.2f} |")
    print()
    runs = defaultdict(list)
    for c, s in sev.items():
        runs[key[c]["run"]].append(s)
    R["by_run_mean"] = {r: float(np.mean(v)) for r, v in runs.items()}
    print("mean severity by run:", {r: round(m, 2) for r, m in sorted(R["by_run_mean"].items(), key=lambda kv: kv[1])}, "\n")

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
    }
    gen = [c for c in sev if key[c]["kind"] == "gen"]
    y = np.array([sev[c] for c in gen])
    rng = np.random.default_rng(0)
    boots = [rng.integers(0, len(gen), len(gen)) for _ in range(2000)]
    vals = {m: np.array([f(c) for c in gen], float) for m, f in METRICS.items()}
    R["metrics"] = {}
    print(f"## Agreement with severity (generated crops, n = {len(gen)})\n")
    print("| metric | Spearman [95% boot] | Kendall τb | AUC ≥1 vs 0 | AUC ≥2 vs ≤1 | AUC 3 vs ≤2 | mean at 0 / 1 / 2 / 3 |\n|---|---|---|---|---|---|---|")
    for m, v in vals.items():
        ok = ~np.isnan(v)
        rho = spearmanr(v[ok], y[ok])[0]
        bs = [spearmanr(v[b][ok[b]], y[b][ok[b]])[0] for b in boots]
        tau = kendalltau(v[ok], y[ok])[0]
        a1, a2, a3 = auc(v[y >= 1], v[y == 0]), auc(v[y >= 2], v[y <= 1]), auc(v[y == 3], v[y <= 2])
        means = [float(np.nanmean(v[y == k])) if (y == k).any() else float("nan") for k in range(4)]
        R["metrics"][m] = {"spearman": float(rho), "ci": [float(np.nanquantile(bs, .025)), float(np.nanquantile(bs, .975))],
                           "kendall": float(tau), "auc_ge1": a1, "auc_ge2": a2, "auc_3": a3, "mean_by_level": means}
        print(f"| {m} | {rho:+.2f} [{R['metrics'][m]['ci'][0]:+.2f}, {R['metrics'][m]['ci'][1]:+.2f}] | {tau:+.2f} | "
              f"{a1:.3f} | {a2:.3f} | {a3:.3f} | " + " / ".join("n/a" if np.isnan(x) else f"{x:.3f}" for x in means) + " |")
    print()
    ref_m = "stampscore excess p4 (crop)"
    print("## Paired bootstrap: Spearman(stampscore excess p4) minus Spearman(other)\n\n| other | Δρ | 95% | P(Δ ≤ 0) |\n|---|---|---|---|")
    R["paired"] = {}
    for m in vals:
        if m == ref_m:
            continue
        d = []
        for b in boots:
            ok = ~np.isnan(vals[m][b]) & ~np.isnan(vals[ref_m][b])
            d.append(spearmanr(vals[ref_m][b][ok], y[b][ok])[0] - spearmanr(vals[m][b][ok], y[b][ok])[0])
        d = np.array(d)[~np.isnan(d)]
        R["paired"][m] = {"delta": float(d.mean()), "ci": [float(np.quantile(d, .025)), float(np.quantile(d, .975))],
                          "p_le0": float((d <= 0).mean())}
        print(f"| {m} | {d.mean():+.3f} | [{np.quantile(d, .025):+.2f}, {np.quantile(d, .975):+.2f}] | {(d <= 0).mean():.3f} |")
    print()
    for thr in (0.005, 0.01, 0.02):
        pred = vals[ref_m] > thr
        R.setdefault("flag_rate_by_level", {})[thr] = [float(pred[y == k].mean()) if (y == k).any() else None for k in range(4)]
    print("crop p4 excess flag rate at severity 0 / 1 / 2 / 3:", R["flag_rate_by_level"], "\n")

    # round-1 vs round-2 consistency on the same image (different crop locations)
    r1k = json.loads((LAB / "key.json").read_text())
    r1p = LAB / "felix_raw.json"
    if r1p.exists():
        r1 = json.loads(r1p.read_text())["labels"]
        img1 = {(v["run"], v["pid"], v["seed"]): r1[c]["label"] for c, v in r1k.items() if v["kind"] == "gen" and c in r1}
        pairs = [(img1[(key[c]["run"], key[c]["pid"], key[c]["seed"])], sev[c]) for c in gen
                 if (key[c]["run"], key[c]["pid"], key[c]["seed"]) in img1]
        t = defaultdict(Counter)
        for a, b in pairs:
            t[a][b] += 1
        R["round1_vs_round2"] = {a: dict(v) for a, v in t.items()}
        print("## Round 1 (binary) vs round 2 (severity), same image, different crop\n\n| round 1 | 0 | 1 | 2 | 3 |\n|---|---|---|---|---|")
        for a in ("not", "unsure", "stamped"):
            print(f"| {a} | " + " | ".join(str(t[a][k]) for k in range(4)) + " |")
        print()
    print("## Real photos\n")
    for c in sorted(sev):
        if key[c]["kind"] == "real":
            print(f"- {c} {key[c]['pid']}: severity {sev[c]}; crop dup p4 {st[c]['dup']['p4']:.3f}")
    R["notes"] = {c: {"note": v["note"], "run": key[c]["run"], "pid": key[c]["pid"], "severity": v.get("label")}
                  for c, v in raw["labels"].items() if v.get("note")}
    if R["notes"]:
        print("\n## Notes\n")
        for c, n in R["notes"].items():
            print(f"- {c} ({n['run']}, {n['pid']}, severity {n['severity']}): {n['note']}")
    (OUT / "severity_results.json").write_text(json.dumps(R, indent=1, default=float))


if __name__ == "__main__":
    main(sys.argv[1])

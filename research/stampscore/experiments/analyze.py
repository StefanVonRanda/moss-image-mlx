"""All numbers for REPORT.md from the caches: out/results.json plus printed markdown tables.

Usage: analyze.py
"""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cache import dup, get, path  # noqa: E402
from dataset import OUT, TQ, items  # noqa: E402
from stampscore.core import DETAIL_MIN_RATIO, LATTICE_MAX_RATIO, bands, load  # noqa: E402
from stats import auc  # noqa: E402
from synth import VARIANTS, bases  # noqa: E402

R = {}


def auc_ci(pos, neg, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    b = [auc(rng.choice(pos, len(pos)), rng.choice(neg, len(neg))) for _ in range(n)]
    return auc(pos, neg), float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))


def f3(x):
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.3f}"


def table(head, rows):
    print("| " + " | ".join(head) + " |\n|" + "---|" * len(head))
    for r in rows:
        print("| " + " | ".join(str(c) for c in r) + " |")
    print()


IT = items()
GEN = [i for i in IT if i.get("anchor")]


def excess(net, ph, i, **kw):
    return dup(get(net, ph, i["key"]), **kw) - dup(get(net, ph, i["anchor"]), **kw)


def by_group(vals):
    g = defaultdict(list)
    for i, v in zip(GEN, vals):
        g[i["group"]].append(v)
    return g


# 1. reproduction of the original grid numbers
sep = json.loads((TQ / "novelty/sep_rows.json").read_text())
diffs = []
for grp, run, pid, seed, ge, pe in sep:
    k = f"{run}/{pid}_s{seed}"
    i = next((x for x in GEN if x["key"] == k), None)
    if i:
        diffs.append((abs(excess("dinov2-small", "grid", i) - ge), abs(excess("dinov2-small", "p4", i) - pe)))
R["reproduction"] = {"n": len(diffs), "max_abs_diff_grid": max(d[0] for d in diffs),
                     "max_abs_diff_p4": max(d[1] for d in diffs)}
print("## Reproduction\n", R["reproduction"], "\n")

# 2. separation per phase mode (layer 6, .85, 56)
print("## Separation by phase mode (small, L6, t .85, d 56)\n")
rows = []
R["separation"] = {}
for ph in ("grid", "p4", "p9"):
    g = by_group([excess("dinov2-small", ph, i) for i in GEN])
    a = auc_ci(g["dmd"], g["clean"])
    b = auc(g["blend"], g["clean"])
    med = {k: float(np.median(v)) for k, v in g.items()}
    R["separation"][ph] = {"auc_dmd_clean": a, "auc_blend_clean": b, "median": med}
    rows.append([ph, f"{a[0]:.3f} [{a[1]:.2f}, {a[2]:.2f}]", f3(b), f3(med["dmd"]), f3(med["blend"]), f3(med["clean"])])
table(["phases", "AUC dmd vs clean [95% boot]", "AUC blend vs clean", "median dmd", "median blend", "median clean"], rows)

# paired held-out: H_quality vs H_s2F4
for ph in ("grid", "p4", "p9"):
    wins = 0
    for i in GEN:
        if i["run"] == "H_quality":
            j = next(x for x in GEN if x["run"] == "H_s2F4" and x["pid"] == i["pid"])
            wins += excess("dinov2-small", ph, i) > excess("dinov2-small", ph, j)
    R["separation"][ph]["heldout_pairs_dmd_higher"] = int(wins)
print("held-out pairs H_quality > H_s2F4:", {ph: R["separation"][ph]["heldout_pairs_dmd_higher"] for ph in R["separation"]}, "/ 8\n")

# 3. synthetic: stamps, degradations, shifts
print("## Synthetic perturbations (excess over the unperturbed base; mean over bases)\n")
B = bases()
R["synth"] = {}
rows = []
for v in VARIANTS:
    if v.startswith("shift") or v == "orig":
        continue
    row = [v]
    for ph in ("grid", "p4", "p9"):
        e = [dup(get("dinov2-small", ph, f"synth/{v}/{k}")) - dup(get("dinov2-small", ph, f"synth/orig/{k}")) for k, _ in B]
        R["synth"][f"{v}/{ph}"] = {"mean": float(np.mean(e)), "min": float(np.min(e)), "max": float(np.max(e)),
                                   "frac_pos": float(np.mean(np.array(e) > 0.005))}
        row.append(f"{np.mean(e):+.3f} ({np.mean(np.array(e) > 0.005):.0%})")
    rows.append(row)
table(["variant", "grid", "p4", "p9"], rows)
print("(% = share of the", len(B), "bases with excess > .005)\n")

print("## Phase invariance: |dup(shift k) - dup(shift 0)|, k = 3, 7, 10 px\n")
rows = []
R["shift"] = {}
for ph in ("grid", "p4", "p9"):
    d0 = {k: dup(get("dinov2-small", ph, f"synth/shift0/{k}")) for k, _ in B}
    dev = [abs(dup(get("dinov2-small", ph, f"synth/shift{s}/{k}")) - d0[k]) for s in (3, 7, 10) for k, _ in B]
    rel = [abs(dup(get("dinov2-small", ph, f"synth/shift{s}/{k}")) - d0[k]) / max(d0[k], 1e-3)
           for s in (3, 7, 10) for k, _ in B]
    R["shift"][ph] = {"mean_abs": float(np.mean(dev)), "max_abs": float(np.max(dev)), "median_rel": float(np.median(rel))}
    rows.append([ph, f3(np.mean(dev)), f3(np.max(dev)), f"{np.median(rel):.0%}"])
table(["phases", "mean abs dev", "max abs dev", "median rel dev"], rows)

# 4. detail guard
print("## Detail guard (mid-band ratio < %.1f -> low_detail; fine/mid ratio > %.1f -> decoder_lattice)\n" % (DETAIL_MIN_RATIO, LATTICE_MAX_RATIO))


def ratios(a, r):
    (f, m), (rf, rm) = bands(a), bands(r)
    return m / rm, (f / m) / (rf / rm)


def flagged(dr, lr):
    return dr < DETAIL_MIN_RATIO or lr > LATTICE_MAX_RATIO


gr = {i["key"]: ratios(load(i["path"]), load(TQ / "runs" / (i["anchor"] + ".png"))) for i in GEN}
g = by_group(list(gr.values()))
R["guard"] = {"gen": {k: {"mid_ratio": [float(min(x[0] for x in v)), float(max(x[0] for x in v))],
                          "lattice_ratio": [float(min(x[1] for x in v)), float(max(x[1] for x in v))]} for k, v in g.items()},
              "gen_flagged": [k for k, v in gr.items() if flagged(*v)]}
rows = [[f"generated: {k}", f"{v['mid_ratio'][0]:.2f}-{v['mid_ratio'][1]:.2f}", f"{v['lattice_ratio'][0]:.2f}-{v['lattice_ratio'][1]:.2f}",
         sum(flagged(*gr[i["key"]]) for i in GEN if i["group"] == k), "", ""] for k, v in R["guard"]["gen"].items()]
for v in ("noise6", "blur1.5", "blur3", "blur6", "aligned", "jitter"):
    rr = []
    for k, p in B:
        a = load(p, 1024)
        rr.append(ratios(VARIANTS[v](a), a))
    ex = R["synth"][f"{v}/p4"]["mean"]
    R["guard"][f"synth_{v}"] = {"flagged": int(sum(flagged(*x) for x in rr)), "n": len(rr), "excess_p4": ex}
    rows.append([f"synthetic {v}", f"{min(x[0] for x in rr):.2f}-{max(x[0] for x in rr):.2f}",
                 f"{min(x[1] for x in rr):.2f}-{max(x[1] for x in rr):.2f}", f"{sum(flagged(*x) for x in rr)}/{len(rr)}", f"{ex:+.3f}", ""])
pr = OUT / "stamp_probe.json"
if pr.exists():
    P = json.loads(pr.read_text())
    R["guard"]["probe"] = P
    for d in ("tex_bread_s42", "tex_portrait_s42"):
        fin = P[f"{d}/final_quality"]["dup"]["p4"]
        for k, v in P.items():
            if k.startswith(d) and "step" in k:
                rows.append([f"probe {k.split('/')[1].replace('_sigma', ' σ').replace('_skip', '')}"[:40], f"{v['detail_ratio_vs_final']:.2f}",
                             f"{v['lattice_ratio']:.2f}", "flagged" if v["flagged"] else "scored", f"{v['dup']['p4'] - fin:+.3f}", d[:9]])
table(["image", "mid ratio", "fine/mid ratio", "flagged", "dup p4 excess", "probe"], rows)
print("generated images flagged:", R["guard"]["gen_flagged"], "\n")

# 5. sensitivity (p4): net x layer x threshold x distance
print("## Sensitivity (p4): AUC dmd vs clean / blend vs clean; real-photo max absolute dup (admitted)\n")
R["sens"] = []
rows = []
real_adm = [i for i in IT if i["group"] == "real" and i.get("admitted")]
for net in ("dinov2-small", "dinov2-base"):
    if not path(net, "p4", GEN[-1]["key"]).exists():
        continue
    for L in (3, 6, 9, 12):
        for md in (28, 56, 112):
            for t in (0.75, 0.8, 0.85, 0.9, 0.95):
                g = by_group([excess(net, "p4", i, layer=L, md=md, t=t) for i in GEN])
                a, b = auc(g["dmd"], g["clean"]), auc(g["blend"], g["clean"])
                rmax = max(dup(get(net, "p4", i["key"]), layer=L, md=md, t=t) for i in real_adm)
                R["sens"].append(dict(net=net, layer=L, md=md, t=t, auc_dmd=a, auc_blend=b, real_max=rmax,
                                      med_dmd=float(np.median(g["dmd"])), med_clean=float(np.median(g["clean"]))))
best = sorted(R["sens"], key=lambda r: -r["auc_dmd"])
for r in [x for x in R["sens"] if x["md"] == 56 and x["t"] in (0.8, 0.85, 0.9)]:
    rows.append([r["net"], r["layer"], r["t"], f3(r["auc_dmd"]), f3(r["auc_blend"]), f3(r["med_dmd"]), f3(r["med_clean"]), f3(r["real_max"])])
table(["net", "layer", "t", "AUC dmd", "AUC blend", "med dmd", "med clean", "real max dup"], rows)
rows = []
for md in (28, 56, 112):
    r = next(x for x in R["sens"] if x["net"] == "dinov2-small" and x["layer"] == 6 and x["t"] == 0.85 and x["md"] == md)
    rows.append([md, f3(r["auc_dmd"]), f3(r["auc_blend"])])
table(["min dist (small L6 .85)", "AUC dmd", "AUC blend"], rows)
aucs = np.array([r["auc_dmd"] for r in R["sens"]])
R["sens_summary"] = {"n_configs": len(aucs), "frac_auc_ge_0.95": float((aucs >= 0.95).mean()),
                     "frac_auc_ge_0.9": float((aucs >= 0.9).mean()), "top5": best[:5]}
print("sensitivity summary:", {k: v for k, v in R["sens_summary"].items() if k != "top5"})
print("top 5:", [(r["net"], r["layer"], r["md"], r["t"], round(r["auc_dmd"], 3)) for r in best[:5]], "\n")

# synth jitter response at layer 9 vs 6
for L in (6, 9):
    e = [dup(get("dinov2-small", "p4", f"synth/jitter/{k}"), layer=L) - dup(get("dinov2-small", "p4", f"synth/orig/{k}"), layer=L) for k, _ in B]
    n6 = [dup(get("dinov2-small", "p4", f"synth/noise6/{k}"), layer=L) - dup(get("dinov2-small", "p4", f"synth/orig/{k}"), layer=L) for k, _ in B]
    R.setdefault("synth_layer", {})[L] = {"jitter_mean": float(np.mean(e)), "noise_mean": float(np.mean(n6))}
print("synth by layer (p4):", R["synth_layer"], "\n")

# 6. no-reference variant: absolute dup (p4, L6, .85)
print("## No-reference variant (absolute dup_frac, p4)\n")
absd = {i["key"]: dup(get("dinov2-small", "p4", i["key"])) for i in IT}
grp = defaultdict(list)
for i in IT:
    grp[i["group"] if i["group"] != "real" else ("real_admitted" if i.get("admitted") else "real_other")].append(absd[i["key"]])
thr = max(grp["real_admitted"] + grp["ref"])
R["noref"] = {"auc_dmd_clean": auc_ci(grp["dmd"], grp["clean"]), "auc_blend_clean": auc(grp["blend"], grp["clean"]),
              "threshold_max_real_or_ref": thr,
              "tpr_dmd": float(np.mean(np.array(grp["dmd"]) > thr)), "fpr_clean": float(np.mean(np.array(grp["clean"]) > thr)),
              "range": {k: [float(min(v)), float(np.median(v)), float(max(v))] for k, v in grp.items()}}
table(["group", "min / median / max abs dup"], [[k, " / ".join(f3(x) for x in v)] for k, v in R["noref"]["range"].items()])
print({k: v for k, v in R["noref"].items() if k != "range"}, "\n")

# 7. standard metrics
print("## Standard metrics\n")
sm = OUT / "std_main.json"
if sm.exists():
    S = json.loads(sm.read_text())
    R["std"] = {}
    rows = []
    for m, sign in (("lpips", 1), ("sifid", 1), ("clipiqa", -1), ("musiq", -1)):
        g = by_group([sign * S[i["key"]].get(m, float("nan")) for i in GEN])
        a = auc(g["dmd"], g["clean"])
        R["std"][m] = {"auc_dmd_clean_badness": a, "auc_blend_clean_badness": auc(g["blend"], g["clean"]),
                       "median": {k: sign * float(np.median(v)) for k, v in g.items()}}
        rows.append([m, f3(a), f3(R["std"][m]["auc_blend_clean_badness"]), *(f3(R["std"][m]["median"][k]) for k in ("dmd", "blend", "clean"))])
    # stampscore row for comparison
    rows.insert(0, ["stampscore p4 (excess)", f3(R["separation"]["p4"]["auc_dmd_clean"][0]), f3(R["separation"]["p4"]["auc_blend_clean"]),
                    *(f3(R["separation"]["p4"]["median"][k]) for k in ("dmd", "blend", "clean"))])
    table(["metric", "AUC dmd vs clean (worse=dmd)", "AUC blend vs clean", "median dmd", "median blend", "median clean"], rows)
    # partial: does a standard metric explain stampscore? Spearman within generated images
    from scipy.stats import spearmanr
    st = [excess("dinov2-small", "p4", i) for i in GEN]
    R["std_spearman_vs_stamp"] = {m: float(spearmanr(st, [S[i["key"]].get(m, np.nan) for i in GEN], nan_policy="omit")[0])
                                  for m in ("lpips", "sifid", "clipiqa", "musiq")}
    print("Spearman vs stampscore excess:", R["std_spearman_vs_stamp"], "\n")
ss = OUT / "std_synth.json"
if ss.exists():
    S = json.loads(ss.read_text())
    rows = []
    R["std_synth"] = {}
    for v in ("aligned", "unaligned", "jitter", "soft", "noise6", "blur1.5", "blur3", "blur6"):
        r = {m: float(np.mean([S[f"synth/{v}/{k}"][m] - (S[f"synth/orig/{k}"][m] if m in ("clipiqa", "musiq") else 0) for k, _ in B]))
             for m in ("lpips", "sifid", "clipiqa", "musiq")}
        r["stampscore_p4"] = R["synth"][f"{v}/p4"]["mean"]
        R["std_synth"][v] = r
        rows.append([v] + [f"{r[m]:+.3f}" for m in ("stampscore_p4", "lpips", "sifid", "clipiqa", "musiq")])
    table(["variant", "stampscore p4", "LPIPS", "SIFID", "Δ CLIP-IQA", "Δ MUSIQ"], rows)
pf = OUT / "std_patchfid.json"
if pf.exists():
    P = json.loads(pf.read_text())
    R["patchfid"] = P
    rows = [[k, v["group"], v["n_img"], f"{v['patch_fid']:.1f}", f"{v['patch_kid'][0] * 1000:.2f}"] for k, v in sorted(P.items(), key=lambda kv: kv[1]["group"])]
    table(["run", "group", "images", "patch-FID", "patch-KID ×1e3"], rows)
    g = defaultdict(list)
    for v in P.values():
        g[v["group"]].append(v["patch_kid"][0])
    R["patchfid_group_median_kid"] = {k: float(np.median(v)) for k, v in g.items()}

(OUT / "results.json").write_text(json.dumps(R, indent=1, default=float))

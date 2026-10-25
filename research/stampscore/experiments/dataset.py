"""Evaluation images: admitted organic materials (Phase 2a) on E and the 8 held-out materials/nature prompts on H.

group: ref (50-step reference), dmd (community DMD turbo adapters), blend (DMD blended with ours), clean (non-DMD
fast methods), real (Wikimedia photos, centre-cropped to 1024). anchor = destriped same-seed 50-step reference.
"""
from pathlib import Path

TQ = Path(__file__).resolve().parents[2]
RUNS, REAL = TQ / "runs", TQ / "realphotos"
OUT = Path(__file__).resolve().parents[1] / "out"
E_PIDS = ["tex_bread", "u_moss_wall", "u_bark"]
H_PIDS = [f"heldout_materials_{i:02d}" for i in (1, 5, 16, 17)] + [f"heldout_nature_{i:02d}" for i in (3, 8, 11, 19)]
GROUPS = {
    "ref": ["E_ref", "E_ref_d", "H_ref_d"],
    "dmd": ["E_quality", "E_akh4", "H_quality"],
    "blend": ["E_M6F2_l20", "E_M6F2_l50", "E_M6F2_ramp"],
    "clean": ["E_v2_r2k", "E_s3F4", "E_s2F4", "E_oracle463", "E_T6S2d", "E_pilot", "E_alloc_T2", "H_s2F4", "H_s3F4"],
}
ADMITTED_REAL = ["lichen_stone", "pine_bark", "bread_crust"]
FROZEN = 1792711217  # 2026-10-22 19:20 local: later files (other sessions still render into runs/) are excluded


def items():
    """[{key, path, group, run, pid, seed, anchor}] for generated images; anchor None for references themselves."""
    out = []
    for group, runs in GROUPS.items():
        for run in runs:
            pids = H_PIDS if run.startswith("H_") else E_PIDS
            for pid in pids:
                for seed in (101, 202):
                    p = RUNS / run / f"{pid}_s{seed}.png"
                    a = RUNS / ("H_ref_d" if run.startswith("H_") else "E_ref_d") / f"{pid}_s{seed}.png"
                    if p.exists() and a.exists() and p.stat().st_mtime < FROZEN:
                        out.append(dict(key=f"{run}/{pid}_s{seed}", path=str(p), group=group, run=run, pid=pid,
                                        seed=seed, anchor=None if run.endswith("ref_d") else f"{a.parent.name}/{a.stem}"))
    for d in sorted(REAL.iterdir()):
        if d.is_dir():
            for f in sorted(d.glob("*.jpg")):
                out.append(dict(key=f"real/{d.name}/{f.stem}", path=str(f), group="real", run="real", pid=d.name,
                                seed=0, anchor=None, admitted=d.name in ADMITTED_REAL))
    return out


if __name__ == "__main__":
    from collections import Counter
    it = items()
    print(len(it), Counter(i["group"] for i in it), Counter(i["run"] for i in it))

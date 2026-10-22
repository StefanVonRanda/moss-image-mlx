# Training proposal v2: prefix student at 1024² (2026-10-20; gate passed 07:20 UTC; not started; needs a go-ahead for spend)

**Gate: passed.** The oracle `E_oracle463` (1024², seed 101, 7 triage prompts) takes the exact 50-step guided state at σ .463, then runs the existing 2-step turbo finish.
- **By eye it is nearly indistinguishable from the 50-step reference:**
  - the same composition, lichen, bread crumb and bark;
  - sign "THE GOLDEN CRUST BAKERY" correct;
  - infographic labels correct, with the reference's own layout.
- **Metrics:**
  - repetition excess .011 / .001 / .003 on bread / lichen / bark (quality .037 / .017 / .015);
  - text bag .99;
  - mean luma .452 (reference .458, quality .367).

**Seed 202 (6 prompts, 12:35 UTC):** again nearly indistinguishable from the reference. The lichen has no rosettes; every fast method has them on this seed. Luma .251 vs reference .259.

**Earlier handoff points** (5d/5e):
- exact to .926 + 6 turbo steps stamps;
- exact to .671 + 3 is almost clean.

**So the finish is not the problem.** The whole gap between the turbo profiles and the original is how accurately the prefix reaches the σ .463 state. A better-trained prefix is the right investment.

## What the pilot showed (M6b Phase 4)

**Training:** 253 teacher samples at 768², of which ~25% were text (≈ 63, from 10 sign + 6 label + poster templates); 221 used for training. Rank 64, LR 1e-4, 2,339 iterations in the 60-min cap.

**Validation** (relative state error at σ .463 vs the 50-step teacher, 32 held-out samples):
- the curve is noisy and never clearly converged: .455 → .427 → .455 → **.399** → .417;
- the cheapest no-training bar is guided Euler with 12 forwards: .395.

**At 1024², eval set E:**
- it removes stamping on lichen, bark and bread (repetition excess ≈ 0);
- **but text fails** (Chinese note illegible, sign garbled);
- images are washed out (luma .343 vs ref .388), detail Δ −.032.

**Mixing** (teacher steps first, pilot after; T2S4F2) made text *worse*. The pilot is fragile off its training distribution.

## Hypothesis

The text failure is under-training plus a resolution mismatch, not a flaw in the objective:
- **data:** ~60 text samples, all at 768²;
- **resolution:** glyph strokes at 1024² occupy different token scales.

**Falsifier:** with 4× the data at 1024², 45% text, validation error must fall clearly below the guided-Euler bar (target ≤ .30). If it plateaus near .40, the 6-step CFG-free regression is capacity- or step-limited, and the route is closed.

## Run

| item | pilot (done) | v2 |
|---|---|---|
| resolution | 768² | **1024²** |
| teacher samples | 253 | **1,000** (val 64) |
| text share | ~25% (templates) | **~45%**: multi-line EN signs, menus with prices, CN handwritten notes, labelled charts; all templates disjoint from E (checked) |
| student steps to σ .463 | 6 (8 passes in total) | **8** (10 passes in total, ~78 s on the Mac: the same cost as quality). Smaller jumps per step. All grid points are on the teacher's 50-step grid, so the targets stay exact |
| iterations | 2,339 (60-min cap) | up to 6,000; val every 500 |
| rank / LR | 64 / 1e-4 | 64 / 1e-4, cosine decay after 1,000 |
| self-rollout share | 50% after it 500 | unchanged |

## Cost (from measured pilot rates, A100-SXM4-80GB SECURE at $1.79/h)

| stage | estimate |
|---|---|
| teacher | 20 s/sample at 768² (84 forwards) → ~40 s at 1024² → ~11 GPU-h for 1,000 |
| training | 1.54 s/iteration at 768² → ~3 s at 1024² → ~5 h for 6,000 |
| overhead | boot, 27 GB download, encode ~0.5 h |

- **Total ≈ 17 A100-hours ≈ $30.** Wall clock is ~17 h on one GPU, or ~8 h if the teacher stage runs on a 4-GPU pod at the same cost.
- **Hard cost guard: $45** (`pod_guard.sh` deadline). The guard also deletes the pod on completion, no heartbeat, or bootstrap failure.

Each sample stores 12 latent states in float16 (the 6- and 8-step grid points, ~12 MB), ~13 GB in total. They are uploaded to the private dataset so a later retrain (another grid or rank) costs no new teacher time.

**Evaluated on the Mac afterwards:**
- the student alone + finish;
- the student blended with the turbo adapter (λ .2 / .5, the Phase 4d trick, 0 extra passes).

## Stop rules

1. **Early stop:** at iteration 3,000 the best validation error is above 0.9× the guided-Euler error on the same 8-step grid (16 forwards), i.e. no clear gain over the cheapest no-training bar. Fetch and delete.
2. **Local acceptance** (M6b decision rule against `quality` on E, 1024², 2 seeds):
   - (a) no text regression;
   - (b) ≥ 60% reference-anchored preference;
   - (c) no new defect class;
   - (d) ≤ 115 s. At 10 passes it would be ~78 s.

## Readiness (08:30 UTC)

- Code: `tq/pod_v2/` (teacher, train, prompts, run.sh); `BUNDLE=pod_v2 NAME=moss-prefix-v2` in `pod_launch.py`; guard `tq/pod_guard_v2.sh` (cap from the live costPerHr).
- Smoke-tested end to end on a tiny random model on CPU: states at the 12 grid points (fp16), 8-step student, validation, best checkpoint, early stop.
- Prompts: 1,100, 44.5% text; none contains a quoted string from the eval set.
- Results go to `results_v2/`, so the pilot's old "PILOT COMPLETE" status can't trigger the new guard.

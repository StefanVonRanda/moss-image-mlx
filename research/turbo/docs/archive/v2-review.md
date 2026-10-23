# MOSS turbo v2: review before the next training run (2026-10-21)

Every claim below points to measured evidence in `LOG.md`, Phases 6a–6d. The tables are in `tq/compare/`.

## What the raw evidence says

1. **The finish is solved without training.**
   - From the exact 50-step state at σ .463, the original model with DPM-Solver++(2M), 3 evals (6 passes), lands at mean |img − ref| .0047. The 2-step turbo finish lands at .0218.
   - Repetition and tone match the reference.
   - 4 evals give .0044, so 3 evals is saturated.
2. **Stamping is committed before .463 and no finish undoes it.**
   - quality's own prefix + the same original finish keeps the stamped lichen daisies and bread lattice.
   - The dose-response is clear: exact to .926 + 6 turbo steps stamps; .671 + 3 barely; .463 + 2 not at all.
3. **A regression-trained prefix does not stamp, even when inaccurate.**
   - pilot prefix + original finish: repetition excess ≤ 0 on all five calibrated images, luma +.004 vs reference.
   - But every structural failure of the prefix survives: ghosted letters, an illegible ZH note, a double-exposure face.
4. **The error budget is demanding.**
   - The pilot's state at .463 is off by rel .31–.55 (mean .40).
   - Mixing the exact state with the pilot's: at 25% of that error (≈ .10) the result is indistinguishable from exact; at 50% (≈ .20) ghosting appears.
   - Mixing makes double exposures by construction, so this is the budget for averaging-type errors specifically.
5. **The pilot's training was broken by its targets, not only by too little data.**
   - Teacher-forced losses fell steadily: step 0 .165 → .12, steps 3–5 ≈ .02.
   - From it 500, own-rollout states with "teleport" targets `(x*_{k+1} − x̂_k)/Δσ` took the losses to .3–1.3; validation never left .40–.46.
   - Those targets aren't valid flow velocities. MSE answers them by averaging trajectories, which is the same signature as the ghosting.
6. **Architecture** (official code):
   - single-stream DiT with joint attention over image and text tokens, 2-D RoPE;
   - the σ grid is fixed at shift 5 (no dynamic shifting);
   - the timestep reaches all 42 blocks through **one shared modulation**. Our LoRA (252 block projections) is σ-agnostic.

## Highest-leverage changes

| # | change | mechanism | training cost | inference cost | test |
|---|---|---|---|---|---|
| 1 | Original-model finish (DPM++ 2M, 3 evals, CFG 3 + skip 22); no DMD adapter anywhere | removes the only stamping source after .463 (evidence 1–2) | 0 | 6 passes, about the 6 the turbo steps cost in quality | done (finish bench) |
| 2 | Valid on-policy targets: relabel own-rollout states with the guided teacher's continuation over the same interval, plus a curriculum ramp | stops the trajectory averaging that produces ghosting (evidence 5) | ~1.6× per iteration (+~10 guided forwards on half the iterations) | 0 | paired short runs, teleport vs relabel, same data and seed: val error and paired image metric |
| 3 | Data: 2,000 dense trajectories at the serving pixel count, prompts matched to real use, aspect ratios if wanted | the pilot saw 221 samples and the teacher-forced losses were still falling | teacher ≈ 22 A100-h | 0 | learning curve from data subsets |
| 4 | Student steps front-loaded at high σ | step-0 loss is 5–8× the others; composition forms there | 0 (dense data) | 0 | grid ablation on the same data |
| 5 | (if needed) σ-specific capacity: LoRA on the shared modulation and time embedding, or rank 128 | the only σ path in MOSS is one shared modulation | ~1.1× | ~0 | ablation, only if #2–4 plateau above budget |

**Not recommended for our architecture:**
- **Higher-resolution or crop supervision.** The trajectory is a global function of all tokens, so crops aren't valid targets. A downsampled 2 MP image is a different image, not a better 1 MP target. Fine detail at 1 MP already comes from the original finish (F ≈ A).
- **DMD/DMD2.** Mode-seeking is the measured stamping mechanism.
- **TDM** (distribution matching over trajectory segments): a fallback only if relabelled regression still ghosts. It needs a fake-score network and trades away seed fidelity.

**Correction training** contributes the on-policy idea (kept, with valid targets) and the warning about repeated application (heeded: no student below .463). **DP-DMD** supports regression for the composition steps.

## Budget

- **Per-pass time at 1024²:** bf16 7.0 s, q8 ~7.8 s, plus ~3 s encode and decode.
- **8-step student + 3-eval finish = 14 passes:** ~101 s bf16 / ~112 s q8.
- **10-step student = 16 passes:** ~115 s bf16 (fits 120 s); ~128 s q8 (doesn't).
- The finish is saturated at 3 evals, so extra budget goes to the student.
- Handoff at .671 is under test.

## Staged plan (each stage gated by a stop rule)

| stage | what | A100-h | cost |
|---|---|---|---|
| 1 | 2,000 dense teacher trajectories (+ 50-step finals for 64 val samples) | ~22 | ~$40 |
| 2 | paired short runs: relabel vs teleport, 2,000 its each | ~6 | ~$11 |
| 3 | full run of the winner, ~8,000 its, image-space validation every 1,000 | ~14 | ~$25 |

- **Total ≈ $76; guard cap $85.** The 1024² rates are extrapolated (×2) from measured 768² rates.
- **Stop rules:**
  - Stage 2: relabel must beat teleport on paired image error.
  - Stage 3: stop if validation isn't heading below rel .15 by it 4,000.

## Evaluation (pre-registered)

- **Paired fidelity vs the reference at the same seed** (the student preserves the seed map): |img − ref| and luma Δ. Target ≈ .025, the 25%-error level.
- **Regularity:** DINO dup_frac on real-photo-calibrated prompts (lichen, bark, bread), plus pixel recurrence and a probe set (gravel, foliage, clouds, crowd, knit, freckles; tweed as the legitimate-regularity control) × 4 seeds. Pass: excess ≤ the noise floor (±.005).
- **Text:** exact lines, verified manually. **Structure:** a defect scan for ghosting and malformed faces or hands on 36 held-out images.
- **Diversity:** pairwise DINO distance across seeds vs the reference's.
- **Reference-anchored sheets** for your judgement.

## Update (02:55 UTC): three more measurements

1. **Handoff at .671 is as faithful as at .463.** The 4-eval original finish gives |img − ref| .0051, vs .0053 for .463 + 3 evals.
   - .671 skips the most curved stretch of the trajectory (.62 → .46): a single guided jump errs .03–.05 there, vs ≤ .02 elsewhere.
   - It also cuts teacher work by 17%.
   - Same total budget: 6 student steps + 8 finish passes = 14.
2. **Errors compound strongly.** Single guided jumps err ≤ .05, but chained 6- or 8-step coarse rollouts end at rel .32–.51.
3. **Ghosting depends on the kind of error, not its size.**
   - Those coarse rollouts (rel ≈ .4, valid-flow error) give clean, legible but darker and softer images.
   - Averaging-type error of rel ≈ .2 (50% of the pilot's) ghosts at both handoffs.
   - Hence: relabelled (valid-flow) targets, validation on decoded images after the original finish, and TDM only as a fallback.

**Revised recommendation:**
- Store dense trajectories to .463 and decide the handoff (.671 + 4-eval finish vs .463 + 3-eval finish) by a training-only ablation.
- Default to .671 if the two are tied: easier student job, cheaper data.

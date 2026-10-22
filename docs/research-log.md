# Turbo quality research log (M5, night of 2026-10-18/19)

Goal: recover the 50-step model's visual quality (fine detail, microcontrast, materials, light) at turbo speed,
without smoothing or painted-on texture. Baseline: `--turbo` = lodestones 8-step DMD `lora_latest` @ 0.7, release
scheduler (shift 5), 8 steps, no CFG, 8-bit tier. ~34 s at 768², ~70 s at 1024².

Everything runs through `tq/` (engine.py, run.py, queue_runner.sh); results in `tq/runs/<job>/`, scores via
`score.py` (hand metrics), `dino_metrics.py` (learned features), `sheets.py` (crop sheets, blind mode).

## Evaluation design

- **Prompts** (`prompts.py`): 16 texture/material/light prompts (skin x4, fur, knit, leather, denim+wood, metal,
  stone, water, foliage, food x2, noir lighting, landscape) + 3 controls (EN sign text, small ZH handwriting,
  3-object composition). Screening at 768², seed 42; confirmation at 1024² and a second seed.
- **References**: the original model, 50 steps, CFG 3 (8-bit tier; 8-bit is texture-identical to bf16, M2).
- **Hand metrics** (`metrics.py`), validated on controlled manipulations of a 50-step render:
  - **Waxiness:** interior fine-band RMS ↓, while edge RMS is unchanged and the slope is steeper.
  - **Added noise:** cross-scale coherence ↓ and kurtosis ↓.
  - **Sharpening:** edge RMS ↑↑ and the slope flatter than the reference.

  Genuine recovery has to move interior energy and slope toward the reference without either tell.
- **Learned features** (`dino_metrics.py`):
  - **Gram distance:** layout-free texture statistics, per prompt;
  - **KID:** set-level only.
- **Visual**: 1:1 crops; paired crops when methods share structure; blind sheets (shuffled, key saved).

## Calibration (tex_portrait)

| case | interior fine | edge fine | coherence | kurtosis | slope |
|---|---|---|---|---|---|
| 50-step reference | 0.0063 | 0.0187 | 0.881 | 27.9 | -2.37 |
| turbo | 0.0051 | 0.0187 | 0.906 | 27.6 | -2.70 |
| reference blurred σ1 | 0.0030 | 0.0099 | 0.936 | 21.2 | -3.80 |
| turbo + gray noise σ4 | 0.0070 | 0.0194 | 0.847 | 20.1 | -2.62 |
| turbo + color noise σ4 | 0.0063 | 0.0191 | 0.859 | 23.1 | -2.65 (chroma ratio 0.15 vs 0.02) |
| turbo + unsharp 1.5 | 0.0088 | 0.0311 | 0.883 | 23.3 | -1.96 |

## Findings (chronological)

### D1 diagnosis (03:30-04:28 UTC; 4 prompts with existing references, 768², seed 42)

Log-ratio to the 50-step reference (0 = like the original; negative = less detail):

| method | nfe | interior fine (whole) | in-focus fine/mid | slope Δ | notes |
|---|---|---|---|---|---|
| turbo (0.7, shift 5, 8 st) | 8 | -0.15 | -0.17 | -0.25 | baseline |
| strength 1.0 | 8 | -0.23 | -0.24 | -0.50 | waxier: the distilled weights smooth |
| 16 steps | 16 | -0.18 | -0.18 | -0.23 | more steps do not help |
| shift 2 / shift 1 | 8 | -0.12 / -0.31 | -0.14 / -0.15 | -0.24 / -0.09 | shift 2 slightly better |
| data-space sampler, lodestones | 8 | -0.49 | -0.12 | -0.33 | waxy surfaces + geometric motifs (peakiness 1.35) |
| data-space, darask0 4-step (adversarial) | 4 | -0.27 | -0.07 | -0.20 | crosshatch "screen-door" grid (peakiness 1.60) |
| turbo then 3 base steps from σ0.19 (cfg 1 / 3) | 11 / 14 | -0.17 | -0.17 | -0.26 | late refinement adds nothing |
| handoff σ0.5 → base tail cfg1 | 14 | -0.10 | -0.14 | -0.15 | small gain |
| stochastic re-noise η0.5 | 8 | **-0.04** | -0.14 | -0.31 | coherence/kurtosis like reference |
| base 8 steps + CFG | 16 | -0.10 | +0.25 | +0.17 | noise signature (coherence -0.018, kurtosis -2.3, chroma 0.10) |
| base 50 steps, no CFG | 50 | -0.30 | +0.06 | -0.03 | CFG itself carries texture |
| (harness check) base 50 + CFG | 100 | +0.003 | | -0.007 | reproduces quant/D8T8 |

**Visual (1:1 crops), which the numbers alone would have missed:**
- **Turbo's deficit is not only wax.** Textures collapse into stylized repeating motifs:
  - bread: cross/star motif on the flour and a regular cell pattern in the crust;
  - mug: sparse, even dots on a glossy glaze where the original has a matte, granular glaze.
- **darask0's extra detail is a synthetic grid,** so the adversarial adapter trades wax for artifacts.
- **Whole-frame energy metrics are confounded by depth of field** (bokeh reads as wax). Hence the in-focus fine/mid metric and a periodic-peak detector.

**Traces (decoded clean estimates per step, portrait):**
- **The original model commits its wrinkle structure by σ≈0.76.** The band below 0.5 only refines it.
- **The turbo's estimate at σ0.97 has more wrinkle texture than its final image.** Each later step smooths a little more.
- So the loss is progressive averaging along the distilled trajectory. It isn't a lack of capacity at the start, and it isn't the final jump. Where texture forms (σ 0.5–0.8) is where intervention has to happen.

### P1: stochastic sampling and layer-selective adapters (04:40-05:13 UTC; same 4 prompts)

All 8 nfe. Scores are in-focus fine/mid, in-focus interior, slope Δ and coherence Δ, against the reference:

| method | fine/mid | interior | slope Δ | coh Δ | visual |
|---|---|---|---|---|---|
| turbo | -0.17 | -0.18 | -0.25 | +0.002 | glossy, saturated, stylized motifs |
| η0.3 everywhere | -0.11 | -0.09 | -0.25 | -0.003 | more pores; motifs unchanged |
| η0.5 late steps only | -0.16 | -0.18 | -0.26 | 0 | no gain: **fresh noise only helps early** |
| η0.5 early steps only | -0.15 | -0.10 | -0.29 | -0.005 | all of the η gain |
| η1.0 | -0.06 | -0.17 | -0.45 | **-0.033** | noise signature |
| adapter off, front 12 / mid 18 blocks | ≈ turbo | | | | not where the look comes from |
| adapter off, rear 12 blocks | +0.35 | -0.25 | +0.09 | -0.021 | hazy, unconverged |
| rear 12 at 0.4 | +0.15 | -0.14 | -0.07 | -0.013 | **matte natural mug glaze, organic flour, natural skin tones; slightly softer** |
| FFN-only adapter (attention adapters off) | -0.02 | -0.12 | -0.23 | +0.001 | matte mug; stronger flour stars |
| attention-only adapter | +0.21 | -0.13 | -0.01 | -0.022 | noise tell |

**Interpretation:**
- **The distillation adapter's rear blocks (30–41) carry the "turbo look":** gloss, saturation, stylized repeating micro-motifs and the plastic finish.
- **The front and middle blocks carry the few-step capability:** switching their adapters off changes nothing visible.
- **The trade-off:** weakening the rear adapters restores the base model's material rendering (matte glaze with a natural speckle gradient, organic flour mottling, photographic skin tones) at zero extra cost. But the base-model rear blocks converge less sharply in 8 steps.
- **The large motifs** (the star pattern on the flour) are decided earlier: front or middle blocks, or early steps.

**Weight-space check:** relative adapter delta ‖0.7·BA‖/‖W‖, averaged over projections:
- blocks 0–11: 0.0002;
- blocks 12–29: 0.0003;
- blocks 30–41: 0.0006, peaking at 0.0007 in blocks 32–36.

The distillation concentrated its edits in the rear blocks, the same blocks whose adapters produce the stylized look. All deltas are under 0.1% of the weights, consistent with the author's warning that merging into bf16 erases them.

### P2: rear-block strength, depth and schedule (05:50-06:12 UTC; 4 prompts)

- **Every rear attenuation moves the spectral shape toward the reference**, but lowers absolute fine energy at edges (edge Δ -0.19 to -0.32, vs -0.12 for turbo): less crisp. Fine/mid goes from -0.17 to between -0.07 and +0.19, and the slope gap from -0.25 to between -0.03 and -0.15.
- **Attention-only rear attenuation does almost nothing;** the FFN rear adapters matter more.
- **Full adapter only for σ ≤ 0.5 ("lastfull")** brings crispness back, and with it part of the look.
- **Rear ×0.4 + shift 3 is visually the best so far.** Portrait: deep natural wrinkles and photographic tone, like the reference. Bread crust: the most organic. Shift 3 places more of the 8 steps in the mid/low band, so the base-like rear blocks converge.
- **The star motif on the flour survives every rear variant:** it's a composition-stage habit of the adapter.
- **Ceilings:** turbo structure plus 50-step base + CFG in the tail below 0.5 (22 nfe) or below 0.35 (18 nfe) barely changes the metrics (fine/mid -0.15 / -0.18). Paying for base + CFG late buys almost nothing.

### P3: rear × shift grid on 7 prompts (06:35-07:33 UTC), a negative result

The 3 new prompts reversed the rear-block lead:
- **Leather, knitting hands, freckled skin:** rear attenuation (×0.4–0.7, any shift) is softer and hazier. Leather edge log-ratio reaches -0.83 to -1.22, and freckles get *smoother* skin than turbo.
- **The mug glaze and old-man skin-tone gains do not generalize.** Not a robust fix; dropped as a primary direction.
- **Plain shift 3 is a small, consistent gain** at zero cost: surface detail -0.05 vs -0.08, fine/mid -0.10 vs -0.13.

**The deeper finding (visual):**
- **The distilled model idealizes *content*,** not just texture. Against the 50-step model it draws:
  - glossy uniform pebbled leather where the reference is worn and scuffed;
  - young clean hands where the reference has elderly wrinkled ones;
  - beauty-filter skin with sparse freckles where the reference has dense natural freckling.
- **These are mode-seeking choices made at high noise** (composition), which every band swap so far left in turbo's hands.
- **Next test:** base model + CFG for the first 2–4 coarse composition steps, then turbo. The 2-step version costs the same 8 nfe.

### P4: base-model composition, turbo finish (07:53-08:16 UTC; 7 prompts), the first robust win

Segments: base model (adapter off) + CFG 3 on the first steps, then the turbo adapter (0.7, no CFG) to σ=0.

- **Base 4 steps (1→.968→.926→.871→.791, 8 nfe) + turbo 4 (12 nfe, 48 s at 768²):**
  - leather comes out worn and distressed, as in the reference (turbo: glossy, uniform);
  - freckled skin gets dense natural freckles with no beauty filter;
  - knitting hands are *weathered*, as the prompt asks (turbo drew young, clean hands);
  - the warm lamplight mood matches the reference.

  Slightly soft in the finest detail.
- **Base 3 coarse (1→.953→.883→.771) or 2 (1→.91→.717) + turbo 4:** fails. Ghosted double exposures on the face, unresolved leather, blur. Big first jumps leave a blurry average composition that turbo can't resolve.
- **The hand metrics mislead here.** Lower fine energy and big kurtosis come from moodier lighting and larger dark or defocused areas, which is exactly why every candidate gets a visual check.

**Conclusion:**
- **The "idealization" (young, clean, glossy, smooth) is decided in the first ~4 high-noise steps.** Giving those steps to the original model with CFG restores the original's content character *and* prompt adherence.
- **Turbo can finish from σ≈0.79** without re-idealizing the content.
- **Open:** cost (12 nfe = +50%) and the last bit of crispness.

### P5: cheaper composition handoffs (08:37-09:27 UTC; 11 prompts incl. text)

| variant | nfe | fine/mid | slope Δ | OCR EN / ZH (768²) | visual |
|---|---|---|---|---|---|
| turbo | 8 | -0.13 | -0.23 | 0.98 / 0.96 | idealized |
| turbo, shift 3 | 8 | -0.11 | -0.24 | 0.90 / 0.96 | ≈ turbo |
| A: base 2 fine steps (1→.968→.926) + turbo 6 | 10 | -0.14 | -0.16 | 0.94 / 0.96 | natural freckles, irregular cobbles, worn leather |
| base 3 fine + turbo 5 | 11 | -0.11 | -0.08 | 0.92 / 0.83 | |
| B: base 3 fine + turbo 6 on a shift-3 grid | 12 | -0.14 | -0.09 | 0.92 / 0.96 | densest, most reference-like freckles/skin |
| base 4 + turbo 4 | 12 | -0.08 | -0.06 | 0.88 / 0.91 | darker, softer |
| base 4 + turbo 5 (shift 3) | 13 | -0.11 | -0.07 | 0.82 / 0.87 | |

- **The sign headline is correct in every variant.** The OCR differences come from the tiny chalkboard, which is smaller and darker in the base-composition scenes (realistic dusk). Turbo misreads it too.
- **Prompt adherence on the 3-object layout is preserved,** and `quality` has more natural tones.

### Finalist A (`quality`) vs turbo on all 16 texture prompts, seed 42

- **Hand metrics:** fine/mid -0.16 vs -0.13 and slope -0.16 vs -0.20. So `quality` does **not** add fine-scale energy (its finish is still turbo, and its lighting is moodier).
- **DINOv2:**
  - KID 12.7 vs 25.7 (×10⁻³);
  - per-prompt Gram texture distance closer on 11/16, mean Δ -0.035 (95% bootstrap CI -0.066 to -0.005).

  The *distribution* moves toward the original model.
- **Composition agreement with the 50-step image from the same seed:** low-res correlation 0.66 vs 0.58, higher on 11/16 (watch 0.09 → 0.60). Modest.
- **Where Gram says "worse"** (leather, denim, elder hands, mug, portrait), the crops show `quality` closer to the original:
  - worn creased leather;
  - faded denim folded as in the original;
  - the original's knit sleeves and pose, where turbo invents a scarf and strawberries.

  The Gram metric is not reliable per image.
- **Package check:** `--turbo quality` / `--turbo quality-plus` reproduce these runs bit for bit, with the same 9.3 GB peak.

### Final evaluation (09:48-12:13 UTC)

- **Blind, seed 7, unseen images** (`tq/blind7/`): `quality` preferred over old turbo on 12/16 (9 clearly), turbo on 2 (salad clearly; wave slightly), 2 ties. Sign test one-sided p = 0.0065. One judge who knew the style by then; supporting evidence only.
- **Blind, `quality` vs `quality-plus`** (3 composition steps + shift-3 finish, 12 nfe; `tq/blind7ab/`): 3 slight wins, 2 losses (one a mangled hand/pen), 3 ties. No measurable benefit for +20% time: not shipped.
- **Reusing the unconditional branch in step 2** (9 nfe): ~20 dB from `quality` (not identical), keeps the non-idealized look, slightly rougher bracelet geometry on the watch. Promising follow-up (+12.5% instead of +21%), not validated blind.
- **1024², seed 101:** `quality` 78 s vs turbo 64.5 s of sampling (+21%), with the same 9.3 GB peak.
  - Against 50-step references, `quality` matches the original's mood, framing and freckles (skin) and its street layout and irregular stones (cobbles). It's clearly closer on leather, though not as distressed as the original.
  - Text is spelled correctly (EN sign, ZH note OCR 1.0).
- **Shipped:** `--turbo` = profile `quality` (`--turbo fast` = old). Repo commit `9290878` (local, unpushed), 17 tests pass. The CLI reproduces the research runs bit for bit.

## M6: continuation toward the best quality within ~120 s at 1024² (started 2026-10-19 15:12 UTC)

Brief from the user + GPT Pro: preserve `quality`. Attack the three failures separately:
- content idealization;
- excessively regular small structures (bread lobes, beard bundles, tiered branches, subdivided clouds);
- lost fine detail.

Budget is ≤ ~115 s at 1024²; compare every candidate against `quality`, with the 50-step original as the reference.

### C1 (15:12-15:25 UTC): evidence already on disk

**Seed-7 originals vs fast vs quality** (the 4 references never analysed before; `tq/runs/ref_base50/*_s7.png`):

| prompt | original (50 st) | fast | quality |
|---|---|---|---|
| bread | fibrous torn ear, irregular smooth flour dusting | scalloped lobed crumb along the score; flour in regular star blotches | **same lobes and blotches as fast** (residual in quality) |
| portrait | irregular mixed beard hair | evenly curled beard bundles | **same curls**; better weathered skin and wrinkles |
| forest | foggy, irregular trunks | evenly spaced trunks, tiered pines | more natural spacing than fast; some regular trunk spacing remains |

**Where the regular structures arise** (seed 42, using the earlier band-swap runs):
- **Bread lobes** appear when *either* band is turbo:
  - teacher 4 steps (to σ .791) + turbo: lobes;
  - turbo above .8 + teacher 50-density + CFG below .8 (50 nfe): lobes, softer;
  - turbo above .8, teacher only in .8–.5: lobes.
- **The flour star blotches** disappear only with the teacher through σ .791 (T4). T2 keeps them, so they are decided at σ .926→.791.
- **Beard curls** are present in fast, T2 and T4.

So the turbo continuation itself manufactures the morphology: from a teacher-composed state (T4), and it commits lobes above .8 that a later teacher cannot undo.

**OCR "THE" (1024², seed 101, `quality`):** the image is correct; this was an OCR miss, not a rendering error.
- **Requested:** "THE GOLDEN CRUST BAKERY"; chalkboard "Fresh sourdough daily", "Open 7am - 6pm".
- **Visible at 1:1:** all of it, including a small "THE" above-left of "GOLDEN" and "7am - 6pm".
- **Apple Vision:** "GOLDEN CRUST BAKERY" (confidence 0.5; "THE" skipped) and "Tam-opm".

**New diagnostic `recurrence.py`:**
- **What it measures:** the DINOv2-small layer-6 patch tokens (14 px, 1:1 tiles), restricted to textured tokens. For each token, its nearest-neighbour cosine to tokens ≥ 56 px away. `dup_frac` = share ≥ 0.85.
- **Calibration:**
  - noise σ6 and blur 1.5 leave it unchanged (0.016 / 0.013 vs 0.015);
  - token-aligned stamped motifs over 5% of the area raise it to 0.044;
  - unaligned stamps barely register (phase sensitivity; it's a relative tell, not an absolute measure).
- **Seed-7 results, ref / fast / quality:**
  - bread: 0.015 / 0.075 / 0.078;
  - portrait: 0.007 / 0.024 / 0.017;
  - forest: 0.010 / 0.040 / 0.012.

  Consistent with the crops: quality fixes forest spacing and some beard curls, not bread lobes.

**akhaliq/MOSS-Image-SFT-4-steps** (pinned `7258a71`; root `lora.safetensors` = `moss_image_sft_4step_v2.0`):
- v2.0 is Euler-native: 4 release-grid steps (1, .91, .717, .024), CFG 1, strength 1.0;
- warm-started from lodestones; differs from both adapters tested before.

Registered as `akh4`; a bounded check is queued after the teacher-solver screens.

**Decisions and queue:**
- T3 (3 teacher + 5 turbo on the release grid, 11 passes) and T4 are being filled on the dev set at seeds 42 and 7.
- Teacher-only DPM-Solver++(2M) with CFG is queued at 7 and 6 evals (14 and 12 passes), plus a shift-3 variant and the teacher on the release grid to .463 with turbo for the last 2 steps.
  - The solver passes exactness tests on analytic flows: an exact point mass, and a Gaussian target that converges at 2nd order.
- The no-skip (full-depth negative) prefix is queued for decomposition B.

### C2 (15:25-15:35 UTC): the morphology is committed by the student's FIRST estimate

Decoded clean estimates along `quality` (seed 42; `tq/runs/m6_trace_T2/trace_png`):
- **The teacher steps' estimates are blurry means:** σ1 and σ.968 (the grid pattern there is the VAE decoding an off-manifold mean latent).
- **The student's first estimate already commits the final morphology.** At σ .926 it contains:
  - the bread's scalloped crumb lobes;
  - the beard's fan of evenly curved strands;
  - plus a transient crosshatch at σ .926–.871 that vanishes by .791.

  Later student steps only refine.
- **So the regular structures are a mode-seeking jump:** from a blurry high-noise state, the student predicts a sharp, stereotyped x0. The 50-step teacher instead commits texture gradually, mostly by σ≈0.5–0.75 (D1 traces).
- **Prediction:** the student must not be asked for its first x0 before the teacher has committed the relevant morphology, around σ 0.6–0.46. That makes "teacher to ~0.46 with a 2nd-order solver, student finishes the last 1–3 steps" the main candidate family.
- **Queued as m044/m046/m047/m048** (10–14 passes), alongside full teacher solvers (12–14 passes).
- **Execution profile at 1024²** (`runs/profile/profile.json`):
  - per forward: q8 7.50 s; q8 + LoRA 8.07 s; dense bf16 with the same dequantized weights 6.92 s (−7.8%); dense + LoRA 7.49 s; `mx.compile` 7.37 s (bit-identical);
  - dense raises the peak to ~12 GB (fine within the guard).
  - Dense vs q8 base output differs 1.0% (kernel rounding). A 14% discrepancy with the adapter on is being re-checked (`profile2.py`); it's probably a lazy-evaluation artefact in the profiler.

### C3 (15:35-16:00 UTC): matched-state probes, execution check, first teacher-solver result

**Matched states along quality (bread, seed 42; `runs/probe_states/`):**

At the same latent, relative x0 difference between the student and the guided teacher:

| σ | student vs teacher | skip-22 guided vs full-depth guided |
|---|---|---|
| .926 | 0.62 | 0.17 |
| .871 | 0.58 | 0.13 |
| .791 | 0.47 | 0.13 |
| .671 | 0.35 | 0.10 |
| .463 | 0.20 | 0.07 |

- **Decoded at the same state:** the teacher's estimate is a blurry mean until ~.67, while the student's is already sharp, with the lobed crumb at .926.
- **The block-22 skip is not negligible.** It changes the guided clean estimate by 10–17% at the prefix states. The trajectory-level effect is in m05 (queued).

**Dense vs q8 with the adapter** (clean re-check, `runs/profile/check_dense_lora.json`): the base output differs by 1.1%, but with the adapter on, the outputs differ by 13% at σ .9.
- The adapter's own effect there is 72%, so the student is very sensitive to tiny base perturbations at high noise.
- **Consequence:** dense execution is a legitimate 7.8% speed-up for teacher passes, but it changes student-dependent renders in detail. Adopt it only together with a render check.

**Teacher-only DPM-Solver++(2M), 7 guided evals (14 passes), seed 42, 768²:**
- **Morphology** is the best of any fast method so far:
  - bread: irregular fibrous crumb, no scalloped lobes, no star blotches;
  - beard: irregular hair, no fan of curls;
  - leather: worn and distressed;
  - oak: irregular branching;
  - clouds: natural.
- **But text collapses:**
  - EN OCR 0.35 ("GOLDEN ♥ GRUST / EANERN"); the ZH note is unreadable (0.0) in a washed-out, blurry frame;
  - compose is correct but desaturated;
  - the portrait is somewhat over-contrasted (gritty skin).

**Reading:** the teacher at this step count gets texture organisation right but integrates glyphs poorly. The student is excellent at glyphs but stamps texture. Text and texture are decided in overlapping noise ranges, so a simple handoff point may not serve both. Pending: m042–m048 (6 evals; shift 3; Euler control; teacher to .46 + student 2/3; 4–5-eval 2M prefixes). DPM-Solver++(3M) added (verified 3rd-order-ish on a Gaussian flow).

### C4 (16:00-16:35 UTC): the decisive split. The teacher organises texture above σ≈0.46; the student finishes text below it

**T6S2** (`m6_rel6_2m_turbo2`) = guided original model with DPM-Solver++(2M) for 6 evals on the stored release grid (1 → .968 → .926 → .871 → .791 → .671 → .463), then the turbo adapter (0.7, no CFG) for the last 2 release steps (.463 → .024 → 0). That's 14 branch passes; 55 s at 768² vs 41 s for quality.

| run | passes | EN OCR | ZH OCR | repetition excess (DINO dup_frac over ref, 3 prompts) | bread / portrait / forest dup |
|---|---|---|---|---|---|
| reference (50 st) | 100 | 0.86 | 0.96 | 0 | .025 / .059 / .014 |
| fast | 8 | | | 0.079 | .111 / .173 / .050 |
| quality | 10 | 0.94 | 0.96 | 0.043 | .086 / .120 / .022 |
| T4 (Euler) | 12 | | | 0.033 | .061 / .120 / .014 |
| teacher-only 2M, 7 evals | 14 | 0.35 | 0.00 | 0 | .004 / .019 / .002 |
| teacher-only 2M, 6 evals | 12 | 0.31 | 0.04 | 0 | |
| teacher-only 2M, 7 evals, shift 3 | 14 | 0.47 | 0.00 | | |
| **T6S2** | **14** | **0.96** | **0.91** | **0.019** | **.025 / .114 / .015** |

Visual (1:1 crops):
- **Bread:** T6S2 matches the original's character: irregular fibrous crumb and smooth flour dusting, with no scalloped lobes and no star blotches.
- **Text:** the sign reads "THE GOLDEN CRUST BAKERY" in the right serif; the chalkboard is legible.
- **Elsewhere:** clouds, oak branching, the beard, weathered hands and worn leather all look natural.
- **Teacher-only runs** garble letters and wash out the sticky note.

So glyph precision is set in the low-noise tail, where the student is excellent, while texture organisation is set above ~0.46, where the student stamps templates. This explains why late teacher refinement and early teacher prefixes each fixed only part of the problem.

**Banding** (MATLOWAI's bandviz, re-implemented in `bandscore.py`; ≥ ~16 at the 16 px / 48-cycle period means visible):
- Teacher-heavy runs inherit SFT's stripes: oak sky ts7 31.7, T6S2 26.4; compose 20.9 / 17.0.
- Quality stays ≤ 13; the student finish removes stripes.
- At colour ×5 the stripes are visible in the oak sky of both, and faintly in quality.
- nynxz's destripe LoRA (SVD-extracted, pinned `7b071d5`, kohya keys matching ours) is queued on T6S2 (m0482).

**The user's two links** (both read; method notes):
- **MATLOWAI:** exact layer swaps plus logit-lens stripe spectra.
  - The stripes need SFT's changes in layers 21–41, are fed by 37–41 first, and compound through the steps.
  - Reverting 37–41 to Base is the shipped LoRA.
- **nynxz:** SVD of Base→SFT deltas per layer, then one backward pass scoring every direction by d(stripe energy)/d(direction). Subtracting the top 64 directions gives a 1.9 MB LoRA with ranks 1–7, which beats layer reversion at 0.99 SSIM.
- **Transfer to our regularity problem:** `attrib.py` does the same for the turbo adapter.
  - Exact SVD of each layer's BA with a per-direction gain.
  - Objective: soft latent patch recurrence of the student's clean estimate at the states where the morphology commits (σ .926/.871/.791 of quality).
  - The latent proxy is validated on saved final latents: mean NN cosine is 0.28 / 0.34 / 0.35 for ref / fast / quality on bread s7, with similar orderings on portrait and forest.
  - A feasibility run is queued.
- **Caveat:** T6S2 makes this less important, because the student no longer makes the morphology decisions. It matters for any cheaper (≤ 12-pass) design where the student starts earlier.

### C5 (16:35-17:55 UTC): variants of the hybrid, destripe, and direction attribution

All at 768², seed 42, dev set. OCR is Apple Vision bag-coverage. "Excess" = repetition over the reference (3 prompts).

| variant | passes | sec | EN / ZH OCR | excess | oak / compose banding | visual |
|---|---|---|---|---|---|---|
| quality | 10 | 41 | .94 / .96 | .043 | 9.1 / 12.1 | lobes, curls |
| **T6S2 (2M)** | 14 | 55 | .96 / .91 | .019 | 26.4 / 17.0 | correct sign; natural texture; sky stripes |
| **T6S2 (2M) + destripe 1.0** | 14 | 55 | **.96 / .96** | .022 | **4.7 / 9.6** | same image, stripes gone |
| T6S2, Euler teacher | 14 | 55 | .92 / 1.0 | .024 | 25.0 / 12.5 | composition closest to the reference; slightly grainier |
| T6S2, 3M teacher | 14 | 55 | .88 / .91 | .014 | 22.9 / 16.3 | **text breaks** ("GOLDIN CRUST BAIKEAY") |
| T5S3 (teacher to .671, student 3) | 13 | 52 | .92 / .96 | .024 | | |
| C5S2 (custom coarse grid 1, .95, .87, .75, .6, .463) | 12 | 47 | .78 / .83 | .000 | 20.9 / 8.7 | **text garbled** ("GOQDENN") |
| teacher 7 evals, Euler only | 14 | 54 | | | | close to the reference's composition, but grainy and soft |

**Lessons:**
- **Glyph identity needs the fine high-σ release-grid steps** (1 → .968 → .926 → ...). Coarse first jumps and higher-order extrapolation garble letters; the student's last 2 steps sharpen glyphs but can't fix wrong ones.
- **Texture organisation is set between ~.9 and ~.46,** where the teacher must be in charge.
- **The 2M teacher** sharpens slightly beyond the reference; Euler stays closer but grainier. The student finish removes the grain in both.
- **Destripe** (nynxz, rank 1–7 directions in blocks 24–41, applied to every pass) removes the inherited SFT stripes with no visible side effect.

**Direction attribution** (`runs/attrib/`, 6 repetition prompts × 2 seeds × 3 states):
- 15–17 s per backward at 768²; MLX peak 24.6 GB.
- **Caveat:** the process footprint reached 38.8 GB and swapped (1.6 M swap-outs), the guard didn't trip; `mx.clear_cache()` per backward added since.
- **Concentrated:** top 3 directions = 0.40 / 0.30 / 0.19 of the normalised sum; top 64 carry 69% of positive attribution, top 256 83%.
- **Location:** mostly the leading singular directions (index 0–4) of the rear blocks 34–41: ffn_up_gate, o, ffn_down, v (plus a few in blocks 0–1).
- Removal tests (top 3/16/64/256, gain 0, on the quality profile) are queued; this is the nynxz method applied to our artifact.

### C6 (17:55-18:45 UTC): direction surgery on the turbo adapter is a negative result; the hybrid holds at seed 7; timing

**Removing the top-K regularity-attributed adapter directions** (gain 0, quality profile):

| K removed | dirs / layers | excess repetition | EN / ZH OCR | visual |
|---|---|---|---|---|
| 0 (quality) | – | .043 | .94 / .96 | lobes, stars |
| 3 | 3 / 3 | .043 | .92 / 1.0 | identical |
| 16 | 16 / 13 | .043 | .90 / .87 | identical |
| 64 | 64 / 41 | .044 | .88 / 1.0 | identical |
| 256 | 256 / 96 | .040 | .92 / .91 | identical lobes and star blotches |

- The latent proxy the attribution optimised barely moves on the final latents either (bread .4396 → .4383).
- **Conclusion:**
  - the stripe artifact is narrow-band, crisply measurable and carried by a few SFT directions, so nynxz's surgery works for it;
  - the student's stamped morphology is not carried by a sparse set of adapter directions that a first-order latent-recurrence gradient can find.
- Two final probes are queued (K = 1024 at gain 0; K = 256 at gain −1); otherwise this lane is closed.
- The useful transfer from the two repos is the destripe LoRA itself (on teacher-heavy schedules) and the methodology (exact swaps and probes, which C1–C4 used).

**T6S2 + destripe at seed 7** (768²):
- bread: fibrous torn crumb like the original, no lobes, natural flour;
- beard: irregular;
- OCR EN .98, ZH .96; repetition excess .008 vs quality .025.

**Timing at 1024²:** T6S2 + destripe 1.5 on the official bf16 SFT weights (dense) takes **97.8 s of sampling** for 14 passes (7.0 s/pass), about 101 s end to end with encode and decode. That's within the ~115 s budget. The q8 tier is about 107 s.

**Queued:**
- 10 held-out prompts at 1024² (seed 101), with quality and T6S2d on all of them and 50-step references for 4. They cover:
  - textures: moss wall, tweed, dunes, bark, crumpled paper, blood orange, crowd;
  - text: a menu with prices, a Chinese poster, an infographic with percentages.
- The 18-prompt 1024² validation, the CFG skip decomposition (m05) and the dev-set references (m07).

### C7 (19:15-19:35 UTC): held-out 1024² blind check. The hybrid is NOT a uniform win

10 never-used prompts, seed 101, 1024², `quality` vs T6S2+destripe 1.5, sealed shuffled keys, picks recorded before reveal (`tq/blind_unseen/`). Tally: **T6S2d 2, quality 4 (2 slight), 4 ties.**

| prompt | pick | why |
|---|---|---|
| bark | **T6S2d** | quality stamps bold uniform plate outlines; T6S2d has varied plates with fine fissures |
| moss wall | **T6S2d** | quality covers the stones with identical repeated lichen rosettes and a regular brick pattern; T6S2d has organic lichen |
| crowd | quality (slight) | T6S2d has smeared, ghosted pedestrians |
| infographic | quality | T6S2d garbles "62%" on the ring; label mapping wrong in both |
| orange | quality (slight) | T6S2d pulp softer and hazier |
| paper | quality | T6S2d crumple hazy and low-contrast |
| dunes, menu, ZH poster, tweed | tie | all text correct in both; ripples and herringbone legitimately regular in both |

**Reading:**
- T6S2 fixes exactly the targeted failure: stamped repetition in bark, lichen, bread and beard, on held-out prompts too.
- But at 1024² it is softer and hazier on some materials (paper, orange), ghosts small figures (crowd), and can break a numeral.
- Those are the failure modes of coarse teacher steps (cf. coarse prefixes in M5).
- The 768² dev-set results overstated it.

**Diagnosis queued** (1024², 6 held-out prompts):
- T6S2 without destripe (is destripe 1.5 softening?);
- finish strength 1.0 (crisper finish?);
- T5S3 (the student finishes from .671: more crispness, possibly more stamping);
- T7S2 with an extra teacher point at σ .567 (16 passes, finer mid-band; within budget at ~112 s bf16).

### P (20:00-20:20 UTC): RunPod pilot launched (user approved "runpod. go for it")

- **Target changed from the proposal, for a reason found in C7:** the student imitates the *50-step* guided teacher's states at the release-grid points 1 → .463, not the 6-eval 2M teacher, whose coarse steps cause the haze and ghosting seen on held-out 1024² prompts.
  - Student: 6 CFG-free steps on the release grid, then the lodestones finish for 2 steps: 8 passes in total.
  - Training is trajectory regression on teacher states plus the student's own rollout states (50% after 500 iterations).
  - LoRA rank 64 on all 252 projections; 768²; ~720 teacher samples from 515 template prompts disjoint from all eval prompts; time caps: teacher 80 min, training 60 min.
- **Code:** `tq/pod/` (official PyTorch MOSS code). Smoke-tested end to end on a tiny random model on CPU (validation error fell 0.041 → 0.036 in 120 iterations). Bundle and results go through the private dataset `VisualInference/moss-turbo-train`.
- **Capacity:**
  - H100s and A100s were unavailable in both clouds;
  - pod `2l8ib0rnx4msmt` = 1× H200 SECURE (SE), $5.29/h.
- **Mistake:** my retry loop missed a successful create, so a second pod (RTX PRO 6000, $2.49/h) also started. I deleted it after ~2 min; its Blackwell GPU couldn't have run the CUDA 12.6 image anyway.
- **Cost guard** (`tq/pod_guard.sh`, logs/pod_guard.log): polls STATUS every 3 min; on PILOT COMPLETE it downloads `results/` to `tq/pod_results/` and deletes the pod; hard deadline 210 min (~$19 max).
- **Unrelated, untouched:** the account also holds an EXITED pod "ideogram4-finetune-a100" from July (not this project).

### C8 (20:15-21:50 UTC): haze diagnosis, and what the 50-step references say about the held-out disagreements

**Haze variants** (1024², 6 held-out prompts): T6S2 without destripe, finish strength 1.0, T5S3 (teacher to .671), T7S2 (extra teacher point at σ .567, 16 passes, 120 s q8).
- All share the same look as T6S2d: no variant removes the crowd motion-ghosting or changes the paper's softness.
- Destripe is not the cause.
- T5S3 and T7S2 bring back a few lichen rosettes. The "haze" is the 2M-teacher trajectory's character, not a solver-density or finish-strength issue.

**References for the disputed prompts** (50 steps, CFG 3, `runs/u_ref`):

| prompt | blind pick (no reference) | against the reference |
|---|---|---|
| paper | quality (crisp facets) | **reversed: T6S2d.** The original crumples softly and irregularly like T6S2d; quality's crisp regular faceting is itself a manufactured pattern |
| crowd | quality (slight) | **confirmed:** the original has sharp pedestrians; T6S2d's motion-ghosting is a real defect |
| orange | quality (slight) | tie: both are brighter, more uniform red than the original's dark, varied blood orange |
| infographic | quality | quality (slight): the original has a different layout with correct labels ("Solar 62%", "Wind 38%"); both fast methods compose differently and mislabel; quality's digits are cleaner |
| moss wall, bark | T6S2d | confirmed: closer to the original's irregular lichen and plates |

So with references: **T6S2d 3, quality 2, ties 5.** T6S2d wins where the user's target failure lives (stamped morphology) and loses on small-figure coherence.

**Notable:** on the infographic, both 8–14-pass trajectories reach a different composition from the 50-step model's, and the 50-step one is the one that gets the labels right. Early-trajectory discretisation, not just texture, separates them. That is exactly what the RunPod pilot trains: a 6-step CFG-free student regressed onto the 50-step teacher's states.

**RunPod relaunch** (P2):
- Both `pytorch/pytorch:2.14.1` pods sat silent: no heartbeat in 48 min, then 27 min; deleted, ~$6.6 spent with nothing to show.
- Relaunched on `runpod/pytorch:1.4.0-cu1281-torch291-ubuntu2404`, keeping RunPod's sshd, on 1× A100-SXM4-80GB SECURE at $1.79/h (pod `16sh0w2j6c3ifo`). It came up in 2 min; the 27 GB download took ~2 min; encoding 516 prompts ~3 min.
- Teacher (50-step guided to σ .463, 768²): ~19 s/sample on the A100, so the 80-min cap gives ~250 samples, not the planned 720.

**Direction-surgery lane closed:** removing the top 1024 attributed adapter directions (gain 0), or reversing the top 256 (gain −1), leaves quality's bread lobes and flour stars visually identical (excess .038–.039 vs .043). The stamped morphology isn't carried by a sparse set of adapter directions that this gradient finds; the mechanism is the student being asked for a sharp x0 from a blurry high-noise state (C2), not a removable direction.

### Literature (user-supplied): Correction Training for Diffusion Models (Ahn et al., arXiv 2610.14136)

**Their Fig. 15:** repeated correction makes "overly coherent structures, such as disk-like pizza toppings or strawberry seeds arranged in a regular grid", because "repeated correction ... reapplies this learned mapping to its own outputs, beyond the single iteration used during training", which "may reinforce coherent structures, sometimes excessively".

**Mitigations they report:**
- a setting-dependent optimum number of correction iterations;
- restricting correction to an interval of steps;
- jointly training the denoiser so the corrector generalises to its own outputs (exposure bias).

**Distillation:** velocity-space MSE (w = 1/t²) onto the refined predictions; about 67% NFE reduction at a small FID cost.

**Our data match that mechanism as a dose-response** (repetition excess over the 50-step reference, 768² dev set, seed 42), ordered by how many times the distilled adapter's mapping is applied to its own outputs:

| student (adapter) steps | config | excess |
|---|---|---|
| 8 | fast | .079 |
| 6 | quality | .043 |
| 4 | T4 (Euler teacher 4) | .033 |
| 3 | T5S3 | .024 |
| 2 | T6S2 | .019 |
| 0 | teacher-only 2M | .000 |

Monotonic. The C2 traces show the stamped morphology appearing in the student's first estimate and being reinforced on each re-application.

**Implications:**
- (i) The fix is to apply the distilled mapping as few times as possible, and only where it's needed: the low-noise tail for glyph and edge sharpening. That is T6S2's structure.
- (ii) The RunPod pilot's design already follows two of their prescriptions: training on the student's own rollout states (exposure bias), and velocity-space MSE regression onto the better trajectory, rather than DMD's mode-seeking objective.

**Code reviewed** (github.com/sunovivid/correction-training, MIT; ImageNet models only, T2I code "coming soon"):
- **Corrector:** a separate network refines x̂ at the same (z, t), with a stop-gradient. It's applied K times inside a `correction_interval`. `base_anchor` holds the denoiser's complementary (noise) component fixed while the corrector rewrites x.
- **Distillation:** student initialised from the base; targets = teacher velocity after 2 corrections on noised real data; velocity MSE.

**Two follow-ups:**
- **"Beyond training" check for the turbo adapter.** lodestones' lora_latest on its *native* data-space sampler (what it was trained with) has repetition excess .044 vs .079 on the Euler / 0.7-strength recipe the demo, community and we use. That's half the regularity, but it's the waxiest condition in D1. So the Euler re-use of the adapter adds regularity on top of the distillation itself.
- **Base-anchored finish:** T6S2 where each student step keeps the guided teacher's noise estimate (16 passes). Queued as m0496.

### Literature batch 2 (user-supplied; read by three parallel reading agents, judged against our data)

**MOSS-Image technical report (2610.23374) and user guide:**
- **Nothing official on SFT, stripes, or turbo.** The report never mentions SFT; the guide is written for Base.
- **Training noise:** the final stage trains at shift 1 with a logit-normal sampler (μ = −1.09); only ~9% of samples fall at σ ≤ .463 and ~2% at σ ≤ .3. Shift 5 is an inference choice. The teacher's low-noise regime is its least-trained one, which fits our observation that the student finish is excellent there.
- **The guide on our negative-branch skip:** "Turning off SLG will make the output image look very realistic, but it may also cause some details to collapse." This matches C3's measurement (the skip changes the guided clean estimate by 10–17%). Teacher-without-skip in T6S2 is queued (m0497a).
- **Official:** "actively working on few-step distillation (DMD)". The pipeline docstring implies internal CFG-distilled variants exist; none are released.
- **lodestones' adapter** was trained on compact-JSON scene_graph captions (natural-language prompts are off-distribution) and with data-space time embeddings that the Euler recipe doesn't load. lodestones also has a Base 8-step DMD.

**Data-Space Iteration (2610.12102):**
- Every prefix of the rollout is trained to be a valid sample, so step 1 must invent sharp detail from noise. That's our first-estimate stamping.
- Later steps "refine from the previous prediction" with no renoising, so the first commitment is never broken.
- Settings were tuned on FID (texture-blind); Euler use is outside the paper; ImageNet only; no code.

**DP-DMD (2602.03139):**
- Recipe: regression to the guided teacher for the first step, DMD for the later steps, with a stop-gradient between them.
- Results: diversity is preserved; texture is left to DMD.
- Our pilot is the regression half with the split at σ .463.
- Suggested follow-up: retrain the finisher on the actual handoff states (stop-gradient), with a CFG ablation in the real score. Code exists for SD3.5.

**TRIG (2610.15164):** a texture-realism metric that should catch wax (phase smoothing), but has no repetition negative and works at a 256² resize; code unreleased.

**Krea 2:**
- Rejects aesthetic filters.
- Defines the "AI look" as blurry backgrounds, waxy skin and boring composition, and attributes it to preference data and aesthetic scorers.
- Runs RL without CFG.
- Distils with TDM (trajectory-level, per-segment timesteps). Nothing is said about texture surviving distillation.
- Takeaway for us: train finishers only on their own segment, and treat strong baked-in CFG as a suspect for idealization. Teacher CFG 2 / 4 in T6S2 queued (m0497b/c).

**StrSR (2603.06275):** studies token-locked 16-px grids from one huge super-resolution step. That doesn't transfer to our object-scale stamping or to the SFT stripes (one-directional, ~48 cycles per width, not token-locked).

### M6b start (22:10 UTC): plan revamped (tq/PLAN_M6b.md), plus raw-source facts from the adapters' training code

**akhaliq `train_dmd.py`** (pinned 7258a71; warm-started from lodestones lora_latest):
- the DMD generator gradient uses the teacher at **CFG 3 + block-22 skip at every τ**;
- generator τ ~ **Beta(2.5, 1)** (mean .71; only ~5% of samples at τ < .3);
- the fake score is trained at uniform τ;
- training prompts are ~70% text/signage scenes, half as bbox-layout JSON.

lodestones' adapter (the one we use) was trained on compact-JSON scene graphs.

**Implication:**
- the distribution-matching signal barely constrains low-noise (texture-scale) statistics;
- guidance is strong at all noise levels;
- natural-material close-ups are off the training prompt distribution.

All three plausibly feed the stamped, idealized micro-texture. Any finisher retrain should put DMD/τ mass at low noise, use lower or interval-limited CFG in the real score, and train on natural-photo prompts.

**Protocol frozen:**
- eval set E (18 prompts, 5 categories) at 1024², seeds 101/202;
- references on the q8 tier (base-only q8 ≡ bf16 by M2);
- decision rule in PLAN_M6b.

**Queue re-ordered:**
- bf16-vs-q8 confound first, then the latent-HF mechanism test (m0496b), then the 768² teacher-lever triage, then E references (seed 101: 10 missing prompts; seed 202: 6-prompt noise-floor subset).
- 768² fill-ins and refs parked.

### Phase 2a (22:15-22:30 UTC): metric calibration against real photographs

**Real-photo set:** `tq/realphotos/`, from Wikimedia Commons, 15 photos with licences in `sources.json`:
- lichen on stone 3, pine bark 3, crumpled paper 3, herringbone tweed 4, bread 2;
- beard searches returned only paintings and bronzes, so that category was dropped.

**DINO dup_frac** (share of near-duplicate textured patches), seed 101, 1024²:

| material | real photos | 50-step ref | quality | T6S2d | fast |
|---|---|---|---|---|---|
| lichen / moss wall | .000–.005 | .003 | **.020** | .005 | |
| pine bark | .000 | .002 | **.016** | .007 | |
| bread | .014 (crust close-up) | .025 (s42) | .044 | .006 | **.065** |
| crumpled paper | .001–.011 | .031 | .016 | .019 | |
| tweed (legit regular) | .006–.112 | (ref pending) | **.513** | **.324** | |

**Verdict:**
- **Admitted** for organic, stochastic materials (lichen, bark, bread): real photos and the reference sit together, quality is 3–7× outside, T6S2d is inside or near.
- **Not admitted** for crumpled paper: no separation; the reference itself is the highest.
- **Not admitted** for woven fabric, where repetition is legitimate. Both fast methods do produce herringbone far more regular than real tweed, so check it against the reference once rendered.
- The pixel-space NCC variant is inconsistent (illumination- and scale-dependent): dropped.

**Anchored finish (m0496), discarded:** T6S2d with student steps that keep the guided teacher's noise estimate (correction-training's base_anchor idea).
- 16 passes; repetition excess .021 vs .022; OCR .96/.91; visually identical to T6S2d.
- With only 2 student steps from σ .463 there is little left for anchoring to constrain.

**Report card** (`tq/reportcard.py`) smoke-tested on held-out seed 101: quality repetition excess .016 vs T6S2d .004; text bag 1.0 vs .99; sampling 78 s vs 106 s on q8.

### Phase 1a (22:45 UTC): precision confound cleared

fast and quality on the bf16 official weights vs the q8 tier (768², seed 42; bread, portrait, beard, moss wall):
- same compositions (PSNR 22–38 dB);
- **identical stamped morphology**: the same lichen rosettes, crumb lobes and flour stars in both;
- repetition dup_frac within ±.01, e.g. bread fast .111 / .100, quality .086 / .088; moss fast .027 / .027, quality .020 / .018.

The stamping is not a quantization artifact, so the q8-based M5/M6 turbo conclusions stand. The 13% adapter-output sensitivity changes details, not the morphology class. The pilot will still be checked on both tiers.

### Phase 2b (22:55 UTC): the latent mechanism test passes; the hybrid's real residual is missing fine-detail energy

**Matched-state test** (`runs/latent_hf/`; 768², seed 42; bread, beard, moss wall, sign): along each trajectory, the same state is scored with the student (adapter 0.7) and the guided teacher.

High-frequency latent energy of the clean estimate, student / teacher:

| trajectory | at σ .968–.791 | at σ .671 | at σ .463 |
|---|---|---|---|
| fast | 1.5–3.4× | 1.3–1.4× | 1.1–1.2× |
| quality | 2–5× | 1.1–1.5× | 1.1–1.2× |
| T6S2 (teacher-driven states, where the student would act) | 1.5–3.2× | 1.3–1.9× | 1.1–1.4× |

- **The prediction holds:** at high noise the student's estimate carries several times the detail energy of the teacher's posterior mean at the same state. It invents committed detail where the teacher is still undecided, and the excess decays toward σ .463.
- **This is consistent with** stamping arising early and with the handoff point. It doesn't by itself prove the *arrangement* is the cause; the repetition metric and the crops cover that.

**Final-latent HF fraction vs the 50-step reference** (same seed): new and decision-relevant.

| prompt | ref | fast | quality | T6S2d | teacher-only 2M |
|---|---|---|---|---|---|
| bread s7 | .579 | .580 | .573 | **.538** | |
| forest s7 | .519 | .515 | .527 | **.475** | |
| portrait s7 | .491 | .464 | .462 | .457 | |
| sign s42 | .609 | .604 | .593 | .569 | .672 |
| leather s42 | .469 | .481 | .453 | **.423** | .537 |
| knit s42 | .424 | .420 | .394 | .392 | .508 |
| freckles s42 | .403 | .398 | .387 | .367 | .468 |

**Reading:**
- fast and quality spend about the reference's *amount* of fine detail, but on stamped arrangements.
- T6S2d has natural arrangement but **~8–10% less fine-detail energy than the original**. That is the quantitative face of the 1024² "softness/haze".
- Teacher-only 2M overshoots (+10–15%, the gritty look).

The residual is therefore the user's third failure class, lost fine detail, not stamping. The likely cause: the coarse 6-eval teacher states at σ .463 are blurrier than the true 50-step states. The pilot regresses to exactly those true states. Final-latent HF delta vs reference is added to the report card.

### Pilot P2 progress (23:03 UTC)

- **Teacher targets:** 253 samples (50-step guided teacher to σ .463 at 768²), ~20 s/sample on the A100.
- **Validation baselines** (32 held-out samples; relative error of the state at σ .463 against the 50-step teacher's):
  - SFT without CFG, 6 Euler steps (the untrained student): 0.453;
  - **guided Euler, 6 steps, 12 forwards: 0.395.** The coarse teacher alone is 40% off the true trajectory at the handoff point, which fits the hybrid's detail deficit.
- **Success bar for the student** (6 CFG-free forwards): below 0.395.

**Pilot validation so far** (rel err of the student's 6-step state at σ .463 vs the 50-step teacher; lower is better):

| point | rel err |
|---|---|
| it 0 (untrained SFT, no CFG) | 0.453 |
| it 500 | 0.455 |
| it 1000 | 0.427 |
| **bar: guided Euler, 12 forwards** | **0.395** |

- After own-rollout training starts (it > 500), later-step losses jump (0.35–1.7). The rollout drifts from the teacher, and the corrective targets are large.
- Improving, but slowly, on 221 training samples. The 60-min cap ends training around it 2700.

### Phase 1b (23:35 UTC): teacher-lever triage at 768² (seed 42), T6S2d variants

| variant | repetition excess | detail Δ vs ref | EN / ZH OCR | visual |
|---|---|---|---|---|
| T6S2d (skip 22, CFG 3) | .022 | −.037 | .96 / .96 | baseline |
| teacher without skip | .016 | −.034 | .88 / 1.0 | sign correct; leather and portrait ≈ baseline; marginal |
| teacher CFG 2 | .009 | −.026 | .88 / .96 | **sign garbled** ("GOLDENS KRUI BAKERY"); smoother leather |
| teacher CFG 4 | pending | | | |

- Lower CFG trades text correctness for slightly less idealization: it fails rule (a).
- No skip is a wash at this screening level.
- Neither lever addresses the detail deficit (−.03); the pilot targets that directly.
- it 1500: 0.455; it 2000: **0.399**. Noisy, but now at the guided-Euler bar (0.395) with half the forwards (6 CFG-free vs 12). Training stops at the 60-min cap around it 2300.

### Phase 4a (00:20 UTC): first pilot results (768² dev set, seed 42)

**Pilot** = trained prefix LoRA, 6 CFG-free steps, then the lodestones 2-step finish. 8 passes, 34.5 s at 768², the same cost as old fast.

| run | passes | repetition excess | detail Δ vs ref | EN / ZH OCR |
|---|---|---|---|---|
| quality | 10 | .043 | −.022 | .94 / .96 |
| T6S2d | 14 | .022 | −.037 | .96 / .96 |
| **pilot** | **8** | **.020** | **−.017** | .96 / .83 |

**Visual:**
- natural bread (no lobes, no flour stars);
- the portrait's composition is closest to the reference's (blue jacket, framing): it follows the true 50-step trajectory;
- **but images look washed and low-contrast:** the sign scene is dark and flat, the portrait faded.

**New, systematic: every fast method is darker and flatter than the original** (9 dev prompts, seed 42, image statistics):

| run | mean luma | luma std (contrast) | saturation |
|---|---|---|---|
| ref (50 steps) | .388 | .213 | .141 |
| quality | .320 | .200 | .142 |
| T6S2d | .321 | .175 | .132 |
| pilot | .343 | .166 | .117 |

The "moodier lighting" I credited to quality last night is a deviation from the original: it's darker.
- **Open question:** is the darkening and flattening introduced by the 2-step lodestones finish, or by the prefix?
- **Isolation test queued** (m0498bb): the exact 50-step teacher to σ .463 plus the lodestones finish, vs the reference.

### Phase 4b (04:50 UTC): E at 1024², seed 101; the finish is exonerated; nothing passes the rule yet

(The GPU queue sat idle 04:10–04:50; I should have queued follow-ups earlier.)

**Finish isolation** (exact 50-step teacher to σ .463, then the lodestones 2-step finish; 768², seed 42; luma mean/std vs the reference):
- sign .199/.161 vs .211/.160; leather .196/.170 vs .202/.169; knit .430/.198 vs .436/.195; bread .322/.237 vs .332/.235.
- The finish preserves tone and contrast. The darkening and flattening of the fast methods comes from their *prefix states*.
- **"Exact 50-step state at .463 + 2-step finish" is the oracle** for this design (8 passes if the prefix were perfect).

**Report card, E, 1024², seed 101** (18 prompts):

| candidate | passes | sampling s | repetition excess | detail Δ | text |
|---|---|---|---|---|---|
| quality | 10 | 78 | .066 | −.025 | all four text prompts correct (manual) |
| pilot (prefix LoRA) | 8 | 65 | **.030** (lichen, bark, bread ≈ 0) | −.032 | **fails:** ZH note illegible, sign garbled; trained only at 768² on 221 samples |
| T6S2d | 14 | 106 | .035 | −.055 | **fails:** ZH note two words garbled (manual), "Cod Brew" |
| akhaliq 4-step v2 | **4** | **32** | .109 (worst) | −.024 | best: all correct |

- **Rule (a)**, no text regression vs quality: pilot and T6S2d fail on this seed.
- **Caveat:** the oak and clouds prompts are in the repetition set without real-photo calibration; they dominate the means (all methods .08–.31).

**Next (queued):** teacher-then-pilot mixes on E, 10 passes, the same cost as quality.
- Guided teacher for steps 1–2 (glyph identity, as in quality), the pilot for steps 3–6 (unstamped morphology), finish 2: **T2S4F2**.
- **T3S3F2:** the same with 3 teacher steps.

Then the seed-202 runs for quality and T6S2d (noise floor).

### Phase 4c (05:15 UTC): teacher-then-pilot (T2S4F2) is worse than the pilot alone

E, 1024², seed 101, 10 passes, 78.7 s:
- repetition excess .035, a little worse than the pilot's .030 (lichen, bark, bread ≈ 0);
- **detail Δ −.064** (worst so far);
- **text bag .65**, worse than the pilot's .70.

Manual check (sheet `scratchpad/text_T2S4F2.jpg`):
- **sign:** the headline is ghosted and illegible, roughly "li GOLDEN … COMPANY";
- **zh note:** faint and garbled;
- the frame is visibly darker than quality.

**Interpretation:**
- The pilot was trained on exact 50-step teacher states. A coarse 2-step Euler teacher prefix hands it states off its training distribution, which makes it worse, not better.
- Glyphs are refined in the mid-σ steps (.87–.67). There the turbo adapter is accurate and the undertrained pilot is not.
- So text needs an accurate mid-σ actor, and repetition grows with turbo-adapter applications there. **This is a real conflict, not an ordering problem.**

**Next:** blend both adapters in one pass (prefix at 1.0 plus the turbo adapter at 0.2, 0.35 or 0.5; 6 steps, then the 2-step finish; 8 passes). Triage set: 4 text prompts plus lichen, bark and bread (m715a–c).
- If no blend restores text without reviving the stamping, the bottleneck is the trained prefix itself: a training-proposal decision.

### Phase 4d (06:10 UTC): both adapters in the same pass ("blend") is the first real lead

**T3S3F2** (3 teacher steps, then the pilot): text bag .75, detail Δ −.072. As bad as T2S4F2; the teacher-prefix mixes are closed.

**Blend M6F2:** prefix LoRA at 1.0 plus the turbo adapter at λ in the same forward, for 6 steps, then the 2-step turbo finish. 8 passes, 69 s at 1024², seed 101, triage prompts:

| run | sign | zh | menu | infographic | rep excess bread / lichen / bark | detail Δ (triage) | mean luma |
|---|---|---|---|---|---|---|---|
| reference | | | | | | | .458 |
| quality | .92 | 1.0 | 1.0 | 1.0 | .037 / .017 / .015 | −.027 | **.367** |
| pilot | .76 | .22 | 1.0 | .83 | .001 / .000 / .000 | −.042 | |
| λ .20 | .96 | 1.0 | 1.0 | 1.0 | .011 / .005 / .001 | −.025 | **.457** |
| λ .35 | .88 | 1.0 | .98 | 1.0 | .018 / .006 / .005 | −.008 | .459 |
| λ .50 | 1.0 | 1.0 | .98 | 1.0 | .028 / .005 / .004 | +.007 | .453 |

**Manual check** (scratchpad blend_a, blend_b and blend_sign sheets):
- **The good:**
  - the blends match the reference's brightness (quality is ~20% darker);
  - the zh note and menu are correct at every λ;
  - at λ .50 the sign headline is perfect.
- **New defect class: stray or doubled text.**
  - λ .35: the sign headline is ghosted and doubled, illegible.
  - λ .35 and .50: a second chalkboard with garbled words.
  - Infographic: a faint garbled subtitle at all λ; at λ .50 an extra "Solar 58%" contradicts its own 62% label. Quality's infographic also mislabels ("Solar Wind 62%" in the centre).
  - λ .20: the sign is correct, with a ghosted roofline above it.
- **Materials:**
  - the lichen is still dotted with small round rosettes (visually only modestly better than quality, despite dup_frac .005 vs .017);
  - bark plates are blockier and heavier-outlined than the reference's;
  - the bread is brighter, with white flour sparkles.
- The menu lettering is cleaner and more graphic than the reference's hand-lettering (mild idealization).

**Takeaway:**
- The two adapters complement each other: the pilot supplies the original's tone and composition, and the turbo adapter supplies glyph competence.
- The remaining defects (ghosting, stray text) are the pilot's signature: it's undertrained.

**Next:**
- fill all of E at seed 101 for λ .20 and λ .50 (m717a/b);
- seed 202 on the 6-prompt subset for both (m74a/b), plus the incumbents m72 and m73;
- the oracle (m716) is running.

### Phase 4e (07:20 UTC): the oracle passes; the finish is innocent, the prefix is everything

`E_oracle463`: the exact 50-step guided state at σ .463, then the 2-step lodestones finish. 1024², seed 101, triage prompts; 86 passes, 637 s per image.

**It is visually nearly indistinguishable from the 50-step reference** (scratchpad oracle.jpg, oracle_crops.jpg):
- the same composition, lichen patches, bread crumb, and bark plates;
- the sign headline is correct (the reference reads "THE GOLDEN CRUST BAKERY", and so does the oracle);
- the infographic has the reference's own layout, with correct labels.

**Metrics:**
- repetition excess bread / lichen / bark .011 / .001 / .003 (quality .037 / .017 / .015);
- detail Δ −.027 (≈ quality);
- text bag .99;
- mean luma .452 (reference .458, quality .367, blend λ .20 .457).

**Conclusion:**
- The 2-step turbo finish introduces essentially no stamping, darkening, or text errors at 1024².
- **Every remaining difference between the fast profiles and the original comes from how accurately the prefix reaches the σ .463 state.**
- The training route (prefix regression to the 50-step teacher) has a near-reference ceiling. The blend lead (4d) is an interim measure, constrained by the undertrained pilot.

**TRAINING_PROPOSAL_v2.md** is updated (gate passed; 8 student steps; ~$30, cap $45). It now waits for the user's go-ahead.

### Phase 5a (08:20–08:45 UTC): blends on all of E (seed 101) and the seed-202 subset; they fail rules (a) and (c)

**Report card** (repetition excess on admitted prompts; detail Δ; text bag; sampling s):

| run | s101 rep | s101 detail | s101 text | s202 rep (bread, lichen, beard) | s202 text | s | luma E s101 / S6 s202 |
|---|---|---|---|---|---|---|---|
| reference | | | | | | | .475 / .259 |
| quality | .066 | −.025 | .98 | .042 / .044 / −.002 | .98 | 78 | .396 / .173 |
| T6S2d | .035 | −.055 | .92 | −.010 / .009 / .006 | .90 | 106 | .410 / .169 |
| blend λ .20 | .029 | −.016 | .99 | .010 / .008 / .023 | .94 | 69 | **.472 / .214** |
| blend λ .50 | .030 | +.008 | .995 | .036 / .010 / .043 | .98 | 69 | **.470 / .247** |

**Manual check** (scratchpad full_a, full_b, s202 sheets).

**Better than quality:**
- brightness and tonality, close to the reference on every prompt;
- oak: λ .20 has the reference's sparse irregular branching; quality's is fractal-regular;
- clouds: two broad masses like the reference, with less cauliflower repetition;
- crowd: sharp pedestrians;
- leather: rich creases.

**Worse, or a new defect:**
- **ghosted or doubled lettering:** λ .20 s202 sign ("CRIIST", visibly double-stroked); λ .35 s101 sign;
- **stray garbled text:** λ .50 s101 extra chalkboard and the infographic's "EROM" and "Solar 58%"; λ .20 s101 roofline "CMIEES"; λ .50 s202 small garbled words;
- **lichen s202:** flower-like rosettes in every fast method, blends included (dup_frac misses them on this seed);
- **identity drift on tex_portrait:** the blends give a different man from the reference's; quality keeps the reference's identity.

**Decision rule:**
- (a) fails: ghosted sign on λ .20 s202; a new wrong number on λ .50 s101.
- (c) fails: stray/ghost text in more than 1 of 36 images.
- (d) passes: 69 s.
- (b) not run, since (a) and (c) already fail.

**Verdict:** no blend replaces quality. The new defects are the undertrained pilot's signature, consistent with the oracle result (4e). This is the evidence behind the v2 training request.

**Queued:** a ramp blend, with the turbo adapter λ rising 0.1 → 0.7 over the 6 prefix steps. Pilot-dominant at high σ, where stamping is committed; turbo-heavier at mid σ, where glyphs form (m75a/b).

### v2 training prep (08:25–08:35 UTC, CPU only, no spend)

- **`tq/pod_v2/`** (copied from the pilot bundle, so the pilot's code is untouched):
  - student grid set by `GRID_IDX` (default for v2: 50-grid indices 0, 5, 11, 16, 21, 26, 32, 37, 42 = σ 1 → .463 in 8 steps);
  - teacher stores fp16 states at the union of the release-grid and student-grid points;
  - training adds cosine LR decay after it 1000, keeps the best checkpoint only (`prefix_lora_best.safetensors`, uploaded in the background after each validation), and early-stops at it 3000 if the best error is above 0.9× the guided-Euler error on the same grid;
  - `build_v2` prompts: 1,100, 44.5% text (EN signs, labels, posters, menus with prices, charts with numbers, handwritten notes, CN); no quoted eval string appears.
- **Results prefix:** `results_v2/`, so the pilot's old "PILOT COMPLETE" can't trigger the new guard.
- **`tq/pod_guard_v2.sh`:** deadline computed from the pod's live costPerHr and COST_CAP (default $45), minus 10 min.
- **`pod_launch.py`:** `BUNDLE`/`NAME` env.
- **Smoke test** (TINY=1, CPU, 128², 12 samples, 60 its): all stages pass (state keys, val, best checkpoint, early-stop path).
- **`run.py actors`:** optional `grid50_idx` for evaluating an n-step student on 50-grid points, then the release finish. Checked: g50[42] == release .463.
- **Bug fixed:** the per-step `mix_steps` indexing crashed on the finish steps (m75a/b failed at 08:24, requeued 08:25).

### Phase 5b (08:45 UTC): ramp blend, the best no-training variant so far

**Setup:** turbo-adapter strength rising over the 6 prefix steps (0.1, 0.1, 0.2, 0.4, 0.6, 0.7) with the pilot prefix at 1.0, then the 2-step finish. 8 passes, 69 s.

**Results** (triage seed 101 + S6 seed 202):

| | text bag s101 / s202 | rep excess bread / lichen / bark-or-beard s101 | s202 | detail Δ s101 / s202 |
|---|---|---|---|---|
| ramp | .985 / .98 | .013 / .006 / .000 | .015 / .009 / .013 | +.007 / +.010 |
| quality | .98 / .98 | .037 / .017 / .015 | .042 / .044 / −.002 | −.025 / −.014 |

**Manual check** (scratchpad ramp.jpg):
- **sign:** headline correct on both seeds, with no ghosting (λ .20 ghosted on s202); the chalkboards are correct;
- **s202:** extra small "GOLDEN CRUST" decals in the windows (plausible, one slightly malformed);
- **infographic s101:** a **stray grey label "Nowysor"** and a duplicate "Solar" label (stray-text defect, 1 image so far);
- **lichen s202:** rosettes, as in quality and every other fast method; not a regression against the incumbent.

**Queued:** all of E for the ramp at seeds 101 and 202, and quality at 202 (m76a–c, ~45 min). Then the rule can be applied on the full 36 images.

### Phase 5c (09:30–09:45 UTC): the ramp on all of E at both seeds fails rule (c); no blend replaces quality

**Full E, seeds 101 and 202** (36 images each):

| | text bag s101 / s202 | rep excess s101 / s202 | detail Δ s101 / s202 | s |
|---|---|---|---|---|
| quality | .98 / .995 | .066 / .028 | −.025 / −.014 | 78 |
| ramp | .985 / .995 | .018 / .013 | +.013 / +.010 | 69 |

**Manual check, every text image** (scratchpad text_101, text_202): **rule (a) passes.**
- Every line correct in quality is correct in the ramp. OCR's "Cola Brew" is a misread of "Cold Brew".
- s101 infographic: the ramp labels Wind 38% / Solar 62% correctly (quality writes "Solar Wind 62%" in the centre), but adds a stray "Nowysor".
- Stray invented words also occur in quality (s202 infographic "Isneels"), so they are not a new class. The ramp has more of them (s202 "MADLICT"; a smudged chalkboard line).

**Defect scan, all non-text images** (s202_non_a/b, ramp101_faces): **rule (c) fails.**
- **s202 skin_freckles:** the ramp's face is badly malformed (melted, distorted mouth and eye). A new structural defect.
- **s202 knit_hands:** the ramp's hands are muddy and ambiguous; quality's are clean.
- s101 faces, hands, crowd and compose are fine. tex_portrait again changes identity from the reference's.
- Lichen rosettes at s202 appear in both quality and the ramp.

**Verdict:**
- Every blend (λ .20, .35, .50, and the ramp) fails (a) or (c). The incumbent stays.
- The failures are the undertrained pilot's: occasional structural collapse and stray text.
- Nothing more is planned for blend tuning (diminishing returns).

**Queued** (free, design-relevant for v2): **how far along the trajectory must the state be exact?**
- Exact 50-step teacher to σ .926 (14 steps), then the turbo adapter for 6 steps (m77a).
- Exact 50-step teacher to σ .671 (35 steps), then the turbo adapter for 3 steps (m77b).
- **If .926 already gives oracle-like images:** the student only needs to reach .926, the teacher data costs a third, and inference is ~2+6 passes.
- **If only .463 works:** v2 as proposed.

### Phase 5d (10:00 UTC): handing off at σ .926 stamps; the stamping happens in the turbo adapter's mid-σ steps

`E_oracle926`: the exact 50-step guided state at σ .926 (14 steps), then the turbo adapter for 6 release-grid steps. 34 passes, 256 s; triage, seed 101.

| | rep excess bread / lichen / bark | text bag | detail Δ | luma |
|---|---|---|---|---|
| reference | | | | .458 |
| oracle .463 (4e) | .011 / .001 / .003 | .99 | −.027 | .452 |
| **oracle .926** | **.055 / .010 / .024** | .99 | −.015 | .434 |
| quality (coarse T2 to .926) | .037 / .017 / .015 | .98 | −.027 | .367 |

**Manual check** (scratchpad oracle926.jpg, oracle926_crop.jpg):
- **Composition, sign and infographic** match the reference exactly. Even the correct "Wind 38% / Solar 62%" labels, which quality gets wrong.
- **But:**
  - the bread crust gets a regular star-shaped flour pattern;
  - the lichen gets flower rosettes on fish-scale stone, at least as stamped as quality.

**Conclusions:**
1. **Layout and glyph identity** are fixed by σ .926, and coarse T2 is what costs quality its infographic labels.
2. **Stamping** is produced by the turbo adapter's applications between .926 and .463, even from a perfect starting state. This is the dose-response mechanism, now isolated from prefix error.
3. **Darkening** has two parts: coarse guided teacher steps (quality .367 → oracle926 .434) and the turbo adapter's mid-σ steps (.434 → .452 when they are replaced by the exact teacher).
4. **For v2:**
   - the student must cover σ 1 → .463, with no turbo-adapter steps before .463;
   - a cheaper student ending at .926 would not fix stamping;
   - the proposal's endpoint is confirmed.

m77b (exact to .671, then 3 turbo steps) is running: does one turbo application above .463 already stamp?

### Phase 5e (11:05 UTC): the dose-response of turbo-adapter applications, measured from exact states

**`E_oracle671`:** the exact 50-step state at σ .671 (35 steps), then the turbo adapter for 3 steps (.671 → .463 → .024 → 0). 73 passes, 542 s.

| exact teacher to σ | turbo-adapter steps | rep excess bread / lichen / bark | luma | visual (scratchpad oracle671.jpg) |
|---|---|---|---|---|
| .926 | 6 | .055 / .010 / .024 | .434 | **stamped:** star flour, flower rosettes, fish-scale stone |
| .671 | 3 | .017 / .004 / .004 | .445 | ≈ reference; lichen slightly more speckled |
| .463 | 2 | .011 / .001 / .003 | .452 | ≈ reference |
| reference | 0 | | .458 | |

**Interpretation:**
- A clean dose-response. Stamping needs several turbo-adapter applications above σ ≈ .67; one application above .463 adds only a trace.
- Text is correct in all three (layout and glyphs are fixed by .926).

**v2:**
- keep the .463 endpoint (cleanest);
- the stored teacher states include index 35 (.671), so a 7-step student to .671 + 3 turbo steps could be trained from the same data if .463 proves too hard.

**Queued** (free; a no-training upgrade to the incumbent):
- quality's coarse 2-step guided Euler prefix to .926 costs it brightness (.367 vs .434) and the infographic labels;
- replace it with 3 or 4 DPM-Solver++(2M) teacher evals on 50-grid points to .926, then the same 6 turbo steps;
- 12 / 14 passes, ~94 / ~109 s (m78a/b, triage seed 101).
- Expectation: layout, labels and tone improve; stamping stays at quality's level.

### Phase 5f (11:30 UTC): a more accurate teacher prefix to .926 does not upgrade quality; no-training options exhausted

**Setup:** 3 or 4 DPM-Solver++(2M) guided evals on 50-grid points to .926, then the 6 turbo steps (triage, s101).

| | passes / s | rep excess bread / lichen / bark | text bag | luma |
|---|---|---|---|---|
| quality | 10 / 78 | .037 / .017 / .015 | .98 | .367 |
| T3dpm→.926 + 6 turbo | 12 / 94 | .067 / .014 / .022 | .99 | .396 |
| T4dpm→.926 + 6 turbo | 14 / 108 | .046 / .011 / .019 | .98 | .409 |
| exact→.926 + 6 turbo | 34 / 256 | .055 / .010 / .024 | .99 | .434 |

**Manual check** (scratchpad t34.jpg):
- The composition moves toward the reference's (the sign's "THE" is restored).
- The bread's stamped star pattern is *stronger* than quality's; lichen and bark are as stamped.
- **Text regressions:**
  - T4 infographic: "ENERGY CUS FROM / FROM" and "Solar 65%";
  - T3 infographic: duplicated "62%".

**Verdict:**
- Rule (a) fails (T4), and (b) is implausible (stamping is unchanged or worse).
- A better prefix *to .926* cannot fix stamping, consistent with 5d/5e.
- The lane is closed. **All no-training avenues tested tonight fail the rule; the incumbent stands.** The remaining lever is the trained prefix to .463 (v2).

### Phase 5g (12:35 UTC): the oracle holds on seed 202

`E_oracle463` at seed 202 (S6: lichen, bread, sign, beard, crowd, leather): the exact 50-step state at σ .463 plus the 2-step turbo finish.

**Visually nearly indistinguishable from the reference on all six** (scratchpad oracle_s202.jpg, oracle_s202_crop.jpg):
- **lichen:** the reference's irregular foliose patches, **no rosettes**. Quality, T6S2d, every blend and the ramp all show flower rosettes on this seed;
- **bread, sign, beard, crowd and leather:** same composition and detail as the reference.

**Metrics:**
- rep excess bread / lichen / beard .016 / .001 / .011 (quality .042 / .044 / −.002);
- luma .251 (reference .259, quality .173).

**The near-reference ceiling of the v2 design is confirmed on a second seed.**

### Phase 6a (2026-10-20 22:30 UTC): user critique. The oracle (B) is better than quality but not as good as the reference; slight regularity creeps in

**User:** "the target of the training run [is an] improvement but not as good as A (I can see the regular pattern issue creeping in just a bit even though the images look similar to A overall)". Table: `tq/compare/texture_table.jpg` (A reference, B oracle .463, C quality, D ramp).

**Implication:**
- B involves no trained component. Its residual regularity comes from the **2-step lodestones finish** (σ .463 → .024 → 0), the last two turbo-adapter applications.
- A better prefix (v2 as proposed) is bounded by B. **To approach A, the finish must change too.**

**Finish bench:**
- `run.py` gains `segments` "save" (stores x at a segment end), explicit "sigmas", a "solver": "dpm" segment, and kind `from_state`. Smoke-tested at 256².
- **Cache:** the exact 50-step state at σ .463 for lichen, bread, sign and ZH note (s101) and lichen, bread and beard (s202) (m80, ~80 min). The run's images are B again, which also checks determinism.
- **Finishes from the identical state** (m81a–f):

| variant | finish | passes |
|---|---|---|
| Flode2 | lode 2 steps, λ .7 | 2 (= B, control) |
| Flode2s5 | lode 2 steps, λ .5 | 2 |
| Flode3 | lode 3 steps | 3 |
| FT1L1 | original 1 step to .193, then lode last step | 3 |
| FT2dpm | original DPM++(2M), 2 evals | 4 |
| FT3dpm | original DPM++(2M), 3 evals | 6 |

**Budget check:** an 8-pass student prefix + a 4–6-pass finish = 12–14 passes ≈ 94–109 s at 1024². That is within 115 s.

**User question on resolution:** the MOSS user guide recommends 1, 1.5 or 2 MP (extrapolates 0.25–4 MP). Measured cost scales ~tokens^1.2 (768² 3.9 s/pass, 1024² 7.8 s/pass).
- 1.5 MP ≈ 11.8 s/pass; 2 MP ≈ 16.8 s/pass.
- A 10-pass turbo would be ~2 min at 1.5 MP and ~3 min at 2 MP.
- Training at 2 MP ≈ 2.15× compute (~$65).
- The student must be trained at the resolution it serves (the pilot's 768²→1024² mismatch broke text).

### Phase 6b (2026-10-21 00:00 UTC): the finish bench. An original-model finish (3 DPM++ evals) reproduces the reference

**Setup:** finishes from the identical exact 50-step state at σ .463 (7 images: lichen and bread s101/s202, sign and ZH s101, beard s202). Table: `tq/compare/finish_table.jpg`.

| finish | passes | rep excess (lichen101, bread101, lichen202, bread202, beard202) | detail Δ | mean \|img − ref\| | luma Δ |
|---|---|---|---|---|---|
| lode 2 (= B) | 2 | .001 .011 .001 .016 .011 | −.029 | .0218 | −.008 |
| lode 2, λ .5 | 2 | .002 .012 .001 .013 .012 | −.041 | .0180 | −.006 |
| lode 3 | 3 | .000 .006 .000 .010 .005 | −.006 | .0187 | −.006 |
| original 1 step + lode last | 3 | .001 .008 .000 .005 .009 | −.028 | .0101 | −.004 |
| original DPM++(2M), 2 evals | 4 | .000 .008 .001 .004 .009 | −.033 | .0087 | −.002 |
| **original DPM++(2M), 3 evals** | **6** | **−.001 −.000 −.000 −.001 −.003** | **+.006** | **.0047** | **+.001** |

**Results:**
- **The 3-eval original finish (σ .463 → .321 → .115 → 0) is ~4.6× closer to the reference than B**, with zero excess repetition and matching detail and tone.
- **Text** (manual): the sign and the ZH note are correct in every variant. OCR failed to run in this harness (bag 0 for every variant, including B), so the check was visual.
- **Design consequence:** student prefix (8 passes) + original 3-eval finish (6) = 14 passes ≈ 109 s at 1024², within the 115 s budget. The finish needs no training.

**Open question (queued, m82a–c):** is the original finish robust to an *imperfect* prefix? Three prefixes, each followed by the 3-eval original finish, on 9 images (incl. s202 skin_freckles and knit_hands, where the ramp failed):
- quality's prefix (T2 + 4 turbo steps; 8 + 6 = 14 passes);
- the pilot (6 + 6);
- the ramp blend (6 + 6).

**Bug noted:** the first m82a run crashed because `set_prefix` was called without a prefix LoRA. `set_prefix` now tolerates that; the run was requeued.

### Phase 6c (2026-10-21 00:20–01:00 UTC): raw-evidence review for v2 (user request: minimally lossy distillation)

**Official code** (SFT snapshot `pipeline_moss_image.py`, scheduler config):
- **Single-stream DiT:** image and text tokens concatenated, 2-D RoPE, 42 blocks × (q, k, v, o, ffn_up_gate, ffn_down) = the 252 projections our LoRA covers.
- **Timestep conditioning is a single `shared_block_modulation`** (one Linear: time → 6 × 3072, shared by all blocks) plus `video_head_modulation`.
  - Our LoRA is σ-agnostic; σ-specific behaviour can only come through that one shared modulation path.
  - Candidate change: LoRA on `time_embedding`, `shared_block_modulation` and `video_head_modulation`, or per-step LoRA banks.
- **Scheduler:** `use_dynamic_shifting: false`, shift 5. The σ grid is identical at every resolution.

**Pilot training history** (`train_hist.json`): velocity MSE by student step.
- **Teacher-forced phase (it ≤ 500):** decreasing (step 0 .165 → .12; steps 3–5 ≈ .02). **Step 0 (σ 1 → .968) is 5–8× harder than the rest.**
- **After own-rollout started (it 500, p = .5, "teleport" targets `(x*_{k+1} − x̂_k)/Δσ`):** losses jumped to .3–1.3 and stayed noisy; validation rollout error never left .40–.46 (guided-Euler bar .395).
- **Hypothesis:** teleport targets from drifted states are not valid flow velocities. The MSE-optimal response to them averages trajectories, a plausible source of the pilot's ghosted and double-stroked letters.
- **Fix candidate:** relabel own-rollout states with the guided teacher's continuation over the same interval, plus a curriculum (ramp).

**quality's prefix + original 3-eval finish** (`fin_qpre_FT3`, 14 passes):
- rep excess (lichen101, bread101, lichen202, bread202, beard202) .008 / .019 / .017 / .013 / −.002 vs quality .017 / .037 / .044 / .042 / −.002;
- |img − ref| .135 (quality .142); luma −.075 (quality −.082).
- **Visual: quality's stamped lichen daisies and bread lattice remain.**
- So stamping is committed above σ .463. A finish cannot undo it; the prefix must be stamp-free.

**Code (pod_v2, smoke-tested on TINY, CPU):**
- `TARGET_MODE` = teleport | relabel | forced, with a `RAMP` curriculum;
- `DENSE=1` stores every 50-grid state to the handoff (grid ablations from one teacher pass);
- `ASPECTS` (HxW list at ~equal pixels) for multi-aspect data.

**Free runs** (scored next):
- pilot and ramp prefixes + original finish (m82b/c);
- error budget: exact state mixed with the pilot's at α .25 / .5 / 1, then the original finish (m83c);
- 4-eval finish (m83b);
- handoff at .671 with 4- and 5-eval original finishes (m84).

**m82 results** (original 3-eval finish after imperfect prefixes; sheet scratchpad/pilotFT3.jpg):

| prefix + original finish | passes | rep excess (lichen101, bread101, lichen202, bread202, beard202) | detail Δ | \|img − ref\| | luma Δ |
|---|---|---|---|---|---|
| exact (F) | 6 + 42 teacher | ≈ 0 | +.006 | .0047 | +.001 |
| pilot | 12 | ≤ .002 everywhere | +.018 | .106 | +.004 |
| ramp | 12 | ≈ 0 except bread202 −.008 | +.059 | .116 | +.008 |
| quality | 14 | .008–.019 | +.023 | .135 | −.075 |

**Visual:**
- **pilot + finish:** natural, unstamped lichen and bread at the reference's tone.
- **But every structural failure of the pilot prefix survives:**
  - ghosted sign letters;
  - an illegible ZH note;
  - the s202 freckles portrait is a *double exposure* (two overlaid faces), which fits the trajectory-averaging hypothesis for the teleport targets.
- **ramp + finish:** correct text, but stamped rosettes (committed by the turbo adapter in the prefix) and the same malformed face.

**Conclusion:** the original finish is faithful to whatever the prefix commits, texture *and* structure. The quality of the trained prefix is the whole problem.

### Phase 6d (01:20 UTC): error budget and finish saturation

**Pilot prefix error at σ .463 (1024², rel to the exact state):** beard .31, sign .55, bread .38 / .33, lichen .39 / .42, ZH .41 (mean ≈ .40, as at 768² validation).

**Error budget:** x = (1 − α)·exact + α·pilot at .463, then the original 3-eval finish (sheet `tq/compare/error_budget.jpg`):

| α (≈ rel err) | \|img − ref\| | rep excess | visual |
|---|---|---|---|
| 0 (exact) | .0047 | ≈ 0 | = reference |
| .25 (≈ .10) | .025 | ≤ 0 | indistinguishable from exact: sign, note, lichen, beard correct |
| .50 (≈ .20) | .054 | ≤ 0 | **ghosting appears:** doubled sign letters, faded/double ZH note, slightly doubled eyes |
| 1 (≈ .40, pilot) | .106 | ≤ 0 | ghosted sign, illegible note, different face |

**Caveat:** linear mixing of two states produces double exposures by construction. The budget is exact for *averaging-type* errors (the failure mode of teleport targets) and pessimistic for coherent drift (a different but self-consistent image).

**Finish saturation:** the 4-eval original finish (8 passes) gives |img − ref| .0044 vs .0047 for 3 evals (6 passes). **3 evals is saturated;** extra budget should go to the student.

### Phase 6e (02:15 UTC): handoff at .671 and trajectory curvature

**Handoff at σ .671** (exact 50-step state, 4 items):

| handoff | finish | finish passes | \|img − ref\| | rep excess | detail Δ | luma Δ |
|---|---|---|---|---|---|---|
| .463 | original 3-eval | 6 | .0053 (cache671 run) | ≈ 0 | +.007 | +.001 |
| **.671** | original 4-eval (.671 → .536 → .373 → .193 → 0) | 8 | **.0051** | ≈ 0 | −.004 | +.000 |
| .671 | original 5-eval | 10 | .0086 | ≈ 0 | +.022 | +.003 |

- **An earlier handoff is equally faithful at +2 finish passes.** The student would cover only σ 1 → .671 (the composition steps), and the teacher data costs ~17% less (35 vs 42 steps).
- Open question: is the error budget at .671 more forgiving? Queued as m85: pilot state at .671 mixed with the exact state, α .25 / .5 / 1, then the 4-eval finish.

**Curvature of the guided trajectory:**
- Trajectories were reconstructed from the trace (exact to 1e-4 against the saved states).
- Measure: one guided-Euler jump from the exact state at the interval start vs the exact end state (relative):

| grid | per-interval error, first → last |
|---|---|
| uniform 8 on the 50-grid (to .463) | .003–.010 for the first five; **.010–.016, then .028–.029 for the last** (.623 → .463) |
| release 6 | first interval .015–.018; **last (.671 → .463) .048–.050** |
| 6 steps to .671 | .004–.024; all small |

- **Curvature concentrates late, near .6 → .46, not at σ = 1.**
- The pilot's step-0 loss was the largest (5–8× the rest). That difficulty is CFG emulation at pure noise (the student is CFG-free), not trajectory curvature.
- So: a CFG-related capacity need at high σ, and a curvature need near .5.
- Ending the student at .671 avoids the most curved interval entirely.

**Discrepancy to resolve:** single-jump errors are small (≤ .05), yet the pod's chained guided-Euler-6 rollout error was .395 (768²). Either errors compound strongly along the rollout, or the baseline was measured differently.
- Queued m86: chained guided-Euler 6 (release grid) and 8 (uniform) rollouts at 1024², with the prefix saved, then the original finish. This measures the compounding and whether *coherent* (valid-flow) errors ghost.

### Phase 6f (02:55 UTC): coherent vs averaging error; the .671 budget; compounding

**Chained coarse guided-Euler rollouts at 1024²** (state rel err at .463 vs exact; then the original 3-eval finish):

| rollout | rel err at .463 | \|img − ref\| | luma Δ | detail Δ | visual (`compare/coherent_vs_averaged.jpg`) |
|---|---|---|---|---|---|
| 6 steps (release grid) | .36–.51 | .128 | −.078 | −.030 | |
| 8 steps (uniform 50-grid) | .32–.51 | .106 | −.051 | −.023 | **no ghosting:** legible sign (slightly different lettering), coherent lichen and bread; darker and softer |

- Single-jump errors are ≤ .05 (6e), so **errors compound strongly along the rollout**. Relative state error is dominated by coherent drift.
- The pod's .395 bar was real, not a bug.

**Error budget at .671** (pilot state at .671: rel .20–.29; 4-eval original finish):
- |img − ref| α .25 → .023, α .5 → .056, α 1 → .114, nearly the same as at .463 (.025 / .054 / .106).
- At α .5, the sign is ghosted at both handoffs.
- The earlier handoff is *not* more forgiving per unit of pilot error. It is equally faithful from exact states (6e).

**Conclusions:**
1. **Ghosting is a property of the error type, not its size.**
   - Coherent (valid-flow) error of rel ≈ .4 → a clean, plausible but darker image.
   - Averaging error of rel ≈ .2 → ghosting.
   - Consequences: relabelled targets over teleport targets; validation in image space (after the original finish) rather than state rel err alone; TDM-style distribution matching only as a fallback if capacity-limited regression still averages.
2. **The coarse original model's systematic bias is darkening and softening** (luma −.05 to −.08). A student trained on exact states should avoid it. A student that only follows the flow (relabel) is anchored by teacher-forced steps on exact states.
3. **Handoff .671 vs .463:** equally faithful, equally sensitive.
   - .671 avoids the most curved interval and costs ~17% less teacher time.
   - .463 gives the student 2 more steps at the same total (8 + 6 vs 6 + 8).
   - Dense storage to .463 keeps both open as a training-only ablation.

### Phase 7a (2026-10-21 03:00–03:30 UTC): Spectrum (user-supplied speedup sources)

**Sources:**
- The arXiv link given (2605.24278v2) is a different paper: "Fourier Feature Pyramids for Physics-Informed Neural Networks" (PINNs, not diffusion).
- The repo `Rkkss/ComfyUI-Spectrum-MOSS` (GPL-3.0, @037db25) cites **"Adaptive Spectral Feature Forecasting for Diffusion Sampling Acceleration", arXiv 2603.01623**, with official code `hanjq17/Spectrum` (@4e41f91). Read both.

**Method:** training-free.
- The last attention block's output is cached at real steps.
- Each channel is fitted as a Chebyshev polynomial of the timestep by ridge regression; skipped steps are forecast from the fit (cost ≈ the head only).
- An adaptive schedule spends more real steps early.

**Paper's FLUX.1 results (50-step reference):**
- 3.47× speedup: PSNR 24.3, LPIPS .217;
- 4.79×: PSNR 22.2;
- plain 25 steps (1.97×): PSNR 18.1.

**The MOSS ComfyUI node:**
- forecasts the final-block image hidden state per CFG branch, with separate histories when SLG patches differ;
- Chebyshev degree 4, ridge .1, 8-point history, blended 50/50 with linear extrapolation;
- 8 warmup real steps, 2 tail real steps, ≤ 2 consecutive forecasts (manual), or an adaptive guard;
- "~2–2.5×" claimed.

**Applicability to us:**
- **Few-step turbo** (6–8 student steps + 3–4 finish evals): not applicable. There are too few, too large steps to fit or forecast; every eval is a big jump.
- **Full 50-step model:** directly applicable, potentially ~2× (12½ → ~6 min).
- **Teacher data:** could cut the dominant training cost ~2×, but it perturbs the trajectories the student is trained on. Only acceptable if the measured state/image shift is tiny.

**Implementation:**
- Our own, written from the paper rather than copied from the GPL code: `run.py` kind `spectrum`, `SpectralForecaster`, with a `_hidden`/`_head` split of the MLX forward.
- Sanity check at 256²: max_consec 0 reproduces the plain 50-step sampler exactly (rel 0.0); max_consec 2 uses 46/100 passes (2.2×) with final-latent rel diff .31.
- **Running (m88):** 1024², 5 items with references and cached exact states; max_consec 2 and 1; states saved at .671 and .463.

**Questions page for the user:** `release/x/site/v2/questions.html` at (private page)
- 11 questions with radio/checkbox answers, optional notes, "fill my defaults", localStorage, and Copy as JSON.
- Checked: axe 0 violations; Playwright flow test passes (defaults → change → note → copy JSON → reload persistence); phone-width screenshot OK.

### Phase 8a (2026-10-21 04:00–04:40 UTC): user decisions; prompts v3; stage 1–2 launched

**User answers** (questions page JSON):
- ~1 MP multi-aspect; ≤ 120 s on the q8 tier;
- prompts: short / long / JSON layout / EN text / ZH text / people / products / landscapes. "Be very thoughtful with the training prompts";
- official look (CFG 3 + skip 22); "same character" over seed parity (→ relabel targets);
- **destripe** in the teacher and finish, with a destriped 50-step baseline for fidelity (design call delegated);
- handoff default step 35 (.671);
- **spend: stages 1–2, ~$50 cap**;
- **Spectrum: don't use**;
- release 0.2.0: "let's discuss"; keep the 8791 site.

**Prompts v3** (`tq/prompts_v3/`):
- Spec `PROMPT_SPEC.md`, written to match MOSS's own caption schema (long / medium / short + official layouts) and the user's use cases.
- 8 domain writers × 250, plus a 160-prompt held-out set by a separate writer.
- `assemble.py`: compiles layouts with the official OpenMOSS `compile_prompt.py` (Apache-2.0, `tools/`); exact and near-duplicate removal; leakage check against tq/prompts.py and the held-out set (parsed layouts; shared visible strings ≥ 16 Latin or ≥ 5 CJK characters); stratified val split (first 16 = long + medium/layout per domain); shuffled.
- **Result:** 1,998 prompts (1,934 train / 64 val), 0 invalid, 2 dropped as leaks.
  - formats: long 635, medium 485, short 465, layout 323, zh_prose 90;
  - aspects: 1:1 17.5%, portrait 42%, landscape 40%;
  - text 41.5% (EN 26%, ZH 13%, bilingual 3%); compositional 15.5%.
- Spot-read: natural, specific, exact strings in both scripts.

**Destriped references** (Mac, m90a/b) are running; the first two show destripe changes nothing but the stripes:
- |Δ| .009–.018;
- luma +.002;
- HF and dup unchanged;
- beard banding 9.0 → 7.0.

**pod_v2 (stage 1–2):**
- encode.py;
- teacher.py: sharded per GPU; destripe 1.5 merged into the weights; dense fp16 states to index 35; val prompts continued to σ 0 and decoded as references;
- compact.py: grid states for reuse, uploaded;
- train.py:
  - lazy loading;
  - relabel / teleport / forced targets;
  - ramp curriculum;
  - grid 0, 4, 9, 15, 21, 28, 35;
  - image validation: student rollout, then the original 4-eval DPM++(2M) finish, decode, |luma − destriped 50-step reference| on 16 val prompts every 500 its, with sheets uploaded;
  - best checkpoint chosen by the image metric;
- run.sh: two arms in parallel on 2 GPUs.
- **Checks:**
  - TINY end-to-end smoke test passes;
  - real-weight checks: destripe merge (25 modules); torch-vs-MLX VAE decode |luma Δ| 1–3e-5;
  - a decode no_grad bug was found and fixed.

**Launch:**
- First pod (2× A100-SXM4 SECURE, $3.58/h) was placed on a machine flagged by RunPod ("critical error on this machine…"), so I deleted it.
- **Mistake:** my relaunch loop mis-parsed pod ids (`tr -dc` kept the "id" key letters), so two more flagged pods started before I deleted them, ~1–2 min each. ≈ $0.3–0.5 wasted in total.
- Final pod `fuvpqi0hcs15nt` (no machine warning). Guard `pod_guard_v2.sh`, COST_CAP $49.5 → deadline 819 min.
- TEACHER_MINUTES 480; ITERS 2,000 per arm; TRAIN_MINUTES 230.
- Downloads done (26 GB) at 3 min.

### Phase 8b (05:07–05:25 UTC): the stage 1–2 pod was stopped; RunPod credits exhausted

**What happened:**
- Pod `fuvpqi0hcs15nt` ran for 30 min: encode done (1,999 prompts in 2.5 min); teacher at 38 s/sample per GPU on the first 20 val samples (val samples run 50 steps + decode).
- Then: `lastStatusChange: "Exited by Runpod"` at 05:06:36. The guard saw EXITED, fetched STATUS and run.log, and exited without deleting.

**Cause:** the account balance is −$0.24 (GraphQL `myself.clientBalance`; spendLimit 80). The remaining credit ran out mid-run and RunPod stopped the pod. My first reading ("flagged machine") was wrong.

**Lesson:** I launched without checking the balance.

**Actions:**
- deleted the stopped pod (no storage billing);
- removed its stale `results_v2/` status;
- guard now deletes on EXITED/TERMINATED;
- `pod_launch.py` gains `balance` and refuses `create` when `clientBalance < MIN_BALANCE` (tested: refuses at −0.24 < 48);
- run.sh TEACHER_MINUTES 440 (more margin under the cap).

**Today's spend:** the remaining balance (~$2) plus the brief duplicate pods. Billing for 2026-10-21 isn't posted yet.

**Needs the user:** add funds (≥ ~$55 recommended for stages 1–2 at ~$43 with margin). Relaunch command:
`MIN_BALANCE=48 BUNDLE=pod_v2 NAME=moss-prefix-v2 GPU_COUNT=2 CLOUD=SECURE CDISK=40 VOL=200 GPUS="NVIDIA A100-SXM4-80GB,NVIDIA A100 80GB PCIe" pod_launch.py create`, then `COST_CAP=49.5 pod_guard_v2.sh <id>`.

### Phase 8c (05:33 UTC): relaunched on the user's funded account

- The user topped up RunPod credits (the previous balance had run out).
- New API key installed locally (never printed); balance verified at $150.
- `launch_v2.py` (balance wait, rejects machines with a RunPod note, robust JSON parsing) launched pod `s62jw76x762tsz` (2× A100-SXM4, US-MD-1, no note).
- Guard: COST_CAP $47.5 (keeping the ~$2 spent on the exhausted account inside the ~$50 approval) → deadline 786 min.
- Downloads done at 3 min.

### Phase 8d (08:15 UTC): destripe 1.5 vs the original 50-step references (E, seed 101, all 18)

**Metrics:**
- **Stripe score** (bandviz, 48-cycle period): mean 13.9 → 5.2. Strongly striped images: oak 23.8 → 3.7, paper 29.1 → 4.9, tweed 29.1 → 1.5, lichen 25.0 → 3.6, knit 21.0 → 3.5, compose 16.9 → 4.1.
- **Side effects:**
  - mean |Δimg| .018;
  - luma +.001;
  - dup_frac on lichen/bark/bread unchanged (≤ +.0015);
  - final-latent HF −.0067 on average (≤ −.019 on tweed, menu, paper).

**Visual** (`compare/destripe_check.jpg`): tweed weave, paper creases, chalk lettering and oak bark are unchanged in detail. Only the faint vertical banding disappears. The HF reduction is the stripes themselves.

**Conclusion:** destripe meets the user's condition ("without introducing other issues") on E. Destriped 50-step images (`E_ref_d`, `H_ref_d`) are the fidelity target for v2.

**Pod status 08:07:** ~300/999 per shard; 27.7 s/sample cumulative (≈ 26 s steady). Stage 1 hits its 440-min cap at ~12:55 UTC with ~1,900 samples.

### Phase 8e (12:40 UTC): the guard failed on a local network outage; replaced (pod_guard_v3.py)

**What happened:**
- At 08:30 UTC (177 min) the Mac lost DNS (`nodename nor servname provided`). pod_guard_v2.sh read "status unavailable" as a lost heartbeat and tried to delete the pod. The delete also failed (same outage), and the guard exited.
- The pod ran unguarded 08:30–12:42 (~4.2 h). Spend stayed within the cap: ~$25.6 of $47.5 at 12:42.

**pod_guard_v3.py fixes:**
- spend from the pod's `lastStartedAt` × costPerHr, so restarts are safe;
- network/API errors are never acted on;
- stuck = status fetch *succeeds* but the content is unchanged for 45 min;
- EXITED → fetch and delete;
- deletes retried until verified gone.
- Restarted 12:42 with cap $47.5.

**Progress:** teacher shard 0 at 940/999 (26.5 s/sample) at 12:36; stage 1 ends ~13:00. Balance $124.19 at 12:41.

### Phase 8f (14:40 UTC): stage 2 mid-run; the relabel arm is healthy and teleport ghosts again

**Training curves** (pod STATUS):
- **relabel:** per-step losses .02–.14 throughout; own-rollout loss .06–.09 (stable).
- **teleport:** own-rollout loss .8–1.3 after the ramp; later-step losses .4–1.3, the pilot's explosion again.

**Image metric** (mean |luma − destriped 50-step ref|, 16 val prompts; lower is better):

| it | relabel | teleport |
|---|---|---|
| 0 | .118 | .118 |
| 500 | .094 | .104 |
| 1000 | .104 | .105 |
| 1500 | (pending) | .108 |

- Noise across arms at it 500 (identical teacher-forced training) suggests about ±.005–.01.
- The paired metric is dominated by coherent composition drift ("same character" allows this).

**Sheets** (left = student 6 steps + original 4-eval finish; right = reference; `scratchpad/valimg_progress.jpg`):
- **it 0 (untrained):** washed out, wrong compositions, garbled card text, blank parcel label.
- **it 1000** (relabel, by upload time): compositions, faces and text match the references (EN card, ZH parcel, xiangqi board); natural textures (leather, glaze, lupins); no ghosting seen.
- **it 1500** (teleport): mostly good, **but the train reader's face is a ghosted double exposure** and the parcel label is messier. This is the averaging failure predicted for teleport targets.

**Bug (cosmetic):** both arms upload `valimg_<tag>.jpg` to the same dataset path. Arm identity was inferred from upload times. The full `arm_*` folders, with their own sheets, upload at the end.

### Phase 8g (15:30 UTC): stage 2 complete; relabel wins; pod deleted; Mac evaluation queued

**Spend:** pod `s62jw76x762tsz` ~$35.5 (guard-estimated), plus ~$2 on the earlier attempt that ran out of credit. ≈ $37.5 of the ~$50 approval. Balance $114.26. Pod deleted (verified).

**Image metric** (16 val prompts, |luma − destriped 50-step ref|):

| arm | it 0 | 500 | 1000 | 1500 | 2000 |
|---|---|---|---|---|---|
| relabel | .118 | .094 | .104 | .096 | **.090** |
| teleport | .118 | .104 | .105 | .108 | .088 |

State rel err at 2000: relabel .184, teleport .179.

**Visual, it 2000** (`scratchpad/arms_it2000.jpg`):
- **relabel:** clean: train reader coherent; parcel "易碎品 小心轻放" correct; card, xiangqi and textures correct.
- **teleport:** the train reader is a smeared double exposure; the parcel's Chinese characters are garbled.
- **Winner: relabel.** The paired metric can't separate the arms; the visual defect check can.

The relabel curve is still improving, so a longer stage 3 (reusing the compact states uploaded to the dataset) would likely help.

**Queued:** Mac evaluation of `arm_relabel/prefix_lora_best.safetensors` (it 2000) via `queue_v2_eval.py`:
- E s101 (18) and S6 s202;
- 32 held-out prompts at their own sizes;
- quality on the same held-out prompts;
- student = 6 CFG-free steps on 50-grid indices 0, 4, 9, 15, 21, 28, 35, then the original 4-eval DPM++ finish, destripe 1.5 throughout (14 passes).

### Phase 8h (16:25 UTC): first Mac evaluation of the v2 student (relabel, it 2000), E seed 101

**Speed:** 14 passes, 109.4 s sampling per image on q8 (~113 s end to end), within the 120 s budget. Peak 10.9 GB.

**Report card vs the original references:**

| run | rep excess (all 7 admitted) | bread / lichen / bark | oak / clouds | detail Δ | text bag | \|img − destriped ref\| | luma vs destriped ref |
|---|---|---|---|---|---|---|---|
| destriped 50-step reference | .003 | | | −.007 | .99 | | |
| **v2 student** | **.012** | .003 / −.001 / .001 | .040 / .041 | **−.009** | .99 | **.098** | +.037 |
| quality | .066 | .037 / .017 / .015 | .174 / .151 | −.025 | .98 | .145 | −.080 |

**Visual** (`compare/v2_E_s101.jpg`):
- **Textures:** lichen, bread and oak are at reference character. No rosettes, lattice or fractal stamping, where quality shows all three.
- **People and scenes:** freckles portrait, knitting hands and crowd are natural and close to the reference.
- **Two text failures:**
  - **en_sign headline ghosted/garbled** ("THE GOLDEN" doubled; "BAKERY" split). quality has it correct.
  - **infographic:** "COMES FIS FROM" plus a stray garbled line "(FRTD)". The labels Wind 38% / Solar 62% are correct; quality mislabels them.
- These are averaging-type errors in glyph regions. The student at it 2000 is still undertrained on hard text, and the curve was still falling.

**Decision rule:** (a) fails on en_sign at this checkpoint. The repetition / idealization categories are clearly better. Pending: E s202, the 32 held-out prompts, and quality on the held-out prompts.

### Phase 8i (16:45 UTC): fusion tests (user idea: hybrid / alternating actors)

**T1S4** (one guided original Euler step 1 → .957, then the student's own intervals 9→15→21→28→35, then the original 4-eval finish; 14 passes, 107.8 s):
- **sign:** the headline is now mostly correct ("THE GOLDEN CRUST / BAKERY", minor glyph wobble) instead of ghosted;
- **infographic:** pale, washed-out colours, and the title garbled differently ("COM MBES FROM");
- **lichen:** natural but softer and lower in contrast;
- **oak:** composition changed to a bare winter tree;
- rep excess ≈ 0 (−.0007); **detail Δ −.037** (student-only −.009).
- **Read:** the one big guided jump from pure noise is coarse and softens/washes the state. It helps the large headline and hurts elsewhere. Not a clear win.

**Queued: an allocation map, measured rather than guessed.** One guided-teacher step (2 passes) replaces the student at interval k (k = 0..5 on grid 0, 4, 9, 15, 21, 28, 35); student elsewhere; same finish; 15 passes. Run on 7 prompts (4 text + lichen, bread, oak) at s101 (m94c0–c5).
- This is the marginal value of an extra pass at each phase. The greedy next round adds a second teacher step or trades the finish down to 3 evals.
- T2S3 (two guided steps up to .919, 15 passes) is running.

### Phase 8j (18:30 UTC): allocation map, one guided step per student interval (7-prompt panel, s101; sheet `compare/alloc_slots.jpg`)

| slot (interval) | text bag | rep (bread / lichen / oak) | detail | \|d\| | luma |
|---|---|---|---|---|---|
| student only | .990 | .003 / −.001 / .040 | −.002 | .111 | +.029 |
| T0 (1 → .983) | .957 | .007 / .001 / .031 | −.006 | .109 | +.009 |
| T1 (.983 → .957) | .949 | .002 / −.001 / .019 | −.011 | .100 | −.009 |
| **T2 (.957 → .919)** | .990 | −.002 / −.002 / .035 | −.003 | .107 | +.015 |
| T3 (.919 → .871) | .975 | .002 / −.001 / .040 | −.004 | .109 | +.028 |
| T4 (.871 → .791) | .973 | .001 / −.003 / .048 | −.006 | .110 | +.029 |
| T5 (.791 → .671) | .970 | .002 / −.002 / .049 | −.008 | .110 | +.029 |

**Visual** (the text bag can't see ghosting):
- **T2:** fixes the en_sign headline ("THE GOLDEN CRUST BAKERY") **and** the infographic title (a small stray yellow line remains).
- **T1:** fixes the infographic title (stray small text) but not the headline.
- **T0, T3, T4, T5:** headline ghosted, like the student alone.
- ZH note correct everywhere; oak and lichen natural in every variant.

**Interpretation:** glyph formation for large text is decided around σ .96 → .92. A single exact-ish guided step there repairs it without reintroducing stamping. Brightness moves toward the reference when the teacher acts early (T0/T1).

**Next:** T2 + 3-eval finish (13 passes); T1 + T2 + 3-eval finish (14 passes); then the winner on held-out text prompts and E s202.

### Phase 8k (19:05 UTC): allocation round 2; held-out validation queued

| schedule | passes | sampling s (q8) | text bag | rep (bread / lichen / oak) | detail | \|d\| | luma |
|---|---|---|---|---|---|---|---|
| student only | 14 | 109.6 | .990 | .003 / −.001 / .040 | −.002 | .111 | +.029 |
| **slot2 + F4** | 15 | 116.2 (~119 e2e) | .990 | −.002 / −.002 / .035 | −.003 | .107 | +.015 |
| slot2 + F3 | 13 | 101.2 | .980 | .003 / −.002 / .049 | −.017 | .107 | +.013 |
| slots1+2 + F3 | 14 | 108.1 | .985 | .003 / .001 / .025 | −.027 | .103 | −.018 |

**Visual** (`compare/alloc_round2.jpg`):
- all three slot-2 schedules render the headline "THE GOLDEN CRUST BAKERY" (a small flourish on the C) and the infographic title;
- menu correct; lichen, bread and oak natural;
- slots1+2 pulls the bread composition toward the reference but loses the most detail and gets darker;
- F3 loses some detail (−.017) vs F4 (−.003), which is hardly visible in crops.

**Lead: slot2 + F4** (15 passes, ~119 s end to end on q8, at the 120 s limit). Fallback: slot2 + F3 (13 passes, ~104 s).

**Queued** (the plain-student held-out jobs that had not yet run are parked in tq/hold):
- quality on held-out 32;
- slot2 + F4 on E s202 (S6) and held-out 32;
- slot2 + F3 on the 8 held-out text prompts.

### Phase 8l (20:35 UTC): slot2 + F4 on E seed 202 (S6)

| run | rep bread / lichen / beard | detail | \|d\| vs destriped ref | luma |
|---|---|---|---|---|
| **slot2 + F4** | .003 / .002 / .004 | −.006 | **.082** | −.034 |
| quality | .042 / .044 / −.002 | −.014 | .134 | −.081 |
| student only | −.008 / .002 / .021 | +.000 | .091 | −.031 |

**Visual** (`compare/s2F4_E_s202.jpg`):
- sign correct ("THE GOLDEN CRUST BAKERY"; chalkboard legible);
- lichen irregular, like the reference (quality: rosettes);
- bread and crowd compositions nearly identical to the reference;
- beard and leather natural.

Held-out (32) still running.

### Phase 8m (21:20 UTC): held-out evaluation (32 prompts, s101, own sizes)

**Schedule:** slot2 + F4 = student (relabel, it 2000) on grid 0, 4, 9, 15, 21, 28, 35, with the guided original model on interval .957 → .919, then the original 4-eval DPM++ finish; destripe 1.5; 15 passes.

| run | \|d\| vs destriped ref | luma | detail | dup excess (materials + nature, 8) | OCR bag: EN text (4) | ZH text (4) |
|---|---|---|---|---|---|---|
| **slot2 + F4** | **.090** | +.011 | **−.004** | **−.006** | .947 | .543* |
| quality | .124 | −.049 | −.022 | +.063 | .951 | .543* |
| destriped reference | 0 | 0 | 0 | 0 | .934 | .543* |

*The ZH OCR is identical for the reference (vertical / calligraphic text is unread), so it is uninformative. Checked visually.

**Visual** (`compare/heldout_p1..p4.jpg`): slot2 + F4 is close to the reference on all 32.
- textures natural (pebbles: quality stamps uniform beads);
- people natural, no malformed hands or faces;
- EN poster, book cover, receipt, highway sign correct;
- ZH tea cake, chart, couplets, banner correct;
- only miss: the tram's small destination sign differs (places_08).

**Decision rule (pre-registered) vs quality:**
- (a) no text regression: met (the E en_sign headline is fixed by slot 2; held-out text is on par);
- (b) reference-anchored preference: clearly favoured (|d| .090 vs .124; textures and composition closer). The user's judgement is pending;
- (c) no new defect class in E (36) + held-out (32): none seen;
- (d) ≤ 120 s: 15 passes ≈ 119 s end to end on q8 (at the limit).

**Verdict:** passes, pending the user's review.

### Phase 9a (2026-10-22 00:30 UTC): user decisions on v2 (results page JSON)

**Verdict:** "new turbo is clearly better".
- **Exception (user note):** legitimate geometric regularity gets "organicised". A chain-link fence shadow should fall as a grid but becomes organic blobs (heldout_people_portrait fence portrait). Laundromat washers should be neatly aligned but are scattered (heldout laundromat).
- I confirm the fence shadow on the sheet. This is the flip side of anti-stamping: a regression student that averages fine periodic structure.

**Decisions:**
- small text: fix first (stage 3 before release);
- schedule: only the 15-pass version (slot 2 + F4);
- next: stage 3 first, then package;
- **stage 3 spend approved, hard cap $15**;
- replace the old (DMD) turbo;
- release: keep private.

**Stage 3 design (to address both):**
- **New prompts:** ~250 new training prompts targeting (a) legitimate regularity (fences and mesh shadows, tiles, window grids, rows of machines, shelving, keyboards, chessboards, brickwork, crop rows, solar panels, stairs) and (b) dense or small text. Teacher only for the new ones.
- **Warm start:** resume from the relabel it-2000 LoRA, reusing the compact states of the 1,983 stage-1 samples.
- **Sampling:** oversample the new categories.
- **Evaluation:** a new held-out regular-structure plus small-text set, with destriped 50-step references on the Mac.

### Phase 9b (00:45 UTC): stage 3 launched

**New prompts:** 260 (140 regular-geometry, 120 small text); 1 dropped as too close to the new held-out set. Combined file `pod_v3/prompts_s3_all.jsonl`: the 1,998 stage-1 prompts, then 259 new (16 new val: 8 regular, 8 smalltext).

**New held-out set:** `prompts_v3/heldout_regular.jsonl` (16 regular, 8 small text). Destriped 50-step references (H2_ref_d) and the stage-2 schedule (H2_s2F4) are queued on the Mac.

**pod_v3:**
- warm start from relabel it 2000;
- stage-1 compact states plus stage-2 valrefs downloaded from the dataset;
- teacher runs only on indices ≥ 1998;
- sampling weights: regular / smalltext ×3, text_en / text_zh ×1.5;
- LR 7e-5, cosine decay after 1500, 4,500 iterations, self-rollout 0.5 from the start (relabel);
- image check every 750 its on 16 stage-2 val + 16 new val prompts, reported separately.
- TINY smoke test passed.

**Pod `ucyuz3irbth19v`** (1× A100-SXM4, $1.79/h). Guard v3: cap $15, prefix results_s3, marker "STAGE3 COMPLETE". Encode underway at 6 min.

### Phase 9c (05:30 UTC): stage 3 mid-run

Warm start verified (252 modules; state rel err .1836, equal to the stage-2 end).

**Image metric** (|luma − destriped ref|):

| it | stage-2 val (16) | new regular / smalltext val (16) |
|---|---|---|
| 0 | .090 | .104 |
| 750 | .112 | .116 |
| 1500 | .096 | .116 |

- The resume perturbed the student first: full self-rollout from it 0 at LR 7e-5 plus the shifted sampling mix. It is now recovering on the stage-2 set; the new set is flat.
- This paired metric is dominated by composition drift, so the deciding test is visual on the 24-prompt H2 set (fence-shadow grids, aligned machines, small text).
- Note: train.py saves `prefix_lora_best` only when the metric beats it 0; `prefix_lora_last` is always saved.
- Speed: 5.0 s/it, so the 300-min cap ends at ~it 3,500. Spend $8.4 at 05:27.

### Phase 9d (08:05 UTC): stage 3 complete ($13.10; pod deleted, verified)

**Image metric:**

| it | stage-2 val | new val |
|---|---|---|
| 0 | .090 | .104 |
| 750 | .112 | .116 |
| 1500 | .096 | .116 |
| 2250 | .090 | .107 |
| 3000 | .091 | .103 |
| 3565 (time cap) | **.089** | .105 |

Best checkpoint: it 3565.

**New-val sheet** (`arm_s3/valimg_it3565.jpg`):
- blind-stripe shadows straight; window grid, pool lanes, crop rows and grid paper regular;
- newspaper headline, bar-chart values, ZH receipt and medicine label correct;
- miss: a subway map drawn as a line instead of a loop.

**Queued on the Mac:** H2 (24 new held-out prompts) and E s101 with the stage-3 student plus the slot-2 schedule. H2_ref_d and H2_s2F4 are already done. Balance $100.88.

## Phase 9e: stage 3 evaluation (2026-10-22)
- H2 (24 new held-out, s101): |d| s2 .1012 / s3 .1017; regular dup vs ref s2 -.072 / s3 -.059; smalltext OCR s2 .957 / s3 .944 (ref .970).
- E s101 (slot2+F4): s2 (E_alloc_T2) text .990, rep +.011, detail -.003, |d| .107, luma +.015; s3 (E_s3F4) text .927, rep +.006, detail -.007, |d| .092, luma +.034.
- By eye: s3 fence-shadow grid more regular (closer to ref); washers aligned in both; s3 REGRESSES u_infographic headline (garbled, OCR 1.0 -> .75), heldout2_regular_13 stadium (rows warp, camera drifts), timetable text set smaller.
- Verdict: stage 3 does not replace stage 2. Stage 2 (results_v2/46a7d1237777 arm_relabel best) stays the candidate. Sheets: compare/s3_h2_p1-3.jpg.
- Owner review (2026-10-22): "stage 3 is a clear improvement; improvements outnumber the regressions in quantity and intensity"; the stadium warp is acceptable. Owner asked for more Mac tests before deciding. Queued (queue_s3_tests.py, 28 jobs / 86 images): H2 seed 202 for s2 and s3, s3 on the 32 earlier held-out prompts (H_s3F4, s101), s3 on E S6 at s202. Scoring: score_s3_tests.py.

## Phase 9f: stage 3 follow-up tests and the stage-2 release package (2026-10-22)

**Stage 3 follow-up tests (Mac, free):**
- H2 seed 202 (no reference; stage 2 vs stage 3): small-text OCR s2 .914 / s3 .955 (map .47 / .72); regular-geometry dup .247 / .246. By eye, the seed-101 regressions (stadium warp, shrunken timetable) do not recur.
- H (32 earlier held-out, s101), vs destriped ref: |d| s2 .0899 / s3 .0887; luma +.011 / +.009; dup excess −.006 / −.010; EN OCR .947 / .932 (ref .934).
- E S6 seed 202, vs E_ref_d: dup excess mean s2 +.004 / s3 −.005 (leather +.027 → −.001); en_sign OCR .959 / 1.0; |d| .082 / .098.
- Read: stage 3 ≈ stage 2 overall, better on small text and stamping, slightly more composition drift.

**Owner decision:** release stage 2 now; continue stage 3 work in parallel.

**Package (moss-image-mlx, branch `turbo-v2`, commit d16cee9; not pushed):**
- `--turbo` is now the 15-pass sampler (`moss_image_mlx/turbo.py`), replacing the DMD-adapter turbo.
- Adapter groups (destripe always on; student switched per step).
- Parity: bit-identical to the research H_s2F4 images (3 prompts, max pixel diff 0).
- Timing: 117 s sampling + 2 s decode at ~1 MP; single-image peak 9.4 GB.
- Weights staged at `out/MOSS-Image-SFT-Turbo-MLX/` (turbo_student.safetensors = stage-2 relabel best, sha256 1859687695670d32…; model card, LICENSE, assets); not uploaded.

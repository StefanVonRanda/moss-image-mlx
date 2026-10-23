# Turbo quality: overnight research report (2026-10-19)

**Status:** complete. The 1024² comparison with 50-step references is on the report page.

## The short version

- **The waxy look is mostly not a texture problem; it's an *idealization* problem, decided in the first two or three of turbo's eight steps.** The distilled adapter steers each picture toward a cleaner, younger, glossier, more uniform version of the prompt:
  - glossy pebbled leather instead of worn, scuffed leather;
  - young clean hands where the prompt says "weathered";
  - beauty-filter skin with a few freckles;
  - grid-regular pavers instead of irregular wet cobbles;
  - star-shaped flour motifs on bread.

  The smoothing that people notice is what that idealized content looks like.
- **The fix that works:**
  - run the **first 2 composition steps with the original model (with CFG)**, then hand over to the turbo adapter for the remaining 6;
  - cost: 10 model evaluations instead of 8: 41 s vs 34 s per image at 768², 78 s vs 64.5 s of sampling at 1024² (about 80 s vs 67 s end to end, +21%), same 9.3 GB peak;
  - result: the original model's content character, material realism and prompt adherence come back (weathered hands, worn leather, natural freckles, irregular cobbles, darker realistic dusk light). Turbo keeps its few-step finish.
- **Blind check (seed 7, images not seen before in any form):**
  - `quality` was preferred over old turbo on 12/16 texture prompts (9 clearly), turbo on 2, with 2 ties (sign test p = 0.0065).
  - Caveat: one judge (me), who by then knew `quality`'s typical look, so treat this as supporting evidence. The page has the same blind sheets for you.
- **A costlier profile was tested and not shipped:** 3 composition steps plus a finish with more low-noise steps (12 evaluations). Blind against `quality` it was a wash: 3 slight wins, 2 losses (one a mangled hand), 3 ties.
- **Shipped as `--turbo` (default profile `quality`); `--turbo fast` is the old behaviour.**

## What I learned about where the quality goes

1. **The smoothing is in the distilled weights, not the step count.**
   - 16 turbo steps are no better than 8.
   - Adapter strength 1.0 is waxier than 0.7.
   - The adapter's own trained sampler (data-space) is the waxiest of all.
   - DMD's reverse-KL objective is mode-seeking; the literature reports the same over-smoothing for DMD/DMD2.
2. **It is progressive.** Decoded per-step estimates show turbo's first clean estimate (σ 0.97) has *more* wrinkle texture than its final image; each later step averages a little more away. The original model commits its texture by σ≈0.76 and only refines below 0.5.
3. **CFG carries texture.** The original model *without* CFG, at 50 steps, is about as waxy as turbo. Any fix that keeps CFG out of the formative steps inherits this.
4. **The look lives in the adapter's rear blocks; the content choices happen at high noise.**
   - Switching the adapter off in blocks 30–41 removes the glossy, saturated, stylized finish (matte glaze, natural skin tone); switching it off in the front or middle blocks changes nothing.
   - The weights agree: the adapter's edits are ~3× larger, relative to the base weights, in the rear blocks.
   - But this softens other prompts badly, and the content idealization (young hands, glossy leather) is decided by the rear adapters during the first few high-noise steps.
5. **Late refinement cannot fix it.** Handing the low-noise tail (σ<0.5, even σ<0.8) to the original model with full CFG and 50-step density barely moves anything. By then the idealized content is committed.
6. **Handing the *first* steps to the original model fixes it.** Two composition steps (σ 1 → 0.968 → 0.926) with the original model and CFG are enough; turbo can finish from there without re-idealizing.
   - The steps must stay small: a coarse first jump (1 → 0.91 → 0.72) leaves a blurry, averaged composition that turbo turns into ghosted double exposures.

## What worked, what didn't

| Direction | Cost | Result |
|---|---|---|
| More turbo steps (16) | 2× | no gain |
| Adapter strength 1.0 / 0.85 | same | waxier |
| Adapter's native data-space sampler (lodestones) | same | waxiest; geometric motifs |
| darask0 adversarial 4-step adapter | 0.5× | fine detail, but a synthetic crosshatch grid (caught by the peak detector) |
| Lower schedule shift (2–3) | same | small, consistent gain; kept in `quality-plus` |
| Stochastic re-noising (fresh noise, η 0.3–0.5) | same | more pores; motifs unchanged; η 1.0 adds noise-like texture |
| Original model + CFG on the low-noise tail (σ<0.5 or <0.8) | +40–75% | almost no gain (content already committed) |
| Adapter attenuated in rear blocks | same | right finish on some prompts, soft and hazy on others; not robust |
| **Original model + CFG for the first 2 steps, then turbo** | **+25%** | **content character, materials and prompt adherence restored** |
| Same with 3 steps and a shift-3 finish | +45% | strongest skin and material detail |
| Original model, 2 coarse steps (1 → 0.91 → 0.72), then turbo | same | fails: ghosting |

## How it was evaluated

- **Prompts:** 16 hard texture/material/light prompts plus 3 controls (EN sign text, small handwritten Chinese, a 3-object layout). The texture prompts cover:
  - skin ×4;
  - fur, knit, leather, denim + wood, brushed metal, wet stone, water, foliage, food ×2;
  - noir lighting and a landscape.

  Screening at 768², seed 42. Finalists on all prompts at seeds 42 and 7, then 1024² at a fresh seed (101) with 50-step references.
- **References:** the original model, 50 steps, CFG 3, same seed and size. The harness reproduces the earlier stage-2 references exactly.
- **Hand metrics, calibrated on controlled manipulations before use:**
  - **Wax** (Gaussian smoothing): in-focus fine/mid energy and interior energy drop, edges stay put, the slope steepens.
  - **Painted noise:** cross-scale coherence and kurtosis drop; colour noise also raises the chroma ratio.
  - **Sharpening:** edge energy rises well past the reference and the slope flattens.
  - **Periodic artifacts** (grids, screen-door): spectral peakiness rises.

  Whole-frame energy turned out to be confounded by depth of field, so the main numbers are restricted to in-focus regions.
- **Learned features:** DINOv2 Gram-matrix texture distance (layout-free) and patch KID, used only at set level.
- **Visual:** labelled sheets with a 1:1 centre crop and each image's most-structured 1:1 window, for every round; blind A/B sheets with the original labelled and the two turbo variants shuffled. My own blind calls were made on seed-7 images I had not seen in any form.
- **Why the metrics alone were not enough:** they said the adversarial adapter had the best fine detail (it is a synthetic grid). They also said rear-block attenuation was a clear win, and the wider prompt set showed it softens most prompts. Every decision here was checked on crops.
- **Consistency:** the shipped `--turbo` profiles reproduce the research engine's renders bit for bit.

## Bottlenecks: fundamental vs solvable

**Fundamental, with this adapter and inference-time changes only:**
- **The adapter is mode-seeking.** Whenever it makes the composition decisions it idealizes, and no sampler, schedule or noise trick tried tonight undoes that without handing those steps to the original model.
- **It also polishes micro-texture in the finishing steps.** The finishing stays a little smoother than 50 steps of the original model with CFG. The original's last ~8% of detail costs its own CFG steps (the tail ceiling experiments: +40–75% time for small gains).

**Solvable, with training or engineering:**
- **Retrain or fine-tune the adapter with a texture-preserving objective:**
  - an adversarial term (DMD2-style) or continuous-time distribution matching (CDM);
  - a regulariser on the rear blocks, where the stylization lives.

  This is a GPU-hours job, not something the Mac can do overnight.
- **CFG-distill the two composition steps** (or train a tiny head for them). That would bring `quality` back to 8 evaluations with the same result.
- **Cheaper guidance for the composition steps.** The unconditional branch could run a pruned network (skip-layer guidance), cutting the +25% roughly in half. Untested.
- **darask0's adversarial adapter shows real detail can be distilled.** Its grid artifact is a training defect (likely the discriminator's patching), not a law.

## What `quality` does not fix

- **Micro-texture is still turbo's.** In-focus fine/mid energy is unchanged or slightly lower (its lighting is darker and moodier), and pores and fibres are a touch smoother than 50 steps of the original. The gain is in content and materials, not in sharpness.
- **Some stylized motifs survive** when they are decided in the very first step on this seed: the speckle dots on the mug, the flour pattern on the bread (both blind ties).
- **It sometimes picks a different scene than turbo** (it's a different trajectory); seeds aren't interchangeable between `fast` and `quality`.
- **Small, dim text** reads slightly worse in OCR because the scenes are darker. At 1024² the sign and the handwritten Chinese note are spelled correctly (Chinese OCR 1.0); OCR skipped a small "THE" and misread a dim chalkboard line.

## Reproduce

- **Shipped:**

  ```bash
  moss-mlx "prompt" --turbo
  ```

  `--turbo fast` gives the old behaviour.
- **Research harness** (`tq/`): `engine.py` (samplers), `run.py` (JSON jobs), `queue_runner.sh` (GPU queue under the accelerator lock), `score.py` / `dino_metrics.py` / `sheets.py` / `make_report.py`.
  - Every job spec is in `tq/done/`, every image in `tq/runs/<job>/`, and every log in `tq/logs/`.
  - Chronological notes: `tq/LOG.md`.
- **Consistency:** `tq/check_cli.py` shows the shipped profile reproduces the research run bit for bit.

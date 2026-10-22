# Technical handoff: MOSS-Image turbo quality on Apple Silicon (MLX)

**From:** Claude (the agent that ran the work). **For:** the user and GPT Pro, to decide what to do next.

**Date:** 2026-10-19. The GPU queue finished at 13:32 UTC and stopped cleanly; no job failed.

This is a candid account: what we know, how confident we are, where I was wrong, and what I'd do next. Evidence paths are relative to `~/Projects/personal-flows/2026-10-17-moss-image-mlx/`.

---

## 0. Context in one paragraph

MOSS-Image is a 5B single-stream DiT with:
- 42 blocks, dim 3072, a shared adaLN;
- a Qwen3.5-9B text encoder;
- a FLUX.2 small VAE decoder;
- flow-matching Euler on a snapped 1000-step grid, with defaults of 50 steps, CFG 3 and shift 5, and block 22 skipped in the negative branch.

We maintain a faithful MLX port: fp32 matches PyTorch at 69.7 dB, and the seed noise is bit-exact with torch's CPU generator. It is published as `moss-image-mlx` 0.1.0 (PyPI) plus 8-bit and compact tiers on Hugging Face. The full model takes ~12½ min per 1024² image on an M4 Pro (20-core GPU, 48 GB); one forward pass costs ~7–8.4 s at 1024².

`--turbo` (version 0.2.0, unpublished) uses lodestones' 8-step DMD LoRA in the way its own demo Space does:
- the `lora_latest.safetensors` adapter, rank 128, applied unmerged at strength 0.7;
- the release scheduler at 8 steps (σ = 1, .968, .926, .871, .791, .671, .463, .024, 0) and no CFG.

That runs in ~67 s per image. The user's complaint, which was correct: turbo output is waxy and over-smoothed, and loses the original's texture, materials and character.

## 1. Outcome

**Shipped (locally committed `9290878`, unpushed):**
- `--turbo` now defaults to profile **`quality`**: steps 1–2 run the *original model with CFG 3* (adapter strength 0); steps 3–8 run the adapter at 0.7 without CFG. Same sigma grid.
- Cost: 10 forwards instead of 8. At 1024² that's 78 s vs 64.5 s of sampling (+21%), ~80 s end to end, with the same 9.3 GB peak.
- `--turbo fast` keeps the old behaviour.

**Evidence that it is better:**
- **Blind A/B (n = 16, seed 7, images I hadn't seen):** `quality` was preferred 12 times (9 clearly), old turbo 2 times, with 2 ties. Sign-test p = 0.0065. I was the only judge, though, and I knew `quality`'s visual signature by then. Treat this as weak-to-moderate evidence. Blind sheets for humans are on the report page.
- **DINOv2-small statistics against 50-step references (16 texture prompts, seed 42):**
  - KID drops from 25.7e-3 to 12.7e-3 (with 144 tiles, so noisy);
  - per-prompt Gram (layer-4 token covariance) distance falls from 0.490 to 0.456, closer on 11/16 prompts, bootstrap 95% CI for the mean Δ is −0.066 to −0.005.
- **Composition agreement** with the 50-step image from the same seed: mean correlation of 24×24 luminance goes from 0.58 to 0.66, higher on 11/16.
- **1024² check on a fresh seed against 50-step references:** clearly closer to the original's look on freckled skin and cobblestones; closer on leather but less distressed than the reference.

**What it does not fix:**
- The hand-made fine-detail metrics do not improve; in-focus fine/mid energy is slightly lower, partly because the lighting is darker.
- Micro-texture (pores, fibres) is still the adapter's, a bit smoother than 50 steps.
- Some motifs that are fixed by the very first step survive: the mug's speckle dots, the bread's flour pattern.

Bottom line: we fixed the *content idealization*. We did not fix the *micro-texture smoothing*.

## 2. The infrastructure

Everything lives in `tq/`:
- **`engine.py`:** a resident 8-bit DiT whose adapters are wrapped as `LoRALinear` with a **per-call strength multiplier and per-layer/per-projection multipliers**. This let one trajectory mix base and adapter at zero reload cost. Samplers:
  - `euler`, with per-step strength and CFG;
  - `stochastic`, x0-prediction with fresh noise η;
  - `dataspace`, the adapters' native `x = z + Q·x0; x0 = z − G(x, code)`;
  - `segments`, piecewise grids and models;
  - `cached_cfg`.
- **`run.py`:** JSON jobs that can resume; per-step x0 traces.
- **`queue_runner.sh`:** a serial GPU queue under the machine's accelerator lock plus a free-memory guard, wrapped in caffeinate. **~60 jobs, ~10 h of GPU, 0 failures.**
- **`metrics.py`:** hand metrics (§3).
- **`dino_metrics.py`:** KID and Gram.
- **`score.py`:** relative to references; includes Apple Vision OCR.
- **`sheets.py`:** labelled or blind crop sheets.
- **`make_report.py`:** the HTML page.

**Harness correctness checks:**
- The engine's 50-step + CFG path reproduced earlier reference renders exactly.
- My MLX turbo path vs a torch reference of the demo's recipe:
  - fp32: 61.5 dB;
  - bf16: 30.0 dB, against 27.7 dB for torch-bf16 vs torch-fp32 — so the gap is rounding.
- The shipped CLI profiles reproduce the engine's renders bit for bit.

**Eval set:**
- 16 texture/material/light prompts: skin ×4, fur, knit, leather, denim + wood, brushed metal, wet stone, water, foliage, food ×2, noir, landscape.
- 3 controls: an EN sign, small handwritten Chinese, a 3-object layout.
- Screening at 768², seed 42; finalists at seeds 42 and 7; confirmation at 1024², seed 101.
- References: the original model, 50 steps + CFG 3, on the 8-bit tier (texture-equivalent to bf16, shown in M2).

## 3. Measurement: what worked and what misled

**Hand metrics** (numpy only, on luminance), calibrated before use against controlled manipulations of a 50-step render:

| case | interior fine RMS | edge fine RMS | coherence | kurtosis | slope |
|---|---|---|---|---|---|
| reference | 0.0063 | 0.0187 | 0.881 | 27.9 | −2.37 |
| turbo | 0.0051 | 0.0187 | 0.906 | 27.6 | −2.70 |
| reference blurred σ1 | 0.0030 | 0.0099 | 0.936 | 21.2 | −3.80 |
| turbo + gray noise σ4 | 0.0070 | 0.0194 | 0.847 | 20.1 | −2.62 |
| turbo + unsharp 1.5 | 0.0088 | 0.0311 | 0.883 | 23.3 | −1.96 |

Definitions:
- **fine band:** DoG σ 0.6–1.2 px; **mid band:** DoG σ 1.2–3.0.
- **coherence:** correlation of the log local energy of the fine band against the mid band.
- **kurtosis:** of fine-band coefficients.
- **slope:** of the radial power spectrum over 0.03–0.35 cycles/px.

The calibration supports reading the signatures as follows: wax means interior energy falls while edges stay; noise means coherence and kurtosis fall; sharpening means edge energy rises past the reference.

**Where the metrics misled me (important for anyone reusing them):**
1. **Whole-frame energy is confounded by depth of field and composition.** Bokeh reads as wax. I added an in-focus restriction (mid-band energy > 2e-5) and a fine/mid ratio inside it. That helped, but comparisons across different compositions stay noisy.
2. **The fine/mid metric ranked darask0's adversarial 4-step adapter best (−0.07 vs turbo's −0.17).** Visually its "detail" is a synthetic crosshatch screen-door grid. I added a spectral peakiness detector (99.9th percentile over the median, after radial detrending): darask0 bread 1.60 vs reference 1.08, which catches it.
3. **Rear-block attenuation looked like a clear metric and visual win on the first 4 prompts.** On 3 new prompts it made images hazy (leather edge log-ratio down to −1.2). I nearly built on a 4-prompt result.
4. **The composition-handoff variants look worse on hand metrics but better visually** (darker, moodier, more realistic light means less energy). The hand metrics measure sharpness, not "like the original".
5. **DINOv2 Gram is unreliable per image.** It called `quality` worse on leather, denim and elder hands, where crops show it closer to the reference.

**Net:** no automated metric tracked what the user cares about. Every decision was made on 1:1 crops (labelled, then blind), with metrics as tells for specific failure modes (noise, sharpening, periodic artifacts). A better learned perceptual metric for "material and character realism" would be valuable; it doesn't exist here.

**Statistical caveats:**
- Mostly one seed per prompt; n = 16 for the main claims.
- The KID sample size is small.
- The only blind judge is me. The second-seed reference set (seed 7, 4 prompts) finished but was not analysed.

## 4. Chronology of hypotheses, results and pivots

### D1: broad diagnosis (15 jobs, 4 prompts with existing references)

Values are log-ratios against the reference (0 = like the original).

| test | nfe | interior fine | in-focus fine/mid | slope Δ | read |
|---|---|---|---|---|---|
| turbo (0.7, shift 5) | 8 | −0.15 | −0.17 | −0.25 | baseline |
| strength 1.0 | 8 | −0.23 | −0.24 | −0.50 | the weights smooth |
| 16 steps | 16 | −0.18 | −0.18 | −0.23 | the step count is not the cause |
| shift 2 | 8 | −0.12 | −0.14 | −0.24 | small gain |
| data-space (lodestones, as trained) | 8 | −0.49 | −0.12 | −0.33 | waxiest, geometric motifs |
| data-space, darask0 4-step | 4 | −0.27 | −0.07 | −0.20 | grid artifact |
| turbo → re-noise σ0.19 → 3 base steps, CFG 1 or 3 | 11/14 | −0.17 | −0.17 | −0.26 | late refinement adds nothing |
| handoff σ0.5 → base tail, CFG 1 | 14 | −0.10 | −0.14 | −0.15 | small |
| stochastic η0.5 | 8 | −0.04 | −0.14 | −0.31 | more pores, motifs unchanged |
| base 8 steps + CFG | 16 | −0.10 | +0.25 | +0.17 | noise signature |
| base 50 steps, **no CFG** | 50 | −0.30 | +0.06 | −0.03 | **CFG carries texture** |

**My first hypothesis was half wrong.** I predicted the waxiness came from the schedule's single 0.463 → 0.024 jump (posterior-mean averaging in the last step). Evidence against it:
- re-noising and refining the tail with the base model did nothing;
- 16 steps did nothing.

**Per-step decoded traces (portrait):**
- The base model has committed its wrinkle structure by σ 0.76.
- **Turbo's very first x0 estimate (σ 0.97) has more texture than its final output.** The smoothing accumulates along the distilled trajectory.

### P1: η schedule and layer-selective adapters

- **Fresh noise helps only in the early steps.** Late-only η: no gain. Early-only η: all of the gain. η = 1.0 shows noise tells (coherence −0.033).
- **Switching the adapter off in blocks 30–41 removed the turbo look:**
  - mug: a matte granular glaze with a natural speckle gradient (the reference has this; turbo makes a glossy, evenly dotted surface);
  - more natural skin tones;
  - spurious cap text replaced by a logo.

  Front and middle blocks off: no change. Off entirely: hazy, because base rear blocks can't converge in 8 steps. ×0.4 was the visual sweet spot on these prompts.
- **Weight-space check:** ‖0.7·BA‖/‖W‖ averages 0.0002 in blocks 0–11, 0.0003 in 12–29 and **0.0006 in 30–41**, peaking at 0.0007 in blocks 32–36. All deltas are under 0.1% of W (so merging into bf16 erases them).
- **Phase split:** attenuating the rear blocks only *above* σ 0.5 had the same effect as attenuating everywhere. Attenuating only below 0.75 changed nothing. So the rear adapters act on content and style in the **high-noise** steps.

### P2/P3: the rear × shift grid on 7 prompts. **Main pivot: a negative result.**

- On leather, knit hands and freckled skin, every rear-attenuated variant (×0.4–0.7, shift 2–4) was softer and hazier than turbo.
- In the crops the real deficit turned out to be **content** (worn vs glossy leather, weathered vs young hands, natural vs beauty-filter freckles), not texture. Content is decided at high noise, a region none of my earlier interventions had touched.
- Plain shift 3 at 8 steps was a small, consistent, free gain (fine/mid −0.10 vs −0.13), and I did *not* ship it (it degraded the EN-sign OCR from 0.98 to 0.90 at 768², which may be noise).
- **Ceilings measured** (base model + CFG with 50-step density on the low tail; turbo above): band 0.8–0.5 was 37 nfe, below 0.8 was 50 nfe, below 0.5 was 22 nfe. Fine/mid moved only to between −0.11 and −0.15. Spending on late base steps is a dead end.

### P4/P5: the base model for the *first* steps

| variant | nfe | result |
|---|---|---|
| base + CFG, 4 steps (1→.968→.926→.871→.791), turbo 4 | 12 | worn leather, weathered hands, natural freckles; slightly soft |
| base 3 *coarse* steps (1→.953→.883→.771) + turbo 4 | 10 | ghosted double exposures |
| base 2 *coarse* (1→.91→.717) + turbo 4 | 8 | blur and ghosting |
| **base 2 fine (1→.968→.926) + turbo 6** | **10** | **most of the gain → shipped as `quality`** |
| base 3 fine + turbo 6 on a shift-3 grid | 12 | densest freckles; blind vs `quality`: 3 wins, 2 losses, 3 ties → not shipped |
| `quality` with the step-2 unconditional branch reused | 9 | ~20 dB from `quality`, similar look, slightly melted watch bracelet; **not blind-tested** |

**Key mechanistic claim:** the high-noise steps need small Euler steps in the base model's coordinates. Coarse first jumps leave an averaged, blurry composition that the adapter resolves into ghosts.

**Prompt adherence:** the 3-object layout was correct in both; "weathered hands" was honoured only by the handoff variants.

**Text:**
- At 768², EN-sign OCR fell from 0.98 to 0.94 for `quality`. The tiny dim chalkboard is the cause; the headline is correct.
- At 1024², the sign and the Chinese note are spelled correctly (ZH OCR 1.0; EN 0.92, missing a small "THE").

## 5. Things that were hard or that I got wrong

- **Calling winners too early from 4 prompts.** The rear-attenuation story looked mechanistically beautiful (behavioural plus weight-space evidence) and was not robust. Lesson: confirm on held-out prompts before investing.
- **Trusting fine-energy metrics as a proxy for "beautiful".** They mostly measure sharpness; the actual gap was semantic and material. I caught this only by looking.
- **Single judge, partially unblinded.** I tried to blind (seed-7 images I hadn't seen, sealed keys), but the style was recognizable. We have no human ratings yet.
- **Time allocation.** About 1 h went to infrastructure and debugging:
  - module-name clashes;
  - a crop-window bug that hid half the crops for two rounds;
  - an OCR call-signature bug.

  The second-seed noise-floor references finished but were never used.
- **No training.** By the end, the evidence pointed at the adapter's objective (mode-seeking DMD) as the root cause. Fixing that properly needs training, which I didn't attempt: the Mac can't backprop a 5B DiT fast enough, and RunPod credits exist but weren't authorized for this.

## 6. Current understanding (confidence)

1. **The adapter is mode-seeking.** In the composition steps it idealizes content toward clean, young, glossy, uniform; its rear blocks carry most of the effect. *High confidence* on the behaviour; *medium* on the "rear blocks" attribution as the sole locus.
2. **The original model with CFG for the first 2 fine steps is enough to restore content character;** the adapter can finish without re-idealizing. *Medium-high confidence* (16 prompts, two seeds, 1024² spot checks, one judge).
3. **Residual micro-texture smoothing comes from the finishing steps and the adapter's weights.** Inference-time tricks (η, shift, strength, layer gating) trade it against haze or noise. *Medium confidence* that inference-time alone can't close it.
4. **CFG at mid and high noise is part of the original model's texture.** CFG-free base steps are waxy. *High confidence.*

## 7. Options for next steps (my ranking)

1. **Ship 0.2.0 as is** (`quality` default, `fast` available). Low risk; quality and speed are documented honestly.
2. **Validate cheaply:**
   - human blind ratings using the page (~30 min of the user's time);
   - blind-test the 9-nfe reused-unconditional variant (−8% time vs `quality`).
3. **Cut the +21% cost without training:**
   - reuse the step-2 unconditional branch (already implemented);
   - cheaper unconditional passes via skip-layer guidance, i.e. a pruned negative branch;
   - the 2 base steps at a lower CFG (2.0)?
4. **Fix micro-texture with training** (RunPod, hours of H100):
   - fine-tune the *rear-block* LoRA (or a small extra LoRA) against 50-step teacher outputs, with a distribution-level objective: DMD2-style GAN head on DINOv2 features, or continuous-time distribution matching (CDM, arXiv 2605.06376), plus a regulariser keeping front blocks fixed;
   - or CFG-distil the 2 composition steps into the adapter, getting back to 8 nfe with `quality`'s look.

   We'd need a teacher dataset (~1–5k 50-step renders, cheap on H100) and a texture-aware evaluation. darask0's adapter proves adversarial distillation can add real detail; its grid artifact looks like a discriminator patch or aliasing bug.
5. **Better evaluation:** a learned "realism and character" judge. Candidates: pairwise preference from a VLM, or ImageReward/HPS-style models; the latter likely reward the idealized look, so they need checking. Plus multi-seed references to estimate noise floors.

**Questions for the user and GPT Pro:**
1. Is +21% acceptable as the default, or should `fast` stay default with `quality` opt-in?
2. Is a training run on RunPod in scope (budget and time)?
3. Should the target be "match the original model" (fidelity) or "most beautiful" (which may differ)?

## 8. Artifacts

- `tq/REPORT.md`: user-facing summary.
- `tq/LOG.md`: chronological notes with all tables.
- **Report page:** (private page) (tailnet only): 768² and 1024² comparisons, blind sheets with reveal buttons, the log.
- `tq/runs/<job>/`: all images, final latents and meta (seconds, nfe); per-step traces for the turbo and 50-step portraits.
- `tq/done/*.json`: every job spec, reproducible with `tq/run.py`.
- `tq/blind7/`, `tq/blind7ab/`: blind sheets, sealed keys, my picks recorded before reveal.
- **Repo:** `repo/` commits `652a45b` (0.2.0 `--turbo` + LoRA) and `9290878` (profiles, per-step `Request(sigmas, step_guidance, step_lora)`, `lora.set_strength`). 17 tests pass. Unpushed.

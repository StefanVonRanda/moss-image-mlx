# MOSS-Image turbo, M6b report (2026-10-20)

**Question:** can a turbo at 1024² keep the original model's character (no stamped textures, no waxy darkening, correct text) within ~115 s on the M4 Pro?

**Answer so far:** not without training. The incumbent `--turbo quality` (10 passes, 78 s) still stands under the pre-registered rule.
- **The problem is now located precisely:** the turbo adapter's steps before σ ≈ .67.
- **A trained prefix that reaches the original model's state at σ .463 has a near-reference ceiling.** That is the v2 training proposal, awaiting a go-ahead.

Detailed chronology: `LOG.md` (Phases 4a–5f). Protocol: `PLAN_M6b.md`.

## Protocol

**Eval set E:** 18 prompts in five categories (repetition and material, idealization, structure, text, legitimate regularity), at 1024², seeds 101 and 202.

**References:** 50-step guided originals (CFG 3, block-22 skip).

**Decision rule** (fixed before the finalist runs). A candidate replaces the incumbent only if:
- (a) no text line correct in the incumbent becomes wrong;
- (b) ≥ 60% reference-anchored preference;
- (c) no new structural defect class in more than 1 of 36 images;
- (d) ≤ 115 s.

**Report card** (`reportcard.py`):
- Apple Vision OCR, plus manual verification of every text image;
- DINO recurrence, calibrated against real photos for lichen, bark and bread;
- final-latent HF delta;
- luma;
- cost.

## Key results

### 1. The finish is innocent; the prefix state is everything

"Oracle" runs run the original model exactly (50-step grid) to a handoff σ, then let the lodestones turbo adapter finish on the release grid. Triage prompts, seed 101:

| exact original to σ | turbo steps | stamping (bread / lichen / bark) | text | luma | look |
|---|---|---|---|---|---|
| .926 | 6 | .055 / .010 / .024 | correct | .434 | stamped (star flour, flower rosettes, fish-scale stone) |
| .671 | 3 | .017 / .004 / .004 | correct | .445 | ≈ reference, lichen slightly speckled |
| **.463** | **2** | **.011 / .001 / .003** | correct | **.452** | **≈ reference: same composition, lichen, crumb, bark** |
| reference (50 steps) | 0 | | | .458 | |
| *quality* (2 coarse guided steps to .926) | 6 | .037 / .017 / .015 | correct | .367 | stamped, ~20% darker |

**Seed 202 confirms it** (6 prompts):
- the exact-to-.463 oracle is again nearly indistinguishable from the reference, including the lichen, where every fast method shows rosettes on this seed;
- luma .251 vs reference .259 (quality .173).

**What this shows:**
- **Stamping** is a dose-response of turbo-adapter applications above ~.67. It appears even from a perfect starting state. This matches correction training's "repeated correction" mechanism.
- **Layout and glyph identity** are fixed by .926. Quality's coarse 2-step start is what costs it the infographic labels.
- **Darkening** has two causes: coarse guided steps, plus the turbo adapter's mid-σ steps.

### 2. Everything tried without new training fails the rule

All at 1024², seed 101 (full E) unless noted.

| candidate | passes / s | stamping excess | text | why it fails |
|---|---|---|---|---|
| **quality (incumbent)** | 10 / 78 | .066 | all correct | — |
| T6S2d (6 guided DPM++ evals, then 2 turbo, destripe) | 14 / 106 | .035 | ZH note garbled | (a) |
| akhaliq 4-step v2 | 4 / 32 | .109 | all correct | stamping worse (b) |
| RunPod pilot prefix (768², 221 samples) | 8 / 65 | .030 | sign and ZH illegible | (a) |
| teacher steps, then pilot (T2S4F2, T3S3F2) | 10–11 / 79–85 | .034 | worse than the pilot | (a) |
| pilot + turbo blended per step, λ .2 / .35 / .5 | 8 / 69 | .029–.030 | ghosted signs, stray words | (a), (c) |
| **ramp blend** (turbo λ .1 → .7) | 8 / 69 | .018 (s101), .013 (s202) | (a) passes over 36 images | **(c): malformed face (s202), muddy hands** |
| 3–4 DPM++ teacher evals to .926, then 6 turbo | 12–14 / 94–108 | ≈ quality | new label errors | (a), and stamping unchanged |

**What the blends do show:** a CFG-free prefix trained on the original's trajectory gives back the reference's brightness (luma .47 vs .475; quality .40).
- Its residual failures are those of an undertrained student: occasional structural collapse and stray text.

### 3. Mechanism and measurement notes

- **Latent test:** at high σ, the student's clean estimate has 1.5–5× the teacher's HF energy. It commits detail from a blurry state (mode-seeking).
- **Precision:** bf16 vs q8 makes no difference to stamping.
- **Metrics:**
  - DINO dup_frac is valid for lichen, bark and bread (checked against real photos);
  - it is not valid for paper or tweed;
  - oak, clouds, beard and portrait are uncalibrated.
  - It missed the s202 lichen rosettes, so visual checks remain required.
- **Unanchored blind judgement** reversed once (paper) when the reference was shown. All judgements are now reference-anchored.

## Next step: v2 prefix training (proposal, not started)

**Proposal:** `TRAINING_PROPOSAL_v2.md`.
- **Student:** a CFG-free prefix LoRA regressed onto the original's exact 50-step trajectory, in 8 steps from σ 1 to .463, then the existing 2-step turbo finish. 10 passes, ~78 s: quality's cost.
- **Data:** 1,100 prompts at 1024², 45% text, disjoint from E.
- **Cost:** ~17 A100-hours ≈ $30, with a hard guard at $45.
- **Early stop:** if it isn't clearly beating the guided-Euler bar by iteration 3,000.
- **Teacher states are kept**, so a 7-step student to .671 (the oracle shows .671 is almost as clean) can be retrained without new teacher cost.

**Status:** code ready in `pod_v2/` and smoke-tested end to end on CPU. Guard `pod_guard_v2.sh`. Nothing spent; it needs the user's go-ahead.

## Open items

- If v2 trains well: run the full rule on E (2 seeds), the reference-anchored rating page for the user, and package it as a `--turbo` profile.
- Calibrate or drop the uncalibrated texture prompts in the report card.
- Unpushed commits 652a45b and 9290878 (0.2.0, `--turbo` and `--turbo quality`) stay unpublished until the user decides.

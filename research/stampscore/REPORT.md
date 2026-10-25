# stampscore: a hardened within-image stamping score (2026-10-22)

**Scope.** This turns the DINO recurrence diagnostic (`tq/recurrence.py`, `tq/novelty/rec_phase.py`, both left
untouched) into a validated local tool, `tq/stampscore/`. Everything ran on CPU (about 3 s per 1024² image). All
numbers come from `experiments/analyze.py` (raw output in `out/analysis.md`, `out/results.json`).

**Data.** The data are the 107 images behind `novelty/sep_rows.json`:
- admitted materials (bread, moss/lichen wall, bark) on E at s101/s202;
- 8 held-out materials/nature prompts on H;
- 15 Wikimedia photos.

Groups: 16 DMD (E_quality, E_akh4, H_quality), 15 blends, 43 clean (non-DMD fast methods), 18 references, 15 real.
The set is frozen to files present at 19:20 (another session is still rendering into `runs/`). The anchor is always
the destriped same-seed 50-step reference.

**Human validation:** Felix labelled 82 blind crops (§9) and then rated 80 new crops 0–3 for severity (§10), both on
2026-10-23. Sections 1–7 are image-group-based (DMD vs clean) at t .85.

## Headline

| question | answer |
|---|---|
| Does it reproduce the original? | Yes: grid and p4 values match `sep_rows.json` exactly (74 rows, max diff 0.000). |
| Separation, DMD vs clean (per image) | AUC **.977** grid → **.984** p4 → **.985** p9 (95% bootstrap ≈ [.93–.95, 1.00]); held-out H: DMD higher on 8/8 pairs. |
| Phase invariance | Jittered stamps (independent 0–13 px offsets) are detected on 38% of bases with grid, **76% with p4, 86% with p9**. |
| Blur confound | Fixed by a guard: blur 3/6 px and every posterior-mean state inflated by more than +.05 are flagged; 0/79 finished generated images are flagged. |
| Robust to settings? | Yes, for DINOv2 layer 6 (small or base, t .80–.85, 28–112 px): AUC .91–.99. The last layer fails (.60–.82). |
| No-reference variant | Screening only: AUC .86 overall (.96 on E only); legitimately regular references reach dup .33. |
| Agreement with Felix's blind labels | Crop excess AUC **.92** (p4) / .93 (p9) / .89 (grid); whole-image excess **.95**. LPIPS (.90) and SIFID (.92) tie on these labels; see §9. |
| Severity (Felix, 0–3, 80 new crops) | At the recalibrated t .80, crop excess ρ **.76** (t .85: .62); severe vs clear AUC **.94** vs LPIPS .82 / SIFID .78. The default threshold moved to .80 (§10). |
| Do standard metrics see stamping? | Not specifically. LPIPS/SIFID separate DMD from clean (.87/.86) because DMD drifts from the reference overall. A jittered stamp moves LPIPS less than noise sd 6 does, and 8× less than blur 1.5. CLIP-IQA and MUSIQ are near chance. |

## 1. Phase invariance

`p4` pools tilings offset by 0/7 px (3.5 px worst-case misalignment); `p9` uses 0/5/9 (2.5 px).

**Separation.** Small, L6, t .85, d 56. Excess over the destriped reference:

| phases | AUC DMD vs clean [95% boot] | AUC blend vs clean | median DMD | median blend | median clean |
|---|---|---|---|---|---|
| grid | .977 [.93, 1.00] | .884 | .037 | .005 | −.001 |
| p4 | .984 [.95, 1.00] | .887 | .069 | .010 | −.002 |
| p9 | .985 [.95, 1.00] | .907 | .085 | .014 | −.003 |

**Synthetic perturbations of 21 clean bases.** The bases are 13 references and 8 admitted real photos. Stamps are a
48-px motif on a 70-px grid over the central third. Values are the mean excess over the unperturbed base; the share in
brackets is the share of bases above .005.

| variant | grid | p4 | p9 |
|---|---|---|---|
| aligned stamps | +.029 (95%) | +.025 (90%) | +.024 (90%) |
| unaligned (+7 px, all copies) | +.022 (100%) | +.024 (95%) | +.022 (90%) |
| **jittered (independent 0–13 px)** | **+.003 (38%)** | **+.011 (76%)** | **+.016 (86%)** |
| soft (50% alpha, jittered) | −.002 (0%) | −.001 (0%) | −.002 (0%) |
| noise sd 6 | +.000 | +.001 | +.002 |
| blur σ 1.5 | −.004 | −.004 | −.004 |
| blur σ 3 | +.011 | +.014 | +.016 |
| blur σ 6 | +.065 | +.083 | +.088 |

**What the phase failure really is.** The grid only fails when the copies sit at *different* phases relative to each
other. A constant +7 px offset keeps every copy at the same phase, so they still match each other; this is why the
earlier "unaligned" test looked fine on the grid. Real stamps are jittered, so p4/p9 matter.

**Whole-image shift by 3/7/10 px.** The median relative change in dup_frac is 21% (grid), 15% (p4) and 10% (p9). The
mean absolute change is .002–.003.

**Seed-202 lichen (the earlier miss).** p4/p9 now register the blends (p4 excess +.010/+.019/+.010 for l20/l50/ramp;
grid .004–.007). T6S2d is at +.011, and LOG Phase 5g records rosettes in T6S2d on this seed too. The v2 student is at
.000. The miss is reduced, not eliminated.

**Default.** p4 stays the default, for continuity with `rec_phase.py` and the cross-family replication. p9 is better
on every row at about 1.5× the cost (`--phases p9`).

## 2. Detail guard (blur and posterior means)

Two checks against the reference, implemented in `core.StampScorer.score`:
- **mid-band (1–4 px) energy ratio < 0.6** gives `low_detail`;
- **fine-to-mid band ratio > 1.6× the reference's** gives `decoder_lattice`.

**New finding.** Decoded posterior means (`runs/probe_states`) are not just blurry. They carry the VAE decoder's fine
periodic lattice, which is real pixel repetition and reads as stamping (bread σ 1 teacher: dup .60 vs .12 final). A
fine-band-only blur check is fooled: the lattice raises fine-band energy.

| set | mid ratio | fine/mid ratio | flagged | mean p4 excess |
|---|---|---|---|---|
| finished generated, all groups (79) | .69–1.35 | .63–1.46 | **0/79** | |
| noise sd 6 | .99–1.01 | 1.00–1.33 | 0/21 | +.001 |
| blur 1.5 | .54–.75 | .13–.43 | 1/21 | −.004 (harmless) |
| blur 3 | .26–.46 | | **21/21** | +.014 |
| blur 6 | .09–.21 | | **21/21** | +.083 |
| stamps (aligned, jittered) | .99–1.15 | .96–1.04 | 0/21 | +.011–.025 |

**Probe states** (bread and portrait, s42, 768², 32 intermediate x0 decodes, teacher and student):
- 15/32 are flagged, including every state inflated by more than +.05 (up to +.48);
- the residual inflation among states that pass the guard is at most **+.045** (portrait σ .791 teacher), plus three
  at +.026–.027 (portrait σ .926/.871 student, σ .671 teacher).
- Bread states that pass are within ±.012.

Without a reference, only mid-band rms < .010 is flagged (extreme blur). Finished outputs are ≥ .024 and real photos
≥ .013. **Don't score intermediate states.**

## 3. Sensitivity

Settings: p4, 120 configurations = {DINOv2-small, -base} × layers {3, 6, 9, 12} × t {.75–.95} × min distance
{28, 56, 112}. "Real max" is the highest absolute dup_frac on the 8 admitted real photos.

| net | layer | t | AUC DMD | AUC blend | median DMD | median clean | real max |
|---|---|---|---|---|---|---|---|
| small | 3 | .85 | .923 | .693 | .020 | −.001 | .012 |
| small | **6** | .80 | **.994** | .932 | .202 | −.009 | .261 |
| small | **6** | **.85 (default)** | **.984** | .887 | .069 | −.002 | .120 |
| small | 6 | .90 | .828 | .560 | .003 | .000 | .018 |
| small | 9 | .85 | .967 | .744 | .136 | −.006 | .271 |
| small | 12 | .85 | .690 | .388 | .007 | .000 | .995 |
| base | 3 | .80 | .977 | .722 | .064 | .000 | .035 |
| base | 6 | .80 | .994 | .904 | .063 | −.001 | .141 |
| base | 9 | .85 | .984 | .854 | .076 | −.006 | .187 |
| base | 12 | .85 | .814 | .612 | .030 | .000 | .966 |

**Minimum distance** (small, L6, .85): 28 px → .975, 56 → .984, 112 → .974.

Findings:
- **Layer is the main lever.** Mid layers (6; 9 on base) work.
- The last layer is semantic: real photos saturate (dup .8–1.0) and AUC drops to .60–.82.
- Layer 3 is usable but weaker.
- **Layer 9 on small is not stamp-specific.** Jittered stamps don't register (−.002) while noise sd 6 raises it
  (+.011); layer 6 is the reverse (+.011 / +.001).
- **The threshold has to move with the layer.** At L6, t .90 leaves too few near-duplicates (AUC .83). t .80 separates
  slightly better than .85 but raises the natural baseline (real max .26).
- **Network.** DINOv2-base gives no gain over small at its best setting (.994 both), so small stays the default.
- Overall: 30% of the 120 configurations reach AUC ≥ .95 and 46% reach ≥ .90. The top five are all layer 6
  (base, t .75: 1.000).

## 4. No-reference variant

This uses absolute p4 dup_frac at the native size (real photos centre-cropped to 1024).

| group | min / median / max |
|---|---|
| real, admitted materials | .000 / .003 / .120 |
| real, other (tweed, paper) | .003 / .017 / .178 |
| references | .001 / .013 / **.330** (held-out materials 17) |
| clean | .001 / .014 / .311 |
| blend | .004 / .017 / .089 |
| DMD | .012 / .083 / .486 |

**Separation.** AUC DMD vs clean is .86 [.75, .95] on all images and .96 on the E admitted prompts only.

**Operating points:**
- at .05: DMD 75%, clean 14%, references 17%, admitted real 12%;
- at .03: DMD 88%, clean 28%.

No threshold separates once legitimately regular content is present. Use the no-reference mode only to screen
organic-material images; the reference-relative excess is the instrument.

## 5. Operating points (reference mode, p4, t .85; the CLI now uses t .80 with the §10 flags)

| flag at excess > | DMD | blend | clean | E_ref vs E_ref_d |
|---|---|---|---|---|
| 0 | 100% | 93% | 37% | 0% |
| .005 | 100% | 73% | 21% | 0% |
| .01 | 88% | 60% | 9% | 0% |
| .02 | 81% | 20% | 0% | 0% |

Some "clean" images genuinely carry mild stamping (T6S2d on seed 202), so the clean false-positive rate is an upper
bound.

## 6. Standard metrics on the same data

The metrics are LPIPS-alex and SIFID (Inception first-pool patch statistics), both against the same reference, plus
CLIP-IQA and MUSIQ (pyiqa) without one. All run at native resolution. AUC is oriented so that "worse" = DMD.

| metric | AUC DMD vs clean | AUC blend vs clean | Spearman with stampscore |
|---|---|---|---|
| **stampscore excess p4** | **.984** | .887 | 1 |
| LPIPS to reference | .868 | .887 | .50 |
| SIFID to reference | .862 | .904 | .54 |
| −CLIP-IQA | .680 | .634 | −.22 (sign-adjusted .22) |
| −MUSIQ | .555 | .552 | −.09 |

**Specificity on the synthetic set** (mean change over the clean base; the stamp rows are the ones that should move):

| perturbation | stampscore | LPIPS | SIFID | Δ CLIP-IQA | Δ MUSIQ |
|---|---|---|---|---|---|
| jittered stamps | **+.011** | +.060 | +.030 | −.040 | −0.7 |
| aligned stamps | **+.025** | +.061 | +.034 | −.037 | −0.7 |
| noise sd 6 | +.001 | **+.071** | **+.281** | −.001 | −2.6 |
| blur 1.5 | −.004 | **+.455** | **+17.6** | **−.276** | **−23.2** |

The stamp-to-noise response ratio is about 11× for stampscore, 0.85× for LPIPS and 0.1× for SIFID; the IQA scores
respond to blur, not to stamps.

**What this means.** LPIPS/SIFID separate DMD from clean on this set because DMD outputs depart from the reference
everywhere (composition, contrast, detail), not because they detect repetition:
- they separate blends from clean equally well, or better (.89/.90 vs stampscore .89);
- a harmless 1.5-px blur moves them 8× (LPIPS) to 600× (SIFID) more than a stamp does.

The no-reference IQA scores are near chance on DMD vs clean.

**Patch-FID / KID** (Inception pool3 on 299-px crops at 1:1, each run against its references, set-level):
- DMD runs score worst on E (E_akh4 230, E_quality 190; KID 45/35 ×1e-3);
- but the clean E_pilot (171) is worse than H_quality (154), and blends are 144–202;
- clean methods are 63–171, and the undestriped reference E_ref is 24.

Patch-FID sees a distribution shift, but it can't attribute it to stamping or score a single image.

**Crop level** (the 512-px labelling crops, group-based, no labels yet):

| metric | DMD vs clean | blend vs clean |
|---|---|---|
| stampscore excess p9 / p4 / grid | .980 / .969 / .933 | .840 / .810 / .790 |
| stampscore whole-image excess p4 | .980 | .864 |
| stampscore absolute p4 (no ref) | .868 | .636 |
| LPIPS | .904 | .886 |
| SIFID | .855 | .974 |
| −CLIP-IQA / −MUSIQ | .670 / .513 | .595 / .450 |

## 7. Human validation set

**Page:** https://hosts-mac-mini.tail6d2520.ts.net:8791/stamp-label/ (static, under the existing `:8791` route; no new
listener or route).

**Content.** 82 shuffled 512-px crops at 1:1, JPEG q93:
- 72 generated: 13 references, 16 DMD, 15 blends, 28 clean. There is one location per prompt/seed slot, shared by
  every method, drawn from the reference's most textured windows and never from any score.
- 6 real photos.
- 4 catch trials: 2 synthetic jittered stamps, 2 heavy blurs. Both stamp catches score +.06/+.07; both blur catches
  score +.06/+.12 but are guard-flagged (mid ratio .14–.16).

**Using it.** It works one crop at a time, with large Stamped / Not stamped / Unsure buttons, auto-advance, notes, tap
to zoom 2×, an overview grid and Copy as JSON. Answers save on the device. It was tested in iPhone 13 emulation:
label, reload, persist, copy. axe reports 0 violations.

**Key.** `labels/key.json` is sealed and lives outside the site folder.

**Analysis, once the JSON is pasted:**

```bash
cd ~/Projects/personal-flows/2026-10-17-moss-image-mlx/tq/stampscore
.venv/bin/python experiments/analyze_labels.py labels.json
```

It reports:
- labels by hidden group;
- AUC with bootstrap CI and Spearman for every metric, with unsure dropped, counted as stamped and counted as not;
- a paired bootstrap of AUC differences (stampscore vs each standard metric and variant);
- operating points;
- catch trials and real photos;
- notes.

It was dry-run on fake labels (discarded) to confirm it runs end to end.

## 9. Felix's blind labels (82 crops, 2026-10-23)

Labels: 32 stamped, 38 not, 12 unsure. Raw file: `labels/felix_raw.json`. Run `experiments/analyze_labels.py`; output
in `out/labels_analysis.md`.

**By hidden group:**

| group | stamped | not | unsure |
|---|---|---|---|
| reference | 1 | 11 | 1 |
| clean | 3 | 21 | 4 |
| **blend** | **14** | 0 | 1 |
| DMD | 14 | 0 | 2 |
| real photo | 0 | 6 | 0 |
| catch: synthetic stamp | 0 | 0 | 2 |
| catch: heavy blur | 0 | 0 | 2 |

**Agreement.** Generated crops, n = 64 (32 stamped vs 32 not; unsure dropped):

| metric | AUC [95% boot] | Spearman |
|---|---|---|
| stampscore excess p9 (crop) | .932 [.85, .99] | +.75 |
| stampscore excess p4 (crop) | .917 [.83, .98] | +.72 |
| stampscore excess grid (crop) | .890 [.80, .96] | +.67 |
| **stampscore excess p4 (whole image)** | **.952 [.89, .99]** | +.77 |
| stampscore absolute p4 (no reference) | .711 [.57, .84] | +.37 |
| SIFID to reference | .920 [.84, .98] | +.73 |
| LPIPS to reference | .899 [.81, .97] | +.69 |
| −CLIP-IQA | .713 [.58, .83] | +.37 |
| −MUSIQ | .576 [.42, .72] | +.13 |

Counting unsure as stamped or as not moves the stampscore AUCs by at most .04.

**Paired bootstrap, crop p4 vs others:**
- p4 vs LPIPS: +.017 [−.08, +.11];
- p4 vs SIFID: −.004 [−.10, +.08];
- p4 vs grid: +.028 [−.01, +.08];
- p4 beats CLIP-IQA by +.20 and MUSIQ by +.34 (both significant).

**Operating points on the labels** (crop p4 excess):

| threshold | stamped detected | not-stamped flagged |
|---|---|---|
| > .005 | 81% | 6% |
| > .01 | 69% | 3% |
| > .02 | 47% | 0% |

**What the labels show:**
1. **Blends are perceptually stamped.** Felix marked 14/15 blends stamped, as often as DMD. stampscore ranks blends
   between clean and DMD (median image excess .010 vs DMD .069). It under-rates mild stamping relative to a human,
   though it still ranks it. This is the main calibration correction: a blend excess of +.005–.02 already means
   visible stamping.
2. **The labels can't separate stamping from drift.** Every confident DMD/blend crop is stamped (28/28), and only 4/36
   clean/reference crops are. So "stamped" coincides with "far from the reference", and LPIPS/SIFID tie stampscore
   here. Each carries information the others lack:
   - partial rank correlation of label with stampscore given LPIPS: .59; with LPIPS given stampscore: .54;
   - with SIFID, the corresponding values are .58 and .59.

   The specificity argument rests on §6: noise and blur move LPIPS/SIFID more than stamps do, and they don't move
   stampscore. One crop shows it directly: c14 (E_oracle463 bread), labelled stamped, has LPIPS .08 (nearly the
   reference) but stampscore +.027.
3. **Crop vs image.** Whole-image excess (.95) beats crop excess (.92). Several blend misses at crop level register on
   the whole image (c29 +.005 → +.046, c40 +.004 → +.023, c30 −.003 → +.018): the stamping sits elsewhere in the
   frame.
4. **Remaining misses, inspected:**
   - c41 (T6S2d moss s101) and c60 (l20 moss): fine, mildly repetitive lichen dots at ≤ +.005;
   - c76 (l20 bark): repetition at the bark-plate scale, larger than a 14-px token;
   - c45: the H_ref_d bread-crumb reference itself was labelled stamped, so its excess is 0 by construction.
5. **False alarm.** c04 (H_s2F4, materials 17) is a knit-wool swatch: legitimate regularity on a non-admitted
   material. Crop +.016; the whole image is −.019.
6. **Catch trials.** Both synthetic stamp catches got "unsure", not "stamped": the pasted-motif test images don't look
   like generator stamping to a human, so they test the detector but not perception. Both heavy blurs also got
   "unsure", and the guard flags them.
7. **Real photos.** All 6 were labelled not stamped (abs dup .000–.027).

**Round 2: severity (rated 2026-10-23; results in §10).** The page is https://hosts-mac-mini.tail6d2520.ts.net:8791/stamp-severity/.
- 80 new 512-px crops: the same slots and methods at a new location (≤ 23% overlap with round 1), plus 6 real photos,
  with no synthetic catches.
- 18 DMD crops: the E_akh4 s202 renders now exist.
- Felix rates each crop 0 none / 1 mild / 2 clear / 3 severe.
- Key: `labels/key_r2.json`. Scores: `out/stamp_crops_r2.json`, `out/std_crops_r2.json`.
- Analysis: `experiments/analyze_severity.py SEVERITY.json`. It reports Spearman/Kendall with bootstrap CIs; AUCs at
  ≥ 1, ≥ 2 and = 3; mean score per level; flag rates per level; and round-1 vs round-2 consistency on the same images.


## 10. Round 2: severity ratings and threshold recalibration (2026-10-23)

**Data.** Felix rated 80 new blind crops 0–3: 39 at 0, 14 at 1, 14 at 2, 13 at 3. Each is the same image at a new
location (≤ 23% overlap with round 1); there are no synthetic catches. Raw ratings: `labels/felix_severity_raw.json`;
output: `out/severity_analysis.md`, `out/severity_results.json`, `out/severity_threshold_sweep.json`.

**By hidden group** (mean severity): real .00, reference .23, clean .25, **blend 1.87**, DMD 2.39. By run:
- E_quality / E_akh4 2.6, M6F2_l50 2.4, H_quality 2.1, ramp 1.8, l20 1.4;
- oracle463 and T6S2d .6;
- v2_r2k, s3F4 and E_ref_d .0.

**Consistency with round 1** (same image, other crop):
- "stamped" in round 1 → round 2 severity 0/1/2/3 = 1/7/11/13;
- "not" in round 1 → 28/4/0/0.

The ratings are consistent, and stamping is spread across the frame.

**Agreement with severity** (74 generated crops; Spearman [95% boot]; mean per level 0 / 1 / 2 / 3):

| metric | ρ | AUC 3 vs ≤ 2 | means |
|---|---|---|---|
| stampscore crop p4, t .85 (old default) | +.62 [.44, .76] | .922 | .000 / .024 / .017 / .064 |
| stampscore crop p9, t .85 | +.66 | .919 | |
| stampscore whole image p4, t .85 | +.73 [.57, .85] | .893 | |
| **stampscore crop p4, t .80 (new default)** | **+.76** | **.942** | −.005 / .050 / .085 / .199 |
| stampscore crop p4, t .75 | +.78 | .916 | −.007 / .076 / .184 / .299 |
| LPIPS to reference | +.67 [.52, .77] | .817 | .160 / .217 / .459 / .439 |
| SIFID to reference | +.70 [.58, .78] | .781 | .294 / .675 / 1.662 / 1.577 |
| −CLIP-IQA | +.42 | .734 | |
| −MUSIQ | +.13 | .590 | |

**What it shows:**
1. **Severity, not just presence.** At t .85, stampscore already separates severe from clear better than the drift
   metrics. Paired bootstrap ΔAUC(3 vs ≤ 2) vs LPIPS: +.10 [.00, .22], P = .03; vs SIFID: +.14 [.03, .25], P = .003.
   LPIPS and SIFID plateau or fall between 2 and 3: they measure how far the image is from the reference, not how
   repetitive it is. This is the label-based evidence for specificity that round 1 could not give.
2. **The .85 threshold was too strict for mild stamping.** The level-2 misses (s15, s21, s29, s39, s72; mostly ramp and
   l20 blends) are many *similar but not identical* rosettes or flour lace, with cosines of .75–.85. At t .85 the mean
   score is not monotone (level 1 .024 > level 2 .017).
3. **Recalibration, checked on data not used to choose it:**

   | check | t .85 | **t .80** | t .75 |
   |---|---|---|---|
   | DMD vs clean, 107 images (AUC) | .984 | **.994** | .984 |
   | blend vs clean (AUC) | .887 | **.932** | .932 |
   | jittered synthetic stamps detected | 76% | **81%** | 62% |
   | noise sd 6 (excess) | +.001 | +.006 | +.008 |
   | admitted real photos, absolute dup median / max | .003 / .12 | .033 / .26 | .165 / .45 |
   | Felix round 1 (crop AUC) | .917 | .927 | .941 |
   | Felix round 2 (severity ρ) | .62 | .76 | .78 |

   t .75 fits the severity ratings best, but it starts responding to noise and natural texture. **t .80 is the new
   default.**
   - At t .80 vs LPIPS / SIFID on severity: Δρ +.09 [−.06, .24] and +.06 [−.09, .21]. Better, not significantly.
   - The cross-family replication and `rec_phase.py` keep .85; `--threshold 0.85` reproduces them.
4. **New flags** (t .80, crop excess):

   | flag | severity 0 | 1 | 2 | 3 |
   |---|---|---|---|---|
   | `stamping` (> .02) | 12% | 50% | 86% | 100% |
   | `severe stamping` (> .10) | 3% | 21% | 36% | 85% |

   Whole images at > .02: DMD 100%, blend 87%, clean 28%. Felix rated oracle463 and T6S2d .6 on average, so part of
   the clean rate is real.
5. **Remaining disagreements, inspected:**
   - s58 (H_quality, deer in fog): severity 0 but +.10. A low-texture crop; a few trunk and antler tokens dominate.
   - s08 (materials 17 swatches): severity 1 but +.21. The burlap weave is legitimate regularity (a non-admitted
     material), although the cork granules do look stamped.

**Caveats.** One rater. t .80 was chosen after seeing round 2, so the independent checks (rows 1–6) carry the weight.
The image-level tables in §§1–6 are at t .85.

## 11. Limits

- **Relative instrument.** Use it as excess over a same-seed, same-size reference on organic, stochastic materials.
  Woven or legitimately regular content isn't covered, and crumpled paper didn't separate in Phase 2a.
- **Soft stamps are invisible.** A 50%-alpha blended motif doesn't register in any phase mode. Partial,
  low-contrast repetition is a blind spot.
- **Large, sparse motifs** cover few tokens, so the per-image fraction stays small (seed-202 rosettes: improved, not
  solved).
- **Intermediate states.** The guard removes the gross cases; the residual inflation is up to +.045.
- **Size-dependent.** Never resize. The CLI centre-crops with `--max-side`.
- **Small dataset.** 16 DMD images from 3 adapters on one model family. The cross-family replication running on
  RunPod is the needed out-of-family test.
- **One labeller, binary labels.** Severity varied (Felix: "some stampings were more severe than others"), but the
  page recorded only stamped / not / unsure.

## Files

- `stampscore/`: package (`core.py`, `cli.py`); `README.md`: method, calibration, limits.
- `experiments/`:
  - `dataset.py`, `cache.py`, `synth.py`;
  - `metrics_std.py`, `crop_scores.py`, `make_crops.py`;
  - `analyze.py`, `analyze_labels.py`, `stats.py`.
- `out/`: caches, `analysis.md`, `results.json`, `std_*.json`, `stamp_*.json`.
- `labels/key.json`: sealed crop key.
- `.venv/`: torch 2.14.1, transformers, lpips, pyiqa, scipy. Separate from `venvs/ref`, which was not modified.

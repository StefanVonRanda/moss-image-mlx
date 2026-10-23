# M6b: revamped plan (2026-10-19, 22:30 UTC)

**Goal:** the best MOSS-Image turbo at 1024² in ≤ ~115 s end to end on the M4 Pro, decided by a pre-registered protocol.

**Keep from M6:**
- the engine and queue;
- the incumbent `quality` (control);
- the T6S2 hybrid (candidate);
- the RunPod prefix-distillation pilot (running; ~$5);
- destripe;
- literature conclusions (LOG.md).

## Phase 0: freeze the protocol (CPU, now). Nothing gets promoted without it.

**Evaluation set E** (`prompts.EVAL`), 18 prompts, seeds 101 and 202, at 1024²:

| category | prompts |
|---|---|
| repetition and material | tex_bread, beard, tex_portrait, u_moss_wall, u_bark, oak, clouds, u_paper |
| idealization | leather, knit_hands, skin_freckles |
| structure | u_crowd, compose |
| text | en_sign, zh_note, u_menu, u_infographic |
| legitimate regularity (control) | u_tweed |

**References:** 50-step + CFG 3 on the bf16 official weights, for all of E at seed 101 and a 6-prompt subset at 202 (noise floor).

**Report card** (`tq/reportcard.py`), per candidate run:
- **text:** exact-string correctness per line (manual verify sheet) and CER (Apple Vision; a second OCR engine if one is cheaply available);
- **texture:** calibrated texture/recurrence statistics vs the reference and vs real photos (Phase 2);
- **artifacts:** banding score;
- **cost:** wall-clock and passes;
- **visual:** reference-anchored blind sheets (reference shown, labelled; candidates shuffled).

**Decision rule** (written before the finalist runs). C replaces the incumbent if, on E:
- **(a) text:** no line that was correct in the incumbent becomes wrong (manually verified), summed over both seeds;
- **(b) preference:** reference-anchored pairwise preference for C ≥ 60% of non-tie pairs on the repetition, material and idealization prompts. Raters: the VLM judge plus me (blind); the user's votes, when given, override;
- **(c) defects:** no new structural defect class (ghosting, malformed hands/letters/objects) in more than 1 of 36 images;
- **(d) cost:** mean end-to-end ≤ 115 s at 1024² on the tier we ship.

## Phase 1: confounds and noise floor (GPU, starts now)

- **Precision:** fast and quality in bf16 vs q8 at 768² (m0495f–i, queued). If bf16 stamps visibly less, all finalists run on bf16; the pilot is evaluated on both.
- **Noise floor:** references at seeds 101 and 202 (subset); the reference-vs-reference distance sets the "no difference" band for every metric.

## Phase 2: mechanism, measured directly, and calibration against real photos

- **Latent mechanism test** (`latent_hf.py`): along real trajectories (quality, T6S2, fast, pilot), measure the high-frequency energy of the clean estimate x̂₀ per σ for the student vs the guided teacher evaluated at the same states.
  - **Prediction:** the student's excess is largest at high σ and grows with each re-application.
  - **Falsifier:** no excess, or no growth. That would kill the dose-response story.
- **Real-photo calibration set:** ~4 CC0 or Wikimedia photos each of bread crust, lichen on stone, pine bark, grey beard close-up, crumpled paper, herringbone tweed (legitimate regularity).
  - A metric is admitted to the report card only if real-photo and 50-step-reference values overlap while stamped versions (fast) fall outside.
  - Otherwise it is dropped (the DINO dup_frac is on probation).
- **Read the adapters' training code** (akhaliq `train_dmd.py`, darask0 `code/`): the trained timesteps, sampler, captions and CFG in the teacher score. Record their true operating conditions.

## Phase 3: baselines never run

- akhaliq 4-step v2 (Euler-native, pinned `7258a71`) on its own recipe (4 steps, strength 1, CFG 1), and as a 2-step finisher after the T6 teacher;
- lodestones lora (non-latest) as the finisher.

## Phase 4: pilot evaluation (when the pod finishes, ~00:15, jumps the queue)

- **Parity check first:** MLX renders of student6 + finisher vs the pod's PyTorch renders, same prompts and seed.
- **Then the report card on E (seed 101 first):** prefix LoRA (8 passes) vs quality vs T6S2d, on bf16 and q8.
- **Latent test:** does the regression-trained prefix carry teacher-like clean estimates at high σ (no excess HF)?

## Phase 5: finalists at 1024² × 2 seeds on E, then the decision

- **Finalists:** at most 3 among T6S2d (with the best teacher levers: skip and CFG from the m0497 triage, confirmed at 1024²), pilot+finish, and any Phase 3 surprise.
- **External raters:**
  - local Qwen-VL judge through the accelerator lock, pairwise, reference-anchored, with randomised order and both orders scored, calibrated on 10 pairs with known answers (e.g., stamped vs unstamped crops);
  - a blind rating page for the user (and anyone they share it with): an Artifact with a shared database, built when the finalists exist.
- **Apply the decision rule. Report:**
  - raw counts per category;
  - residual failures named per image;
  - the cost / quality frontier (passes, seconds, memory).

## Phase 6: package and next step

- **Package:** ship the winner as a `--turbo` profile (bf16 or q8 per Phase 1), with timing measured end to end.
- **Training follow-up:** a proposal only if the evidence says a trained component is the bottleneck (e.g., DP-DMD-style finisher retrain on real handoff states, or a larger prefix run at 1024²). Each needs a separate go-ahead for spend.

## Explicitly deprioritised

- Rear-block attenuation and direction surgery (closed, caveats noted).
- More 768² triage beyond what is queued.
- Fill-in jobs m08 and m09 (move to the end).

# Training proposal (not started; needs a separate go-ahead for paid compute)

**Status:** prepared on 2026-10-19 during M6. No RunPod spend has happened. Local work continues without it.

## Why train at all

**T6S2** (guided original model, DPM-Solver++(2M), 6 evals to σ .463, then the turbo adapter for the last 2 steps) gives:
- the original's material morphology;
- the student's crisp text;
- ~101 s at 1024², within budget.

The only thing training would buy is **the same picture faster**, which would fund either more fidelity or more speed. 12 of its 14 passes are the guided teacher on the high-noise half.

## Proposal P1: distil the guided high-noise half by regression (not DMD)

- **Student:** SFT with a new LoRA (rank 64, all 6 projections, blocks 0–41).
  - Maps (noise z, prompt) to the T6 state x(σ=.463) in **3 CFG-free steps** on a fixed grid (1 → .871 → .671 → .463).
  - The existing lodestones finish (2 steps) stays unchanged.
- **Target:** the exact T6 teacher state: 2M, 6 guided evals with the block-22 skip.
  - It's a deterministic function of (z, prompt), so the objective is plain trajectory regression (progressive/consistency-style), not distribution matching.
  - The night's evidence is that DMD's mode-seeking is what manufactures the stamped textures; regression to the teacher's own ODE map has no incentive to stamp. Its risk is the opposite one, blur, and that's measurable.
- **Loss:**
  - latent MSE on the final state x(.463), plus MSE on each intermediate state against the teacher's state at the same σ (trajectory-matched);
  - optionally a small weight on the clean-estimate MSE.
  - Train on the learned prefix's own states too, so the second and third steps see their own errors (the brief's point about compounding).
- **Inference after training:** 3 + 2 = **5 passes**, ~37 s at 1024². Or keep 6 teacher-like steps for fidelity at ~45 s.

## Pilot

| item | pilot | full |
|---|---|---|
| resolution | 512² | 1024² |
| prompts | 1,000 (mix: text 25%, materials 50%, scenes 25%; none from tq/prompts.py) | 5,000 |
| teacher targets | 1,000 × 12 passes | 5,000 × 12 passes |
| training steps | 2,000 (batch 4) | 10,000 (batch 4) |
| hardware | 1× H100 80 GB (RunPod) | 1× H100 80 GB |

**Cost estimates:**
- H100: ~0.16 s per 1024² forward pass (upstream's H100 timing of 16.2 s per 100-forward image, consistent with ~37× this Mac), ~0.04 s at 512².
- Targets: pilot ~1,000 × 12 × 0.04 s ≈ 8 min; full ~5,000 × 12 × 0.16 s ≈ 2.7 h.
- Training (3 student forwards + backward ≈ 3× forward each): pilot ~2,000 × 4 × 3 × 0.12 s ≈ 50 min; full ~10,000 × 4 × 3 × 0.48 s ≈ 16 h.
  - Full training is the dominant cost. Reduce it with gradient checkpointing and fewer steps if the pilot converges early.
- **Expected cost** at ~$3/h: pilot **~$5**, full **~$30–60**. A cost guard deletes the pod at completion or a deadline (existing RunPod tooling and memory note).

**Stopping criteria (pilot):**
1. On 32 held-out prompts, latent MSE to the teacher state at σ .463 is below Euler-T6's (the cheapest no-training alternative).
2. Decoded finals after the student finish keep the hybrid's properties:
   - repetition excess ≤ T6S2's;
   - OCR at least quality's on the text prompts;
   - no haze (in-focus fine/mid within 0.05 of T6S2).
3. Blind A/B against T6S2 on 16 unseen prompts: not worse on more than 4.

Fail any of these and stop; the report records why.

**What this does not address:** the student's last 2 steps still finish micro-texture. If the residual smoothness of pores and fibres matters more than speed, a second, separate pilot would fine-tune only the finish adapter's rear blocks against teacher continuations from σ .463. Same budget class, separate decision.

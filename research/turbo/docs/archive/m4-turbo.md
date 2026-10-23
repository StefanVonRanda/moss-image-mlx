# M4: turbo (2026-10-19)

Measured on the M4 Pro (48 GB, MLX 0.32.4) under the heavy-accelerator lock and the free-memory guard. Code: `repo/` commit `652a45b` (0.2.0, local, not yet pushed). Raw data: `parity/*turbo*`, `quant/turbo_*`, `release/turbo/`.

## What shipped

`moss-mlx --turbo` runs the lodestones 8-step DMD adapter for MOSS-Image-SFT:
- **Pinned:** revision `f95df83`, file `lora_latest.safetensors`, Apache-2.0.
- **Recipe:** exactly as its demo Space runs it (`hugging-apps/moss-image-8-step-dmd` @ `2ff429b`):
  - stock transformer, adapter unmerged at strength 0.7;
  - release scheduler, 8 steps;
  - no CFG;
  - CPU-generator noise.
- **Not used: the data-space sampler.** The PLAN assumed its `dataspace.safetensors` Q mix and step codes. The model card now marks that section "[Deprecated]", and the demo no longer uses it.

Also new:
- **`--lora FILE@STRENGTH`:** any kohya-format DiT adapter, applied unmerged (`lora.py`). It works over bf16, fp32 and quantized linears. The fp32 file is cast one module at a time, so the full fp32 copy never sits beside the DiT.
- **`stage2.py --lora/--steps/--guidance`:** the eval harness supports turbo runs.

## Parity (512², 8 steps, seed 42, W1 embeddings from torch)

| Comparison | PSNR |
|---|---|
| MLX fp32 vs torch fp32 | **61.5 dB** (latent cosine 0.99999): implementation exact |
| MLX bf16 vs torch fp32 | 30.0 dB |
| torch bf16 (the demo's own precision) vs torch fp32 | 27.7 dB |
| MLX bf16 vs torch bf16 | 28.1 dB |

The bf16 gap is rounding. Our bf16 path sits closer to the fp32 answer than the reference's own bf16 run.

## Speed and memory (1024², 8bit tier)

| Mode | Per image (model resident) | CLI end to end | Peak footprint |
|---|---|---|---|
| 50 steps, CFG 3 | 743 s | ~12½ min | 9.31 GB |
| **turbo, 8 steps** | **66.8–71 s** (8.4 s per step) | **72 s cold** (encoder 2.7 s, decode 2.1 s) | **9.18 GB** cold; 7.48 GB with a cached prompt |
| turbo, 4 steps | 34.4–36.6 s | — | — |

The 768² photo prompts take 35 s at 8 steps and 18 s at 4 steps. The batch harness peaked at 10.1–10.8 GB. That was because of the fp32 LoRA buffers, fixed before the CLI runs, plus multi-prompt state.

## Quality (14-image stage-2 set vs the 50-step 8-bit renders, `quant/turbo_scores.json`)

| | OCR all / EN / ZH | Layout IoU | Texture, photo set |
|---|---|---|---|
| 50 steps, 8bit (D8T8) | 0.908 / 0.992 / 0.825 | 0.850 | 1.02 |
| **turbo 8 steps** | **0.921 / 0.992 / 0.850** (per-prompt identical; zh_sign 0.4 vs 0.3 is Vision noise on vertical calligraphy) | 0.865 | 0.49 |
| turbo 4 steps | 0.869 / 0.986 / 0.752 (handwritten zh_note: 0.61) | 0.840 | 0.54 |
| 8 steps, `lora.safetensors` @ 1.0 (photos only) | — | — | 0.56 |
| 8 steps, `lora_latest` @ 0.85 (photos only) | — | — | 0.37 |

**Visual check (portrait and mug):**
- **Different compositions:** turbo draws a different picture for the same seed (no CFG, a different trajectory).
- **Smoother fine detail:** skin pores, crease depth and glaze speckle are less crisp, the usual DMD polish. No grain.
- **Spurious text:** the 8-step portrait added a cap reading "ANEAS". darask0's 4-step card reports the same failure ("extra titles").

**Decision:**
- **Turbo is opt-in.** The default stays the full 50-step model.
- **8 steps is the turbo default.** 4 steps is documented, with the small-handwriting caveat.
- The other adapter settings didn't improve detail, so we keep the demo's own recommended setting, which also lets users cross-check against the Space.

## Open

- **Publish 0.2.0:** push to GitHub, tag `v0.2.0` for PyPI, update the HF cards. This waits for the user's go-ahead.
- **Not done in M4:**
  - DeBand (SFT banding; needs the Base DiT, which we have);
  - darask0's 4-step adapter;
  - a turbo × compact tier check. Expected to work, since compact uses the same 8-bit DiT, but it hasn't been run.
- **Possible detail recovery:** a CFG-free "two-stage" approach is untested. For example, turbo for the first steps, then a few full-model steps.

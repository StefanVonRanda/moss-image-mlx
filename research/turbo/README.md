# MOSS-Image turbo: research notes

How the `--turbo` sampler in [moss-image-mlx](../../README.md) was found, built and tested. It renders a ~1 MP [MOSS-Image](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT) image in **15 network passes instead of 100: about 2 minutes instead of 12½ on an M4 Pro (6.2×)**. It stays close to the image the full 50-step model draws for the same seed.

**Use it:** `pip install moss-image-mlx`, then `moss-mlx "your prompt" --turbo`. Weights: [VisualInference/MOSS-Image-SFT-Turbo-MLX](https://huggingface.co/VisualInference/MOSS-Image-SFT-Turbo-MLX). This folder holds the documentation, data summaries and figures; the sampler itself is [`moss_image_mlx/turbo.py`](../../moss_image_mlx/turbo.py).

![Held-out comparison](images/heldout_p1.jpg)
*Prompts never used in training. Each row: the full model at 50 steps | `--turbo` (15 passes) | the earlier turbo built on a community DMD adapter. Note the pebbles.*

## At a glance

| ~1 MP, M4 Pro, 8-bit, 32 held-out prompts | `--turbo` | Earlier DMD-adapter turbo | Full model, 50 steps |
|---|---|---|---|
| Network passes | **15** | 10 | 100 |
| Time per image | **~119 s** | ~80 s | ~743 s |
| Repeated-texture ("stamping") excess vs the full model | **−0.006** | +0.063 | 0 |
| Fine-detail difference | **−0.004** | −0.022 | 0 |
| Mean pixel difference from the full model's image | **0.090** | 0.124 | 0 |
| English text, OCR recall | 0.947 | 0.951 | 0.934 |

Training cost about $37.50 of rented GPU: 1,983 teacher trajectories plus 2,000 training steps.

## What was done

1. **Named the defect.** Community few-step adapters for MOSS-Image (DMD-distilled) turn organic textures into repeated templates: lichen into identical rosettes, pebbles into beads, bread crust into a lattice. Measuring it took a dedicated score, a within-image near-duplicate count from DINOv2 patches compared against the full model's own image. FID and LPIPS don't see it.
2. **Located it with oracles before training anything.** We cached the full model's exact intermediate states and spliced other samplers onto them. Two things showed:
   - **Where stamping comes from:** it grows with every distilled step taken in the middle of the trajectory. With 2, 3 and 6 such steps, bread's repetition excess goes 0.011 → 0.017 → 0.055, even from a perfect starting state.
   - **The finish:** the *undistilled* model finishes faithfully in 3–4 solver steps, 4.6× closer to the reference than a distilled finish.
3. **Built only what was needed.** The released design has three parts:
   - **A student for the first half:** a small rank-64 LoRA, trained by regression onto the full model's own per-seed trajectory, not by distribution matching. It covers the first half of the trajectory in 6 steps, without CFG.
   - **One guided full-model step** at the position where lettering forms, found by testing all six positions.
   - **The full model's own 4-step finish.**
4. **Fixed ghosting at the source.** Training the student on its own drifted states with "jump back to the stored path" targets makes it average two images (double exposures). Relabelling those states with the teacher's own continuation fixed it, in a controlled two-arm ablation. A related measurement: a *coherent* error stays clean at twice the size of an *averaged* error that already ghosts.
5. **Shipped it bit-exact.** The packaged MLX sampler reproduces the research images pixel for pixel, at a 9.4 GB peak.
6. **Tested whether the finding generalizes, and it doesn't.** In a pre-registered test on SDXL/DMD2, Qwen-Image/Lightning and FLUX.1-dev/Turbo-Alpha, the distilled students did not repeat texture more than their teachers. Stamping is a property of MOSS-Image's community DMD adapters, not of distillation in general ([details](docs/cross-family-test.md)).

## Read next

| Document | Read it for |
|---|---|
| [**Interim write-up**](docs/interim-writeup.md) | **Start here.** The whole story for technical readers outside image generation: a short primer, the defects shown and annotated, the oracle experiments, the design, results, latest findings and honest limits |
| [Technical write-up](docs/technical-writeup.md) | The release, decision by decision: what each choice fixed, how much it contributed, how well isolated that evidence is, and what failed |
| [How the numbers work](docs/how-the-numbers-work.md) | Every metric, with formulas, and where each can mislead |
| [Cross-family test](docs/cross-family-test.md) | The pre-registered generality test and its negative result |
| [**stampscore**](../stampscore/README.md) | The stamping score itself: code, validation against blind human labels, the labelled crops, and its limits |
| [Results and decisions](docs/results-and-decisions.md) | Result tables, comparison sheets and the decisions taken at each stage |
| [Prompt specification](docs/prompt-spec.md) | How the 2,000 training prompts and the held-out sets were designed |
| [Research log](docs/research-log.md) | The full chronological lab notebook, dead ends and mistakes included |
| [Archive](docs/archive/) | Earlier phase reports, plans and training proposals |

## Known limits

- **Regular geometry:** legitimately regular structure, such as a chain-link fence shadow or rows of washing machines, comes out too organic.
- **Small, dense text** slips more often than with the full model.
- **Composition** follows the full model's image for the same seed approximately, not exactly.
- **Evidence scale:** most comparisons use one model, 1–2 seeds and 7–32 prompts; visual judgements were mostly unblinded.
- **Not yet public:** the training code and prompt files.

## Credits and licenses

- **Base model:** MOSS-Image SFT by OpenMOSS (Apache-2.0).
- **Destripe LoRA:** [nynxz/MOSS-Image](https://huggingface.co/nynxz/MOSS-Image) (Apache-2.0).
- **Community turbo adapters evaluated:** [lodestones/MOSS-Image-SFT-8-steps-dmd](https://huggingface.co/lodestones/MOSS-Image-SFT-8-steps-dmd) and [akhaliq/MOSS-Image-SFT-4-steps](https://huggingface.co/akhaliq/MOSS-Image-SFT-4-steps).
- **Metrics:** DINOv2 (Meta) for the repetition score; Apple Vision for OCR.
- **Prior work** is discussed in the write-ups.

Documentation and images in this folder: CC BY 4.0 ([LICENSE](LICENSE)). The code in this repository and the student weights: Apache-2.0.

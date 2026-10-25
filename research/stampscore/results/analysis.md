## Reproduction
 {'n': 74, 'max_abs_diff_grid': 0.0, 'max_abs_diff_p4': 0.0} 

## Separation by phase mode (small, L6, t .85, d 56)

| phases | AUC dmd vs clean [95% boot] | AUC blend vs clean | median dmd | median blend | median clean |
|---|---|---|---|---|---|
| grid | 0.977 [0.93, 1.00] | 0.884 | 0.037 | 0.005 | -0.001 |
| p4 | 0.984 [0.95, 1.00] | 0.887 | 0.069 | 0.010 | -0.002 |
| p9 | 0.985 [0.95, 1.00] | 0.907 | 0.085 | 0.014 | -0.003 |

held-out pairs H_quality > H_s2F4: {'grid': 8, 'p4': 8, 'p9': 8} / 8

## Synthetic perturbations (excess over the unperturbed base; mean over bases)

| variant | grid | p4 | p9 |
|---|---|---|---|
| aligned | +0.029 (95%) | +0.025 (90%) | +0.024 (90%) |
| unaligned | +0.022 (100%) | +0.024 (95%) | +0.022 (90%) |
| jitter | +0.003 (38%) | +0.011 (76%) | +0.016 (86%) |
| soft | -0.002 (0%) | -0.001 (0%) | -0.002 (0%) |
| noise6 | +0.000 (19%) | +0.001 (14%) | +0.002 (14%) |
| blur1.5 | -0.004 (14%) | -0.004 (14%) | -0.004 (14%) |
| blur3 | +0.011 (43%) | +0.014 (67%) | +0.016 (76%) |
| blur6 | +0.065 (90%) | +0.083 (100%) | +0.088 (100%) |

(% = share of the 21 bases with excess > .005)

## Phase invariance: |dup(shift k) - dup(shift 0)|, k = 3, 7, 10 px

| phases | mean abs dev | max abs dev | median rel dev |
|---|---|---|---|
| grid | 0.003 | 0.021 | 21% |
| p4 | 0.002 | 0.022 | 15% |
| p9 | 0.002 | 0.020 | 10% |

## Detail guard (mid-band ratio < 0.6 -> low_detail; fine/mid ratio > 1.6 -> decoder_lattice)

| image | mid ratio | fine/mid ratio | flagged | dup p4 excess | probe |
|---|---|---|---|---|---|
| generated: ref | 0.98-1.01 | 0.97-1.06 | 0 |  |  |
| generated: dmd | 0.69-1.35 | 0.63-1.01 | 0 |  |  |
| generated: blend | 1.01-1.26 | 0.84-1.30 | 0 |  |  |
| generated: clean | 0.69-1.14 | 0.88-1.46 | 0 |  |  |
| synthetic noise6 | 0.99-1.01 | 1.00-1.33 | 0/21 | +0.001 |  |
| synthetic blur1.5 | 0.54-0.75 | 0.13-0.43 | 1/21 | -0.004 |  |
| synthetic blur3 | 0.26-0.46 | 0.06-0.27 | 21/21 | +0.014 |  |
| synthetic blur6 | 0.09-0.21 | 0.06-0.25 | 21/21 | +0.083 |  |
| synthetic aligned | 0.99-1.15 | 0.96-1.04 | 0/21 | +0.025 |  |
| synthetic jitter | 0.99-1.15 | 0.96-1.04 | 0/21 | +0.011 |  |
| probe step0 σ1.000_student | 0.11 | 4.61 | flagged | +0.211 | tex_bread |
| probe step0 σ1.000_teacher | 0.14 | 9.36 | flagged | +0.476 | tex_bread |
| probe step1 σ0.968_student | 0.31 | 3.03 | flagged | +0.097 | tex_bread |
| probe step1 σ0.968_teacher | 0.25 | 4.54 | flagged | +0.138 | tex_bread |
| probe step2 σ0.926_student | 0.51 | 2.08 | flagged | -0.004 | tex_bread |
| probe step2 σ0.926_teacher | 0.20 | 3.65 | flagged | +0.108 | tex_bread |
| probe step3 σ0.871_student | 0.66 | 1.52 | scored | -0.012 | tex_bread |
| probe step3 σ0.871_teacher | 0.29 | 3.28 | flagged | +0.022 | tex_bread |
| probe step4 σ0.791_student | 0.78 | 1.29 | scored | +0.004 | tex_bread |
| probe step4 σ0.791_teacher | 0.41 | 2.54 | flagged | -0.018 | tex_bread |
| probe step5 σ0.671_student | 0.89 | 1.12 | scored | +0.009 | tex_bread |
| probe step5 σ0.671_teacher | 0.63 | 1.65 | flagged | -0.012 | tex_bread |
| probe step6 σ0.463_student | 0.99 | 1.01 | scored | +0.002 | tex_bread |
| probe step6 σ0.463_teacher | 0.89 | 1.15 | scored | -0.010 | tex_bread |
| probe step7 σ0.024_student | 1.00 | 1.00 | scored | +0.000 | tex_bread |
| probe step7 σ0.024_teacher | 1.00 | 1.00 | scored | +0.001 | tex_bread |
| probe step0 σ1.000_student | 0.16 | 4.98 | flagged | -0.110 | tex_portr |
| probe step0 σ1.000_teacher | 0.06 | 7.55 | flagged | +0.055 | tex_portr |
| probe step1 σ0.968_student | 0.64 | 1.60 | flagged | +0.077 | tex_portr |
| probe step1 σ0.968_teacher | 0.33 | 6.00 | flagged | -0.042 | tex_portr |
| probe step2 σ0.926_student | 0.78 | 1.35 | scored | +0.027 | tex_portr |
| probe step2 σ0.926_teacher | 0.75 | 1.72 | flagged | +0.129 | tex_portr |
| probe step3 σ0.871_student | 0.86 | 1.22 | scored | +0.026 | tex_portr |
| probe step3 σ0.871_teacher | 0.64 | 1.83 | flagged | +0.048 | tex_portr |
| probe step4 σ0.791_student | 0.89 | 1.18 | scored | +0.011 | tex_portr |
| probe step4 σ0.791_teacher | 0.72 | 1.57 | scored | +0.045 | tex_portr |
| probe step5 σ0.671_student | 0.93 | 1.09 | scored | +0.010 | tex_portr |
| probe step5 σ0.671_teacher | 0.81 | 1.35 | scored | +0.027 | tex_portr |
| probe step6 σ0.463_student | 1.00 | 1.00 | scored | -0.000 | tex_portr |
| probe step6 σ0.463_teacher | 0.93 | 1.13 | scored | +0.005 | tex_portr |
| probe step7 σ0.024_student | 1.00 | 1.00 | scored | +0.000 | tex_portr |
| probe step7 σ0.024_teacher | 1.00 | 1.00 | scored | -0.000 | tex_portr |

generated images flagged: [] 

## Sensitivity (p4): AUC dmd vs clean / blend vs clean; real-photo max absolute dup (admitted)

| net | layer | t | AUC dmd | AUC blend | med dmd | med clean | real max dup |
|---|---|---|---|---|---|---|---|
| dinov2-small | 3 | 0.8 | 0.898 | 0.699 | 0.119 | -0.006 | 0.125 |
| dinov2-small | 3 | 0.85 | 0.923 | 0.693 | 0.020 | -0.001 | 0.012 |
| dinov2-small | 3 | 0.9 | 0.821 | 0.571 | 0.001 | -0.000 | 0.000 |
| dinov2-small | 6 | 0.8 | 0.994 | 0.932 | 0.202 | -0.009 | 0.261 |
| dinov2-small | 6 | 0.85 | 0.984 | 0.887 | 0.069 | -0.002 | 0.120 |
| dinov2-small | 6 | 0.9 | 0.828 | 0.560 | 0.003 | -0.000 | 0.018 |
| dinov2-small | 9 | 0.8 | 0.958 | 0.685 | 0.155 | -0.010 | 0.505 |
| dinov2-small | 9 | 0.85 | 0.967 | 0.744 | 0.136 | -0.006 | 0.271 |
| dinov2-small | 9 | 0.9 | 0.967 | 0.721 | 0.036 | 0.000 | 0.134 |
| dinov2-small | 12 | 0.8 | 0.609 | 0.349 | 0.001 | -0.000 | 0.999 |
| dinov2-small | 12 | 0.85 | 0.690 | 0.388 | 0.007 | 0.000 | 0.995 |
| dinov2-small | 12 | 0.9 | 0.766 | 0.490 | 0.037 | 0.000 | 0.959 |
| dinov2-base | 3 | 0.8 | 0.977 | 0.722 | 0.064 | -0.000 | 0.035 |
| dinov2-base | 3 | 0.85 | 0.927 | 0.564 | 0.010 | -0.000 | 0.003 |
| dinov2-base | 3 | 0.9 | 0.814 | 0.619 | 0.001 | 0.000 | 0.000 |
| dinov2-base | 6 | 0.8 | 0.994 | 0.904 | 0.063 | -0.001 | 0.141 |
| dinov2-base | 6 | 0.85 | 0.927 | 0.693 | 0.006 | -0.001 | 0.055 |
| dinov2-base | 6 | 0.9 | 0.616 | 0.461 | 0.000 | 0.000 | 0.004 |
| dinov2-base | 9 | 0.8 | 0.984 | 0.840 | 0.166 | -0.016 | 0.308 |
| dinov2-base | 9 | 0.85 | 0.984 | 0.854 | 0.076 | -0.006 | 0.187 |
| dinov2-base | 9 | 0.9 | 0.969 | 0.816 | 0.011 | -0.001 | 0.078 |
| dinov2-base | 12 | 0.8 | 0.769 | 0.588 | 0.006 | 0.000 | 0.995 |
| dinov2-base | 12 | 0.85 | 0.814 | 0.612 | 0.030 | -0.000 | 0.966 |
| dinov2-base | 12 | 0.9 | 0.818 | 0.693 | 0.072 | 0.005 | 0.801 |

| min dist (small L6 .85) | AUC dmd | AUC blend |
|---|---|---|
| 28 | 0.975 | 0.930 |
| 56 | 0.984 | 0.887 |
| 112 | 0.974 | 0.842 |

sensitivity summary: {'n_configs': 120, 'frac_auc_ge_0.95': 0.3, 'frac_auc_ge_0.9': 0.4583333333333333}
top 5: [('dinov2-base', 6, 56, 0.75, 1.0), ('dinov2-base', 6, 112, 0.75, 0.997), ('dinov2-small', 6, 56, 0.8, 0.994), ('dinov2-base', 6, 28, 0.75, 0.994), ('dinov2-base', 6, 56, 0.8, 0.994)] 

synth by layer (p4): {6: {'jitter_mean': 0.011418318381164413, 'noise_mean': 0.0009721283053522637}, 9: {'jitter_mean': -0.0017764493002576983, 'noise_mean': 0.011270716432826493}} 

## No-reference variant (absolute dup_frac, p4)

| group | min / median / max abs dup |
|---|---|
| ref | 0.001 / 0.013 / 0.330 |
| dmd | 0.012 / 0.083 / 0.486 |
| blend | 0.004 / 0.017 / 0.089 |
| clean | 0.001 / 0.014 / 0.311 |
| real_admitted | 0.000 / 0.003 / 0.120 |
| real_other | 0.003 / 0.017 / 0.178 |

{'auc_dmd_clean': (0.8590116279069767, 0.752906976744186, 0.9477107558139533), 'auc_blend_clean': 0.6155038759689923, 'threshold_max_real_or_ref': 0.32978133417841865, 'tpr_dmd': 0.0625, 'fpr_clean': 0.0} 

## Standard metrics

| metric | AUC dmd vs clean (worse=dmd) | AUC blend vs clean | median dmd | median blend | median clean |
|---|---|---|---|---|---|
| stampscore p4 (excess) | 0.984 | 0.887 | 0.069 | 0.010 | -0.002 |
| lpips | 0.868 | 0.887 | 0.446 | 0.417 | 0.230 |
| sifid | 0.862 | 0.904 | 0.969 | 1.035 | 0.218 |
| clipiqa | 0.680 | 0.634 | 0.641 | 0.658 | 0.717 |
| musiq | 0.555 | 0.552 | 66.968 | 67.589 | 67.986 |

Spearman vs stampscore excess: {'lpips': 0.5015092502434275, 'sifid': 0.5370740019474197, 'clipiqa': -0.21940116845180138, 'musiq': -0.09432814021421615} 

| variant | stampscore p4 | LPIPS | SIFID | Δ CLIP-IQA | Δ MUSIQ |
|---|---|---|---|---|---|
| aligned | +0.025 | +0.061 | +0.034 | -0.037 | -0.715 |
| unaligned | +0.024 | +0.060 | +0.033 | -0.040 | -0.668 |
| jitter | +0.011 | +0.060 | +0.030 | -0.040 | -0.711 |
| soft | -0.001 | +0.033 | +0.007 | -0.029 | -0.516 |
| noise6 | +0.001 | +0.071 | +0.281 | -0.001 | -2.632 |
| blur1.5 | -0.004 | +0.455 | +17.598 | -0.276 | -23.241 |
| blur3 | +0.014 | +0.694 | +29.319 | -0.335 | -42.192 |
| blur6 | +0.083 | +0.877 | +35.978 | -0.370 | -50.796 |

| run | group | images | patch-FID | patch-KID ×1e3 |
|---|---|---|---|---|
| E_M6F2_l20 | blend | 5 | 144.2 | 7.26 |
| E_M6F2_l50 | blend | 5 | 201.7 | 45.83 |
| E_M6F2_ramp | blend | 5 | 177.0 | 28.00 |
| E_T6S2d | clean | 5 | 129.8 | 3.63 |
| E_alloc_T2 | clean | 2 | 130.3 | -12.05 |
| E_oracle463 | clean | 5 | 63.3 | -5.54 |
| E_pilot | clean | 3 | 170.9 | 12.02 |
| E_s2F4 | clean | 2 | 112.6 | -15.57 |
| E_s3F4 | clean | 5 | 113.8 | -3.93 |
| E_v2_r2k | clean | 5 | 93.4 | -5.12 |
| H_s2F4 | clean | 8 | 75.7 | -6.99 |
| H_s3F4 | clean | 8 | 82.9 | -6.24 |
| E_akh4 | dmd | 3 | 230.3 | 44.66 |
| E_quality | dmd | 5 | 189.6 | 35.05 |
| H_quality | dmd | 8 | 153.8 | 6.48 |
| E_ref | ref | 5 | 23.9 | -13.30 |


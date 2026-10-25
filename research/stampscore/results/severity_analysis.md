## Severity by (hidden) group

| group | 0 | 1 | 2 | 3 | mean |
|---|---|---|---|---|---|
| real | 6 | 0 | 0 | 0 | 0.00 |
| ref | 10 | 3 | 0 | 0 | 0.23 |
| clean | 22 | 5 | 1 | 0 | 0.25 |
| blend | 0 | 4 | 9 | 2 | 1.87 |
| dmd | 1 | 2 | 4 | 11 | 2.39 |

mean severity by run: {'E_ref_d': 0.0, 'E_s3F4': 0.0, 'E_v2_r2k': 0.0, 'real': 0.0, 'H_s2F4': 0.12, 'H_ref_d': 0.38, 'E_oracle463': 0.6, 'E_T6S2d': 0.6, 'E_M6F2_l20': 1.4, 'E_M6F2_ramp': 1.8, 'H_quality': 2.12, 'E_M6F2_l50': 2.4, 'E_akh4': 2.6, 'E_quality': 2.6} 

## Agreement with severity (generated crops, n = 74)

| metric | Spearman [95% boot] | Kendall τb | AUC ≥1 vs 0 | AUC ≥2 vs ≤1 | AUC 3 vs ≤2 | mean at 0 / 1 / 2 / 3 |
|---|---|---|---|---|---|---|
| stampscore excess p4 (crop) | +0.62 [+0.44, +0.76] | +0.49 | 0.805 | 0.816 | 0.922 | 0.000 / 0.024 / 0.017 / 0.064 |
| stampscore excess p9 (crop) | +0.66 [+0.48, +0.78] | +0.52 | 0.840 | 0.819 | 0.919 | -0.001 / 0.027 / 0.021 / 0.075 |
| stampscore excess grid (crop) | +0.56 [+0.36, +0.71] | +0.43 | 0.765 | 0.779 | 0.913 | -0.001 / 0.019 / 0.009 / 0.038 |
| stampscore abs p4 (crop, no-ref) | +0.39 [+0.17, +0.59] | +0.31 | 0.672 | 0.696 | 0.820 | 0.061 / 0.071 / 0.029 / 0.085 |
| stampscore excess p4 (whole image) | +0.73 [+0.57, +0.85] | +0.60 | 0.888 | 0.872 | 0.893 | -0.001 / 0.022 / 0.023 / 0.064 |
| LPIPS to ref | +0.67 [+0.52, +0.77] | +0.51 | 0.821 | 0.928 | 0.817 | 0.160 / 0.217 / 0.459 / 0.439 |
| SIFID to ref | +0.70 [+0.58, +0.78] | +0.52 | 0.867 | 0.923 | 0.781 | 0.294 / 0.675 / 1.662 / 1.577 |
| -CLIP-IQA | +0.42 [+0.21, +0.60] | +0.33 | 0.715 | 0.732 | 0.734 | -0.704 / -0.659 / -0.601 / -0.543 |
| -MUSIQ | +0.13 [-0.10, +0.34] | +0.09 | 0.561 | 0.566 | 0.590 | -66.629 / -65.522 / -64.324 / -63.250 |

## Paired bootstrap: Spearman(stampscore excess p4) minus Spearman(other)

| other | Δρ | 95% | P(Δ ≤ 0) |
|---|---|---|---|
| stampscore excess p9 (crop) | -0.034 | [-0.08, +0.00] | 0.970 |
| stampscore excess grid (crop) | +0.063 | [-0.00, +0.14] | 0.025 |
| stampscore abs p4 (crop, no-ref) | +0.228 | [+0.04, +0.43] | 0.008 |
| stampscore excess p4 (whole image) | -0.085 | [-0.22, +0.02] | 0.934 |
| LPIPS to ref | -0.040 | [-0.24, +0.15] | 0.653 |
| SIFID to ref | -0.071 | [-0.27, +0.11] | 0.767 |
| -CLIP-IQA | +0.201 | [-0.05, +0.45] | 0.050 |
| -MUSIQ | +0.491 | [+0.23, +0.74] | 0.001 |

crop p4 excess flag rate at severity 0 / 1 / 2 / 3: {0.005: [0.12121212121212122, 0.42857142857142855, 0.6428571428571429, 1.0], 0.01: [0.030303030303030304, 0.35714285714285715, 0.6428571428571429, 1.0], 0.02: [0.030303030303030304, 0.2857142857142857, 0.2857142857142857, 0.8461538461538461]} 

## Round 1 (binary) vs round 2 (severity), same image, different crop

| round 1 | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| not | 28 | 4 | 0 | 0 |
| unsure | 4 | 3 | 1 | 0 |
| stamped | 1 | 7 | 11 | 13 |

## Real photos

- s18 lichen_stone: severity 0; crop dup p4 0.007
- s27 bread_crust: severity 0; crop dup p4 0.025
- s34 pine_bark: severity 0; crop dup p4 0.003
- s35 pine_bark: severity 0; crop dup p4 0.001
- s41 lichen_stone: severity 0; crop dup p4 0.002
- s44 bread_crust: severity 0; crop dup p4 0.014

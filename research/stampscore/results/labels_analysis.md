## Labels by (hidden) group

| group | stamped | not | unsure |
|---|---|---|---|
| ref | 1 | 11 | 1 |
| clean | 3 | 21 | 4 |
| blend | 14 | 0 | 1 |
| dmd | 14 | 0 | 2 |
| real | 0 | 6 | 0 |
| catch_stamp | 0 | 0 | 2 |
| catch_blur | 0 | 0 | 2 |

## Agreement with labels (generated crops, n = 64: 32 stamped, 32 not; unsure dropped)

| metric | AUC [95% boot] | Spearman | AUC, unsure = stamped | AUC, unsure = not |
|---|---|---|---|---|
| stampscore excess p4 (crop) | 0.917 [0.83, 0.98] | +0.72 | 0.916 | 0.873 |
| stampscore excess p9 (crop) | 0.932 [0.85, 0.99] | +0.75 | 0.930 | 0.891 |
| stampscore excess grid (crop) | 0.890 [0.80, 0.96] | +0.67 | 0.870 | 0.849 |
| stampscore abs p4 (crop, no-ref) | 0.711 [0.57, 0.84] | +0.37 | 0.698 | 0.696 |
| stampscore excess p4 (whole image) | 0.952 [0.89, 0.99] | +0.77 | 0.890 | 0.916 |
| LPIPS to ref | 0.899 [0.81, 0.97] | +0.69 | 0.848 | 0.872 |
| SIFID to ref | 0.920 [0.84, 0.98] | +0.73 | 0.854 | 0.891 |
| -CLIP-IQA | 0.713 [0.58, 0.83] | +0.37 | 0.676 | 0.692 |
| -MUSIQ | 0.576 [0.42, 0.72] | +0.13 | 0.548 | 0.590 |
| 1 - detail ratio (blur) | 0.320 [0.18, 0.47] | -0.31 | 0.335 | 0.326 |

## Paired bootstrap: AUC(stampscore excess p4) minus AUC(other)

| other | Δ AUC | 95% | P(Δ ≤ 0) |
|---|---|---|---|
| stampscore excess p9 (crop) | -0.015 | [-0.05, +0.01] | 0.867 |
| stampscore excess grid (crop) | +0.028 | [-0.01, +0.08] | 0.099 |
| stampscore abs p4 (crop, no-ref) | +0.205 | [+0.07, +0.35] | 0.002 |
| stampscore excess p4 (whole image) | -0.036 | [-0.12, +0.05] | 0.795 |
| LPIPS to ref | +0.017 | [-0.08, +0.11] | 0.366 |
| SIFID to ref | -0.004 | [-0.10, +0.08] | 0.529 |
| -CLIP-IQA | +0.203 | [+0.06, +0.35] | 0.003 |
| -MUSIQ | +0.342 | [+0.18, +0.50] | 0.000 |
| 1 - detail ratio (blur) | +0.596 | [+0.41, +0.76] | 0.000 |

operating points (crop excess p4 > t): {0.0: {'tpr': 0.90625, 'fpr': 0.25}, 0.005: {'tpr': 0.8125, 'fpr': 0.0625}, 0.01: {'tpr': 0.6875, 'fpr': 0.03125}, 0.02: {'tpr': 0.46875, 'fpr': 0.0}} 

## Catch trials and real photos

- c10 catch_blur u_bark: label **unsure**; crop dup p4 0.127, excess +0.123, detail ratio 0.16
- c17 catch_blur tex_bread: label **unsure**; crop dup p4 0.076, excess +0.058, detail ratio 0.14
- c20 real bread_crust: label **not**; crop dup p4 0.027
- c31 catch_stamp tex_bread: label **unsure**; crop dup p4 0.072, excess +0.062, detail ratio 1.03
- c43 catch_stamp u_moss_wall: label **unsure**; crop dup p4 0.074, excess +0.071, detail ratio 1.02
- c55 real pine_bark: label **not**; crop dup p4 0.000
- c63 real bread_crust: label **not**; crop dup p4 0.006
- c66 real lichen_stone: label **not**; crop dup p4 0.005
- c71 real pine_bark: label **not**; crop dup p4 0.001
- c79 real lichen_stone: label **not**; crop dup p4 0.027

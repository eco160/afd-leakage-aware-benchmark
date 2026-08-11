# Antioxidant benchmark - final aggregate

Generated 2026-08-11 from `results/*.json`. All statistics are mean +/- sample SD (ddof=1) over seeds 0-4, real mode unless stated. `perm floor` is the mean R2 of the label-permutation control (fusion models have no perm runs). Delta R2 = mean R2(random) - mean R2(grouped).

## 1. Main table (real runs)

| Model | Regime | R2 | RMSE | Spearman | Perm floor R2 | Delta R2 (rand-grp) |
|---|---|---|---|---|---|---|
| median_global | grouped | -0.216 +/- 0.025 | 1.356 +/- 0.033 | n/a | -0.216 +/- 0.025 | 0.014 |
| median_global | random | -0.202 +/- 0.015 | 1.298 +/- 0.066 | n/a | -0.202 +/- 0.015 |  |
| median_category | grouped | 0.442 +/- 0.052 | 0.918 +/- 0.036 | 0.707 +/- 0.032 | -0.213 +/- 0.040 | 0.022 |
| median_category | random | 0.464 +/- 0.036 | 0.866 +/- 0.055 | 0.695 +/- 0.013 | -0.197 +/- 0.019 |  |
| knn | grouped | 0.610 +/- 0.058 | 0.766 +/- 0.057 | 0.816 +/- 0.023 | -0.099 +/- 0.067 | 0.053 |
| knn | random | 0.663 +/- 0.071 | 0.681 +/- 0.050 | 0.839 +/- 0.017 | -0.109 +/- 0.091 |  |
| ridge | grouped | 0.674 +/- 0.027 | 0.702 +/- 0.033 | 0.828 +/- 0.020 | -0.030 +/- 0.044 | 0.013 |
| ridge | random | 0.687 +/- 0.060 | 0.657 +/- 0.039 | 0.838 +/- 0.015 | -0.009 +/- 0.036 |  |
| rf | grouped | 0.573 +/- 0.011 | 0.804 +/- 0.016 | 0.801 +/- 0.029 | -0.093 +/- 0.065 | 0.048 |
| rf | random | 0.621 +/- 0.041 | 0.727 +/- 0.021 | 0.815 +/- 0.018 | -0.134 +/- 0.035 |  |
| xgb | grouped | 0.610 +/- 0.020 | 0.767 +/- 0.019 | 0.738 +/- 0.029 | -0.011 +/- 0.011 | 0.018 |
| xgb | random | 0.629 +/- 0.044 | 0.719 +/- 0.020 | 0.729 +/- 0.037 | -0.007 +/- 0.013 |  |
| lgbm | grouped | 0.616 +/- 0.017 | 0.762 +/- 0.021 | 0.765 +/- 0.025 | -0.009 +/- 0.010 | 0.028 |
| lgbm | random | 0.644 +/- 0.055 | 0.703 +/- 0.028 | 0.798 +/- 0.018 | -0.003 +/- 0.001 |  |
| mlp | grouped | 0.657 +/- 0.008 | 0.721 +/- 0.018 | 0.833 +/- 0.013 | -0.043 +/- 0.013 | -0.001 |
| mlp | random | 0.656 +/- 0.056 | 0.691 +/- 0.031 | 0.837 +/- 0.020 | -0.067 +/- 0.023 |  |
| cnn | grouped | 0.617 +/- 0.019 | 0.761 +/- 0.018 | 0.783 +/- 0.034 | -0.025 +/- 0.038 | 0.021 |
| cnn | random | 0.638 +/- 0.042 | 0.710 +/- 0.028 | 0.801 +/- 0.035 | -0.003 +/- 0.017 |  |
| bilstm | grouped | 0.608 +/- 0.031 | 0.769 +/- 0.033 | 0.772 +/- 0.042 | -0.030 +/- 0.044 | 0.047 |
| bilstm | random | 0.656 +/- 0.064 | 0.690 +/- 0.038 | 0.814 +/- 0.030 | 0.004 +/- 0.027 |  |
| copypaste | grouped | 0.442 +/- 0.052 | 0.918 +/- 0.036 | 0.707 +/- 0.032 | -0.213 +/- 0.040 | 0.111 |
| copypaste | random | 0.553 +/- 0.027 | 0.791 +/- 0.045 | 0.781 +/- 0.012 | -0.465 +/- 0.110 |  |
| abl_ridge_text | grouped | 0.597 +/- 0.026 | 0.781 +/- 0.027 | 0.789 +/- 0.020 | -0.020 +/- 0.025 | 0.037 |
| abl_ridge_text | random | 0.634 +/- 0.055 | 0.712 +/- 0.028 | 0.805 +/- 0.018 | -0.010 +/- 0.023 |  |
| abl_ridge_cat | grouped | 0.503 +/- 0.065 | 0.865 +/- 0.050 | 0.702 +/- 0.036 | -0.021 +/- 0.040 | -0.009 |
| abl_ridge_cat | random | 0.494 +/- 0.032 | 0.842 +/- 0.040 | 0.680 +/- 0.015 | -0.001 +/- 0.033 |  |
| hybrid_xgb_emb | grouped | 0.625 +/- 0.023 | 0.753 +/- 0.019 | 0.806 +/- 0.024 | -0.002 +/- 0.014 | 0.042 |
| hybrid_xgb_emb | random | 0.667 +/- 0.053 | 0.679 +/- 0.026 | 0.830 +/- 0.023 | -0.004 +/- 0.008 |  |
| hybrid_xgb_full | grouped | 0.627 +/- 0.019 | 0.751 +/- 0.017 | 0.806 +/- 0.023 | -0.013 +/- 0.011 | 0.040 |
| hybrid_xgb_full | random | 0.666 +/- 0.051 | 0.680 +/- 0.024 | 0.831 +/- 0.022 | -0.005 +/- 0.004 |  |
| hybrid_ridge_full | grouped | 0.629 +/- 0.013 | 0.749 +/- 0.007 | 0.805 +/- 0.029 | -0.305 +/- 0.241 | 0.044 |
| hybrid_ridge_full | random | 0.673 +/- 0.055 | 0.673 +/- 0.031 | 0.827 +/- 0.021 | -0.127 +/- 0.049 |  |
| fusion_all | grouped | 0.684 +/- 0.011 | 0.692 +/- 0.020 | 0.845 +/- 0.020 | n/a | 0.022 |
| fusion_all | random | 0.706 +/- 0.042 | 0.639 +/- 0.020 | 0.854 +/- 0.020 | n/a |  |
| fusion_ltd | grouped | 0.679 +/- 0.013 | 0.697 +/- 0.022 | 0.827 +/- 0.024 | n/a | 0.015 |
| fusion_ltd | random | 0.694 +/- 0.046 | 0.652 +/- 0.024 | 0.841 +/- 0.022 | n/a |  |

## 2. Paired comparison vs ridge (shared splits, per-seed R2 differences)

### grouped regime

| Model | Per-seed diff (s0..s4) | Mean diff | SD | Seeds beating ridge | Sign test (two-sided) |
|---|---|---|---|---|---|
| mlp | +0.042, -0.033, -0.035, -0.020, -0.042 | -0.0176 | 0.0342 | 1/5 | two-sided exact sign test: 1/5 seeds beat ridge, p = 0.3750 (not significant at 0.05; n=5 seeds can reach at best p=0.0625) |
| cnn | -0.018, -0.046, -0.075, -0.044, -0.103 | -0.0571 | 0.0327 | 0/5 | two-sided exact sign test: 0/5 seeds beat ridge, p = 0.0625 (not significant at 0.05; n=5 seeds can reach at best p=0.0625) |
| bilstm | -0.073, -0.061, -0.073, -0.063, -0.059 | -0.0659 | 0.0066 | 0/5 | two-sided exact sign test: 0/5 seeds beat ridge, p = 0.0625 (not significant at 0.05; n=5 seeds can reach at best p=0.0625) |
| lgbm | -0.039, -0.051, -0.070, -0.062, -0.066 | -0.0577 | 0.0126 | 0/5 | two-sided exact sign test: 0/5 seeds beat ridge, p = 0.0625 (not significant at 0.05; n=5 seeds can reach at best p=0.0625) |
| knn | -0.119, -0.055, -0.052, -0.042, -0.050 | -0.0638 | 0.0314 | 0/5 | two-sided exact sign test: 0/5 seeds beat ridge, p = 0.0625 (not significant at 0.05; n=5 seeds can reach at best p=0.0625) |
| hybrid_xgb_emb | -0.004, -0.039, -0.077, -0.027, -0.099 | -0.0490 | 0.0383 | 0/5 | two-sided exact sign test: 0/5 seeds beat ridge, p = 0.0625 (not significant at 0.05; n=5 seeds can reach at best p=0.0625) |
| hybrid_xgb_full | -0.003, -0.039, -0.069, -0.034, -0.093 | -0.0476 | 0.0344 | 0/5 | two-sided exact sign test: 0/5 seeds beat ridge, p = 0.0625 (not significant at 0.05; n=5 seeds can reach at best p=0.0625) |
| hybrid_ridge_full | -0.003, -0.038, -0.077, -0.037, -0.072 | -0.0453 | 0.0301 | 0/5 | two-sided exact sign test: 0/5 seeds beat ridge, p = 0.0625 (not significant at 0.05; n=5 seeds can reach at best p=0.0625) |
| fusion_all | +0.042, +0.009, +0.004, +0.003, -0.011 | +0.0096 | 0.0195 | 4/5 | two-sided exact sign test: 4/5 seeds beat ridge, p = 0.3750 (not significant at 0.05; n=5 seeds can reach at best p=0.0625) |
| fusion_ltd | +0.033, +0.003, +0.003, -0.001, -0.012 | +0.0050 | 0.0168 | 3/5 | two-sided exact sign test: 3/5 seeds beat ridge, p = 1.0000 (not significant at 0.05; n=5 seeds can reach at best p=0.0625) |

### random regime

| Model | Per-seed diff (s0..s4) | Mean diff | SD | Seeds beating ridge | Sign test (two-sided) |
|---|---|---|---|---|---|
| fusion_ltd | +0.018, +0.012, -0.027, +0.028, +0.003 | +0.0069 | 0.0209 | 4/5 | two-sided exact sign test: 4/5 seeds beat ridge, p = 0.3750 (not significant at 0.05; n=5 seeds can reach at best p=0.0625) |
| hybrid_ridge_full | -0.023, +0.015, -0.030, -0.009, -0.025 | -0.0144 | 0.0184 | 1/5 | two-sided exact sign test: 1/5 seeds beat ridge, p = 0.3750 (not significant at 0.05; n=5 seeds can reach at best p=0.0625) |

## 3. Ablation - ridge input channels (grouped regime)

| Model | R2 | RMSE | Spearman |
|---|---|---|---|
| ridge | 0.674 +/- 0.027 | 0.702 +/- 0.033 | 0.828 +/- 0.020 |
| abl_ridge_text | 0.597 +/- 0.026 | 0.781 +/- 0.027 | 0.789 +/- 0.020 |
| abl_ridge_cat | 0.503 +/- 0.065 | 0.865 +/- 0.050 | 0.702 +/- 0.036 |
| median_category | 0.442 +/- 0.052 | 0.918 +/- 0.036 | 0.707 +/- 0.032 |

ridge = full feature set; abl_ridge_text = text features only; abl_ridge_cat = categorical features only; median_category = per-category median baseline (no learned weights).

## 4. Leakage mechanics (grouped vs random)

| Probe | Grouped R2 | Random R2 | Delta R2 | Grouped Spearman | Random Spearman | Duplicate-name fraction (grouped / random) |
|---|---|---|---|---|---|---|
| copypaste | 0.442 +/- 0.052 | 0.553 +/- 0.027 | 0.111 | 0.707 +/- 0.032 | 0.781 +/- 0.012 | 0.000 +/- 0.000 / 0.365 +/- 0.013 |
| knn | 0.610 +/- 0.058 | 0.663 +/- 0.071 | 0.053 | 0.816 +/- 0.023 | 0.839 +/- 0.017 | - |
| median_category | 0.442 +/- 0.052 | 0.464 +/- 0.036 | 0.022 | 0.707 +/- 0.032 | 0.695 +/- 0.013 | - |

## 5. Flags

- none (no missing runs, no perm floor above 0.05, no unexpected NaN in reported cells)

Notes (informational, not failures):
- median_global grouped spearman is NaN for all seeds (constant predictor; Spearman undefined) - expected, reported as n/a
- median_global random spearman is NaN for all seeds (constant predictor; Spearman undefined) - expected, reported as n/a

# Leakage-aware ML/DL benchmarking on the Antioxidant Food Table

Reproduction package for the manuscript on predicting food antioxidant capacity (FRAP) from
product description and category, benchmarked under leakage-free evaluation. Submitted to
the MDPI *Antioxidants* Special Issue "Artificial Intelligence and Other Bioinformatic
Modern Technologies Approaches to the Study of Antioxidant Capacity in Food Production".

## Headline results (test R², log-transformed FRAP, mean over 5 seeds)

| Model | Grouped split | Random split | ΔR² (inflation) |
|---|---|---|---|
| **Fusion (8-model average)** | **0.684 ± 0.011** | 0.706 | — |
| Ridge (TF-IDF) | 0.674 ± 0.027 | 0.687 | 0.013 |
| MLP (same TF-IDF) | 0.657 | 0.656 | −0.001 |
| Hybrid Ridge (TF-IDF+emb) | 0.629 ± 0.013 | 0.673 | 0.044 |
| Hybrid XGBoost (TF-IDF+emb) | 0.627 | 0.666 | 0.040 |
| XGBoost on DL embeddings | 0.625 | 0.667 | 0.042 |
| char-CNN | 0.617 | 0.638 | 0.021 |
| LightGBM | 0.616 | 0.644 | 0.028 |
| k-NN (cosine) | 0.610 | 0.663 | 0.053 |
| XGBoost | 0.610 | 0.629 | 0.018 |
| BiLSTM | 0.608 | 0.656 | 0.047 |
| Random Forest | 0.573 | 0.621 | 0.048 |
| Memorization baseline (duplicate-name lookup) | 0.442 | 0.553 | **0.111** |
| Category median | 0.442 | 0.464 | 0.022 |

Key findings: (1) 39.4% of records share a normalised product name; under the literature's
default random split, **45% of k-NN's advantage over the category median is reproduced by a
memorization baseline that merely retrieves identically named training records**.
(2) Engineered TF-IDF features beat learned character embeddings at this sample
size (Ridge > char-CNN/BiLSTM on 0/5 seeds each). (3) The equal-weight fusion of all eight
learned models is the only method that beats Ridge (4/5 seeds) and it cuts seed-to-seed SD
by a factor of ~2.6 (0.027 → 0.011). All models are controlled by y-permutation floors ≈ 0.

## Reproduce

```bash
pip install -r requirements.txt          # exact versions used
python3 code/build_afd_csv.py            # PDF -> processed/antioxidant_food_table.csv
python3 code/prepare.py                  # frozen splits (grouped + random x 5 seeds)
for m in median_global median_category knn ridge rf xgb lgbm mlp; do
    python3 code/runner.py --model $m    # classical ML (checkpointed, resumable)
done
python3 code/dl.py --arch cnn            # deep models + embeddings
python3 code/dl.py --arch bilstm
for m in copypaste abl_ridge_text abl_ridge_cat hybrid_xgb_emb \
         hybrid_xgb_full hybrid_ridge_full fusion_all fusion_ltd; do
    python3 code/day4.py --model $m      # hybrids, ablations, diagnostics, fusion
done
python3 code/make_figures.py             # all manuscript figures from results/*.json
```

Total runtime ≈ 2–3 h on a 4-core laptop (no GPU needed). Every run checkpoints to
`results/<model>.json`; interrupted runs resume. `results/` in this repository already
contains the exact run outputs behind the manuscript (including per-run test predictions
in `results/preds/` and the DL embeddings in `results/embeds/`), so figures and tables can
be regenerated without any training.

## Protocol notes (the paper's core)

- **Grouped split**: GroupShuffleSplit by normalised product name — no name straddles
  train/test. **Random split**: same sizes, the literature's default; 34.6–38.0% of its
  test records share a product name with a training record.
- All feature fitting (TF-IDF vocabularies, category encoder, char vocab, embedding
  standardisation) on the fit partition only; val (group-aware) used solely for early
  stopping / hyperparameter selection; test untouched.
- One y-permutation control per (regime, seed); for two-stage hybrids the permutation
  covers the downstream learner (the embedder carries its own permutation runs).
- xgb/lgbm/cnn/bilstm train on fit (val reserved for early stopping); ridge/knn/rf refit
  on fit+val after selection — stated because cross-family comparisons then differ by
  ~15% effective training data.
- Under the grouped regime the memorization baseline finds no duplicate names and reproduces
  the category-median baseline **bit-identically** — a built-in correctness check.

## Data provenance

`processed/antioxidant_food_table.csv` is extracted (99.9%, validated against the source's
own per-category statistics) from Additional file 1 of Carlsen et al., *Nutrition Journal*
2010;9:3, doi:10.1186/1475-2891-9-3 (CC-BY). See `processed/DATASET.md` for the validation
record. **Cite Carlsen et al. (2010) whenever you use the dataset.** Code is MIT-licensed.

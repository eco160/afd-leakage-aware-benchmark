# antioxidant_food_table.csv — extraction and validation record

Machine-readable release of the **Antioxidant Food Table** (Additional file 1 of Carlsen et al.,
*The total antioxidant content of more than 3100 foods…*, Nutrition Journal 2010;9:3,
doi:10.1186/1475-2891-9-3, **CC-BY**). Built by `../build_afd_csv.py` from word-level bounding
boxes (`pdftotext -bbox-layout`); each numeric antioxidant value anchors a row band, words are
assigned to bands by y-centre and to columns by x-centres read from each page's own header.
Frozen 2026-08-10 (v1).

## Contents

**3,135 records** (published total: 3,139 → 99.9%), 24 categories, 7 columns:

| column | completeness | note |
|---|---|---|
| category_no, category | 100% | 1–24, names canonicalised per number |
| product | 100% | multi-line names reconstructed; 0.1% residual wrap defects (3 records) |
| manufacturer_origin | 91.3% | blank where blank in the PDF |
| procured_in | 100% | country |
| antioxidant_mmol_per_100g | 100% | FRAP, mmol/100 g |
| comment_ref | 99.1% | reference-note number(s) from the source, e.g. "1, 3" |

## Validation record

Validation proceeded in two stages, both fully independent of the extraction code.

**Stage 1 — faithfulness to the printed supplement.** An independent value-level census of
the supplement (values anchored on the antioxidant/comment columns, no product-name parsing)
was compared with this CSV. Per-category record counts and value sums match **exactly in 23
of 24 categories**. The single difference: one record of 0.17 mmol/100 g in *Mixed food
entrees* (supplement count 189, CSV 188) was not recovered, because of an unusually wrapped
line. This is the only extraction loss relative to the printed supplement.

**Stage 2 — agreement with the article's summary table.** Recomputed per-category
n/mean/median/max agree with the article's Table 1 within published rounding for most
categories. Every residual difference traces to an inconsistency internal to the source:

- **19-product reassignment**: 11 "Biscuits, refrigerated…" and 8 "English muffins…" are
  printed under *Grains and grain products* in the supplement but tallied under *Snacks* and
  *Desserts and cakes* in the article. The article's tallies imply a twentieth such product
  that is absent from the printed supplement.
- **Three phantom records**: the article's category counts sum to 3139, but the printed
  supplement contains 3136 extractable records. One record each in *Beverages* (article 283
  vs supplement 282), *Spices and herbs* (425 vs 424), and the reassignment group is tallied
  in the article yet absent from the supplement.
- **Two article-side typographical values**: cat 21 maximum (article 4.67; the supplement's
  own product listing gives 4.66 for "Sauce, taco, medium") and cat 1 median (article 3.34;
  the exact 60th of 119 sorted values is 3.35).
- **One source-inherited duplicate row**: "Rice, white, long grain / Canilla / USA / 0.1"
  appears twice, exactly as printed twice in the supplement. It is retained faithfully.

The supplement's own placements and values are preserved throughout; nothing was "corrected"
toward the article. Coverage: 3135 of 3136 supplement records (99.97%), or 3135 of the
article's 3139 tallied records (99.87%).

## Properties relevant to modelling

- Target range 0–2897.11 mmol/100 g, median 0.5, 40 exact zeros → model on log1p.
- **Leakage surface: 1,234 of 3,135 records (39.4%) share a normalised product name with at
  least one other record** ("tomato juice" ×11, "oregano, dried" ×9, "apple juice" ×9 — same
  food, different brand/origin/lot). Random splits place these on both sides of the train/test
  boundary; grouped splits by normalised name are required for honest evaluation. This number is
  the empirical basis of the paper's protocol argument (it was 30.9% under the earlier flawed
  parse; the cleaner names raised it).

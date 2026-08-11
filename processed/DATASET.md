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

## Validation against the source article (Table 1 of the paper)

Per-category n, mean, median and max recomputed from this CSV and compared with the published
descriptive statistics: **21 of 24 categories reproduce exactly** (within rounding). Overall
n = 3,135 vs 3,139 (−4).

The three non-matching categories are a **source-level inconsistency, not an extraction error**:
19 products ("Biscuits, refrigerated…" ×11, "English muffins…" ×8) are printed under
*Category 11 Grains and grain products* in the PDF but counted under *Snacks* (cat 20) and
*Desserts and cakes* (cat 6) in the article's Table 1 (published: cat 6 n=134, cat 11 n=227,
cat 20 n=66; PDF: 125/246/55). Means/medians/maxima of all other 21 categories match, including
the extremes (cat 12 max 2897.11 = dried amla-based preparation; cat 2 max 1347.83; cat 24 mean
98.58). We keep the PDF's own placement and note the discrepancy rather than silently "fixing"
either source.

Remaining −4 vs (−1 net of the 19/20 shift): one record each lost in categories 2, 17, 22 (line
pairs merged where two records share one wrapped block) and the 19↔20 balance. Documented, not
hidden.

## Properties relevant to modelling

- Target range 0–2897.11 mmol/100 g, median 0.5, 40 exact zeros → model on log1p.
- **Leakage surface: 1,234 of 3,135 records (39.4%) share a normalised product name with at
  least one other record** ("tomato juice" ×11, "oregano, dried" ×9, "apple juice" ×9 — same
  food, different brand/origin/lot). Random splits place these on both sides of the train/test
  boundary; grouped splits by normalised name are required for honest evaluation. This number is
  the empirical basis of the paper's protocol argument (it was 30.9% under the earlier flawed
  parse; the cleaner names raised it).

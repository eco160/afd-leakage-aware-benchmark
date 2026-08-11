#!/usr/bin/env python3
"""Methodological flow diagram (Methods-opening figure).

Same visual tokens as the data figures: blue = grouped/leakage-free protocol,
orange = the leakage surface and the random-split default. Counts and shares are
read from the processed data so the diagram cannot drift from the manuscript.
"""
import csv
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

BLUE, ORANGE = "#2a78d6", "#eb6834"
FILL, INK, INK2 = "#eaf2fc", "#0b0b0b", "#52514e"
EDGE = "#9ec5f4"
CM = 1 / 2.54

rows = list(csv.DictReader(open("processed/model_data.csv")))
n = len(rows)
share = sum(v for v in Counter(r["group"] for r in rows).values() if v > 1) / n

fig = plt.figure(figsize=(17.8 * CM, 12.2 * CM))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100)
ax.set_ylim(0, 68)
ax.axis("off")


def box(x, y, w, h, title, lines, edge=EDGE, fill=FILL, title_color=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6",
                                facecolor=fill, edgecolor=edge, linewidth=1.2))
    ax.text(x + w / 2, y + h - 1.4, title, ha="center", va="top", fontsize=8.2,
            fontweight="bold", color=title_color)
    ax.text(x + w / 2, y + h - 4.6, "\n".join(lines), ha="center", va="top",
            fontsize=7.2, color=INK2, linespacing=1.45)


def arrow(x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=13, linewidth=1.4, color=BLUE,
                                 shrinkA=2, shrinkB=2))


# ---- stage 1: data construction ------------------------------------------------
box(1, 58, 29, 9, "Source (open data)",
    ["Antioxidant Food Table, Carlsen et al.",
     "2010 (CC-BY) — 138-page PDF supplement"])
box(35.5, 58, 29, 9, "Structured extraction",
    [f"word-level bounding boxes → {n:,} records",
     "21/24 categories reproduce published stats"])
box(70, 58, 29, 9, "Modelling table",
    ["target: log(1+FRAP)  [mmol/100 g]",
     "inputs: product description + category"])
arrow(30.4, 62.5, 35.1, 62.5)
arrow(64.9, 62.5, 69.6, 62.5)

# ---- stage 2: leakage surface and frozen splits --------------------------------
box(1, 45.5, 45, 9, "Leakage surface", [
    f"name normalisation → {share:.1%} of records",
    "share a product name (same food, other brand/lot)"],
    edge=ORANGE, title_color=INK)
box(52, 45.5, 47, 9, "Frozen evaluation splits  (× 5 seeds)",
    ["grouped by product name  vs  random (literature default)",
     "each: fit / group-aware validation / 20% test"])
arrow(84.5, 57.6, 84.5, 55.0)          # modelling table -> splits
arrow(40, 57.6, 40, 55.0)              # extraction -> leakage surface
arrow(46.4, 50, 51.6, 50)              # leakage surface -> splits

# ---- stage 3: model families ---------------------------------------------------
Y3, H3 = 26, 13.5
box(1, Y3, 22.5, H3, "Baselines",
    ["global median", "category median", "memorization", "(duplicate lookup)"])
box(26.5, Y3, 22.5, H3, "Classical ML",
    ["TF-IDF (word + char)", "+ category one-hot", "Ridge · k-NN · RF",
     "XGBoost · LightGBM · MLP"])
box(52, Y3, 22.5, H3, "Deep learning",
    ["character vocabulary", "char-CNN · BiLSTM", "+ category embedding",
     "→ 192-d representations"])
box(77.5, Y3, 21.5, H3, "ML–DL integration",
    ["embeddings → XGBoost", "embeddings → Ridge",
     "8-model equal-weight", "fusion"])
for xc in (12.25, 37.75, 63.25, 88.25):
    arrow(75.5 if False else xc, 45.0, xc, Y3 + H3 + 0.4)
arrow(74.9, Y3 + 4.5, 77.1, Y3 + 4.5)  # DL embeddings feed the integration

# ---- stage 4: evaluation, then diagnostics (stacked full-width bands) ----------
box(1, 13.5, 98, 10, "Leakage-aware evaluation",
    ["all feature fitting on the fit partition only · validation used for early "
     "stopping / selection only · test untouched until scoring",
     "R² · RMSE · Spearman ρ on log(1+FRAP), mean ± SD over 5 paired seeds · "
     "y-permutation control per split (chance floor ≈ 0)"])
box(1, 2.5, 98, 8, "Diagnostics",
    ["leakage inflation ΔR² = random − grouped  ·  memorization decomposition "
     "of k-NN  ·  text / category ablations"])
for xc in (12.25, 37.75, 63.25, 88.25):
    arrow(xc, Y3 - 0.6, xc, 24.1)
arrow(50, 12.9, 50, 11.1)

for ext in ("png", "pdf"):
    fig.savefig(f"figures/fig_methodology.{ext}", bbox_inches="tight",
                facecolor="white", dpi=600)
print("wrote figures/fig_methodology.png/.pdf")

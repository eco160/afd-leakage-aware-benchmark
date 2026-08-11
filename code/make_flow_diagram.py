#!/usr/bin/env python3
"""Methodological flow diagram in the author's published flowchart convention
(cf. Ozkat et al., Physica C 616 (2024) 1354430, Fig. 1): numbered stage
containers with header bars, sharp rectangles, and orthogonal
elbow-routed arrows. Monochrome. Live numbers are read from the processed data.
"""
import csv, json
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle

BK = "#1a1a1a"
CM = 1 / 2.54

rows = list(csv.DictReader(open("processed/model_data.csv")))
n = len(rows)
share = sum(v for v in Counter(r["group"] for r in rows).values() if v > 1) / n
S = json.load(open("processed/splits.json"))
leak = np.mean([S["random"][str(s)]["test_leak_fraction"] for s in range(5)])

fig = plt.figure(figsize=(17.8 * CM, 13.6 * CM))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100)
ax.set_ylim(0, 78)
ax.axis("off")


def panel(x, y, w, h, title):
    ax.add_patch(Rectangle((x, y), w, h, facecolor="white", edgecolor=BK,
                           linewidth=1.5, zorder=1))
    ysep = y + h - 3.6
    ax.plot([x, x + w], [ysep, ysep], color=BK, lw=1.2, zorder=2)
    ax.text(x + 1.4, ysep + 1.1, title, fontsize=8.8, fontweight="bold",
            color=BK, va="bottom", zorder=3)


def box(x, y, w, h, title, body=None):
    ax.add_patch(Rectangle((x, y), w, h, facecolor="white", edgecolor=BK,
                           linewidth=1.1, zorder=4))
    if body:
        ax.text(x + w / 2, y + h - 1.1, title, ha="center", va="top",
                fontsize=7.6, fontweight="bold", color=BK, zorder=5)
        ax.text(x + w / 2, y + h - 3.6, body, ha="center", va="top",
                fontsize=6.9, color=BK, linespacing=1.4, zorder=5)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                fontsize=7.6, fontweight="bold", color=BK, zorder=5)


def route(pts, head=True):
    """Orthogonal connector through waypoints; arrowhead on the final leg."""
    if len(pts) > 2:
        xs, ys = zip(*pts[:-1])
        ax.plot(xs, ys, color=BK, lw=1.2, zorder=3,
                solid_capstyle="projecting")
    style = "-|>" if head else "-"
    ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle=style,
                                 mutation_scale=13, linewidth=1.2, color=BK,
                                 shrinkA=0, shrinkB=0, zorder=3))


# ============ 1. Data construction ==============================================
panel(0.5, 45.5, 31.5, 32, "1. Data Construction")
box(3, 65, 26.5, 7.5, "Antioxidant Food Table",
    "Carlsen et al. 2010 (CC-BY)\n138-page PDF supplement")
box(3, 55.5, 26.5, 7.5, "Structured Extraction",
    "word bounding boxes\n+ statistical validation")
box(3, 46.5, 26.5, 7.5, "Modelling Table",
    f"{n:,} records · log(1+FRAP)\nproduct text + category")
route([(16.25, 65), (16.25, 63)])
route([(16.25, 55.5), (16.25, 54)])

# ============ 2. Split design ===================================================
# The partitioning step is a PROCESS box with two dataset outputs (cf. the
# author's published "Execute DoE" convention) -- not a decision diamond.
panel(34.5, 45.5, 65, 32, "2. Leakage-Aware Split Design")
box(37, 59.5, 24, 9, "Name Normalisation",
    f"{share:.1%} of records share\na product name")
box(67, 59.5, 29, 9, "Data Partitioning",
    "5 seeds per regime\n(fit / validation / 20% test)")
box(63.5, 46.5, 16.5, 8, "Grouped by Name",
    "no shared name\nacross partitions")
box(82, 46.5, 16.5, 8, "Random Split",
    f"literature default\n({leak:.0%} test duplicates)")
route([(29.5, 50.25), (33, 50.25), (33, 64), (37, 64)])
route([(61, 64), (67, 64)])
route([(71.75, 59.5), (71.75, 54.5)])
route([(90.25, 59.5), (90.25, 54.5)])

# ============ 3. Model development ==============================================
panel(0.5, 20.5, 99, 22, "3. Model Development")
# integration sits BETWEEN its two input families: deep learning supplies the
# embeddings (left arrow), classical ML supplies fusion-member predictions
# (right arrow) -- both in-edges drawn, no crossings
box(3, 24, 21, 11.5, "Baselines",
    "global median\ncategory median\nmemorization\n(duplicate lookup)")
box(28, 24, 21, 11.5, "Deep Learning",
    "char-CNN · BiLSTM\n+ category embedding\n→ 192-d representations")
box(53, 24, 21, 11.5, "ML–DL Integration",
    "embeddings + TF-IDF →\nXGBoost / Ridge hybrids\nfusion of 8 models")
box(78, 24, 19.5, 11.5, "Classical ML",
    "TF-IDF + category\nRidge · k-NN · RF\nXGBoost · LightGBM · MLP")
route([(49, 29.75), (53, 29.75)])
route([(78, 29.75), (74, 29.75)])
# both regime datasets feed model development: short drops merge on a rail in
# the inter-panel gap, then one trunk enters the distribution rail
route([(71.75, 46.5), (71.75, 43.8)], head=False)
route([(90.25, 46.5), (90.25, 43.8)], head=False)
ax.plot([71.75, 90.25], [43.8, 43.8], color=BK, lw=1.2, zorder=3)
route([(81, 43.8), (81, 37.6)], head=False)
ax.plot([13.5, 87.75], [37.6, 37.6], color=BK, lw=1.2, zorder=3)
# the split data feeds the three base families only; ML-DL Integration consumes
# its inputs entirely through the lateral edges (embeddings from Deep Learning,
# features/predictions from Classical ML), so it gets no data drop
for xc in (13.5, 38.5, 87.75):
    route([(xc, 37.6), (xc, 35.5)])

# ============ 4. Evaluation and diagnostics =====================================
panel(0.5, 1, 99, 16.5, "4. Evaluation and Diagnostics")
box(3, 3, 45, 9, "Leakage-Aware Scoring",
    "R² · RMSE · MAE · Spearman ρ on log(1+FRAP)\n"
    "5 paired seeds · y-permutation controls")
box(55, 3, 42.5, 9, "Diagnostics",
    "ΔR² = random − grouped · memorization\n"
    "decomposition · text / category ablations")
route([(48, 7.5), (55, 7.5)])
# bus: model families merge onto a rail, single entry into scoring
for xc in (13.5, 38.5, 63.5, 87.75):
    ax.plot([xc, xc], [24, 19], color=BK, lw=1.2, zorder=3)
ax.plot([13.5, 87.75], [19, 19], color=BK, lw=1.2, zorder=3)
route([(32, 19), (32, 12)])

for ext in ("png", "pdf"):
    fig.savefig(f"figures/fig_methodology.{ext}", bbox_inches="tight",
                facecolor="white", dpi=600)
print("wrote figures/fig_methodology.png/.pdf")

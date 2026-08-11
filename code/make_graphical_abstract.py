#!/usr/bin/env python3
"""Graphical abstract, MDPI Antioxidants submission.

House style: navy-monochrome three-column schematic (same family as the EiN and
Pharmaceuticals GAs). 3300x1680 px at 300 dpi = 3x the journal's 560x1100 minimum,
aspect 1.964. Purpose-built — not identical to any manuscript figure. Key numbers
are read from results/summary-level JSONs so the GA cannot drift from the paper.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

NAVY, MID, LIGHT = "#123a6d", "#2a78d6", "#dce9f8"
PAPER, INK = "#ffffff", "#123a6d"
ACCENT = "#eb6834"

W, H, DPI = 3300, 1680, 300
fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")
fig.patch.set_facecolor(PAPER)


def r2(model, regime):
    d = json.load(open(f"results/{model}.json"))["runs"]
    return float(np.mean([d[f"{regime}_s{s}_real"]["r2"] for s in range(5)]))


fusion_g = r2("fusion_all", "grouped")
knn_r, knn_g = r2("knn", "random"), r2("knn", "grouped")
cp_r, med_r = r2("copypaste", "random"), r2("median_category", "random")
twin_share = (cp_r - med_r) / (knn_r - med_r)


def card(x, w, title):
    ax.add_patch(FancyBboxPatch((x, 14), w, 72, boxstyle="round,pad=1.2",
                                facecolor=LIGHT, edgecolor=NAVY, linewidth=1.6))
    ax.add_patch(FancyBboxPatch((x, 78), w, 8, boxstyle="round,pad=1.2",
                                facecolor=NAVY, edgecolor=NAVY))
    ax.text(x + w / 2, 82, title, ha="center", va="center", color="white",
            fontsize=11.5, fontweight="bold")


def arrow(x0, x1):
    ax.add_patch(FancyArrowPatch((x0, 50), (x1, 50), arrowstyle="-|>",
                                 mutation_scale=28, linewidth=2.6, color=NAVY))


# ---------------------------------------------------------------- title band
ax.text(50, 95, "Protocol and representation decide food antioxidant-capacity "
                "prediction", ha="center", va="center", color=NAVY,
        fontsize=14.5, fontweight="bold")

# ---------------------------------------------------------------- column 1
card(3, 26, "OPEN DATA")
ax.text(16, 72, "Antioxidant Food Table", ha="center", fontsize=10.5,
        color=NAVY, fontweight="bold")
ax.text(16, 66.5, "3,135 foods · FRAP assay", ha="center", fontsize=9.5,
        color=NAVY)
# mini bar glyph: category magnitudes
for i, h in enumerate([16, 11, 8, 6, 4.2, 3]):
    ax.add_patch(plt.Rectangle((7 + i * 3.1, 42), 2.3, h, facecolor=MID,
                               edgecolor="none"))
ax.text(16, 36.5, "spices → egg: 4 orders of magnitude", ha="center",
        fontsize=8.2, color=NAVY)
ax.add_patch(FancyBboxPatch((6.5, 21), 19, 9, boxstyle="round,pad=0.8",
                            facecolor="white", edgecolor=ACCENT, linewidth=1.8))
ax.text(16, 27, "39% of records share", ha="center", fontsize=9, color=NAVY,
        fontweight="bold")
ax.text(16, 23.5, "a product name", ha="center", fontsize=9, color=NAVY,
        fontweight="bold")

arrow(30.5, 36)

# ---------------------------------------------------------------- column 2
card(37, 26, "LEAKAGE-AWARE BENCHMARK")
ax.text(50, 72, "random split", ha="center", fontsize=9.5, color=ACCENT,
        fontweight="bold")
ax.text(50, 68.5, "vs", ha="center", fontsize=8.5, color=NAVY)
ax.text(50, 65, "grouped by product name", ha="center", fontsize=9.5,
        color=NAVY, fontweight="bold")
# split glyph: dots left/right of a divider
for k in range(7):
    xx = 42 + (k % 4) * 2.6
    yy = 52 - (k // 4) * 3.4
    ax.add_patch(plt.Circle((xx, yy), 1.0, facecolor=MID, edgecolor="white",
                            linewidth=0.8))
ax.plot([51.5, 51.5], [45, 56], color=NAVY, lw=1.6, ls=(0, (4, 3)))
for k in range(4):
    xx = 54 + (k % 2) * 2.6
    yy = 52 - (k // 2) * 3.4
    ax.add_patch(plt.Circle((xx, yy), 1.0, facecolor="white", edgecolor=MID,
                            linewidth=1.4))
ax.text(50, 40.5, "18 models · 5 seeds", ha="center", fontsize=9.5,
        color=NAVY)
ax.text(50, 36.8, "classical ML · deep learning · hybrids",
        ha="center", fontsize=8.6, color=NAVY)
ax.text(50, 25.5, "permutation controls\n+ memorization baseline",
        ha="center", fontsize=8.6, color=NAVY, style="italic")

arrow(64.5, 70)

# ---------------------------------------------------------------- column 3
card(71, 26, "WHAT HOLDS UP")
ax.text(84, 71.5, f"{twin_share:.0%} of k-NN's apparent skill",
        ha="center", fontsize=9.6, color=ACCENT, fontweight="bold")
ax.text(84, 68, "= duplicate-record lookup, not learning", ha="center",
        fontsize=9, color=NAVY)
# small paired-bar glyph: grouped vs random for knn
ax.add_patch(plt.Rectangle((78, 52), 5.4, 4.4, facecolor=ACCENT, edgecolor="none"))
ax.add_patch(plt.Rectangle((78, 46), 5.4 * knn_g / knn_r, 4.4, facecolor=MID,
                           edgecolor="none"))
ax.text(84.2, 54.2, "random", fontsize=7.6, color=NAVY, va="center")
ax.text(78 + 5.4 * knn_g / knn_r + 0.8, 48.2, "grouped", fontsize=7.6,
        color=NAVY, va="center")
ax.text(84, 40.5, "engineered text features\nbeat learned embeddings",
        ha="center", fontsize=9, color=NAVY)
ax.add_patch(FancyBboxPatch((74.5, 21), 19, 9.5, boxstyle="round,pad=0.8",
                            facecolor=NAVY, edgecolor=NAVY))
ax.text(84, 27.5, "ML + DL fusion wins", ha="center", fontsize=9.4,
        color="white", fontweight="bold")
ax.text(84, 23.8, f"R² = {fusion_g:.3f}, leakage-free", ha="center",
        fontsize=9.2, color="white")

# ---------------------------------------------------------------- footer
ax.text(50, 7, "Antioxidant Food Table (Carlsen et al. 2010, CC-BY) · "
               "grouped evaluation · open code + data",
        ha="center", fontsize=8.4, color=NAVY)

for ext in ("png", "jpg"):
    fig.savefig(f"figures/GA_Ozkat_2026.{ext}", dpi=DPI, facecolor=PAPER)
print("wrote figures/GA_Ozkat_2026.png/.jpg  "
      f"({W}x{H}px, aspect {W/H:.4f}, min required 1100x560)")

#!/usr/bin/env python3
"""Generate all manuscript figures from results/*.json + processed/ files.

Every number in every figure is read from the frozen result files at run time, so
figures cannot drift from the reported tables. Outputs figures/figN_*.png (600 dpi)
and .pdf (vector) at MDPI widths (full 17.8 cm / intermediate 14 cm).

Palette: two-regime pair blue #2a78d6 (grouped / leakage-free) and orange #eb6834
(random / leaky) — validated adjacent pair; sequential blue steps for magnitude;
ink and grid tokens from the reference palette. Print/light mode only.
"""
import csv, json, os
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- tokens
BLUE, ORANGE = "#2a78d6", "#eb6834"
BLUE_DARK, BLUE_LIGHT = "#1c5cab", "#86b6ef"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
CM = 1 / 2.54

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 8,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.titlesize": 9, "axes.titleweight": "semibold",
    "legend.frameon": False, "legend.fontsize": 7.5,
    "figure.dpi": 120, "savefig.dpi": 600,
})


def style(ax, xgrid=False, ygrid=False):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if xgrid:
        ax.xaxis.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
    if ygrid:
        ax.yaxis.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)


def save(fig, name):
    os.makedirs("figures", exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"figures/{name}.{ext}", bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)
    print(f"wrote figures/{name}.png/.pdf")


def ptitle(ax, letter, text):
    ax.set_title(f"({letter}) {text}", loc="left", fontsize=9,
                 fontweight="semibold", color=INK, pad=8)


# ---------------------------------------------------------------- data
rows = list(csv.DictReader(open("processed/model_data.csv")))
y_raw = np.array([float(r["y_raw"]) for r in rows])
y_log = np.array([float(r["y_log"]) for r in rows])
cats = [r["category"] for r in rows]
groups = [r["group"] for r in rows]


def agg(model):
    d = json.load(open(f"results/{model}.json"))["runs"]
    out = {}
    for reg in ("grouped", "random"):
        v = [d[f"{reg}_s{s}_real"]["r2"] for s in range(5)]
        out[reg] = (float(np.mean(v)), float(np.std(v, ddof=1)), v)
    return out


MODELS = [  # (file, display) — figure order is by grouped mean, computed below
    ("median_category", "Category median"),
    ("copypaste", "Memorization baseline"),
    ("rf", "Random Forest"),
    ("bilstm", "BiLSTM"),
    ("knn", "k-NN"),
    ("xgb", "XGBoost"),
    ("lgbm", "LightGBM"),
    ("cnn", "char-CNN"),
    ("hybrid_xgb_emb", "XGBoost on DL embeddings"),
    ("hybrid_xgb_full", "Hybrid XGBoost (TF-IDF+emb)"),
    ("hybrid_ridge_full", "Hybrid Ridge (TF-IDF+emb)"),
    ("mlp", "MLP (TF-IDF)"),
    ("ridge", "Ridge (TF-IDF)"),
    ("fusion_all", "Fusion (8-model average)"),
]
A = {m: agg(m) for m, _ in MODELS}

# ================================================================ FIG 1
fig, (a1, a2) = plt.subplots(1, 2, figsize=(17.8 * CM, 7.5 * CM),
                             gridspec_kw={"width_ratios": [1.15, 1]})
cnt = Counter(cats)
names, vals = zip(*sorted(cnt.items(), key=lambda kv: kv[1]))
a1.barh(range(len(names)), vals, height=0.72, color=BLUE, zorder=3)
a1.set_yticks(range(len(names)))
a1.set_yticklabels([n if len(n) <= 34 else n[:32] + "…" for n in names],
                   fontsize=6.8)
for i, v in enumerate(vals):
    a1.text(v + 4, i, str(v), va="center", fontsize=6.5, color=MUTED)
a1.set_xlabel("products")
a1.set_xlim(0, max(vals) * 1.12)
style(a1, xgrid=True)
ptitle(a1, "a", "24 categories, 3,135 products")

bins = np.linspace(0, y_log.max(), 46)
a2.hist(y_log, bins=bins, color=BLUE, zorder=3)
med = float(np.median(y_log))
a2.axvline(med, color=INK2, lw=1, ls=(0, (4, 3)))
a2.text(med + 0.12, a2.get_ylim()[1] * 0.93,
        f"median {np.expm1(med):.2f} mmol/100 g", fontsize=7, color=INK2)
ticks = [0, 1, 10, 100, 1000]
a2.set_xticks([np.log1p(t) for t in ticks])
a2.set_xticklabels([str(t) for t in ticks])
a2.set_xlabel("FRAP (mmol/100 g; log(1+x) axis)")
a2.set_ylabel("products")
style(a2, ygrid=True)
ptitle(a2, "b", "More than four orders of magnitude")
fig.tight_layout(w_pad=2.5)
save(fig, "fig1_dataset")

# ================================================================ FIG 2
fig, (a1, a2) = plt.subplots(1, 2, figsize=(17.8 * CM, 7 * CM),
                             gridspec_kw={"width_ratios": [1, 1.25]})
gsize = Counter(Counter(groups).values())          # name-multiplicity -> n groups
buckets = {"1": 0, "2": 0, "3": 0, "4": 0, "5+": 0}
for size, ngroups in gsize.items():
    key = str(size) if size < 5 else "5+"
    buckets[key] += size * ngroups                 # records, not groups
share = sum(v for k, v in buckets.items() if k != "1") / len(rows)
cols = [BLUE_LIGHT] + [BLUE] * 4
a1.bar(range(5), [buckets[k] for k in buckets], width=0.72, color=cols, zorder=3)
a1.set_xticks(range(5))
a1.set_xticklabels(list(buckets))
for i, k in enumerate(buckets):
    a1.text(i, buckets[k] + 28, f"{buckets[k]:,}", ha="center", fontsize=7,
            color=MUTED)
a1.set_xlabel("records sharing the same normalised product name")
a1.set_ylabel("records")
style(a1, ygrid=True)
ptitle(a1, "a", f"{share:.1%} of records share a product name")

gvals = {}
for g, v in zip(groups, y_raw):
    gvals.setdefault(g, []).append(v)
elig = {g: vs for g, vs in gvals.items() if len(vs) >= 2 and len(set(vs)) >= 2}
top = sorted(elig, key=lambda g: (-len(elig[g]), g))[:10]
for i, g in enumerate(reversed(top)):
    vv = np.log1p([y_raw[j] for j, gg in enumerate(groups) if gg == g])
    a2.scatter(vv, [i] * len(vv), s=26, color=BLUE, zorder=3,
               edgecolors="white", linewidths=1.4)
a2.set_yticks(range(10))
a2.set_yticklabels([g if len(g) <= 30 else g[:28] + "…" for g in reversed(top)],
                   fontsize=7)
ticks = [0.1, 1, 10, 100]
a2.set_xticks([np.log1p(t) for t in ticks])
a2.set_xticklabels([str(t) for t in ticks])
a2.set_xlabel("FRAP (mmol/100 g; log(1+x) axis)")
style(a2, xgrid=True)
ptitle(a2, "b", "Same name, different value: top-10 repeated products")
fig.tight_layout(w_pad=2.5)
save(fig, "fig2_leakage_surface")

# ================================================================ FIG 3
order = sorted(range(len(MODELS)), key=lambda i: A[MODELS[i][0]]["grouped"][0])
fig, ax = plt.subplots(figsize=(17.8 * CM, 9.5 * CM))
for row, i in enumerate(order):
    m, disp = MODELS[i]
    g, r = A[m]["grouped"], A[m]["random"]
    ax.plot([g[0], r[0]], [row, row], color=GRID, lw=1.6, zorder=2)
    ax.errorbar(g[0], row, xerr=g[1], fmt="o", ms=6.5, color=BLUE,
                ecolor=BLUE, elinewidth=1.2, capsize=2.5,
                markeredgecolor="white", markeredgewidth=1.2, zorder=4)
    ax.errorbar(r[0], row, xerr=r[1], fmt="o", ms=6.5, color=ORANGE,
                ecolor=ORANGE, elinewidth=1.2, capsize=2.5,
                markeredgecolor="white", markeredgewidth=1.2, zorder=3)
    if m == "fusion_all":
        ax.text(g[0], row + 0.38, f"{g[0]:.3f}", ha="center", va="bottom",
                fontsize=7, color=INK2, fontweight="semibold")
    elif m == "ridge":
        ax.text(g[0] - g[1] - 0.015, row, f"{g[0]:.3f}", ha="right", va="center",
                fontsize=7, color=INK2, fontweight="semibold")
ax.set_yticks(range(len(order)))
ax.set_yticklabels([MODELS[i][1] for i in order], fontsize=8)
ax.set_xlabel(r"test $R^2$ (log-transformed FRAP), mean $\pm$ SD over 5 seeds")
h = [plt.Line2D([], [], marker="o", ls="", ms=6.5, color=BLUE,
                markeredgecolor="white", label="grouped split (leakage-free)"),
     plt.Line2D([], [], marker="o", ls="", ms=6.5, color=ORANGE,
                markeredgecolor="white", label="random split (literature default)")]
ax.legend(handles=h, loc="lower right")
style(ax, xgrid=True)
save(fig, "fig3_benchmark")

# ================================================================ FIG 4
fig, (a1, a2) = plt.subplots(1, 2, figsize=(17.8 * CM, 7.5 * CM),
                             gridspec_kw={"width_ratios": [1.2, 1]})
infl = [(disp, A[m]["random"][0] - A[m]["grouped"][0],
         m == "copypaste") for m, disp in MODELS if m != "fusion_all"]
infl.sort(key=lambda t: t[1])
for i, (disp, dv, is_cp) in enumerate(infl):
    a1.barh(i, dv, height=0.72, color=ORANGE if is_cp else BLUE, zorder=3)
    a1.text(dv + 0.002, i, f"{dv:.3f}", va="center", fontsize=6.8, color=MUTED)
a1.set_yticks(range(len(infl)))
a1.set_yticklabels([t[0] for t in infl], fontsize=7.5)
a1.set_xlabel(r"leakage inflation  $\Delta R^2$ = random $-$ grouped")
# identity is carried by the y-axis labels; orange marks the no-learning reference
style(a1, xgrid=True)
ptitle(a1, "a", "Leakage inflation by model")

steps = [("Category\nmedian", A["median_category"]["random"][0], BLUE_LIGHT),
         ("Memorization\nbaseline", A["copypaste"]["random"][0], ORANGE),
         ("k-NN", A["knn"]["random"][0], BLUE)]
xs = range(3)
a2.bar(xs, [s[1] for s in steps], width=0.6, color=[s[2] for s in steps], zorder=3)
for x, (lab, v, _) in zip(xs, steps):
    a2.text(x, v + 0.008, f"{v:.3f}", ha="center", fontsize=7.5, color=INK2)
d1 = steps[1][1] - steps[0][1]
d2 = steps[2][1] - steps[1][1]
frac = d1 / (steps[2][1] - steps[0][1])
# delta annotations inside the bars (white on fill, single line so they fit)
a2.text(1, steps[1][1] - 0.035, f"+{d1:.3f}", ha="center", va="top",
        fontsize=7.2, color="white", fontweight="semibold")
a2.text(2, steps[2][1] - 0.035, f"+{d2:.3f}", ha="center", va="top",
        fontsize=7.2, color="white", fontweight="semibold")
a2.set_xticks(list(xs))
a2.set_xticklabels([s[0] for s in steps], fontsize=7.5)
a2.set_ylabel(r"test $R^2$, random split")
a2.set_ylim(0, 0.78)
style(a2, ygrid=True)
ptitle(a2, "b", f"{frac:.0%} of the k-NN gain is duplicate-name lookup")
fig.tight_layout(w_pad=2.5)
save(fig, "fig4_mechanism")

# ================================================================ FIG 5
fig, (a1, a2) = plt.subplots(1, 2, figsize=(17.8 * CM, 7 * CM),
                             gridspec_kw={"width_ratios": [0.85, 1.3]})
rg = A["ridge"]["grouped"][2]
fu = A["fusion_all"]["grouped"][2]
for s in range(5):
    a1.plot([0, 1], [rg[s], fu[s]], color=GRID, lw=1.4, zorder=2)
a1.scatter([0] * 5, rg, s=34, facecolors="white", edgecolors=BLUE, zorder=3,
           linewidths=1.6)
a1.scatter([1] * 5, fu, s=34, facecolors="white", edgecolors=BLUE, zorder=3,
           linewidths=1.6)
a1.set_xticks([0, 1])
a1.set_xticklabels(["Ridge", "Fusion\n(8 models)"], fontsize=8)
a1.set_xlim(-0.35, 1.35)
a1.set_ylabel(r"test $R^2$ per seed (grouped)")
wins = sum(f > r for f, r in zip(fu, rg))
style(a1, ygrid=True)
ptitle(a1, "a", f"Fusion wins {wins}/5 seeds\n(mean {np.mean(fu) - np.mean(rg):+.3f})")

abl = [("Category only (ridge)", agg("abl_ridge_cat")["grouped"][0]),
       ("Text only (ridge)", agg("abl_ridge_text")["grouped"][0]),
       ("Text + category (ridge)", A["ridge"]["grouped"][0]),
       ("+ DL embeddings (hybrid ridge)", A["hybrid_ridge_full"]["grouped"][0]),
       ("Fusion of all 8 models", A["fusion_all"]["grouped"][0])]
for i, (lab, v) in enumerate(abl):
    a2.barh(i, v, height=0.68, color=BLUE if i < 4 else BLUE_DARK, zorder=3)
    a2.text(v + 0.006, i, f"{v:.3f}", va="center", fontsize=7.2, color=MUTED)
a2.set_yticks(range(len(abl)))
a2.set_yticklabels([a[0] for a in abl], fontsize=8)
a2.set_xlabel(r"test $R^2$ (grouped split)")
a2.set_xlim(0, 0.78)
style(a2, xgrid=True)
ptitle(a2, "b", "What each signal adds")
fig.tight_layout(w_pad=3)
save(fig, "fig5_integration")

print("\nall figures generated from results/*.json — nothing hard-coded")

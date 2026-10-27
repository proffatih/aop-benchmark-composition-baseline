#!/usr/bin/env python3
"""All manuscript figures, regenerated from the result artefacts and the raw data.

Running this script is the only way figures are produced, so a figure can never
drift away from the numbers it depicts.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))  # vendored pro_style.py
import pro_style  # noqa: E402

pro_style.apply()
C = pro_style.PALETTE

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data_raw/anoxpepred/repo/Data"
RES = ROOT / "results"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
AA = "ACDEFGHIKLMNPQRSTVWY"

PUB = {
    "AnOxPePred": {"P60": 63.87, "P70": 74.13, "P80": 74.32, "P90": 74.25},
    "UniDL4BioPep": {"P60": 63.32, "P70": 71.94, "P80": 73.01, "P90": 72.22},
    "DGAOP": {"P60": 64.68, "P70": 73.22, "P80": 73.50, "P90": 73.43},
    "ChemBERTa": {"P60": 66.10, "P70": 72.93, "P80": 72.15, "P90": 72.65},
    "ESM-2": {"P60": 67.57, "P70": 74.07, "P80": 73.93, "P90": 75.07},
    "AOP-DRL": {"P60": 69.17, "P70": 74.93, "P80": 75.14, "P90": 76.21},
    "BLSTM": {"P60": 64.11, "P70": 74.22, "P80": 74.00, "P90": 74.93},
    "LSTM": {"P60": 64.38, "P70": 73.36, "P80": 74.93, "P90": 74.64},
    "FC": {"P60": 61.45, "P70": 74.57, "P80": 73.08, "P90": 73.01},
}
SPLITS = ["P60", "P70", "P80", "P90"]


def composition(seqs):
    X = np.zeros((len(seqs), len(AA) + 1))
    for i, s in enumerate(seqs):
        s = str(s)
        for j, a in enumerate(AA):
            X[i, j] = s.count(a) / len(s)
        X[i, -1] = len(s)
    return X


d = pd.read_csv(DATA / "03_p60_AO_db.csv", index_col=0)
d["aop"] = ((d.FRS == 1) | (d.Chelator == 1)).astype(int)
d["random_neg"] = (d.Source == "-").astype(int)
conv = pd.read_csv(RES / "08_conventions.csv").set_index("split")

# ====================================================== Figure 1
PUB_AUC = {
    "AnOxPePred": {"P60": 67.75, "P70": 79.03, "P80": 79.11, "P90": 79.00},
    "UniDL4BioPep": {"P60": 71.21, "P70": 79.97, "P80": 80.73, "P90": 80.15},
    "DGAOP": {"P60": 71.86, "P70": 80.16, "P80": 80.38, "P90": 81.09},
    "ChemBERTa": {"P60": 71.03, "P70": 80.14, "P80": 80.29, "P90": 80.33},
    "ESM-2": {"P60": 71.13, "P70": 80.65, "P80": 80.81, "P90": 81.67},
    "AOP-DRL": {"P60": 73.10, "P70": 80.52, "P80": 81.69, "P90": 82.22},
    "BLSTM": {"P60": 71.10, "P70": 79.36, "P80": 80.59, "P90": 81.73},
    "LSTM": {"P60": 68.32, "P70": 80.33, "P80": 80.67, "P90": 81.82},
    "FC": {"P60": 67.51, "P70": 80.39, "P80": 80.90, "P90": 80.55},
}
bsplit = pd.read_csv(RES / "11_baseline_by_split.csv", index_col=0)

fig, axd = plt.subplot_mosaic([["a", "b"], ["c", "c"]], figsize=(10.0, 7.6),
                              height_ratios=[1.0, 0.78])
axes = {0: axd["a"], 1: axd["b"], 2: axd["c"]}

ax = axes[0]
x = np.arange(len(SPLITS))
_MK = ["o", "s", "^", "D", "v", "P", "o", "s", "^"]
for k, (m, v) in enumerate(PUB.items()):
    ax.scatter(x + (k - 4) * 0.070, [v[s] for s in SPLITS], s=24, zorder=3,
               color=C[k % len(C)], marker=_MK[k], label=m,
               edgecolor="white", linewidth=0.4)
base = conv.pooled_acc.reindex(SPLITS).to_numpy()
lo = conv.pooled_lo.reindex(SPLITS).to_numpy()
hi = conv.pooled_hi.reindex(SPLITS).to_numpy()
ax.fill_between(x, lo, hi, color="#444444", alpha=0.16, zorder=1,
                label="composition baseline 95% CI")
ax.plot(x, base, color="#222222", lw=2.0, zorder=4, label="composition baseline (21 features)")
ax.set_xticks(x)
ax.set_xticklabels(SPLITS)
ax.set_ylabel("Accuracy (%)")
ax.set_xlabel("Identity-threshold partitioning")
ax.set_title("(a) Published deep models vs a 21-feature baseline", loc="left")
ax.legend(ncol=2, fontsize=7.8, loc="lower right")

ax = axes[1]
MARKERS = ["o", "s", "^", "D", "v", "P", "o", "s", "^"]
for k, (m, v) in enumerate(PUB_AUC.items()):
    ax.scatter(x + (k - 4) * 0.070, [v[s] for s in SPLITS], s=24, zorder=3,
               color=C[k % len(C)], marker=MARKERS[k], label=m,
               edgecolor="white", linewidth=0.4)
bauc = bsplit.auc.reindex(SPLITS).to_numpy()
blo = bsplit.auc_lo.reindex(SPLITS).to_numpy()
bhi = bsplit.auc_hi.reindex(SPLITS).to_numpy()
ax.fill_between(x, blo, bhi, color="#444444", alpha=0.16, zorder=1)
ax.plot(x, bauc, color="#222222", lw=2.0, zorder=4, label="composition baseline")
ax.set_xticks(x)
ax.set_xticklabels(SPLITS)
ax.set_ylabel("AUC (%)")
ax.set_xlabel("Identity-threshold partitioning")
ax.set_title("(b) On AUC the baseline leads in all 36 cells", loc="left")
ax.legend(ncol=2, fontsize=7.2, loc="lower right")

ax = axes[2]
w = 0.36
ax.bar(x - w / 2, conv.pooled_acc.reindex(SPLITS), w, color=C[0], label="pooled")
ax.errorbar(x - w / 2, conv.pooled_acc.reindex(SPLITS),
            yerr=[conv.pooled_acc.reindex(SPLITS) - conv.pooled_lo.reindex(SPLITS),
                  conv.pooled_hi.reindex(SPLITS) - conv.pooled_acc.reindex(SPLITS)],
            fmt="none", ecolor="#333", capsize=3, lw=1.0)
ax.bar(x + w / 2, conv.fold_acc.reindex(SPLITS), w, color=C[3], label="fold-averaged")
ax.errorbar(x + w / 2, conv.fold_acc.reindex(SPLITS),
            yerr=[conv.fold_acc.reindex(SPLITS) - conv.fold_lo.reindex(SPLITS),
                  conv.fold_hi.reindex(SPLITS) - conv.fold_acc.reindex(SPLITS)],
            fmt="none", ecolor="#333", capsize=3, lw=1.0)
for i, s in enumerate(SPLITS):
    ax.scatter([i], [PUB["AOP-DRL"][s]], marker="_", s=420, color=C[1], zorder=5,
               label="best published" if i == 0 else None)
ax.set_xticks(x)
ax.set_xticklabels(SPLITS)
ax.set_ylim(55, 85)
ax.set_ylabel("Accuracy (%)")
ax.set_xlabel("Identity-threshold partitioning")
ax.set_title("(c) The accuracy gap depends on the convention", loc="left")
ax.legend(fontsize=7.5, loc="upper left")
fig.tight_layout()
fig.savefig(FIG / "fig2_baseline_vs_published.pdf")
fig.savefig(FIG / "fig2_baseline_vs_published.png")
plt.close(fig)
print("fig2 ok")

# ====================================================== Figure 2
fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.0))

ax = axes[0, 0]
counts = [int((d.aop == 1).sum()),
          int(((d.aop == 0) & (d.random_neg == 0)).sum()),
          int(((d.aop == 0) & (d.random_neg == 1)).sum())]
labels = ["antioxidant\n(assayed positive)", "non-antioxidant\n(assayed negative)",
          "assumed negative\n(UniProt fragment,\nnever assayed)"]
bars = ax.bar(range(3), counts, color=[C[2], C[1], C[6]])
for b, c in zip(bars, counts):
    ax.text(b.get_x() + b.get_width() / 2, c + 12, str(c), ha="center", fontsize=9)
ax.set_xticks(range(3))
ax.set_xticklabels(labels, fontsize=7.5)
ax.set_ylabel("Peptides")
ax.set_ylim(0, 800)
ax.set_title("(a) 70% of the negative class was never assayed", loc="left")

ax = axes[0, 1]
comp = pd.read_csv(RES / "c1_composition.csv", index_col=0)
order = comp.sort_values("positives", ascending=False).index
xx = np.arange(len(order))
ax.bar(xx - 0.27, comp.loc[order, "positives"], 0.27, color=C[2], label="antioxidant")
ax.bar(xx, comp.loc[order, "exp_neg"], 0.27, color=C[1], label="assayed negative")
ax.bar(xx + 0.27, comp.loc[order, "random_neg"], 0.27, color=C[6], label="assumed negative")
ax.set_xticks(xx)
ax.set_xticklabels(order, fontsize=7.5)
ax.set_ylabel("Mean residue frequency (%)")
ax.legend(fontsize=7.5)
ax.set_title("(b) Tyr, His and Trp carry the separation", loc="left")

ax = axes[1, 0]
src = d[d.Source != "-"].Source.value_counts()
ax.plot(np.arange(1, len(src) + 1), np.cumsum(src.to_numpy()) / len(d) * 100,
        color=C[0], lw=2.0)
ax.axhline(40.6, color=C[1], ls="--", lw=1.0)
ax.text(len(src) * 0.42, 42.5, "16 papers = 40.6% of the benchmark",
        fontsize=8, color=C[1])
ax.scatter([1], [src.iloc[0] / len(d) * 100], color=C[1], zorder=5, s=30)
ax.annotate(f"largest single paper:\n{src.iloc[0]} peptides ({src.iloc[0]/len(d)*100:.1f}%)",
            xy=(1, src.iloc[0] / len(d) * 100), xytext=(len(src) * 0.34, 20),
            fontsize=8, va="top",
            arrowprops=dict(arrowstyle="->", lw=0.8, color="#555"))
ax.set_ylim(8, 70)
ax.set_xlabel("Source papers, ranked by contribution")
ax.set_ylabel("Cumulative share of benchmark (%)")
ax.set_title("(c) A few studies dominate the data", loc="left")

ax = axes[1, 1]
L = d.Sequence.str.len()
bins = np.arange(1.5, 31.5, 1)
ax.hist(L[d.aop == 1], bins=bins, alpha=0.75, color=C[2], label="antioxidant")
ax.hist(L[d.aop == 0], bins=bins, alpha=0.6, color=C[6], label="negative")
ax.axvline(3.5, color=C[1], ls="--", lw=1.0)
ax.text(3.9, ax.get_ylim()[1] * 0.80,
        f"{(L<=3).mean():.0%} of peptides\nare 2-3 residues", fontsize=8, color=C[1])
ax.set_xlabel("Peptide length (residues)")
ax.set_ylabel("Peptides")
ax.legend(fontsize=7.5)
ax.set_title("(d) The benchmark is dominated by very short peptides", loc="left")
fig.tight_layout()
fig.savefig(FIG / "fig4_benchmark_anatomy.pdf")
fig.savefig(FIG / "fig4_benchmark_anatomy.png")
plt.close(fig)
print("fig3 ok")

# ====================================================== Figure 3
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X = composition(d.Sequence)
m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)).fit(X, d.aop.to_numpy())
coef = m[-1].coef_[0][:20]
idx = np.argsort(coef)
known = set("YWHCM")
colors = [C[2] if AA[i] in known else "#9aa5b1" for i in idx]

fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
ax = axes[0]
ax.barh(range(20), coef[idx], color=colors)
ax.set_yticks(range(20))
ax.set_yticklabels([AA[i] for i in idx])
ax.axvline(0, color="#333", lw=0.8)
ax.set_xlabel("Logistic-regression coefficient (standardised)")
ax.set_title("(a) Classical radical-scavenging residues dominate", loc="left")
ax.legend(handles=[Patch(facecolor=C[2], label="Tyr, Trp, His, Cys, Met"),
                   Patch(facecolor="#9aa5b1", label="other residues")],
          fontsize=8, loc="lower right")

ax = axes[1]
ceil = pd.read_csv(RES / "04_ceiling.csv")
ceil = ceil.sort_values("acc")
cols = [C[1] if "dummy" in n else (C[0] if n.startswith("composition LR") else "#9aa5b1")
        for n in ceil.model]
ax.barh(range(len(ceil)), ceil.acc, color=cols,
        xerr=[ceil.acc - ceil.acc_lo, ceil.acc_hi - ceil.acc], capsize=2.5,
        error_kw=dict(ecolor="#444", lw=0.9))
ax.set_yticks(range(len(ceil)))
ax.set_yticklabels(ceil.model, fontsize=8)
ax.set_xlim(45, 85)
ax.set_xlabel("Accuracy (%), P60 folds")
ax.set_title("(b) Nothing beats plain composition", loc="left")
fig.tight_layout()
fig.savefig(FIG / "fig3_what_was_learned.pdf")
fig.savefig(FIG / "fig3_what_was_learned.png")
plt.close(fig)
print("fig4 ok")

# ====================================================== Figure 4
fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))

ax = axes[0]
ident = pd.read_csv(RES / "c3_identity_violations.csv")
ax.bar(range(4), ident.violation_rate * 100, color=C[0])
for i, (r, w) in enumerate(zip(ident.violation_rate * 100, ident.worst_identity)):
    npair = int(ident.violating_pairs[i])
    ax.text(i, r + 0.00035, f"{npair} pair{'' if npair == 1 else 's'}\nmax id {w:.2f}",
            ha="center", fontsize=8)
ax.set_xticks(range(4))
ax.set_xticklabels(ident.partitioning)
ax.set_ylabel("Cross-fold pairs above the stated threshold (%)")
ax.set_ylim(0, 0.0032)
ax.set_title("(a) Sequence-identity control holds", loc="left")

ax = axes[1]
null = pd.read_csv(RES / "07_placebo_null_A.csv")
A = pd.read_csv(RES / "07_design_A.csv").iloc[0]
ax.hist(null.null_auc_drop, bins=30, color="#9aa5b1", label="placebo groups (null)")
obs = A.auc_pub - A.auc_src
ax.axvline(obs, color=C[1], lw=2.2, label=f"observed, source-disjoint ({obs:+.3f})")
ax.axvline(0, color="#333", lw=0.8, ls=":")
ax.set_xlim(min(null.null_auc_drop.min(), obs) - 0.004, max(null.null_auc_drop.max(), obs) + 0.004)
ax.set_xlabel("Loss in AUC relative to the published folds")
ax.set_ylabel("Placebo draws")
ax.legend(fontsize=8, loc="upper left")
ax.set_title("(b) Grouping costs nothing; grouping by study does", loc="left")
fig.tight_layout()
fig.savefig(FIG / "fig5_protocol_effects.pdf")
fig.savefig(FIG / "fig5_protocol_effects.png")
plt.close(fig)
print("fig5 ok")

print("\nfigures written to", FIG)

# ====================================================== Figure 1 (overview)
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(10.0, 5.6))
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

def box(x, y, w, h, title, body, fc, tc="#111111"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=1.2",
                                linewidth=1.1, edgecolor="#55606b", facecolor=fc))
    ax.text(x + w / 2, y + h - 4.2, title, ha="center", va="top",
            fontsize=9.5, fontweight="bold", color=tc)
    ax.text(x + w / 2, y + h - 10.5, body, ha="center", va="top",
            fontsize=8.0, color=tc, linespacing=1.45)

def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=11, linewidth=1.0, color="#55606b"))

box(26, 76, 48, 21, "Benchmark as distributed (AnOxPePred)",
    "1404 peptides · 687 active / 717 inactive\n"
    "4 partitionings (P60–P90) · median length 4", "#eef2f6")

box(1.5, 40, 30, 27, "Negative class",
    "500 of 717 negatives are\nUniProt fragments, never assayed\n\n"
    "two negative sub-populations\nseparable at AUC 0.770", "#fdf0ee")
box(35, 40, 30, 27, "Study structure",
    "133 source papers;\n113 contribute one class only\n\n"
    "source-disjoint evaluation\ncosts 0.023 AUC (p < 0.002)", "#fdf0ee")
box(68.5, 40, 30, 27, "Sequence identity",
    "11 of 678,120 cross-fold pairs\nexceed the stated threshold\n\n"
    "control is SOUND —\nleakage is not via similarity", "#edf6ee")

box(8, 5, 38, 25, "Baseline comparison",
    "21 features: 20 residue\nfrequencies + length\n\n"
    "exceeds published AUC in\nall 36 model × partition cells", "#e8eef5")
box(54, 5, 38, 25, "What is released",
    "source-disjoint partitions\n+ the baseline as a\nreporting floor\n\n"
    "signal = Tyr, His, Trp (HAT chemistry)", "#e8eef5")

for xs in (16.5, 50, 83.5):
    arrow(50, 76, xs, 67.6)
arrow(16.5, 40, 27, 30.4)
arrow(50, 40, 33, 30.4)
arrow(50, 40, 70, 30.4)
arrow(83.5, 40, 76, 30.4)
fig.tight_layout()
fig.savefig(FIG / "fig1_overview.pdf")
fig.savefig(FIG / "fig1_overview.png")
plt.close(fig)
print("fig1 ok")

# ====================================================== Figure 6 (PPV)
cur = pd.read_csv(RES / "12_ppv_curves.csv")
pts = pd.read_csv(RES / "12_prevalence_precision.csv")
sd = cur[cur.protocol == "source-disjoint"]

fig, ax = plt.subplots(figsize=(7.4, 4.6))
styles = {"default": (C[0], "-", "default cut (p = 0.5)"),
          "high-spec": (C[2], "--", "high-specificity cut (sens $\\geq$ 0.50)")}
for op, g in sd.groupby("operating_point"):
    col, ls, lab = styles[op]
    g = g.sort_values("prevalence")
    ax.plot(g.prevalence * 100, g.ppv, color=col, ls=ls, lw=2.0,
            label=f"{lab}  (sens {g.sens.iloc[0]:.2f}, spec {g.spec.iloc[0]:.2f})")

ax.axvline(48.9, color="#777777", ls=":", lw=1.1)
ax.text(47.4, 0.44, "benchmark\nprevalence 48.9%", fontsize=8, color="#555555",
        ha="right", va="center")

dflt = sd[sd.operating_point == "default"].sort_values("prevalence")
for pi, note, ty in [(0.10, "10%", 0.27), (0.05, "5%", 0.10)]:
    v = float(dflt.iloc[(dflt.prevalence - pi).abs().argmin()].ppv)
    ax.scatter([pi * 100], [v], color=C[1], zorder=5, s=34)
    ax.annotate(f"{note} actives $\\rightarrow$ PPV {v:.2f}",
                xy=(pi * 100, v), xytext=(26, ty), fontsize=8.4, va="center",
                arrowprops=dict(arrowstyle="->", lw=0.8, color="#555",
                                connectionstyle="arc3,rad=0.12"))

ax.set_xlabel("Prevalence of active peptides in the screened pool (%)")
ax.set_ylabel("Positive predictive value")
ax.set_ylim(0, 1.0)
ax.set_xlim(0, 60)
ax.legend(fontsize=8, loc="upper left")
ax.set_title("Benchmark accuracy is not screening precision", loc="left")
fig.tight_layout()
fig.savefig(FIG / "fig6_prevalence_precision.pdf")
fig.savefig(FIG / "fig6_prevalence_precision.png")
plt.close(fig)
print("fig6 ok")

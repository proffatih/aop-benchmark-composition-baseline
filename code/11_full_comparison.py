#!/usr/bin/env python3
"""Complete comparison against all nine published models, on ACC, MCC and AUC.

The published figures below were read directly from the HTML of the source
article (Zhou et al. 2025, Comput Struct Biotechnol J 27:3673-3684,
doi:10.1016/j.csbj.2025.08.014; PMC12800373), Tables for P60, P70, P80 and P90,
not from any secondary summary.  AUC is reported there in per cent.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data_raw/anoxpepred/repo/Data"
OUT = ROOT / "results"
AA = "ACDEFGHIKLMNPQRSTVWY"
NBOOT = 4000

# model -> split -> (ACC%, MCC, AUC%)
PUB = {
    "AOP-DRL":        {"P60": (69.17, 0.38, 73.10), "P70": (74.93, 0.50, 80.52),
                       "P80": (75.14, 0.50, 81.69), "P90": (76.21, 0.52, 82.22)},
    "ESM-2":          {"P60": (67.57, 0.35, 71.13), "P70": (74.07, 0.48, 80.65),
                       "P80": (73.93, 0.48, 80.81), "P90": (75.07, 0.50, 81.67)},
    "BLSTM":          {"P60": (64.11, 0.28, 71.10), "P70": (74.22, 0.48, 79.36),
                       "P80": (74.00, 0.48, 80.59), "P90": (74.93, 0.50, 81.73)},
    "LSTM":           {"P60": (64.38, 0.29, 68.32), "P70": (73.36, 0.46, 80.33),
                       "P80": (74.93, 0.50, 80.67), "P90": (74.64, 0.49, 81.82)},
    "FC":             {"P60": (61.45, 0.23, 67.51), "P70": (74.57, 0.49, 80.39),
                       "P80": (73.08, 0.46, 80.90), "P90": (73.01, 0.46, 80.55)},
    "AnOxPePred-FRS": {"P60": (63.87, 0.32, 67.75), "P70": (74.13, 0.48, 79.03),
                       "P80": (74.32, 0.49, 79.11), "P90": (74.25, 0.49, 79.00)},
    "UniDL4BioPep":   {"P60": (63.32, 0.27, 71.21), "P70": (71.94, 0.44, 79.97),
                       "P80": (73.01, 0.46, 80.73), "P90": (72.22, 0.44, 80.15)},
    "DGAOP":          {"P60": (64.68, 0.30, 71.86), "P70": (73.22, 0.46, 80.16),
                       "P80": (73.50, 0.47, 80.38), "P90": (73.43, 0.47, 81.09)},
    "ChemBERTa":      {"P60": (66.10, 0.30, 71.03), "P70": (72.93, 0.46, 80.14),
                       "P80": (72.15, 0.45, 80.29), "P90": (72.65, 0.45, 80.33)},
}
SPLITS = ["P60", "P70", "P80", "P90"]

lines = []


def say(s=""):
    print(s)
    lines.append(s)


def composition(seqs):
    X = np.zeros((len(seqs), len(AA) + 1))
    for i, s in enumerate(seqs):
        s = str(s)
        for j, a in enumerate(AA):
            X[i, j] = s.count(a) / len(s)
        X[i, -1] = len(s)
    return X


say("# Composition baseline against all nine published models")
say("(published values read from the source article HTML; AUC in per cent)")
say()

rows = []
base = {}
for name in SPLITS:
    d = pd.read_csv(DATA / f"03_p{name[1:]}_AO_db.csv", index_col=0)
    y = (((d.FRS == 1) | (d.Chelator == 1)).astype(int)).to_numpy()
    X = composition(d.Sequence)
    f = d.partition.to_numpy()
    yp, ys = np.zeros(len(y)), np.zeros(len(y))
    for k in np.unique(f):
        te = f == k
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)).fit(X[~te], y[~te])
        yp[te] = m.predict(X[te])
        ys[te] = m.predict_proba(X[te])[:, 1]
    acc, mcc, auc = accuracy_score(y, yp) * 100, matthews_corrcoef(y, yp), roc_auc_score(y, ys) * 100
    rng = np.random.default_rng(0)
    ba, bu = [], []
    for _ in range(NBOOT):
        b = rng.choice(len(y), len(y), replace=True)
        if len(np.unique(y[b])) < 2:
            continue
        ba.append(accuracy_score(y[b], yp[b]) * 100)
        bu.append(roc_auc_score(y[b], ys[b]) * 100)
    base[name] = dict(acc=acc, mcc=mcc, auc=auc,
                      acc_lo=np.percentile(ba, 2.5), acc_hi=np.percentile(ba, 97.5),
                      auc_lo=np.percentile(bu, 2.5), auc_hi=np.percentile(bu, 97.5))

say(f"{'split':>5s} {'baseline ACC':>26s} {'baseline AUC':>26s}")
for s in SPLITS:
    b = base[s]
    say(f"{s:>5s} {b['acc']:>8.2f} [{b['acc_lo']:.2f}, {b['acc_hi']:.2f}]"
        f" {b['auc']:>10.2f} [{b['auc_lo']:.2f}, {b['auc_hi']:.2f}]")
say()

say("## Per-split ranking")
for s in SPLITS:
    b = base[s]
    acc_beat = [m for m in PUB if b["acc"] > PUB[m][s][0]]
    auc_beat = [m for m in PUB if b["auc"] > PUB[m][s][2]]
    best_auc_model = max(PUB, key=lambda m: PUB[m][s][2])
    say(f"  {s}: baseline ACC {b['acc']:.2f} beats {len(acc_beat)}/9"
        f" | baseline AUC {b['auc']:.2f} beats {len(auc_beat)}/9"
        f" (best published AUC: {best_auc_model} {PUB[best_auc_model][s][2]:.2f},"
        f" margin {b['auc']-PUB[best_auc_model][s][2]:+.2f})")
say()

nacc = sum(1 for s in SPLITS for m in PUB if base[s]["acc"] > PUB[m][s][0])
nauc = sum(1 for s in SPLITS for m in PUB if base[s]["auc"] > PUB[m][s][2])
nacc_in = sum(1 for s in SPLITS for m in PUB
              if base[s]["acc_lo"] <= PUB[m][s][0] <= base[s]["acc_hi"])
nauc_in = sum(1 for s in SPLITS for m in PUB
              if base[s]["auc_lo"] <= PUB[m][s][2] <= base[s]["auc_hi"])
tot = len(SPLITS) * len(PUB)
say(f"## Overall across {tot} (model, split) cells")
say(f"  accuracy: baseline exceeds {nacc}/{tot}; {nacc_in}/{tot} published values inside its 95% CI")
say(f"  AUC     : baseline exceeds {nauc}/{tot}; {nauc_in}/{tot} published values inside its 95% CI")
say()
say("  AUC is threshold-free and is not affected by the pooled/fold-averaged ambiguity,")
say("  so it is the cleaner comparison of the two.")

for s in SPLITS:
    for m in PUB:
        rows.append(dict(split=s, model=m, pub_acc=PUB[m][s][0], pub_mcc=PUB[m][s][1],
                         pub_auc=PUB[m][s][2], base_acc=base[s]["acc"], base_mcc=base[s]["mcc"],
                         base_auc=base[s]["auc"],
                         base_acc_lo=base[s]["acc_lo"], base_acc_hi=base[s]["acc_hi"],
                         base_auc_lo=base[s]["auc_lo"], base_auc_hi=base[s]["auc_hi"],
                         beats_acc=base[s]["acc"] > PUB[m][s][0],
                         beats_auc=base[s]["auc"] > PUB[m][s][2]))
pd.DataFrame(rows).to_csv(OUT / "11_full_comparison.csv", index=False)
pd.DataFrame(base).T.to_csv(OUT / "11_baseline_by_split.csv")
(OUT / "11_full_comparison.txt").write_text("\n".join(lines), encoding="utf-8")
print(f"\n[written] {OUT/'11_full_comparison.txt'}")

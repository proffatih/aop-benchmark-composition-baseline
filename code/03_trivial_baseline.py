#!/usr/bin/env python3
"""Does the published benchmark need deep learning?

Evaluates a 21-feature logistic regression (amino-acid composition + length)
using EXACTLY the published fold assignment of the P60..P90 files, so the
numbers are directly comparable with the tables reported by AnOxPePred,
UniDL4BioPep, DGAOP, ChemBERTa and AOP-DRL.

Then re-runs the same model under two stricter protocols:
  S1  experimental negatives only (drop the 500 random strings)
  S2  source-disjoint folds (no source paper spans train and test)
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data_raw/anoxpepred/repo/Data"
OUT = ROOT / "results"
AA = "ACDEFGHIKLMNPQRSTVWY"
RNG = 0

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


def lr():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))


def evaluate(X, y, folds, model_fn):
    """Leave-one-fold-out over a given fold vector; returns pooled ACC/MCC/AUC."""
    yp = np.zeros(len(y))
    ys = np.zeros(len(y))
    for f in np.unique(folds):
        te = folds == f
        tr = ~te
        if len(np.unique(y[tr])) < 2:
            continue
        m = model_fn()
        m.fit(X[tr], y[tr])
        yp[te] = m.predict(X[te])
        ys[te] = m.predict_proba(X[te])[:, 1] if hasattr(m, "predict_proba") else yp[te]
    return (
        accuracy_score(y, yp) * 100,
        matthews_corrcoef(y, yp),
        roc_auc_score(y, ys),
    )


dfs = {}
for t in (60, 70, 80, 90):
    d = pd.read_csv(DATA / f"03_p{t}_AO_db.csv", index_col=0)
    dfs[f"P{t}"] = d

say("# A trivial baseline on the published AnOxPePred partitions")
say()

# --- are the four partitionings actually different? -----------------
say("## 0. Are P60/P70/P80/P90 four different splits?")
base = dfs["P60"].partition.to_numpy()
for name, d in dfs.items():
    same = int((d.partition.to_numpy() == base).sum())
    say(f"  {name}: fold sizes={d.partition.value_counts().sort_index().to_dict()}")
for a, b in [("P70", "P80"), ("P80", "P90"), ("P70", "P90")]:
    eq = bool((dfs[a].partition.to_numpy() == dfs[b].partition.to_numpy()).all())
    say(f"  {a} fold assignment identical to {b}: {eq}")
say()

# --- the headline comparison ----------------------------------------
say("## 1. Published protocol: FRS prediction, published folds")
say(f"  {'split':>6s} {'model':<34s} {'ACC%':>7s} {'MCC':>6s} {'AUC':>6s}")
rows = []
for name, d in dfs.items():
    y = d.FRS.to_numpy()
    folds = d.partition.to_numpy()
    X = composition(d.Sequence)
    for label, fn in [
        ("majority-class dummy", lambda: DummyClassifier(strategy="most_frequent")),
        ("composition LR (21 features)", lr),
        ("1-NN on composition", lambda: KNeighborsClassifier(n_neighbors=1)),
    ]:
        acc, mcc, auc = evaluate(X, y, folds, fn)
        say(f"  {name:>6s} {label:<34s} {acc:7.2f} {mcc:6.3f} {auc:6.3f}")
        rows.append(dict(split=name, model=label, acc=acc, mcc=mcc, auc=auc, protocol="published"))
    say()

say("## 1b. Published numbers for the same splits, for comparison")
say("   (AOP-DRL paper, Tables; ACC% / MCC)")
pub = {
    "AnOxPePred-FRS": {"P60": (63.87, 0.32), "P70": (74.13, 0.48), "P80": (74.32, 0.49), "P90": (74.25, 0.49)},
    "UniDL4BioPep": {"P60": (63.32, 0.27), "P70": (71.94, 0.44), "P80": (73.01, 0.46), "P90": (72.22, 0.44)},
    "DGAOP": {"P60": (64.68, 0.30), "P70": (73.22, 0.46), "P80": (73.50, 0.47), "P90": (73.43, 0.47)},
    "ChemBERTa": {"P60": (66.10, 0.30), "P70": (72.93, 0.46), "P80": (72.15, 0.45), "P90": (72.65, 0.45)},
    "ESM-2": {"P60": (67.57, 0.35), "P70": (74.07, 0.48), "P80": (73.93, 0.48), "P90": (75.07, 0.50)},
    "AOP-DRL": {"P60": (69.17, 0.38), "P70": (74.93, 0.50), "P80": (75.14, 0.50), "P90": (76.21, 0.52)},
}
say(f"  {'model':<18s} " + " ".join(f"{k:>14s}" for k in dfs))
for m, v in pub.items():
    say(f"  {m:<18s} " + " ".join(f"{v[k][0]:7.2f}/{v[k][1]:.2f}" for k in dfs))
say()

# --- stricter protocols ---------------------------------------------
say("## 2. Same model, stricter evaluation protocols (P60 folds)")
d = dfs["P60"].copy()
d["random_neg"] = (d.Source == "-").astype(int)

say(f"  {'protocol':<44s} {'n':>5s} {'ACC%':>7s} {'MCC':>6s} {'AUC':>6s}")

y = d.FRS.to_numpy()
X = composition(d.Sequence)
acc, mcc, auc = evaluate(X, y, d.partition.to_numpy(), lr)
say(f"  {'P0 published folds (as reported above)':<44s} {len(d):5d} {acc:7.2f} {mcc:6.3f} {auc:6.3f}")
rows.append(dict(split="P60", model="composition LR", acc=acc, mcc=mcc, auc=auc, protocol="P0 published"))

# S1: drop the randomly generated negatives
s1 = d[~((d.random_neg == 1))].reset_index(drop=True)
acc, mcc, auc = evaluate(composition(s1.Sequence), s1.FRS.to_numpy(), s1.partition.to_numpy(), lr)
say(f"  {'S1 experimental negatives only':<44s} {len(s1):5d} {acc:7.2f} {mcc:6.3f} {auc:6.3f}")
rows.append(dict(split="P60", model="composition LR", acc=acc, mcc=mcc, auc=auc, protocol="S1 exp-neg only"))

# S2: source-disjoint folds over the literature-sourced subset
s2 = d[d.Source != "-"].reset_index(drop=True)
g = GroupKFold(n_splits=5)
folds_src = np.zeros(len(s2), dtype=int)
for k, (_, te) in enumerate(g.split(s2, s2.FRS, groups=s2.Source)):
    folds_src[te] = k
acc, mcc, auc = evaluate(composition(s2.Sequence), s2.FRS.to_numpy(), folds_src, lr)
say(f"  {'S2 source-disjoint folds (lit. subset)':<44s} {len(s2):5d} {acc:7.2f} {mcc:6.3f} {auc:6.3f}")
rows.append(dict(split="P60", model="composition LR", acc=acc, mcc=mcc, auc=auc, protocol="S2 source-disjoint"))

# S2-control: same subset, same model, but published folds (isolates the effect of S2)
acc, mcc, auc = evaluate(composition(s2.Sequence), s2.FRS.to_numpy(), s2.partition.to_numpy(), lr)
say(f"  {'S2c same subset, published folds (control)':<44s} {len(s2):5d} {acc:7.2f} {mcc:6.3f} {auc:6.3f}")
rows.append(dict(split="P60", model="composition LR", acc=acc, mcc=mcc, auc=auc, protocol="S2c control"))
say()
say("  S2 vs S2c isolates study-level leakage: identical data and model,")
say("  the only change is whether a source paper may span train and test.")

pd.DataFrame(rows).to_csv(OUT / "03_baseline.csv", index=False)
(OUT / "03_trivial_baseline.txt").write_text("\n".join(lines), encoding="utf-8")
print(f"\n[written] {OUT/'03_trivial_baseline.txt'}")

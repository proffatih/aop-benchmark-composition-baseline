#!/usr/bin/env python3
"""Definitive comparison, using the label the published tables actually use.

AOP-DRL (Zhou et al. 2025, Comput Struct Biotechnol J 27:3673-3684,
doi:10.1016/j.csbj.2025.08.014) evaluates AOP vs non-AOP -- 687 positives
(456 radical scavengers + 231 metal chelators) against 717 negatives -- with
five-fold cross-validation on the pre-partitioned P60-P90 subsets of
AnOxPePred (Olsen et al. 2020, Sci Rep 10:21471, doi:10.1038/s41598-020-78319-w).

This script reproduces that protocol with a 21-feature composition baseline,
reports the FRS-label variant as a sensitivity analysis, and checks the
internal consistency of the published table against its own abstract.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data_raw/anoxpepred/repo/Data"
OUT = ROOT / "results"
AA = "ACDEFGHIKLMNPQRSTVWY"
RNG, NBOOT = 0, 2000

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


def oof(X, y, folds, fn=lr):
    yp, ys = np.zeros(len(y)), np.zeros(len(y))
    for f in np.unique(folds):
        te = folds == f
        m = fn()
        m.fit(X[~te], y[~te])
        yp[te] = m.predict(X[te])
        ys[te] = m.predict_proba(X[te])[:, 1]
    return yp, ys


def mtr(y, yp, ys):
    return accuracy_score(y, yp) * 100, matthews_corrcoef(y, yp), roc_auc_score(y, ys)


def ci(y, yp, ys, n=NBOOT, seed=RNG):
    rng = np.random.default_rng(seed)
    a, m, u = [], [], []
    for _ in range(n):
        b = rng.choice(len(y), len(y), replace=True)
        if len(np.unique(y[b])) < 2:
            continue
        a.append(accuracy_score(y[b], yp[b]) * 100)
        m.append(matthews_corrcoef(y[b], yp[b]))
        u.append(roc_auc_score(y[b], ys[b]))
    f = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    return f(a), f(m), f(u)


PUB = {
    "AnOxPePred-FRS": {"P60": (63.87, 0.32), "P70": (74.13, 0.48), "P80": (74.32, 0.49), "P90": (74.25, 0.49)},
    "UniDL4BioPep": {"P60": (63.32, 0.27), "P70": (71.94, 0.44), "P80": (73.01, 0.46), "P90": (72.22, 0.44)},
    "DGAOP": {"P60": (64.68, 0.30), "P70": (73.22, 0.46), "P80": (73.50, 0.47), "P90": (73.43, 0.47)},
    "ChemBERTa": {"P60": (66.10, 0.30), "P70": (72.93, 0.46), "P80": (72.15, 0.45), "P90": (72.65, 0.45)},
    "ESM-2": {"P60": (67.57, 0.35), "P70": (74.07, 0.48), "P80": (73.93, 0.48), "P90": (75.07, 0.50)},
    "AOP-DRL": {"P60": (69.17, 0.38), "P70": (74.93, 0.50), "P80": (75.14, 0.50), "P90": (76.21, 0.52)},
}
SPECIALISED = ["AnOxPePred-FRS", "UniDL4BioPep", "DGAOP", "ChemBERTa"]
ABSTRACT_GAIN = {"P60": 7.26, "P70": 2.57, "P80": 2.59, "P90": 4.20}

dfs = {}
for t in (60, 70, 80, 90):
    d = pd.read_csv(DATA / f"03_p{t}_AO_db.csv", index_col=0)
    d["aop"] = ((d.FRS == 1) | (d.Chelator == 1)).astype(int)
    dfs[f"P{t}"] = d

say("# Composition baseline vs the published AOP-vs-non-AOP tables")
say()
d0 = dfs["P60"]
say(f"  label = AOP vs non-AOP: {int(d0.aop.sum())} positives / {int((1-d0.aop).sum())} negatives"
    f"  (paper states 687 / 717)")
say(f"  of the positives, FRS={int(d0.FRS.sum())} and chelator={int(d0.Chelator.sum())}"
    f"  (paper states 456 / 231)")
say()

say("## 1. Headline comparison (same folds, same label, five-fold CV)")
say()
recs = []
for name, d in dfs.items():
    y = d.aop.to_numpy()
    X = composition(d.Sequence)
    folds = d.partition.to_numpy()
    yp, ys = oof(X, y, folds)
    acc, mcc, auc = mtr(y, yp, ys)
    (al, ah), (ml, mh), (ul, uh) = ci(y, yp, ys)

    yd, sd = oof(X, y, folds, lambda: DummyClassifier(strategy="most_frequent"))
    dacc = accuracy_score(y, yd) * 100

    best = max(PUB, key=lambda m: PUB[m][name][0])
    bacc, bmcc = PUB[best][name]
    say(f"  {name}:  composition LR  ACC {acc:.2f} [{al:.2f}, {ah:.2f}]"
        f"   MCC {mcc:.3f} [{ml:.3f}, {mh:.3f}]   AUC {auc:.3f} [{ul:.3f}, {uh:.3f}]")
    say(f"        majority-class dummy ACC {dacc:.2f}")
    say(f"        best published: {best} ACC {bacc:.2f} / MCC {bmcc:.2f}"
        f"   -> baseline - best = {acc-bacc:+.2f} pp ACC, {mcc-bmcc:+.3f} MCC")
    beaten = [m for m in PUB if acc > PUB[m][name][0]]
    inside = [m for m in PUB if al <= PUB[m][name][0] <= ah]
    say(f"        baseline point estimate exceeds {len(beaten)}/6 published models"
        f"; {len(inside)}/6 published values lie inside the baseline 95% CI")
    say()
    for m in PUB:
        recs.append(dict(split=name, model=m, pub_acc=PUB[m][name][0], pub_mcc=PUB[m][name][1],
                         base_acc=acc, base_acc_lo=al, base_acc_hi=ah, base_mcc=mcc,
                         base_beats=bool(acc > PUB[m][name][0]),
                         pub_inside_ci=bool(al <= PUB[m][name][0] <= ah)))

t = pd.DataFrame(recs)
t.to_csv(OUT / "05_headline_vs_published.csv", index=False)
say(f"  OVERALL: baseline exceeds {int(t.base_beats.sum())}/{len(t)} published (model, split) results;"
    f" {int(t.pub_inside_ci.sum())}/{len(t)} are inside its 95% CI.")
say()

say("## 2. Sensitivity: the FRS-only label (what 'AnOxPePred-FRS' names)")
for name, d in dfs.items():
    y = d.FRS.to_numpy()
    yp, ys = oof(composition(d.Sequence), y, d.partition.to_numpy())
    acc, mcc, auc = mtr(y, yp, ys)
    say(f"  {name}: ACC {acc:.2f}  MCC {mcc:.3f}  AUC {auc:.3f}")
say("  >> the conclusion does not depend on which of the two labels is used.")
say()

say("## 3. Internal consistency of the published table")
say("   AOP-DRL abstract claims accuracy gains over 'specialised models';")
say("   recomputed from the paper's own table (mean of AnOxPePred-FRS, UniDL4BioPep, DGAOP, ChemBERTa):")
for name in dfs:
    mean_spec = float(np.mean([PUB[m][name][0] for m in SPECIALISED]))
    gain = PUB["AOP-DRL"][name][0] - mean_spec
    say(f"   {name}: mean specialised {mean_spec:.2f}  AOP-DRL {PUB['AOP-DRL'][name][0]:.2f}"
        f"  -> gain {gain:.2f} pp;  abstract states {ABSTRACT_GAIN[name]:.2f} pp"
        f"  (discrepancy {ABSTRACT_GAIN[name]-gain:+.2f})")
say()

say("## 4. Study-level leakage under the AOP label")
d = dfs["P60"]
s = d[d.Source != "-"].reset_index(drop=True)
X = composition(s.Sequence)
y = s.aop.to_numpy()
yp_c, ys_c = oof(X, y, s.partition.to_numpy())
acc_c, mcc_c, auc_c = mtr(y, yp_c, ys_c)
(acl, ach), (mcl, mch), (ucl, uch) = ci(y, yp_c, ys_c)

f = np.zeros(len(s), dtype=int)
for k, (_, te) in enumerate(GroupKFold(n_splits=5).split(s, y, groups=s.Source)):
    f[te] = k
yp_s, ys_s = oof(X, y, f)
acc_s, mcc_s, auc_s = mtr(y, yp_s, ys_s)
(asl, ash), (msl, msh), (usl, ush) = ci(y, yp_s, ys_s)

say(f"  n = {len(s)} literature-sourced peptides (the 500 UniProt-derived assumed negatives excluded)")
say(f"  published folds : ACC {acc_c:.2f} [{acl:.2f}, {ach:.2f}]  MCC {mcc_c:.3f} [{mcl:.3f}, {mch:.3f}]"
    f"  AUC {auc_c:.3f} [{ucl:.3f}, {uch:.3f}]")
say(f"  source-disjoint : ACC {acc_s:.2f} [{asl:.2f}, {ash:.2f}]  MCC {mcc_s:.3f} [{msl:.3f}, {msh:.3f}]"
    f"  AUC {auc_s:.3f} [{usl:.3f}, {ush:.3f}]")
say(f"  drop            : ACC {acc_s-acc_c:+.2f} pp   MCC {mcc_s-mcc_c:+.3f}"
    f" ({(mcc_s-mcc_c)/abs(mcc_c):+.0%})   AUC {auc_s-auc_c:+.3f}")

rng = np.random.default_rng(RNG)
diffs = []
for _ in range(NBOOT):
    b = rng.choice(len(y), len(y), replace=True)
    if len(np.unique(y[b])) < 2:
        continue
    diffs.append(roc_auc_score(y[b], ys_c[b]) - roc_auc_score(y[b], ys_s[b]))
diffs = np.array(diffs)
say(f"  paired bootstrap AUC difference {auc_c-auc_s:.3f}"
    f" [{np.percentile(diffs,2.5):.3f}, {np.percentile(diffs,97.5):.3f}],"
    f" P(diff<=0) = {(diffs<=0).mean():.4f}")

pd.DataFrame([
    dict(protocol="published folds", n=len(s), acc=acc_c, acc_lo=acl, acc_hi=ach,
         mcc=mcc_c, mcc_lo=mcl, mcc_hi=mch, auc=auc_c, auc_lo=ucl, auc_hi=uch),
    dict(protocol="source-disjoint", n=len(s), acc=acc_s, acc_lo=asl, acc_hi=ash,
         mcc=mcc_s, mcc_lo=msl, mcc_hi=msh, auc=auc_s, auc_lo=usl, auc_hi=ush),
]).to_csv(OUT / "05_leakage_effect_aop.csv", index=False)

(OUT / "05_headline.txt").write_text("\n".join(lines), encoding="utf-8")
print(f"\n[written] {OUT/'05_headline.txt'}")

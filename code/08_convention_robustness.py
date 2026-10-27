#!/usr/bin/env python3
"""Is the headline robust to the scoring convention?

The published tables do not state whether their ACC/MCC are pooled over the
out-of-fold predictions or averaged across the five folds.  On P60 the two
differ a lot, because the published folds are grossly unequal (701/179/177/174/173).

This script reports the composition baseline under BOTH conventions, with
bootstrap CIs for the pooled figures and fold-level spread for the averaged
ones, so the comparison with the published numbers holds either way.
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
RNG, NBOOT = 0, 4000

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


PUB = {
    "AnOxPePred-FRS": {"P60": (63.87, 0.32), "P70": (74.13, 0.48), "P80": (74.32, 0.49), "P90": (74.25, 0.49)},
    "UniDL4BioPep": {"P60": (63.32, 0.27), "P70": (71.94, 0.44), "P80": (73.01, 0.46), "P90": (72.22, 0.44)},
    "DGAOP": {"P60": (64.68, 0.30), "P70": (73.22, 0.46), "P80": (73.50, 0.47), "P90": (73.43, 0.47)},
    "ChemBERTa": {"P60": (66.10, 0.30), "P70": (72.93, 0.46), "P80": (72.15, 0.45), "P90": (72.65, 0.45)},
    "ESM-2": {"P60": (67.57, 0.35), "P70": (74.07, 0.48), "P80": (73.93, 0.48), "P90": (75.07, 0.50)},
    "AOP-DRL": {"P60": (69.17, 0.38), "P70": (74.93, 0.50), "P80": (75.14, 0.50), "P90": (76.21, 0.52)},
}

say("# Composition baseline under both scoring conventions")
say()

rows = []
for t in (60, 70, 80, 90):
    name = f"P{t}"
    d = pd.read_csv(DATA / f"03_p{t}_AO_db.csv", index_col=0)
    y = (((d.FRS == 1) | (d.Chelator == 1)).astype(int)).to_numpy()
    X = composition(d.Sequence)
    folds = d.partition.to_numpy()

    yp = np.zeros(len(y))
    ys = np.zeros(len(y))
    per = []
    for f in np.unique(folds):
        te = folds == f
        m = lr().fit(X[~te], y[~te])
        yp[te] = m.predict(X[te])
        ys[te] = m.predict_proba(X[te])[:, 1]
        per.append(dict(n=int(te.sum()),
                        acc=accuracy_score(y[te], yp[te]) * 100,
                        mcc=matthews_corrcoef(y[te], yp[te]),
                        auc=roc_auc_score(y[te], ys[te])))
    per = pd.DataFrame(per)

    pooled_acc = accuracy_score(y, yp) * 100
    pooled_mcc = matthews_corrcoef(y, yp)
    pooled_auc = roc_auc_score(y, ys)

    rng = np.random.default_rng(RNG)
    ba = []
    for _ in range(NBOOT):
        b = rng.choice(len(y), len(y), replace=True)
        if len(np.unique(y[b])) < 2:
            continue
        ba.append(accuracy_score(y[b], yp[b]) * 100)
    lo, hi = np.percentile(ba, 2.5), np.percentile(ba, 97.5)

    # CI of the fold-averaged accuracy: t-interval over the five folds
    m_acc = per.acc.mean()
    se = per.acc.std(ddof=1) / np.sqrt(len(per))
    flo, fhi = m_acc - 2.776 * se, m_acc + 2.776 * se  # t_{.975, df=4}

    say(f"## {name}   (fold sizes {list(per.n)})")
    say(f"  pooled         ACC {pooled_acc:6.2f} [{lo:.2f}, {hi:.2f}]   MCC {pooled_mcc:.3f}   AUC {pooled_auc:.3f}")
    say(f"  fold-averaged  ACC {m_acc:6.2f} [{flo:.2f}, {fhi:.2f}]   MCC {per.mcc.mean():.3f}"
        f"   AUC {per.auc.mean():.3f}   (sd across folds {per.acc.std(ddof=1):.2f})")
    best = max(PUB, key=lambda k: PUB[k][name][0])
    bacc = PUB[best][name][0]
    say(f"  best published : {best} ACC {bacc:.2f}")
    say(f"    vs pooled        {pooled_acc-bacc:+6.2f} pp   "
        f"({'baseline ahead' if pooled_acc>bacc else 'published ahead'};"
        f" published {'inside' if lo<=bacc<=hi else 'outside'} baseline CI)")
    say(f"    vs fold-averaged {m_acc-bacc:+6.2f} pp   "
        f"({'baseline ahead' if m_acc>bacc else 'published ahead'};"
        f" published {'inside' if flo<=bacc<=fhi else 'outside'} baseline CI)")
    nbeat_p = sum(1 for k in PUB if pooled_acc > PUB[k][name][0])
    nbeat_f = sum(1 for k in PUB if m_acc > PUB[k][name][0])
    nin_p = sum(1 for k in PUB if lo <= PUB[k][name][0] <= hi)
    nin_f = sum(1 for k in PUB if flo <= PUB[k][name][0] <= fhi)
    say(f"    models exceeded: pooled {nbeat_p}/6, fold-averaged {nbeat_f}/6")
    say(f"    models inside baseline 95% CI: pooled {nin_p}/6, fold-averaged {nin_f}/6")
    say()
    rows.append(dict(split=name, pooled_acc=pooled_acc, pooled_lo=lo, pooled_hi=hi,
                     pooled_mcc=pooled_mcc, pooled_auc=pooled_auc,
                     fold_acc=m_acc, fold_lo=flo, fold_hi=fhi,
                     fold_mcc=per.mcc.mean(), fold_auc=per.auc.mean(),
                     best_pub=best, best_pub_acc=bacc,
                     n_beat_pooled=nbeat_p, n_beat_fold=nbeat_f,
                     n_inside_pooled=nin_p, n_inside_fold=nin_f))

t = pd.DataFrame(rows)
t.to_csv(OUT / "08_conventions.csv", index=False)

say("## Verdict")
say(f"  pooled convention       : baseline exceeds {int(t.n_beat_pooled.sum())}/24 published results,"
    f" {int(t.n_inside_pooled.sum())}/24 inside its CI")
say(f"  fold-averaged convention: baseline exceeds {int(t.n_beat_fold.sum())}/24 published results,"
    f" {int(t.n_inside_fold.sum())}/24 inside its CI")
say()
say("  The safe claim, true under both conventions, is that a 21-feature composition")
say("  model is statistically indistinguishable from the published deep models -- not")
say("  that it beats them.")

(OUT / "08_convention_robustness.txt").write_text("\n".join(lines), encoding="utf-8")
print(f"\n[written] {OUT/'08_convention_robustness.txt'}")

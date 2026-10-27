#!/usr/bin/env python3
"""Benchmark accuracy is not screening precision.

The benchmark is ~49% positive. A hydrolysate fraction being screened is not:
active peptides are a small minority.  Sensitivity and specificity are
prevalence-invariant, positive predictive value is not, so the same classifier
that looks like a reasonable filter on the benchmark returns far fewer true
actives per positive call at the bench.

PPV(pi) = sens*pi / (sens*pi + (1-spec)*(1-pi))

Computed from the baseline's own out-of-fold predictions, under BOTH protocols,
so the figure is not an illustration but a measurement of this model.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data_raw/anoxpepred/repo/Data"
OUT = ROOT / "results"
AA = "ACDEFGHIKLMNPQRSTVWY"

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


def oof(X, y, folds):
    ys = np.zeros(len(y))
    for f in np.unique(folds):
        te = folds == f
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)).fit(X[~te], y[~te])
        ys[te] = m.predict_proba(X[te])[:, 1]
    return ys


def ppv(sens, spec, pi):
    num = sens * pi
    den = num + (1 - spec) * (1 - pi)
    return np.where(den > 0, num / den, np.nan)


d = pd.read_csv(DATA / "03_p60_AO_db.csv", index_col=0).reset_index(drop=True)
d["aop"] = ((d.FRS == 1) | (d.Chelator == 1)).astype(int)
d["group"] = np.where(d.Source != "-", d.Source, ["U%04d" % i for i in range(len(d))])
X = composition(d.Sequence)
y = d.aop.to_numpy()

say("# Benchmark accuracy is not screening precision")
say(f"  benchmark prevalence: {y.mean():.3f}")
say()

scores = {}
scores["published folds"] = oof(X, y, d.partition.to_numpy())
gf = np.zeros(len(d), dtype=int)
for k, (_, te) in enumerate(GroupKFold(n_splits=5).split(X, y, groups=d.group)):
    gf[te] = k
scores["source-disjoint"] = oof(X, y, gf)

PI = np.array([0.50, 0.20, 0.10, 0.05, 0.02, 0.01])
rows = []
for name, s in scores.items():
    fpr, tpr, thr = roc_curve(y, s)
    spec = 1 - fpr
    # operating point 1: the default 0.5 probability cut
    i_def = int(np.argmin(np.abs(thr - 0.5)))
    # operating point 2: the highest-specificity point that still keeps sens >= 0.50
    ok = np.where(tpr >= 0.50)[0]
    i_hi = ok[np.argmax(spec[ok])]
    for label, i in [("default cut (p=0.5)", i_def), ("high-specificity cut (sens>=0.50)", i_hi)]:
        se, sp = tpr[i], spec[i]
        say(f"  {name:16s} | {label:34s} sens={se:.3f} spec={sp:.3f}")
        vals = ppv(se, sp, PI)
        say("      PPV by prevalence: " + "  ".join(f"pi={p:.2f}->{v:.2f}" for p, v in zip(PI, vals)))
        for p, v in zip(PI, vals):
            rows.append(dict(protocol=name, operating_point=label, sens=se, spec=sp,
                             prevalence=p, ppv=v))
    say()

t = pd.DataFrame(rows)
t.to_csv(OUT / "12_prevalence_precision.csv", index=False)

# headline sentences used in the manuscript
sd = t[(t.protocol == "source-disjoint") & (t.operating_point == "default cut (p=0.5)")]
say("## For the manuscript (source-disjoint, default cut)")
for p in (0.50, 0.10, 0.05):
    v = float(sd[sd.prevalence == p].ppv.iloc[0])
    say(f"  prevalence {p:.0%}  ->  PPV {v:.2f}  ({v*100:.0f} true actives per 100 positive calls)")
say()
say("  Sensitivity and specificity are properties of the classifier; PPV is not.")
say("  The same model that looks usable on a balanced benchmark is a weak filter")
say("  in a pool where actives are rare -- and this calculation still assumes the")
say("  classifier's sensitivity and specificity transfer unchanged to that pool,")
say("  which our source-disjoint result suggests is itself optimistic.")

# full curves for the figure
grid = np.linspace(0.005, 0.60, 400)
curves = []
for name, s in scores.items():
    fpr, tpr, thr = roc_curve(y, s)
    spec = 1 - fpr
    i_def = int(np.argmin(np.abs(thr - 0.5)))
    ok = np.where(tpr >= 0.50)[0]
    i_hi = ok[np.argmax(spec[ok])]
    for label, i in [("default", i_def), ("high-spec", i_hi)]:
        curves.append(pd.DataFrame(dict(protocol=name, operating_point=label,
                                        sens=tpr[i], spec=spec[i],
                                        prevalence=grid, ppv=ppv(tpr[i], spec[i], grid))))
pd.concat(curves).to_csv(OUT / "12_ppv_curves.csv", index=False)

(OUT / "12_prevalence_precision.txt").write_text("\n".join(lines), encoding="utf-8")
print(f"\n[written] {OUT/'12_prevalence_precision.txt'}")

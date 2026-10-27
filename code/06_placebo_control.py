#!/usr/bin/env python3
"""Placebo control for the study-level leakage claim.

Grouped cross-validation can depress a score for reasons that have nothing to do
with study-level leakage: grouped folds are unbalanced in size and class ratio,
and grouping always removes some training diversity.  To separate those nuisance
effects from the effect of interest we re-run the identical procedure with
PLACEBO groups -- random partitions of the same peptides whose group-size
distribution matches the real source blocks exactly, but which carry no
study information.

If the real drop sits inside the placebo null distribution, the claim fails.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data_raw/anoxpepred/repo/Data"
OUT = ROOT / "results"
AA = "ACDEFGHIKLMNPQRSTVWY"
RNG, NPLACEBO = 0, 500

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


def oof(X, y, folds):
    yp, ys = np.zeros(len(y)), np.zeros(len(y))
    for f in np.unique(folds):
        te = folds == f
        if len(np.unique(y[~te])) < 2:
            ys[te] = y[~te].mean() if (~te).sum() else 0.5
            continue
        m = lr().fit(X[~te], y[~te])
        yp[te] = m.predict(X[te])
        ys[te] = m.predict_proba(X[te])[:, 1]
    return yp, ys


def folds_from_groups(groups, y, n=5):
    f = np.zeros(len(y), dtype=int)
    for k, (_, te) in enumerate(GroupKFold(n_splits=n).split(np.zeros(len(y)), y, groups=groups)):
        f[te] = k
    return f


d = pd.read_csv(DATA / "03_p60_AO_db.csv", index_col=0)
d["aop"] = ((d.FRS == 1) | (d.Chelator == 1)).astype(int)
s = d[d.Source != "-"].reset_index(drop=True)
X = composition(s.Sequence)
y = s.aop.to_numpy()

say("# Placebo control for study-level leakage")
say(f"  n = {len(s)} literature-sourced peptides, {len(s.Source.unique())} distinct source papers")
say()

# ---- observed -------------------------------------------------------
pub_folds = s.partition.to_numpy()
yp_p, ys_p = oof(X, y, pub_folds)
auc_p = roc_auc_score(y, ys_p)
mcc_p = matthews_corrcoef(y, yp_p)

src_folds = folds_from_groups(s.Source.to_numpy(), y)
yp_s, ys_s = oof(X, y, src_folds)
auc_s = roc_auc_score(y, ys_s)
mcc_s = matthews_corrcoef(y, yp_s)

obs_auc_drop = auc_p - auc_s
obs_mcc_drop = mcc_p - mcc_s

say("## 1. Fold composition")
say(f"  {'scheme':<20s} {'fold sizes':<34s} {'positive rate per fold'}")
for nm, f in [("published", pub_folds), ("source-disjoint", src_folds)]:
    sizes = [int((f == k).sum()) for k in np.unique(f)]
    rates = [f"{y[f == k].mean():.2f}" for k in np.unique(f)]
    say(f"  {nm:<20s} {str(sizes):<34s} {rates}")
say()
say(f"  observed: AUC {auc_p:.3f} -> {auc_s:.3f} (drop {obs_auc_drop:.3f});"
    f"  MCC {mcc_p:.3f} -> {mcc_s:.3f} (drop {obs_mcc_drop:.3f})")
say()

# ---- placebo --------------------------------------------------------
say(f"## 2. Placebo groups ({NPLACEBO} draws, group-size distribution matched)")
sizes = s.Source.value_counts().to_numpy()
rng = np.random.default_rng(RNG)
null_auc, null_mcc = [], []
for _ in range(NPLACEBO):
    perm = rng.permutation(len(s))
    fake = np.empty(len(s), dtype=int)
    pos = 0
    for gi, gsz in enumerate(sizes):
        fake[perm[pos : pos + gsz]] = gi
        pos += gsz
    ff = folds_from_groups(fake, y)
    ypf, ysf = oof(X, y, ff)
    null_auc.append(auc_p - roc_auc_score(y, ysf))
    null_mcc.append(mcc_p - matthews_corrcoef(y, ypf))
null_auc = np.array(null_auc)
null_mcc = np.array(null_mcc)

say(f"  placebo AUC drop: mean {null_auc.mean():+.4f}, 95% range"
    f" [{np.percentile(null_auc,2.5):+.4f}, {np.percentile(null_auc,97.5):+.4f}]")
say(f"  placebo MCC drop: mean {null_mcc.mean():+.4f}, 95% range"
    f" [{np.percentile(null_mcc,2.5):+.4f}, {np.percentile(null_mcc,97.5):+.4f}]")
say()
p_auc = (null_auc >= obs_auc_drop).mean()
p_mcc = (null_mcc >= obs_mcc_drop).mean()
say(f"  observed AUC drop {obs_auc_drop:+.4f}  -> placebo p = {p_auc:.4f}"
    f"  ({'SURVIVES' if p_auc < 0.05 else 'FAILS'})")
say(f"  observed MCC drop {obs_mcc_drop:+.4f}  -> placebo p = {p_mcc:.4f}"
    f"  ({'SURVIVES' if p_mcc < 0.05 else 'FAILS'})")
say()
say("  >> grouping per se costs almost nothing; what costs is grouping BY SOURCE.")
say()

# ---- how much of the signal is source identity? ---------------------
say("## 3. Can source identity alone predict the label?")
top = s.Source.value_counts()
say(f"  a peptide's source paper explains the label to this degree:")
pure = sum(1 for _, g in s.groupby("Source") if g.aop.nunique() == 1)
say(f"  {pure}/{len(top)} source papers ({pure/len(top):.1%}) contribute peptides of a SINGLE class only")
say(f"  those papers cover {sum(len(g) for _, g in s.groupby('Source') if g.aop.nunique()==1)}"
    f"/{len(s)} peptides")
say()
say("  >> when a paper contributes only positives (or only negatives) and its peptides")
say("  >> straddle the train/test boundary, the model can score by recognising the study.")

pd.DataFrame(dict(null_auc_drop=null_auc, null_mcc_drop=null_mcc)).to_csv(
    OUT / "06_placebo_null.csv", index=False)
pd.DataFrame([dict(obs_auc_drop=obs_auc_drop, obs_mcc_drop=obs_mcc_drop,
                   p_auc=p_auc, p_mcc=p_mcc,
                   auc_pub=auc_p, auc_src=auc_s, mcc_pub=mcc_p, mcc_src=mcc_s)]).to_csv(
    OUT / "06_placebo_summary.csv", index=False)
(OUT / "06_placebo_control.txt").write_text("\n".join(lines), encoding="utf-8")
print(f"\n[written] {OUT/'06_placebo_control.txt'}")

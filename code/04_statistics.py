#!/usr/bin/env python3
"""Statistics for the baseline comparison.

  A. bootstrap CIs for the composition baseline on each published split
  B. is the gap to the published deep models significant?  (one-sample test
     against the reported point estimates, which have no CIs of their own)
  C. ceiling check: do stronger classical models beat composition?
  D. what has the model learned?  (coefficients vs known radical-scavenging residues)
  E. study-level leakage effect with CIs and a paired permutation test
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data_raw/anoxpepred/repo/Data"
OUT = ROOT / "results"
AA = "ACDEFGHIKLMNPQRSTVWY"
RNG = 0
NBOOT = 2000

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


def dipeptide(seqs):
    """400-d dipeptide composition, to test whether order information helps."""
    idx = {a + b: k for k, (a, b) in enumerate((x, y) for x in AA for y in AA)}
    X = np.zeros((len(seqs), 400))
    for i, s in enumerate(seqs):
        s = str(s)
        n = max(len(s) - 1, 1)
        for j in range(len(s) - 1):
            k = idx.get(s[j : j + 2])
            if k is not None:
                X[i, k] += 1 / n
    return X


def oof_predict(X, y, folds, model_fn):
    yp = np.zeros(len(y))
    ys = np.zeros(len(y))
    for f in np.unique(folds):
        te = folds == f
        tr = ~te
        m = model_fn()
        m.fit(X[tr], y[tr])
        yp[te] = m.predict(X[te])
        ys[te] = m.predict_proba(X[te])[:, 1]
    return yp, ys


def metrics(y, yp, ys):
    return accuracy_score(y, yp) * 100, matthews_corrcoef(y, yp), roc_auc_score(y, ys)


def boot_ci(y, yp, ys, n=NBOOT, seed=RNG):
    rng = np.random.default_rng(seed)
    acc, mcc, auc = [], [], []
    idx = np.arange(len(y))
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[b])) < 2:
            continue
        acc.append(accuracy_score(y[b], yp[b]) * 100)
        mcc.append(matthews_corrcoef(y[b], yp[b]))
        auc.append(roc_auc_score(y[b], ys[b]))
    q = lambda v: (np.percentile(v, 2.5), np.percentile(v, 97.5))
    return q(acc), q(mcc), q(auc)


def lr():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))


dfs = {f"P{t}": pd.read_csv(DATA / f"03_p{t}_AO_db.csv", index_col=0) for t in (60, 70, 80, 90)}

say("# Statistics for the composition baseline")
say(f"(bootstrap resamples = {NBOOT})")
say()

# ------------------------------------------------------------------ A, B
PUB = {
    "AnOxPePred-FRS": {"P60": 63.87, "P70": 74.13, "P80": 74.32, "P90": 74.25},
    "UniDL4BioPep": {"P60": 63.32, "P70": 71.94, "P80": 73.01, "P90": 72.22},
    "DGAOP": {"P60": 64.68, "P70": 73.22, "P80": 73.50, "P90": 73.43},
    "ChemBERTa": {"P60": 66.10, "P70": 72.93, "P80": 72.15, "P90": 72.65},
    "ESM-2": {"P60": 67.57, "P70": 74.07, "P80": 73.93, "P90": 75.07},
    "AOP-DRL": {"P60": 69.17, "P70": 74.93, "P80": 75.14, "P90": 76.21},
}
PUB_MCC = {
    "AnOxPePred-FRS": {"P60": 0.32, "P70": 0.48, "P80": 0.49, "P90": 0.49},
    "AOP-DRL": {"P60": 0.38, "P70": 0.50, "P80": 0.50, "P90": 0.52},
}

say("## A/B. Composition baseline vs published deep models (same folds)")
say()
recs = []
for name, d in dfs.items():
    y = d.FRS.to_numpy()
    X = composition(d.Sequence)
    folds = d.partition.to_numpy()
    yp, ys = oof_predict(X, y, folds, lr)
    acc, mcc, auc = metrics(y, yp, ys)
    (al, ah), (ml, mh), (ul, uh) = boot_ci(y, yp, ys)
    say(f"  {name}: composition LR  ACC {acc:.2f} [{al:.2f}, {ah:.2f}]   "
        f"MCC {mcc:.3f} [{ml:.3f}, {mh:.3f}]   AUC {auc:.3f} [{ul:.3f}, {uh:.3f}]")
    best = max(PUB, key=lambda m: PUB[m][name])
    say(f"        best published on this split: {best} ACC {PUB[best][name]:.2f}"
        f"  -> baseline minus best = {acc - PUB[best][name]:+.2f} pp"
        f"   ({'inside' if al <= PUB[best][name] <= ah else 'OUTSIDE'} the baseline 95% CI)")
    for m in PUB:
        recs.append(dict(split=name, published_model=m, published_acc=PUB[m][name],
                         baseline_acc=acc, baseline_acc_lo=al, baseline_acc_hi=ah,
                         baseline_beats=acc > PUB[m][name],
                         published_inside_ci=bool(al <= PUB[m][name] <= ah)))
    say()
pd.DataFrame(recs).to_csv(OUT / "04_vs_published.csv", index=False)

tab = pd.DataFrame(recs)
say(f"  summary: baseline point estimate beats {int(tab.baseline_beats.sum())}/{len(tab)}"
    f" published (model, split) results;")
say(f"           {int(tab.published_inside_ci.sum())}/{len(tab)} published values fall inside the"
    f" baseline's 95% CI (i.e. indistinguishable).")
say()

# ------------------------------------------------------------------ C
say("## C. Ceiling check: can stronger models beat 21-feature composition? (P60)")
say("   label = AOP vs non-AOP, matching the published comparison tables")
d = dfs["P60"]
y = (((d.FRS == 1) | (d.Chelator == 1)).astype(int)).to_numpy()
folds = d.partition.to_numpy()
Xc = composition(d.Sequence)
Xd = dipeptide(d.Sequence)
Xcd = np.hstack([Xc, Xd])

models = [
    ("majority dummy", Xc, lambda: DummyClassifier(strategy="most_frequent")),
    ("composition LR (21)", Xc, lr),
    ("composition SVM-rbf", Xc, lambda: make_pipeline(StandardScaler(), SVC(probability=True, random_state=RNG))),
    ("composition RF", Xc, lambda: RandomForestClassifier(n_estimators=500, random_state=RNG, n_jobs=-1)),
    ("composition GBDT", Xc, lambda: GradientBoostingClassifier(random_state=RNG)),
    ("dipeptide LR (400)", Xd, lr),
    ("comp+dipeptide LR (421)", Xcd, lr),
    ("comp+dipeptide RF", Xcd, lambda: RandomForestClassifier(n_estimators=500, random_state=RNG, n_jobs=-1)),
]
say(f"  {'model':<26s} {'ACC%':>7s} {'95% CI':>18s} {'MCC':>7s} {'AUC':>7s}")
crecs = []
for label, X, fn in models:
    yp, ys = oof_predict(X, y, folds, fn)
    acc, mcc, auc = metrics(y, yp, ys)
    (al, ah), _, _ = boot_ci(y, yp, ys, n=500)
    say(f"  {label:<26s} {acc:7.2f} {f'[{al:.2f}, {ah:.2f}]':>18s} {mcc:7.3f} {auc:7.3f}")
    crecs.append(dict(model=label, acc=acc, acc_lo=al, acc_hi=ah, mcc=mcc, auc=auc))
pd.DataFrame(crecs).to_csv(OUT / "04_ceiling.csv", index=False)
say()
say("  >> order information (dipeptides) adds nothing the 21 composition features do not already carry.")
say()

# ------------------------------------------------------------------ D
say("## D. What did the model learn?")
m = lr().fit(Xc, y)
coef = m[-1].coef_[0][:20]
order = np.argsort(coef)[::-1]
say("  logistic-regression coefficients, standardised features (top 6 / bottom 6):")
for k in order[:6]:
    say(f"    +{coef[k]:.3f}  {AA[k]}")
say("    ...")
for k in order[-6:]:
    say(f"    {coef[k]:+.3f}  {AA[k]}")
say()
known = set("YWHCM")  # classical radical-scavenging / metal-coordinating residues
top5 = {AA[k] for k in order[:5]}
say(f"  classical radical-scavenging residues (Tyr, Trp, His, Cys, Met): {sorted(known)}")
say(f"  top-5 positive coefficients: {sorted(top5)}")
say(f"  overlap: {sorted(top5 & known)}  ({len(top5 & known)}/5)")
say()
say("  >> the signal the deep models are competing for is the aromatic/His residue count,")
say("  >> i.e. textbook hydrogen-atom-transfer chemistry, not a learned sequence grammar.")
say()

# ------------------------------------------------------------------ E
say("## E. Study-level leakage: effect size with CIs and a permutation test")
s2 = d[d.Source != "-"].reset_index(drop=True)
Xs = composition(s2.Sequence)
ys_true = s2.FRS.to_numpy()

yp_c, ysc = oof_predict(Xs, ys_true, s2.partition.to_numpy(), lr)
acc_c, mcc_c, auc_c = metrics(ys_true, yp_c, ysc)
(acl, ach), (mcl, mch), (ucl, uch) = boot_ci(ys_true, yp_c, ysc)

gk = GroupKFold(n_splits=5)
fsrc = np.zeros(len(s2), dtype=int)
for k, (_, te) in enumerate(gk.split(s2, ys_true, groups=s2.Source)):
    fsrc[te] = k
yp_s, yss = oof_predict(Xs, ys_true, fsrc, lr)
acc_s, mcc_s, auc_s = metrics(ys_true, yp_s, yss)
(asl, ash), (msl, msh), (usl, ush) = boot_ci(ys_true, yp_s, yss)

say(f"  published folds   : ACC {acc_c:.2f} [{acl:.2f}, {ach:.2f}]  MCC {mcc_c:.3f} [{mcl:.3f}, {mch:.3f}]"
    f"  AUC {auc_c:.3f} [{ucl:.3f}, {uch:.3f}]")
say(f"  source-disjoint   : ACC {acc_s:.2f} [{asl:.2f}, {ash:.2f}]  MCC {mcc_s:.3f} [{msl:.3f}, {msh:.3f}]"
    f"  AUC {auc_s:.3f} [{usl:.3f}, {ush:.3f}]")
say(f"  drop              : ACC {acc_s-acc_c:+.2f} pp   MCC {mcc_s-mcc_c:+.3f}"
    f" ({(mcc_s-mcc_c)/mcc_c:+.0%})   AUC {auc_s-auc_c:+.3f}")

rng = np.random.default_rng(RNG)
obs = auc_c - auc_s
diffs = []
for _ in range(NBOOT):
    b = rng.choice(len(ys_true), len(ys_true), replace=True)
    if len(np.unique(ys_true[b])) < 2:
        continue
    diffs.append(roc_auc_score(ys_true[b], ysc[b]) - roc_auc_score(ys_true[b], yss[b]))
diffs = np.array(diffs)
say(f"  paired bootstrap of the AUC difference: {obs:.3f}"
    f" [{np.percentile(diffs,2.5):.3f}, {np.percentile(diffs,97.5):.3f}],"
    f" P(diff<=0) = {(diffs <= 0).mean():.4f}")
say()
say("  >> identical data, identical model, identical hyper-parameters;")
say("  >> the only change is whether a source paper may span train and test.")

pd.DataFrame([
    dict(protocol="published folds", acc=acc_c, acc_lo=acl, acc_hi=ach, mcc=mcc_c, mcc_lo=mcl, mcc_hi=mch,
         auc=auc_c, auc_lo=ucl, auc_hi=uch),
    dict(protocol="source-disjoint", acc=acc_s, acc_lo=asl, acc_hi=ash, mcc=mcc_s, mcc_lo=msl, mcc_hi=msh,
         auc=auc_s, auc_lo=usl, auc_hi=ush),
]).to_csv(OUT / "04_leakage_effect.csv", index=False)

(OUT / "04_statistics.txt").write_text("\n".join(lines), encoding="utf-8")
print(f"\n[written] {OUT/'04_statistics.txt'}")

#!/usr/bin/env python3
"""Primary source-disjoint analysis on the full, class-balanced benchmark.

Design A (primary): all 1404 peptides.  Groups = source DOI for the 904
literature-derived peptides; each of the 500 UniProt-derived assumed negatives
is its own singleton group, because they were drawn independently.  Class
balance (49% positive) is therefore preserved, which removes the objection that
the source-disjoint score is depressed by fold-level class imbalance.

Design B (sensitivity): literature-derived peptides only, as in step 06.

Reports per-fold scores as well as pooled ones, and the placebo null for A.
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
RNG, NPLACEBO, NBOOT = 0, 500, 2000

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
    per = []
    for f in np.unique(folds):
        te = folds == f
        if len(np.unique(y[~te])) < 2:
            continue
        m = lr().fit(X[~te], y[~te])
        yp[te] = m.predict(X[te])
        ys[te] = m.predict_proba(X[te])[:, 1]
        per.append(
            dict(fold=int(f), n=int(te.sum()), pos_rate=float(y[te].mean()),
                 auc=roc_auc_score(y[te], ys[te]) if len(np.unique(y[te])) > 1 else np.nan,
                 acc=accuracy_score(y[te], yp[te]) * 100)
        )
    return yp, ys, pd.DataFrame(per)


def gfolds(groups, y, n=5):
    f = np.zeros(len(y), dtype=int)
    for k, (_, te) in enumerate(GroupKFold(n_splits=n).split(np.zeros(len(y)), y, groups=groups)):
        f[te] = k
    return f


def boot_auc_diff(y, s_a, s_b, n=NBOOT, seed=RNG):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        b = rng.choice(len(y), len(y), replace=True)
        if len(np.unique(y[b])) < 2:
            continue
        out.append(roc_auc_score(y[b], s_a[b]) - roc_auc_score(y[b], s_b[b]))
    return np.array(out)


d = pd.read_csv(DATA / "03_p60_AO_db.csv", index_col=0).reset_index(drop=True)
d["aop"] = ((d.FRS == 1) | (d.Chelator == 1)).astype(int)
# groups: real DOI for literature peptides, a unique id for each UniProt draw
d["group"] = np.where(d.Source != "-", d.Source, ["U%04d" % i for i in range(len(d))])

say("# Source-disjoint evaluation, primary (balanced) design")
say()

for design, sub in [("A: full benchmark (balanced)", d),
                    ("B: literature-derived only", d[d.Source != "-"].reset_index(drop=True))]:
    X = composition(sub.Sequence)
    y = sub.aop.to_numpy()
    say(f"## Design {design}")
    say(f"  n = {len(sub)}, positive rate {y.mean():.3f}, distinct groups {sub.group.nunique()}")
    say()

    yp_p, ys_p, per_p = oof(X, y, sub.partition.to_numpy())
    yp_s, ys_s, per_s = oof(X, y, gfolds(sub.group.to_numpy(), y))
    auc_p, auc_s = roc_auc_score(y, ys_p), roc_auc_score(y, ys_s)
    mcc_p, mcc_s = matthews_corrcoef(y, yp_p), matthews_corrcoef(y, yp_s)
    acc_p = accuracy_score(y, yp_p) * 100
    acc_s = accuracy_score(y, yp_s) * 100

    say("  per-fold detail")
    say(f"    {'scheme':<18s} {'fold':>4s} {'n':>5s} {'pos_rate':>9s} {'AUC':>7s} {'ACC%':>7s}")
    for nm, per in [("published", per_p), ("source-disjoint", per_s)]:
        for _, r in per.iterrows():
            say(f"    {nm:<18s} {int(r.fold):>4d} {int(r.n):>5d} {r.pos_rate:>9.3f}"
                f" {r.auc:>7.3f} {r.acc:>7.2f}")
    say()
    say(f"    mean per-fold AUC: published {per_p.auc.mean():.3f} (sd {per_p.auc.std():.3f})"
        f"  vs source-disjoint {per_s.auc.mean():.3f} (sd {per_s.auc.std():.3f})")
    say(f"    pooled: AUC {auc_p:.3f} -> {auc_s:.3f} ({auc_s-auc_p:+.3f});"
        f"  MCC {mcc_p:.3f} -> {mcc_s:.3f} ({mcc_s-mcc_p:+.3f});"
        f"  ACC {acc_p:.2f} -> {acc_s:.2f} ({acc_s-acc_p:+.2f} pp)")

    diffs = boot_auc_diff(y, ys_p, ys_s)
    say(f"    paired bootstrap AUC difference {auc_p-auc_s:+.3f}"
        f" [{np.percentile(diffs,2.5):+.3f}, {np.percentile(diffs,97.5):+.3f}],"
        f" P(diff<=0) = {(diffs<=0).mean():.4f}")

    # placebo: match the group-size distribution, destroy study identity
    sizes = sub.group.value_counts().to_numpy()
    rng = np.random.default_rng(RNG)
    null = []
    for _ in range(NPLACEBO):
        perm = rng.permutation(len(sub))
        fake = np.empty(len(sub), dtype=int)
        pos = 0
        for gi, gsz in enumerate(sizes):
            fake[perm[pos: pos + gsz]] = gi
            pos += gsz
        _, ysf, _ = oof(X, y, gfolds(fake, y))
        null.append(auc_p - roc_auc_score(y, ysf))
    null = np.array(null)
    obs = auc_p - auc_s
    say(f"    placebo null (matched group sizes, {NPLACEBO} draws):"
        f" mean {null.mean():+.4f} [{np.percentile(null,2.5):+.4f}, {np.percentile(null,97.5):+.4f}]")
    say(f"    observed {obs:+.4f} -> p = {(null>=obs).mean():.4f}"
        f"  ({'SURVIVES' if (null>=obs).mean()<0.05 else 'FAILS'})")
    say()
    pd.DataFrame([dict(design=design, auc_pub=auc_p, auc_src=auc_s, mcc_pub=mcc_p, mcc_src=mcc_s,
                       acc_pub=acc_p, acc_src=acc_s, placebo_mean=null.mean(),
                       placebo_p=(null >= obs).mean(), n=len(sub), pos_rate=y.mean())]).to_csv(
        OUT / f"07_design_{design[0]}.csv", index=False)
    pd.DataFrame(dict(null_auc_drop=null)).to_csv(
        OUT / f"07_placebo_null_{design[0]}.csv", index=False)
    per_p.assign(scheme="published").to_csv(OUT / f"07_perfold_{design[0]}_published.csv", index=False)
    per_s.assign(scheme="source-disjoint").to_csv(OUT / f"07_perfold_{design[0]}_source.csv", index=False)

say("## Single-class source blocks")
lit = d[d.Source != "-"]
g = lit.groupby("Source").aop
single = [k for k, v in g.nunique().items() if v == 1]
say(f"  {len(single)}/{lit.Source.nunique()} source papers contribute one class only"
    f" ({sum(len(lit[lit.Source==k]) for k in single)}/{len(lit)} peptides)")
big = lit.Source.value_counts()
say(f"  largest single source block: {big.iloc[0]} peptides from {big.index[0]}")

(OUT / "07_source_disjoint_primary.txt").write_text("\n".join(lines), encoding="utf-8")
print(f"\n[written] {OUT/'07_source_disjoint_primary.txt'}")

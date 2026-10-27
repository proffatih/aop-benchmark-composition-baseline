#!/usr/bin/env python3
"""Three leakage channels in the AnOxPePred benchmark.

C1  Negative-class artefact : 500/717 negatives are randomly generated strings.
                              Can a composition-only model separate them from the
                              positives?  From the 217 experimental negatives?
C2  Study-level leakage     : positives arrive in large blocks from single papers.
                              How often does a held-out peptide share its source DOI
                              with a training peptide?
C3  Identity-threshold claim: do the published P60..P90 folds actually keep
                              cross-fold identity below the stated threshold?
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data_raw/anoxpepred/repo/Data"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
RNG = 0
AA = "ACDEFGHIKLMNPQRSTVWY"

lines = []


def say(s=""):
    print(s)
    lines.append(s)


def composition(seqs):
    """20-d amino-acid frequency + length. No positional information at all."""
    X = np.zeros((len(seqs), len(AA) + 1))
    for i, s in enumerate(seqs):
        s = str(s)
        for j, a in enumerate(AA):
            X[i, j] = s.count(a) / len(s)
        X[i, -1] = len(s)
    return X


def cv_auc(X, y, n_splits=5):
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=1.0))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RNG)
    return cross_val_score(clf, X, y, cv=cv, scoring="roc_auc")


dfs = {f"P{t}": pd.read_csv(DATA / f"03_p{t}_AO_db.csv", index_col=0) for t in (60, 70, 80, 90)}
p60 = dfs["P60"]
p60 = p60.assign(
    label=((p60.FRS == 1) | (p60.Chelator == 1)).astype(int),
    random_neg=(p60.Source == "-").astype(int),
)

say("# Three leakage channels in the AnOxPePred benchmark")
say()

# ----------------------------------------------------------------- C1
say("## C1. Is the negative class an artefact?")
say()
pos = p60[p60.label == 1]
neg = p60[p60.label == 0]
rneg = neg[neg.random_neg == 1]
eneg = neg[neg.random_neg == 0]
say(f"  positives={len(pos)}  negatives={len(neg)}  (random={len(rneg)}, experimental={len(eneg)})")
say()

say("  C1a. composition-only logistic regression, positives vs ALL negatives")
X = composition(p60.Sequence)
auc = cv_auc(X, p60.label.values)
say(f"       AUC = {auc.mean():.3f} +/- {auc.std():.3f}")

say("  C1b. positives vs RANDOM negatives only  (the easy half)")
sub = pd.concat([pos, rneg])
auc_r = cv_auc(composition(sub.Sequence), sub.label.values)
say(f"       AUC = {auc_r.mean():.3f} +/- {auc_r.std():.3f}")

say("  C1c. positives vs EXPERIMENTAL negatives only  (the honest half)")
sub = pd.concat([pos, eneg])
auc_e = cv_auc(composition(sub.Sequence), sub.label.values)
say(f"       AUC = {auc_e.mean():.3f} +/- {auc_e.std():.3f}")

say("  C1d. random negatives vs experimental negatives  (SAME label - should be AUC~0.5)")
sub = neg.copy()
auc_nn = cv_auc(composition(sub.Sequence), sub.random_neg.values)
say(f"       AUC = {auc_nn.mean():.3f} +/- {auc_nn.std():.3f}")
say()
say(f"  >> the two negative sub-populations are separable at AUC {auc_nn.mean():.3f};")
say(f"  >> performance against random negatives ({auc_r.mean():.3f}) vs experimental ({auc_e.mean():.3f})"
    f" differs by {auc_r.mean()-auc_e.mean():+.3f} AUC on identical features.")
say()

say("  C1e. amino-acid composition, mean frequency (%) by group")
comp = pd.DataFrame(
    {
        "positives": composition(pos.Sequence)[:, :20].mean(0) * 100,
        "random_neg": composition(rneg.Sequence)[:, :20].mean(0) * 100,
        "exp_neg": composition(eneg.Sequence)[:, :20].mean(0) * 100,
    },
    index=list(AA),
)
comp["pos_minus_rand"] = comp.positives - comp.random_neg
for a, row in comp.sort_values("pos_minus_rand", key=abs, ascending=False).head(8).iterrows():
    say(f"       {a}  pos={row.positives:5.2f}  rand={row.random_neg:5.2f}  exp={row.exp_neg:5.2f}"
        f"   delta(pos-rand)={row.pos_minus_rand:+5.2f}")
comp.to_csv(OUT / "c1_composition.csv")
say()

# ----------------------------------------------------------------- C2
say("## C2. Study-level (source DOI) leakage across folds")
say()
for name, df in dfs.items():
    d = df[df.Source != "-"]
    tot = shared = 0
    for f in sorted(df.partition.unique()):
        test = d[d.partition == f]
        train_src = set(d[d.partition != f].Source)
        tot += len(test)
        shared += int(test.Source.isin(train_src).sum())
    say(f"  {name}: of {tot} literature-sourced peptides, {shared} ({shared/tot:.1%}) sit in a test fold"
        f" whose source paper also appears in training")
say()
src_counts = p60[p60.Source != "-"].Source.value_counts()
say(f"  source blocks: {len(src_counts)} papers; largest contributes {src_counts.iloc[0]} peptides"
    f" ({src_counts.iloc[0]/len(p60):.1%} of the whole benchmark)")
say(f"  papers contributing >=10 peptides: {(src_counts>=10).sum()}"
    f"  covering {src_counts[src_counts>=10].sum()} peptides"
    f" ({src_counts[src_counts>=10].sum()/len(p60):.1%})")
say()
say("  spread of the single largest source across folds (P60):")
big = p60[p60.Source == src_counts.index[0]]
say(f"    DOI {src_counts.index[0]}  n={len(big)}")
say(f"    fold distribution: {big.partition.value_counts().sort_index().to_dict()}")
say()

# ----------------------------------------------------------------- C3
say("## C3. Does the stated identity threshold actually hold between folds?")
say()
try:
    from Bio import Align

    aligner = Align.PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 1
    aligner.mismatch_score = 0
    aligner.open_gap_score = -1
    aligner.extend_gap_score = -1
except Exception as e:  # pragma: no cover
    say(f"  [skipped] biopython unavailable: {e}")
    aligner = None


def identity(a, b):
    """CD-HIT style: identical residues in the best global alignment / shorter length."""
    score = aligner.score(a, b)
    return score / min(len(a), len(b))


if aligner is not None:
    # the four partitionings share one sequence set (verified in step 01),
    # so the pairwise identity matrix is computed once and reused.
    seqs_all = dfs["P60"].Sequence.tolist()
    n_all = len(seqs_all)
    cache = OUT / "identity_matrix.npy"
    if cache.exists():
        ID = np.load(cache)
        say(f"  [cached] identity matrix {ID.shape}")
    else:
        say(f"  computing {n_all*(n_all-1)//2:,} pairwise identities ...")
        ID = np.eye(n_all, dtype=np.float32)
        for i in range(n_all):
            si = seqs_all[i]
            li = len(si)
            for j in range(i + 1, n_all):
                sj = seqs_all[j]
                v = aligner.score(si, sj) / min(li, len(sj))
                ID[i, j] = ID[j, i] = v
        np.save(cache, ID)
        say(f"  [written] results/identity_matrix.npy")
    say()

    rows = []
    for name, df in dfs.items():
        thr = int(name[1:]) / 100
        assert df.Sequence.tolist() == seqs_all, "sequence order differs between partition files"
        folds = df.partition.to_numpy()
        cross = folds[:, None] != folds[None, :]
        iu = np.triu(np.ones((n_all, n_all), dtype=bool), k=1) & cross
        vals = ID[iu]
        pairs = int(iu.sum())
        viol = int((vals > thr + 1e-9).sum())
        worst = float(vals.max()) if pairs else 0.0
        masked = np.where(cross, ID, -1.0)
        maxid_per_seq = masked.max(axis=1)
        n = n_all
        rows.append(
            dict(
                partitioning=name,
                threshold=thr,
                cross_fold_pairs=pairs,
                violating_pairs=viol,
                violation_rate=viol / pairs,
                worst_identity=worst,
                seqs_with_a_violation=int((maxid_per_seq > thr + 1e-9).sum()),
                n_seqs=n,
                median_max_cross_fold_identity=float(np.median(maxid_per_seq)),
                seqs_with_identical_twin=int((maxid_per_seq >= 0.999).sum()),
            )
        )
        r = rows[-1]
        say(f"  {name} (threshold {thr:.2f}): {r['violating_pairs']:,} / {r['cross_fold_pairs']:,} cross-fold pairs"
            f" exceed it ({r['violation_rate']:.2%}); worst={r['worst_identity']:.2f}")
        say(f"        sequences with >=1 cross-fold neighbour above threshold:"
            f" {r['seqs_with_a_violation']}/{r['n_seqs']} ({r['seqs_with_a_violation']/r['n_seqs']:.1%})")
        say(f"        sequences with a 100%-identity cross-fold neighbour (substring/exact):"
            f" {r['seqs_with_identical_twin']}")
        say(f"        median of per-sequence max cross-fold identity: {r['median_max_cross_fold_identity']:.3f}")
    pd.DataFrame(rows).to_csv(OUT / "c3_identity_violations.csv", index=False)
    say()
    say("  [written] results/c3_identity_violations.csv")

(OUT / "02_leakage_channels.txt").write_text("\n".join(lines), encoding="utf-8")
print(f"\n[written] {OUT/'02_leakage_channels.txt'}")

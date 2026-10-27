#!/usr/bin/env python3
"""Emit the source-disjoint partitions and the baseline the paper recommends.

Produces release/ :
  aop_source_disjoint_splits.csv   one row per peptide, with the published fold,
                                   the source-disjoint fold, the grouping key and
                                   a flag marking assumed (never-assayed) negatives
  baseline_composition.py          the 21-feature baseline, standalone
  RELEASE_README.md                how to use them and what to report
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data_raw/anoxpepred/repo/Data"
REL = ROOT / "release"
REL.mkdir(exist_ok=True)

frames = []
for t in (60, 70, 80, 90):
    d = pd.read_csv(DATA / f"03_p{t}_AO_db.csv", index_col=0)
    frames.append(d[["Sequence", "partition"]].rename(columns={"partition": f"fold_P{t}"}))

base = pd.read_csv(DATA / "03_p60_AO_db.csv", index_col=0).reset_index(drop=True)
out = base[["Sequence", "Source", "FRS", "Chelator"]].copy()
out["label_aop"] = ((out.FRS == 1) | (out.Chelator == 1)).astype(int)
out["assumed_negative"] = (out.Source == "-").astype(int)
for t, f in zip((60, 70, 80, 90), frames):
    out[f"fold_P{t}"] = f[f"fold_P{t}"].to_numpy()

# grouping key: real source DOI, or a unique id per independently drawn UniProt fragment
out["group"] = np.where(out.Source != "-", out.Source,
                        ["UNIPROT_%04d" % i for i in range(len(out))])

gk = GroupKFold(n_splits=5)
fold = np.zeros(len(out), dtype=int)
for k, (_, te) in enumerate(gk.split(np.zeros(len(out)), out.label_aop, groups=out.group)):
    fold[te] = k
out["fold_source_disjoint"] = fold

out.to_csv(REL / "aop_source_disjoint_splits.csv", index=False)

# sanity checks that must hold for the release to be valid
assert out.Sequence.is_unique, "duplicate sequences"
for k in range(5):
    tr = set(out.loc[out.fold_source_disjoint != k, "group"])
    te = set(out.loc[out.fold_source_disjoint == k, "group"])
    assert not (tr & te), f"group leak in fold {k}"
print("checks passed: sequences unique, no source group spans train and test")

print("\nfold composition (source-disjoint):")
for k, g in out.groupby("fold_source_disjoint"):
    print(f"  fold {k}: n={len(g):4d}  positive rate {g.label_aop.mean():.3f}"
          f"  assumed-negative share {g.assumed_negative.mean():.3f}")

(REL / "baseline_composition.py").write_text('''#!/usr/bin/env python3
"""The 21-feature baseline this benchmark should be reported against.

    python baseline_composition.py aop_source_disjoint_splits.csv [fold_column]

Default fold_column is fold_source_disjoint; pass fold_P60 .. fold_P90 to
reproduce the published protocol.
"""
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

AA = "ACDEFGHIKLMNPQRSTVWY"


def composition(seqs):
    X = np.zeros((len(seqs), len(AA) + 1))
    for i, s in enumerate(seqs):
        s = str(s)
        for j, a in enumerate(AA):
            X[i, j] = s.count(a) / len(s)
        X[i, -1] = len(s)
    return X


def main(path, fold_col="fold_source_disjoint"):
    d = pd.read_csv(path)
    X, y, f = composition(d.Sequence), d.label_aop.to_numpy(), d[fold_col].to_numpy()
    yp, ys = np.zeros(len(y)), np.zeros(len(y))
    for k in np.unique(f):
        te = f == k
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
        m.fit(X[~te], y[~te])
        yp[te] = m.predict(X[te])
        ys[te] = m.predict_proba(X[te])[:, 1]
    rng = np.random.default_rng(0)
    boot = [accuracy_score(y[b], yp[b]) * 100 for b in
            (rng.choice(len(y), len(y), True) for _ in range(2000))]
    print(f"folds from {fold_col}")
    print(f"  accuracy {accuracy_score(y, yp)*100:.2f}% "
          f"[{np.percentile(boot,2.5):.2f}, {np.percentile(boot,97.5):.2f}]")
    print(f"  MCC      {matthews_corrcoef(y, yp):.3f}")
    print(f"  AUC      {roc_auc_score(y, ys):.3f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "fold_source_disjoint")
''', encoding="utf-8")

(REL / "RELEASE_README.md").write_text('''# Source-disjoint partitions for the antioxidant-peptide benchmark

These files accompany the paper *Amino-Acid Composition as a Sufficient Baseline for
Antioxidant Peptide Prediction: An Audit of the Standard Benchmark*.

They do not replace the AnOxPePred dataset; they add a grouping that the original
distribution does not carry. The peptides, sequences and labels are exactly those
distributed with AnOxPePred (Olsen et al., Sci Rep 10:21471, 2020).

## aop_source_disjoint_splits.csv

| column | meaning |
|---|---|
| `Sequence` | peptide, 20 standard amino acids |
| `Source` | DOI of the publication the peptide was taken from; `-` for UniProt-derived fragments |
| `FRS`, `Chelator` | the two activity flags as distributed |
| `label_aop` | 1 if either flag is set; this is the label the published comparison tables use |
| `assumed_negative` | 1 for the 500 UniProt-derived fragments that were **never assayed** |
| `fold_P60` ... `fold_P90` | the fold assignments as distributed, for reproducing published protocols |
| `group` | grouping key: the source DOI, or a unique id per independently drawn UniProt fragment |
| `fold_source_disjoint` | five-fold assignment in which no source publication spans training and test |

## What to report

1. **Always report the composition baseline.** `baseline_composition.py` implements it in
   40 lines. If a new model does not clearly exceed it, say so.
2. **Report an interval, not a point.** A bootstrap percentile interval over out-of-fold
   predictions takes seconds and prevents one-point differences being read as progress.
3. **State the scoring convention.** Pooled and fold-averaged accuracy differ by almost
   four points on P60, whose folds are unequal (701/179/177/174/173).
4. **Treat `assumed_negative == 1` with care.** Those peptides were not assayed. Consider
   reporting a second figure against assayed negatives only, or using a
   positive--unlabelled formulation.
5. **Do not report chelation metrics without a caveat.** Only 81 of 1404 peptides carry
   the chelation flag and only 11 carry it alone.

## Reproducing the paper's numbers

```bash
python baseline_composition.py aop_source_disjoint_splits.csv fold_P60   # published protocol
python baseline_composition.py aop_source_disjoint_splits.csv            # source-disjoint
```
''', encoding="utf-8")

print(f"\n[written] {REL}")
for p in sorted(REL.iterdir()):
    print("  ", p.name)

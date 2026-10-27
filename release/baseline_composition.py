#!/usr/bin/env python3
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

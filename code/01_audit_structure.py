#!/usr/bin/env python3
"""Structural audit of the AnOxPePred benchmark partitions (P60-P90).

Reads the published CSVs read-only and reports:
  - dataset sizes, duplicate sequences, label conflicts
  - whether the four partitionings share one sequence set
  - composition of the negative class (experimental vs. randomly generated)
  - exact-duplicate leakage across partition folds
"""
import sys
from pathlib import Path
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data_raw/anoxpepred/repo/Data"
OUT = Path(__file__).resolve().parents[1] / "results"
OUT.mkdir(exist_ok=True)

files = {
    "01_raw": "01_AO_db.csv",
    "02_processed": "02_process_AO_db.csv",
    "02_random_neg": "02_random_AO_db.csv",
    "marcatili": "Supplementary_Data_S1_Marcatili.csv",
}
parts = {f"P{t}": f"03_p{t}_AO_db.csv" for t in (60, 70, 80, 90)}

lines = []


def say(s=""):
    print(s)
    lines.append(s)


say("# AnOxPePred benchmark — structural audit")
say(f"source: {DATA}")
say()

say("## 1. File sizes")
dfs = {}
for k, fn in {**files, **parts}.items():
    df = pd.read_csv(DATA / fn, index_col=0)
    dfs[k] = df
    say(f"  {k:14s} rows={len(df):5d}  unique_seq={df.Sequence.nunique():5d}  cols={list(df.columns)}")
say()

say("## 2. Partition files: same sequence set?")
sets = {k: set(dfs[k].Sequence) for k in parts}
base = sets["P60"]
for k, s in sets.items():
    say(f"  {k}: n={len(s)}  identical_to_P60={s == base}  only_here={len(s - base)}  missing={len(base - s)}")
say()

say("## 3. Duplicate sequences inside each partition file")
for k in parts:
    df = dfs[k]
    dup = df[df.Sequence.duplicated(keep=False)].sort_values("Sequence")
    say(f"  {k}: rows={len(df)} unique={df.Sequence.nunique()} duplicated_rows={len(dup)}")
    if len(dup) and k == "P60":
        dup.to_csv(OUT / "duplicates_P60.csv", index=False)
        say(f"      -> written results/duplicates_P60.csv")
        # do duplicates disagree on labels, and do they straddle folds?
        g = dup.groupby("Sequence").agg(
            n=("Sequence", "size"),
            frs=("FRS", "nunique"),
            chel=("Chelator", "nunique"),
            folds=("partition", "nunique"),
        )
        say(f"      duplicate groups={len(g)}  label_conflicting={int(((g.frs > 1) | (g.chel > 1)).sum())}"
            f"  straddling_folds={int((g.folds > 1).sum())}")
        say("      examples:")
        for seq, row in g.head(8).iterrows():
            sub = dup[dup.Sequence == seq]
            say(f"        {seq:28s} n={int(row.n)} FRS={list(sub.FRS)} Chel={list(sub.Chelator)} folds={list(sub.partition)}")
say()

say("## 4. Negative class composition (P60)")
p60 = dfs["P60"]
neg = p60[(p60.FRS == 0) & (p60.Chelator == 0)]
pos = p60[(p60.FRS == 1) | (p60.Chelator == 1)]
say(f"  positives={len(pos)}  negatives={len(neg)}")
say(f"  negatives with Source == '-' (randomly generated): {int((neg.Source == '-').sum())}")
say(f"  negatives with a literature Source (experimental): {int((neg.Source != '-').sum())}")
say(f"  positives with Source == '-': {int((pos.Source == '-').sum())}")
say()
say("  negative Source value counts (top 10):")
for src, n in neg.Source.value_counts().head(10).items():
    say(f"    {str(src)[:60]:60s} {n}")
say()

say("## 5. Source concentration among positives (P60)")
vc = pos.Source.value_counts()
say(f"  distinct sources={len(vc)}; top-5 cover {vc.head(5).sum()}/{len(pos)} = {vc.head(5).sum()/len(pos):.1%}")
for src, n in vc.head(10).items():
    say(f"    {str(src)[:60]:60s} {n}")
say()

say("## 6. Fold sizes and class balance (P60)")
say(f"  {'fold':>5s} {'n':>5s} {'FRS+':>6s} {'Chel+':>6s} {'neg':>6s}")
for f, g in p60.groupby("partition"):
    nneg = int(((g.FRS == 0) & (g.Chelator == 0)).sum())
    say(f"  {int(f):5d} {len(g):5d} {int(g.FRS.sum()):6d} {int(g.Chelator.sum()):6d} {nneg:6d}")
say()

say("## 7. Sequence length distribution (P60)")
L = p60.Sequence.str.len()
say(f"  min={L.min()} max={L.max()} mean={L.mean():.2f} median={L.median()}")
say(f"  length<=3: {(L<=3).sum()} ({(L<=3).mean():.1%})   length<=5: {(L<=5).sum()} ({(L<=5).mean():.1%})")
say("  positives vs negatives mean length: "
    f"pos={pos.Sequence.str.len().mean():.2f}  neg={neg.Sequence.str.len().mean():.2f}")
say()

say("## 8. Non-standard residues")
std = set("ACDEFGHIKLMNPQRSTVWY")
bad = p60[p60.Sequence.apply(lambda s: not set(str(s)) <= std)]
say(f"  sequences with non-standard characters: {len(bad)}")
for s in bad.Sequence.head(10):
    say(f"    {s}")

(OUT / "01_audit_structure.txt").write_text("\n".join(lines), encoding="utf-8")
print(f"\n[written] {OUT/'01_audit_structure.txt'}")

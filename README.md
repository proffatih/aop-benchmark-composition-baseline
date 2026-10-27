# Audit of the antioxidant-peptide prediction benchmark

Reproducibility package for the manuscript *"Amino-Acid Composition as a Sufficient Baseline
for Antioxidant Peptide Prediction: An Audit of the Standard Benchmark"*.

Everything reported in the paper is produced by the scripts here, from public data, with a
fixed seed. Nothing is hand-entered: the published figures we compare against are read from
the source tables and stored in `results/11_full_comparison.csv`.

## The headline, in one command

```bash
cd release
python baseline_composition.py aop_source_disjoint_splits.csv fold_P60
#   accuracy 72.65% [70.30, 75.00]   MCC 0.454   AUC 0.806
python baseline_composition.py aop_source_disjoint_splits.csv
#   accuracy 70.87% [68.45, 73.22]   MCC 0.419   AUC 0.783   (source-disjoint)
```

`baseline_composition.py` is 40 lines: 20 amino-acid frequencies plus peptide length, into a
standardised logistic regression. The first number exceeds the published AUC of all nine deep
models compared on this benchmark, on all four identity partitions — 36 of 36 cells.

## Contents

| Path | Description |
|---|---|
| `code/00_fetch_data.sh` | clones the two upstream repositories into `data_raw/` (not tracked) |
| `code/01_audit_structure.py` | structural audit: duplicates, lengths, source attribution, class balance |
| `code/02_leakage_channels.py` | the three channels: assumed negatives, study structure, sequence identity (984,906 pairwise Needleman–Wunsch identities) |
| `code/03`–`05` | baseline on the published folds, bootstrap intervals, ceiling scan, and the definitive comparison under the published AOP label |
| `code/06_placebo_control.py` | size-matched random groupings as a null for the grouped-CV effect |
| `code/07_source_disjoint_primary.py` | the two source-disjoint designs (balanced primary, literature-only secondary) |
| `code/08_convention_robustness.py` | pooled versus fold-averaged scoring |
| `code/09_figures.py` | all six manuscript figures, from result artefacts only |
| `code/10_release_splits.py` | emits `release/` |
| `code/11_full_comparison.py` | nine published models × four partitions, ACC/MCC/AUC |
| `code/12_prevalence_precision.py` | positive predictive value as a function of screening prevalence |
| `release/` | the source-disjoint partitions and the standalone baseline, for reuse |
| `results/` | every number reported in the paper, as `.txt` logs and `.csv` tables |
| `figures/` | the six figures as submitted (PDF) |

## Data

The benchmark is not redistributed here. `code/00_fetch_data.sh` clones it:

* **AnOxPePred** — Olsen TH, et al. *Sci Rep* **10**:21471 (2020),
  https://github.com/TobiasHeOl/AnOxPePred — 1404 peptides with P60/P70/P80/P90 partitions.
* **AOP-DRL** — https://github.com/Zhouyongzhu/AOP_DRL — source of the published comparison
  tables (doi:10.1016/j.csbj.2025.08.014).

`release/aop_source_disjoint_splits.csv` is the one derived artefact we do distribute. It
carries the sequences, labels and published fold assignments exactly as distributed, plus two
columns the original does not have: `group` (the source publication, or a unique id per
independently drawn UniProt fragment) and `fold_source_disjoint` (a five-fold assignment in
which no source publication spans training and test). `assumed_negative` flags the 500
negatives that were never assayed.

## Reproducing from scratch

```bash
pip install -r requirements.txt
bash code/00_fetch_data.sh
for s in code/0[1-8]*.py code/1[12]*.py; do python -P "$s"; done
python -P code/09_figures.py      # needs results/ from the steps above
```

`02_leakage_channels.py` computes the full 1404×1404 identity matrix and takes a few minutes;
it caches to `results/identity_matrix.npy` (7.9 MB, not tracked — regenerated on demand).
Run scripts with `python -P` so that `data_raw/` is never on the import path.

## Software environment

Python 3.13.12; numpy 2.5.3, pandas 2.3.3, scikit-learn 1.8.0, scipy 1.16.3, matplotlib
3.10.9, biopython 1.88. The analysis is deterministic (seed 0); scikit-learn's logistic
regression solver is stable across recent versions, but pin the above for exact reproduction.

## Checksums (SHA-256)

* `release/aop_source_disjoint_splits.csv` — `bcbfaef5aeecf4580952593c1309708aba03be7a67d269a3c53857dfe81818cb`
* `results/11_full_comparison.csv` — `17ffa96d722b92892c0ee580e3c6dfa6f2c57c9fa666a74338fd2637658d5005`
* `results/12_prevalence_precision.csv` — `c18d230391ebbc0c7788612d1467b8edd22e9c80f958813da1746a53246d7363`

## Licensing

Code is released under the MIT licence (`LICENSE`). `release/aop_source_disjoint_splits.csv`
is a derived work of the AnOxPePred distribution and remains under that project's terms; the
two columns we add (`group`, `fold_source_disjoint`) are released under CC0 1.0.

## Note on scope

This repository contains the analysis code only. The bibliography-harvesting scripts used to
build and Crossref-verify the reference list are deliberately not included: they are
infrastructure, not part of the reported analysis.

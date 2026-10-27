# Source-disjoint partitions for the antioxidant-peptide benchmark

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

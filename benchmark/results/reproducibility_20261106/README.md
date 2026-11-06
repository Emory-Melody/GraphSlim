# GraphSlim comparison results without CentP and GDEM

This artifact records **313 matched settings and 110 method-dataset pairs** across
17 methods and seven datasets. It is the 124-pair roster of the 1.3.1 release with
CentP and GDEM removed from the numerical reference comparison. All other settings,
references and reproduced scores are unchanged.

```bash
python benchmark/results/reproducibility_20261106/verify.py --check
```

The verifier uses only Python's standard library. It first runs the unchanged 1.3.1
verifier on `base_20261105/`, which checks that artifact's hashes and recomputes its
124 pairs. It then removes the methods in `exclusions.json` and recomputes every count
below from the same setting-level scores.

| Metric on the 110-pair roster | <1% | <2% | <5% | <10% |
| --- | ---: | ---: | ---: | ---: |
| One-sided shortfall (primary) | 64/110 | 84/110 | 108/110 (98.18%) | 110/110 (100.00%) |
| Two-sided relative error | 39/110 | 64/110 | 98/110 (89.09%) | 103/110 (93.64%) |

The roster contains 90 historical GC4NC comparisons, four updated GC4NC comparisons
and 16 method-paper comparisons (GECC 6, CGC 5, ClustGDD 5). The four updated GC4NC
comparisons reuse the observations underlying their reference values. Without them,
one-sided shortfall is below 5% for 104/106 pairs and below 10% for 106/106;
two-sided error is below 5% for 94/106 and below 10% for 99/106. Reverting the three
author-confirmed Pubmed corrections gives <5% counts of 107/110 for shortfall and
96/110 for two-sided error. Metric definitions, reference versions and corrections
are documented in `base_20261105/README.md`.

## Why CentP and GDEM are excluded

Defects found while raising test coverage after 1.3.1 also affected the
implementation that generated these methods' GC4NC references:

- **CentP.** The PageRank update multiplied a row-stochastic transition matrix
  without transposing it, so every node with an edge kept the uniform score. The
  defect entered on 2024-05-29 (commit `14499d5`), before GC4NC's first arXiv version.
- **GDEM.** Synthetic features started from uninitialized memory. On Cora and
  Citeseer, the largest-component training map also included validation, test and
  unlabeled nodes with their labels. Both defects date from GDEM's addition on
  2024-08-17.

Both are fixed in commit `6e3974a`. Their published references therefore do not
describe the corrected methods. The decision removes each method on all seven
datasets, including pairs that met the tolerance before and after the fixes. It
was made after the fresh scores below were known.

`rerun_results.json` records those fresh scores for transparency. CentP was rerun on
all seven datasets and GDEM on Cora and Citeseer, using commit `7a2bea0` and the
archived evaluation profiles. These scores do not enter any count.

## Scope

As in the base artifact, these are cumulative records of historical frozen runs,
provenance replays and validation-selected HPO. The verifier recalculates metrics
from frozen scores; it does not retrain methods. K-Center and Herding on the
inductive datasets also changed after the fixes but were not rerun; their archived
scores remain in the roster.

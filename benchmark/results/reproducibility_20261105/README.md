# Fixed GraphSlim 1.3.1 comparison results

This release artifact records **355 matched settings and 124 method-dataset pairs**
across 19 methods and seven datasets. GC4NC is the primary reference under its
unified evaluation protocol. The fixed roster contains 100 historical GC4NC
comparisons, eight updated GC4NC comparisons, and 16 additional method-paper
comparisons (GECC 6, CGC 5, ClustGDD 5). Eight GC4NC comparisons reuse the
observations underlying their reference values; their agreement is not independent
replication. Every pair has a documented numerical reference.

```bash
python benchmark/results/reproducibility_20261105/verify.py --check
```

The directory is portable and requires only Python's standard library.
`settings.json` stores all reference and reproduced scores and their provenance;
`pairs.csv` stores pair aggregates; `summary.json` stores the recalculated results;
`manifest.json` hashes all files. `sources/` retains the extraction inputs.

## Primary metric: one-sided relative shortfall

For a positive reference score a and reproduced score b, compute
`100 * max(0, a - b) / a` for each declared setting. Average these values within
a method-dataset pair, then apply a strict `<` cutoff. Pairs receive equal weight.
Scores at or above reference contribute zero shortfall. This measures performance
retention; it does not establish numerical equivalence. The secondary two-sided
metric uses `100 * abs(a - b) / a` with the same aggregation and roster.

| Metric on the 124-pair roster | <1% | <2% | <5% | <10% |
| --- | ---: | ---: | ---: | ---: |
| One-sided shortfall (primary) | 72/124 | 94/124 | 122/124 (98.39%) | 124/124 (100.00%) |
| Two-sided relative error | 43/124 | 70/124 | 112/124 (90.32%) | 117/124 (94.35%) |

Excluding the eight reused-observation comparisons leaves 116 pairs: one-sided
shortfall is below 5% for 114/116 and below 10% for 116/116; two-sided error is
below 5% for 104/116 and below 10% for 109/116. Original extracted precision is
preserved. Multiple rates, saved graphs and evaluation seeds do not add pairs.
Historical Yelp comparisons use macro-F1; the two updated Yelp comparisons use AP.
Metrics match within each comparison. The shortfall reporting choice followed
the historical results; it is not a preregistered independent benchmark.

## Reference versions and author-confirmed corrections

Reference versions are GC4NC v3 Table 9 (arXiv:2406.16715), GECC v3 Table 9
(arXiv:2502.17614, non-evolving), CGC v2 Table 3 (arXiv:2405.13707, structured),
and ClustGDD v1 Table 3 (arXiv:2505.20807, structured).

[Author-confirmed GC4NC reference corrections](https://github.com/Emory-Melody/GraphSlim/blob/1.3.1/docs/gc4nc_reference_corrections.md)
document the three Pubmed corrections and the exact reference values for the
eight updated GC4NC comparisons. `reference_corrections.json` records the author's
2026-11-05 confirmation. The archived historical commit ID remains provenance;
reviewers can inspect all numerical references directly in this public artifact.

Reverting the three Pubmed corrections gives <5% counts of 121/124 for shortfall
and 110/124 for two-sided error; <10% counts remain 124/124 and 117/124.
The verifier recomputes this sensitivity check.

## Scope

These are cumulative records from historical frozen runs, provenance replays
and validation-selected HPO. The artifact recalculates metrics from frozen scores;
it does not retrain methods or establish a fresh uniform run of this release.
Each pair retains its evaluation profile and source-run labels.
The historical audit inputs remain available for provenance; the manuscript does
not use repository-history reconciliation as its presentation.

The numerical settings and pair results are unchanged from the 2026-11-03
archive. This version updates the public correction status, citation entry point
and documentation while keeping that earlier archive immutable.

# Frozen GraphSlim reproducibility reconciliation, 2026-11-03

This artifact supports the package manuscript's 124-pair result and explains the
previous public README's 123-pair result. The reference and reproduced scores,
rosters, corrections and original extraction files are fixed here. No training,
Torch, network access or files under `runs/` are required to recompute the summary:

```bash
python benchmark/results/reproducibility_20261103/verify.py --check
```

The directory can also be copied anywhere and checked with `python verify.py`.
`manifest.json` contains SHA-256 hashes for every other artifact file, including
the verifier. `settings.json` contains 355 matched setting-level comparisons;
`pairs.csv` contains their 124 pair-level aggregates; `summary.json` contains all
three cohort summaries below. `sources/` preserves the exact extraction inputs
and historical snapshots. The verifier checks hashes, uniqueness, settings,
reference eligibility, both formulas and the complete cohort transition.

## Cohorts and results

All cutoffs are strict `<`, applied after averaging errors within each unique
method–dataset pair. Pairs receive equal weight. Multiple rates, saved edge
graphs and evaluation seeds do not add pairs.

| Cohort | Methods | Pairs | Metric | <5% | <10% |
| --- | ---: | ---: | --- | ---: | ---: |
| Previously public README | 20 | 123 | Two-sided error **plus 7 completion credits** | 111/123 (90.24%) | 116/123 (94.31%) |
| Same historical numerical comparisons, credits removed | 19 | 116 | Two-sided error | 104/116 (89.66%) | 109/116 (93.97%) |
| Same 116 numerical comparisons | 19 | 116 | One-sided shortfall | 114/116 (98.28%) | 116/116 (100.00%) |
| Manuscript, updated numerical comparisons added | 19 | 124 | Two-sided error | 112/124 (90.32%) | 117/124 (94.35%) |
| Manuscript, same 124 pairs | 19 | 124 | One-sided shortfall | 122/124 (98.39%) | 124/124 (100.00%) |

The exact roster transition is **123 − 7 + 8 = 124**. The seven removed Clustering
pairs have no numerical paper reference: their gaps remain missing, and their
completion provides no evidence of numerical agreement. The old numbers remain
reproducible as a historical reporting policy; they are not a numerical-only
reproduction rate. The 116 retained pair records are unchanged between snapshots.

The eight additions are TSpanner and GDEM on each of ogbn-arxiv, Flickr, Reddit
and Yelp. Both methods were already present, so removing Clustering changes 20
methods to 19. All eight additions pass both 5% and 10% cutoffs. They reuse the
archived observations underlying the updated GC4NC reference tables. Their tiny
gaps largely reflect rounding of those observations, **not independent
replication**. Excluding all eight leaves the 116-pair rows above.

The fixed roster is a documented subset of 19 methods × seven datasets; it is
not the whole package method inventory or a complete Cartesian benchmark. All
124 pairs completed their declared settings. No missing references or completion
credits enter either current numerical metric. Published-only GEOM/SFGC rows,
whole-graph baselines and undeclared cells remain excluded.

## Why two metrics are reported

For positive reference score `a` and reproduced score `b`, compute per setting:

- One-sided shortfall: `100 * max(0, a - b) / a`.
- Two-sided relative error: `100 * abs(a - b) / a`.

Then average each quantity over all declared matched settings in a pair, and
apply the strict tolerance to that mean. These are percentages relative to the
reference, not percentage-point differences. An exact 5% error fails `<5%`.
For example, reference 80 and reproduction 88 give zero shortfall but 10%
two-sided error. With two settings, reference `[80, 80]` and reproduction
`[72, 88]`, mean shortfall is 5% and mean two-sided error is 10%; the improvement
at the second setting does not cancel the first setting's shortfall.

Shortfall answers whether the archived implementation loses performance against
a higher-is-better reference. This is useful for assessing performance retention,
but a pass **does not establish numerical agreement, equivalence or identical
protocols**. Two-sided error assesses closeness in both directions and is reported
alongside shortfall on exactly the same roster. The metric change is a disclosed
reporting choice made after historical results existed, not a preregistered
independent benchmark. Neither curve is a statistical significance test.

## References and score precision

There are **100 historical GC4NC comparisons, 16 direct method-paper comparisons
and eight updated GC4NC comparisons**. Source versions and tables are fixed in
`cohorts.json`; every setting has its reference and reproduced percentage in
`settings.json`:

- [GC4NC, arXiv:2406.16715v3](https://arxiv.org/html/2406.16715v3), Table 9,
  with the three declared Pubmed corrections below.
- [GECC, arXiv:2502.17614v3](https://arxiv.org/html/2502.17614v3), Table 9,
  non-evolving GECC: six pairs.
- [CGC, arXiv:2405.13707v2](https://arxiv.org/html/2405.13707v2), Table 3,
  structured CGC: five pairs.
- [ClustGDD, arXiv:2505.20807v1](https://arxiv.org/html/2505.20807v1), Table 3,
  structured ClustGDD: five pairs.
- Updated GC4NC manuscript commit `7ec4c63691f9ba1629e0eee85d73fde78eed4d39`:
  `8appendix.tex` for GDEM and `tables/edge_sparsification.tex` for TSpanner.
  The source snapshot retains their SHA-256 hashes and the 24 result-file hashes.

The adopted Pubmed smallest-budget corrections are:

| Method | Original Table 9 | Adopted reference | Basis |
| --- | ---: | ---: | --- |
| K-Center | 79.18 | 62.91 | Main-text Table 1 |
| Averaging | 74.09 | 75.60 | Main-text Table 1 |
| TSpanner | 62.91 | 79.18 | Inferred Table 9 cell swap; **not an author-confirmed erratum** |

The CLI setting is `0.05` of labeled training nodes; the paper's whole-graph
percentage label is `0.02%`. These units must not be interchanged.
`reference_corrections.json` retains original and adopted values. Reverting all
three corrections gives <5% counts of **121/124 shortfall and 110/124 two-sided
error**, with unchanged <10% counts of 124/124 and 117/124. The verifier also
recomputes this sensitivity analysis.

Historical Yelp settings use macro-F1; only the added TSpanner/Yelp and GDEM/Yelp
settings use AP. Metrics match within each comparison; scores from different
metrics are never directly averaged. Other current Yelp AP tables are separate
results. Original extracted score precision is preserved. The verifier uses the
unrounded archived setting means when provided, rather than the two-decimal
README display values.

## Experiment and verification scope

This is a cumulative historical cohort combining frozen baselines,
provenance-aligned reruns and validation-selected HPO. It is not one uniform run
of the current checkout. Each pair preserves its adopted evaluation profile;
`historical_frozen` is an inherited profile label, not a new run. The setting-level
`source_run` identifies the actual source, including the inherited VN/Yelp repair.
The historical process targeted discrepancies identified in earlier test results;
validation-only parameter selection within later stages does not make the overall
campaign independent of those earlier results. Completed declared stages replaced
predecessors including regressions, rather than choosing the closest test result.

The original GC4NC schedule used declared reduction seeds and ten downstream
GCN fits per reduced graph with validation-selected checkpoints. Individual
provenance replays can differ; this artifact preserves their profile identities.
Original raw-run paths and source/extraction hashes provide provenance. This
artifact reproduces the **reported calculations from frozen setting scores**;
it does not contain datasets, model checkpoints or all original experiment source
archives and does not independently rerun training. Historical hash/finiteness
checks retain their original scope and are not represented as fresh tensor checks.

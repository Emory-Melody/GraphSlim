# GC4NC reference corrections and archived comparison values

**Confirmed by a GC4NC coauthor on 2026-11-05.**
The author confirmed that the following corrections had already been made.
This public note documents the adopted references for GraphSlim's comparison
artifact and makes them inspectable without access to a private manuscript
repository. It does not announce a new arXiv upload or a publisher-issued erratum.

## Pubmed reference corrections

The original source is GC4NC v3 (arXiv:2406.16715), main-text Table 1 and
Appendix Table 9. For the smallest Pubmed budget (the paper's `0.02%` label),
use the following author-confirmed mean scores:

| Method | Original Appendix Table 9 mean | Corrected reference mean |
| --- | ---: | ---: |
| K-Center | 79.18 | 62.91 |
| Averaging | 74.09 | 75.60 |
| TSpanner | 62.91 | 79.18 |

K-Center and Averaging follow the main-text Table 1 values. The K-Center/TSpanner
assignment in Appendix Table 9 was a transcription error. These corrections
are author-confirmed and are not described as an unconfirmed inferred swap.
`reference_corrections.json` in the release artifact records original and adopted
values and the earlier 2026-11-01 decision. Budget labels are preserved from the
reference tables; they are not reinterpreted as retained-edge fractions.

## Updated GC4NC comparison references

The release artifact publicly archives the reference values used for TSpanner
and GDEM on ogbn-arxiv, Flickr, Reddit and Yelp. These eight pairs reuse the
observations underlying their references. The setting identifiers below are
the archived comparison identifiers; the full metric and provenance labels are
in `settings.json`. They provide inspectable reference data, not eight new
independent replications.

| Method | Dataset | Archived setting | Metric | Reference score (%) |
| --- | --- | --- | --- | ---: |
| gdem | flickr | `0.001` | accuracy | 44.61 |
| gdem | flickr | `0.005` | accuracy | 43.88 |
| gdem | flickr | `0.01` | accuracy | 45.3 |
| gdem | ogbn-arxiv | `0.001` | accuracy | 56.25 |
| gdem | ogbn-arxiv | `0.005` | accuracy | 61.41 |
| gdem | ogbn-arxiv | `0.01` | accuracy | 63.68 |
| gdem | reddit | `0.0005` | accuracy | 87.95 |
| gdem | reddit | `0.001` | accuracy | 88.48 |
| gdem | reddit | `0.002` | accuracy | 90.76 |
| gdem | yelp | `0.0005` | AP | 18.05 |
| gdem | yelp | `0.001` | AP | 18.18 |
| gdem | yelp | `0.002` | AP | 18.29 |
| t_spanner | flickr | `graph_1` | accuracy | 47.18 |
| t_spanner | flickr | `graph_2` | accuracy | 47.19 |
| t_spanner | flickr | `graph_3` | accuracy | 47.18 |
| t_spanner | ogbn-arxiv | `graph_1` | accuracy | 69.35 |
| t_spanner | ogbn-arxiv | `graph_2` | accuracy | 69.44 |
| t_spanner | ogbn-arxiv | `graph_3` | accuracy | 69.3 |
| t_spanner | reddit | `graph_1` | accuracy | 94.01 |
| t_spanner | reddit | `graph_2` | accuracy | 93.99 |
| t_spanner | reddit | `graph_3` | accuracy | 94 |
| t_spanner | yelp | `graph_1` | AP | 24.11 |
| t_spanner | yelp | `graph_2` | AP | 24.36 |
| t_spanner | yelp | `graph_3` | AP | 24.1 |

[Fixed release artifact](https://github.com/Emory-Melody/GraphSlim/tree/1.3.1/benchmark/results/reproducibility_20261105)
contains the exact scores, pair-level results, source hashes and offline verifier.

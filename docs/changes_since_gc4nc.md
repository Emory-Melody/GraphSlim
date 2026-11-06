# Changes since GC4NC

GC4NC is a benchmark framework for graph condensation on node classification
(Gong et al., NeurIPS 2025, Datasets and Benchmarks Track;
[published paper](https://proceedings.neurips.cc/paper_files/paper/2025/hash/49caeae02008a3d2c8fc7f178b389c19-Abstract-Datasets_and_Benchmarks_Track.html)).
Its appendix describes an earlier GraphSlim implementation. The current software
contribution is the reusable and maintained graph-reduction package; GC4NC is
one application of that package.

The following inventory describes the current source tree. It does not treat
every shared interface as newly introduced after the earlier paper.

| Area | Current package and extensions beyond the earlier benchmark scope |
| --- | --- |
| Reduction coverage | 49 canonical registry entries: 5 node sparsifiers, 18 edge sparsifiers, 10 coarseners, and 16 condensation methods. Edge sparsification exposes all 15 concrete NetworKit Python sparsifiers plus Spanning Forest, Rank Degree, and t-Spanner. Recent condensation coverage includes GECC, CGC, ClustGDD, and GCGP. |
| Reusable API | Independently callable data, reduction, and evaluation layers; lazy `MethodSpec` registry and `create_reducer`; common `feat_syn`, `adj_syn`, and `labels_syn` output fields; configurable command-line entry points. |
| Compatibility and extension | PyG/DGL graph conversion, eight registered downstream GNN architectures, and reusable loading, initialization, clustering, and evaluation components. |
| Hyperparameter optimization | A shared HPO interface chooses reduction settings using validation data, with explicit CLI overrides taking precedence over configuration defaults. |
| Software quality | Installation diagnostics, regression tests, documentation, examples, and a Python/PyTorch CI matrix. Test results are version-specific; see [testing record](testing_snapshot.md). |
| Reproducibility records | A fixed, offline-verifiable [artifact](https://github.com/Emory-Melody/GraphSlim/blob/1.3.1/benchmark/results/reproducibility_20261105/README.md) stores setting-level inputs, reference provenance, both error formulas, and author-confirmed reference corrections. It explicitly identifies reused observations and historical comparisons. |
| Targeted efficiency work | Shared clustering operators reuse fitted assignments, stream distances, and use sparse or segmented aggregation. [Matched comparisons](../benchmark/results/implementation_comparison_20261031.json) and [GECC comparisons](../benchmark/results/gecc_arxiv_comparison_20261102.json) document method- and budget-dependent behavior, including configurations without a speed improvement. |

The benchmark paper, the present package, and the associated method papers
share algorithms and some experimental observations. This document does not
claim that the 124 reference-backed pairs are independent fresh replications,
that all methods become faster, or that PyPI and Git carry identical features.
Use the permanent source revision specified in the cover letter and installation
instructions for the reviewed feature set.

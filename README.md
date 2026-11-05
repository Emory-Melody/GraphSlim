<div align="center">

# GraphSlim

[English](README.md) | [简体中文](README_zh-CN.md)

A PyTorch library for graph sparsification, coarsening, and condensation.

[Documentation](https://graphslim.readthedocs.io/en/latest/) ·
[GC4NC benchmark paper](https://arxiv.org/abs/2406.16715) ·
[Paper collection](https://github.com/Emory-Melody/awesome-graph-reduction)

<img src="./figures/logo.png" width="300" alt="GraphSlim logo">

</div>

**Frozen source (2026-11-05):** [commit `1a024c6`](https://github.com/Emory-Melody/GraphSlim/tree/1a024c6bf2d3a73e40a3d1facb6dd6b9c1328bab) contains the described 49-method, eight-backbone feature set. This source is versioned 1.3.0; PyPI still distributes the earlier 1.2.1. Install the pinned source below and see its [version-specific testing record](https://github.com/Emory-Melody/GraphSlim/blob/1a024c6bf2d3a73e40a3d1facb6dd6b9c1328bab/docs/testing_snapshot.md).


GraphSlim brings **49 graph reduction methods**, **8 downstream GNN backbones**,
and **validation-only hyperparameter optimization (HPO)** into a shared workflow.
Load a graph, choose a reducer, and evaluate its output through the same interfaces.
The repository includes runnable examples, regression tests, and benchmark
reproduction commands.

The 49 methods comprise **5 node sparsifiers, 18 edge sparsifiers, 10 coarsening
methods, and 16 condensation methods**. Node and edge sparsification together
account for 23 methods; aliases are not counted as additional methods.

The eight backbones are **GCN, SGC, APPNP, ChebNet, GraphSAGE, GAT, GIN, and SGFormer**.
They share the [`DOWNSTREAM_MODELS` registry](graphslim/models/__init__.py) and the
`--eval_model` option; the CLI names for ChebNet and GraphSAGE are `Cheby` and `GraphSage`.

GPU execution uses native PyTorch operations. Install matching Torch and PyG
extension wheels for your GPU and CUDA runtime.
[GPU setup and checks](README.md#gpu-setup-and-checks).

[Quick start](#quick-start) · [Architecture](#architecture) ·
[Reproduction coverage](#reproduction-coverage) · [Clustering acceleration](#clustering-acceleration) ·
[HPO](#hyperparameter-optimization-hpo) · [Methods](#methods)

## Architecture

[![Figure 1. GraphSlim architecture: data, reduction with HPO, and evaluation layers](figures/figure1.png)](figures/figure1.pdf)

Figure 1 depicts the package paper's 49-method development roster. The data, reduction, and evaluation layers are
independently callable. The reduction layer covers sparsification, coarsening,
condensation, and optional HPO. Solid arrows show graph flow; dashed arrows show
node selection or coarsening used to initialize condensation.
[Vector PDF](figures/figure1.pdf) · [Editable source](figures/figure1_architecture.tex).

## Reproduction coverage

The original Yelp pairs retain their historical macro-F1 protocol.
The added TSpanner/Yelp and GDEM/Yelp pairs use AP matched to the revised GC4NC tables;
Other Yelp AP results remain outside this roster.

**124/124 declared method–dataset pairs completed all declared settings.**
**122/124 (98.39%)** meet a mean one-sided relative shortfall below 5%;
**124/124 (100.00%)** meet a mean one-sided relative shortfall below 10%.
The fixed roster spans **19 methods and 7 datasets**, with equal weight per pair.
Pairs without documented numerical references are excluded.

<!-- reproduction-coverage:start -->

[![Cumulative reproduction coverage of 124 documented-reference pairs at 1%, 2%, 5% and 10% one-sided relative-shortfall thresholds](figures/reproduction_coverage.svg)](figures/reproduction_coverage.pdf)

Each column is cumulative for the same **124 pairs with documented numerical references**; blue segments show disjoint measured-shortfall ranges.

[Vector PDF](figures/reproduction_coverage.pdf) · [High-resolution PNG](figures/reproduction_coverage.png) · [Reviewed chart data](benchmark/results/reproduction_coverage_figure.json) · [Figure renderer](benchmark/plot_reproduction_coverage.py)

| Criterion | <1% | <2% | <5% | <10% |
| --- | ---: | ---: | ---: | ---: |
| One-sided relative shortfall | 72/124 (58.06%) | 94/124 (75.81%) | 122/124 (98.39%) | 124/124 (100.00%) |
| Two-sided relative error | 43/124 (34.68%) | 70/124 (56.45%) | 112/124 (90.32%) | 117/124 (94.35%) |

<!-- reproduction-coverage:end -->

One-sided relative shortfall is `100 * max(0, reference - reproduced) / reference`;
a score at or above the reference contributes **0%** shortfall. Compute shortfall per declared
matched setting, average within a method–dataset pair, then apply the strict `<` threshold.
The displayed 1%, 2%, 5% and 10% cutoffs are relative percentages, not percentage-point gaps.
Shortfall measures performance retention for these higher-is-better scores; it does not establish numerical agreement.
We therefore also report two-sided relative error, `100 * abs(reference - reproduced) / reference`,
on the **same 124 pairs**. For reference 80 and reproduction 88, shortfall is 0% and two-sided error is 10%.
Improvements at one setting do not cancel shortfalls at another: clip per setting before averaging.
The metric choice was made after historical results existed; it is not a preregistered independent benchmark.

The denominator is **124 = 100 historical GC4NC pairs + 16 method-paper pairs + 8 updated GC4NC pairs**
(method papers: GECC 6, CGC 5, ClustGDD 5). TSpanner and GDEM each add Arxiv, Flickr, Reddit and Yelp;
their existing three pairs each are not counted again. Each unique method–dataset pair counts once.
Multiple reduction rates, saved edge graphs and evaluation seeds do not add data points.
The eight added pairs use the same archived observations reported in the revised GC4NC tables;
they are documented comparisons rather than independent replication runs. The
[pair-level snapshot](benchmark/results/reproduction_coverage_20261102_gc4nc_extended.json) records all 124 pairs within the 19 × 7 combinations and their provenance.

The previous public README used **123 pairs / 20 methods**, including seven completed Clustering pairs
with unknown numerical gaps that received policy credit at every cutoff. Removing those seven leaves
116 numerical comparisons: two-sided <5% **104/116 (89.66%)**, <10% **109/116 (93.97%)**.
Adding eight documented GC4NC comparisons gives **123 − 7 + 8 = 124 pairs / 19 methods**;
the retained 116 pair records are unchanged. The old 111/123 and 116/123 counts combine measured agreement
with completion credits and are preserved only as historical reporting results.

[Fixed results artifact and full protocol reconciliation](benchmark/results/reproducibility_20261103/README.md)
contains 355 setting-level reference/reproduced scores, per-pair metrics, historical rosters,
source hashes and both metrics. Recompute offline with Python's standard library:

```bash
python benchmark/results/reproducibility_20261103/verify.py --check
```

References are versioned GC4NC and GECC/CGC/ClustGDD tables. The artifact discloses three Pubmed
corrections, including an **inferred, not author-confirmed** TSpanner cell swap. Historical runs,
provenance replays and validation-selected HPO are combined; this is not one uniform current-checkout run.
The artifact reproduces the reported calculations from frozen scores, without rerunning training.

## Clustering acceleration

[Shared clustering operators](graphslim/reduction/cluster_ops.py) reuse fitted
assignments and streamline sparse and segmented aggregation.

<!-- implementation-comparison:start -->

Measured complete reductions on **full ogbn-arxiv: 169,343 nodes and 2,315,598 stored edges after symmetrization**. All three methods share identical saved input tensors, official splits, the same RTX PRO 6000 Blackwell GPU and four CPU threads. GraphSlim/author reduction-time ratios: **GECC 0.84–1.47×**, **CGC 0.19–0.22×**, **ClustGDD 0.43–0.61×**; **lower is better**.

[![Matched GECC, CGC and ClustGDD ogbn-arxiv reductions: time above, peak allocated GPU memory below](figures/clustering_arxiv_comparison.svg)](figures/clustering_arxiv_comparison.pdf)

[Vector PDF](figures/clustering_arxiv_comparison.pdf) · [High-resolution PNG](figures/clustering_arxiv_comparison.png) · [Reviewed chart data](benchmark/results/clustering_arxiv_figure.json)

Columns show GECC, CGC and ClustGDD. **Top: time / matched author median (1), with measured seconds. Bottom: peak allocated GPU memory (GiB). Both metrics: lower is better.** Each bar is the median of three fresh-process reductions; whiskers show observed min–max. All **54 reductions / nine method–budget pairs** are included. Budgets of 0.1%, 0.5% and 1% of the **90,941 labeled training nodes** retain 90, 454 and 909 nodes. GECC adds a separately frozen measurement cohort. Slower results are retained.

<details>
<summary>Measurement protocol and frozen sources</summary>

- Timing includes reducer construction, internal preprocessing and transfers from the common standardized CPU graph, with CUDA synchronization; imports, shared input preparation, saving and GCN evaluation are excluded. GPU peaks reset after warmup, before input transfers.
- Seeds 1–3 alternate implementation order. GECC is measured on the same input tensors and GPU UUID as the existing CGC/ClustGDD cohort, with four CPU threads and TF32 disabled.
- GECC: full transductive propagation, depth two, weights (0.6, 0.5, 0), hard clustering, one restart, maximum 300 iterations. Both implementations use the author's uniform sampled centroid initialization; GraphSlim's native initializer is not measured. Reduced features, adjacency and labels match exactly in all nine rate–seed pairs. The author's dense diagnostic assignments and GraphSlim's sparse diagnostic assignments are included in reduction memory.
- CGC: matched ten initializations / 300 iterations with native FAISS/sklearn convergence rules and Float64 reconstruction. ClustGDD: matched 600 pretraining / 100 refinement updates and three MiniBatchKMeans initializations.
- Outside timing, each reduced graph receives one 300-epoch GCN fit with validation selection. **9/9** method–budget pairs have two-sided relative differences between implementation mean accuracies below 5%. These comparisons use pinned author implementations and have a separate scope from paper-reference reproduction coverage.
- [CGC/ClustGDD records](benchmark/results/arxiv_clustering_comparison_20261101.json) · [measured source archive](benchmark/results/arxiv_clustering_source_20261101.zip).
- [GECC records and pinned author commit](benchmark/results/gecc_arxiv_comparison_20261102.json) · [measured source archive](benchmark/results/gecc_arxiv_source_20261102.zip) · [runner](benchmark/compare_gecc_arxiv.py). GECC requires the pinned author checkout under runs/reference/GECC; the common runtime uses faiss-cpu==1.8.0.post1 with NumPy 1.x.
- Regenerate with `python -m benchmark.plot_clustering_arxiv`; append `--check` to verify all source records and figure hashes.
- Historical synthetic-tensor operator reports remain available separately: [CPU](benchmark/results/gecc_operator_cpu_20261029.json), [CUDA](benchmark/results/gecc_operator_gpu_20261029.json). Their operator timings are excluded from this figure.
</details>

<!-- implementation-comparison:end -->

## Reproduced results

These tables rerun the [GC4NC benchmark](https://arxiv.org/html/2406.16715v3#A1.T9)
with this package. A two-layer GCN with 256 hidden units trains on each reduced
graph for 300 epochs, ten times, and reduced graphs are selected on validation
accuracy. Random, K-Center and Herding use three reduction seeds and the other
methods use one, as in the paper. Cells show mean ± standard deviation of test
accuracy in percent; Yelp shows macro-F1. For node reduction, column headings label
the retained-node budget as a percentage of the
whole graph for Citeseer, Cora, Pubmed and Arxiv, and of the training graph for
Flickr, Reddit and Yelp. These are approximate budget labels; actual node counts depend on integer allocation.

A row is listed when it is on average within 3 points of the published values.
GECC, CGC and ClustGDD are compared with their own papers at the rates those
papers report.
[Frozen results and sources](benchmark/results/table9_reproduction.json).

<details>
<summary>Show the seven-dataset result tables and omitted-row notes</summary>

<!-- reproduction-results:start -->

#### Citeseer — test accuracy (%)

| Family | Method | 0.36% | 0.90% | 1.80% |
| --- | --- | ---: | ---: | ---: |
| Node sparsification | Cent-D | 44.56 ± 2.30 | 59.32 ± 0.24 | 62.91 ± 0.47 |
| Node sparsification | Cent-P | 41.04 ± 1.55 | 52.01 ± 0.78 | 65.42 ± 0.36 |
| Node sparsification | Random | 36.94 ± 4.89 | 51.01 ± 4.46 | 63.13 ± 2.72 |
| Node sparsification | Herding | 46.11 ± 2.80 | 61.67 ± 1.43 | 67.37 ± 1.37 |
| Node sparsification | K-Center | 44.92 ± 4.69 | 53.58 ± 2.70 | 60.16 ± 1.28 |
| Edge sparsification | TSpanner | 71.19 ± 0.44 | 71.57 ± 0.43 | 71.92 ± 0.26 |
| Coarsening | Averaging | 69.68 ± 0.43 | 69.70 ± 0.45 | 69.63 ± 0.41 |
| Coarsening | VNG | 68.83 ± 0.27 | 63.75 ± 0.46 | 67.55 ± 0.55 |
| Condensation | GEOM² | 67.61 ± 0.7 | 70.70 ± 0.5 | 73.03 ± 0.3 |
| Condensation | SFGC² | 66.27 ± 0.8 | 70.27 ± 0.7 | 72.36 ± 0.5 |
| Condensation | GCDMX | 69.91 ± 0.64 | 70.89 ± 0.38 | 71.82 ± 0.63 |
| Condensation | DosCondX¹ | 68.64 ± 0.47 | 69.24 ± 0.79 | 70.64 ± 0.82 |
| Condensation | GCond | 73.42 ± 0.32 | 70.80 ± 0.38 | 72.44 ± 0.42 |
| Condensation | DosCond | 70.29 ± 0.71 | 71.27 ± 0.42 | 71.64 ± 1.01 |
| Condensation | SGDD | 72.39 ± 0.83 | 71.21 ± 0.38 | 71.59 ± 0.50 |
| Condensation | GECC | 73.07 ± 0.33 | 72.82 ± 0.30 | 72.97 ± 0.22 |
| Condensation | CGC | 70.90 ± 0.24 | 71.89 ± 0.25 | 70.96 ± 0.11 |
| Condensation | ClustGDD | 71.92 ± 0.16 | 73.09 ± 0.27 | 72.54 ± 0.30 |

Whole graph: 71.84 ± 0.21.

#### Cora — test accuracy (%)

| Family | Method | 0.50% | 1.30% | 2.60% |
| --- | --- | ---: | ---: | ---: |
| Node sparsification | Cent-D | 57.27 ± 1.15 | 66.86 ± 0.65 | 75.60 ± 0.42 |
| Node sparsification | K-Center | 48.58 ± 5.61 | 64.46 ± 1.47 | 73.17 ± 0.72 |
| Edge sparsification | TSpanner | 81.54 ± 0.21 | 80.62 ± 0.37 | 80.70 ± 0.25 |
| Coarsening | Averaging | 75.83 ± 0.70 | 75.83 ± 0.70 | 75.83 ± 0.70 |
| Coarsening | VNG | 64.59 ± 0.56 | 74.29 ± 0.45 | 75.32 ± 0.21 |
| Condensation | GEOM² | 78.14 ± 0.5 | 82.29 ± 0.6 | 82.82 ± 0.2 |
| Condensation | SFGC² | 75.11 ± 2.2 | 79.55 ± 0.3 | 80.54 ± 0.5 |
| Condensation | GCSNTK | 70.81 ± 1.62 | 68.33 ± 4.01 | 68.83 ± 1.65 |
| Condensation | GCDMX | 80.47 ± 0.46 | 80.59 ± 0.46 | 81.18 ± 0.24 |
| Condensation | DosCondX¹ | 80.51 ± 0.33 | 80.22 ± 0.43 | 78.51 ± 0.45 |
| Condensation | GCond | 80.34 ± 0.34 | 80.93 ± 0.58 | 81.15 ± 0.52 |
| Condensation | DosCond | 79.12 ± 0.70 | 79.96 ± 0.84 | 78.99 ± 0.96 |
| Condensation | MSGC | 79.78 ± 0.45 | 80.39 ± 0.58 | 80.89 ± 0.48 |
| Condensation | SGDD | 80.41 ± 0.49 | 81.00 ± 0.52 | 79.64 ± 0.60 |
| Condensation | GDEM | 53.60 ± 5.81 | 72.23 ± 2.19 | 75.94 ± 0.74 |
| Condensation | GECC | 81.94 ± 0.59 | 82.86 ± 0.35 | 82.61 ± 0.34 |
| Condensation | CGC | 79.75 ± 0.56 | 81.32 ± 0.37 | 80.75 ± 0.60 |

Whole graph: 80.81 ± 0.51.

#### Pubmed — test accuracy (%)

| Family | Method | 0.02% | 0.03% | 0.15% |
| --- | --- | ---: | ---: | ---: |
| Node sparsification | Cent-D | 56.17 ± 2.57 | 55.61 ± 1.55 | 71.75 ± 0.82 |
| Node sparsification | Cent-P | 56.17 ± 2.57 | 62.50 ± 0.99 | 72.48 ± 0.75 |
| Node sparsification | Random | 51.17 ± 3.25 | 57.72 ± 2.02 | 74.08 ± 1.81 |
| Node sparsification | Herding | 63.03 ± 1.61 | 68.99 ± 1.67 | 75.69 ± 0.63 |
| Coarsening | Averaging | 75.60 ± 0.42 | 75.60 ± 0.42 | 75.60 ± 0.42 |
| Coarsening | VNG | 72.85 ± 0.55 | 75.52 ± 0.38 | 76.25 ± 0.42 |
| Condensation | GEOM² | 69.64 ± 1.4 | 76.21 ± 0.7 | 78.49 ± 0.2 |
| Condensation | SFGC² | 67.61 ± 2.0 | 66.89 ± 3.3 | 67.61 ± 4.1 |
| Condensation | GCDMX | 77.60 ± 0.30 | 75.74 ± 1.37 | 77.44 ± 0.36 |
| Condensation | DosCondX¹ | 72.03 ± 1.58 | 72.25 ± 0.97 | 72.56 ± 0.29 |
| Condensation | GCond | 77.70 ± 0.64 | 78.09 ± 0.47 | 77.32 ± 0.43 |
| Condensation | DosCond | 58.03 ± 1.97 | 52.97 ± 0.17 | 76.87 ± 0.20 |
| Condensation | MSGC | 75.36 ± 0.58 | 77.99 ± 0.44 | 78.32 ± 0.25 |
| Condensation | SGDD | 78.16 ± 0.31 | 77.98 ± 0.32 | 76.25 ± 0.21 |
| Condensation | GECC | 75.14 ± 0.22 | 77.75 ± 0.23 | 80.54 ± 0.29 |

Whole graph: 78.60 ± 0.57.

#### Arxiv — test accuracy (%)

| Family | Method | 0.05% | 0.25% | 0.50% |
| --- | --- | ---: | ---: | ---: |
| Node sparsification | Cent-D | 40.07 ± 1.51 | 49.09 ± 1.03 | 52.10 ± 0.49 |
| Node sparsification | Cent-P | 34.45 ± 1.34 | 48.52 ± 0.88 | 57.40 ± 0.63 |
| Node sparsification | Random | 44.91 ± 5.23 | 57.63 ± 1.14 | 59.74 ± 0.39 |
| Node sparsification | Herding | 49.25 ± 1.86 | 57.26 ± 0.92 | 57.54 ± 0.72 |
| Node sparsification | K-Center | 48.66 ± 1.70 | 55.40 ± 1.63 | 59.00 ± 0.55 |
| Edge sparsification | TSpanner³ | 69.35 ± 0.39 | 69.44 ± 0.41 | 69.30 ± 0.28 |
| Coarsening | Averaging | 59.62 ± 0.42 | 59.96 ± 0.33 | 59.94 ± 0.30 |
| Condensation | GEOM² | 64.91 ± 0.4 | 68.78 ± 0.1 | 69.59 ± 0.2 |
| Condensation | SFGC² | 64.91 ± 0.5 | 66.58 ± 0.3 | 67.03 ± 0.5 |
| Condensation | GCDMX | 60.02 ± 0.44 | 60.59 ± 0.44 | 60.72 ± 0.64 |
| Condensation | DosCondX¹ | 59.58 ± 0.48 | 62.28 ± 0.39 | 59.73 ± 0.30 |
| Condensation | DosCond | 56.03 ± 0.60 | 57.41 ± 0.48 | 54.00 ± 0.58 |
| Condensation | MSGC | 61.09 ± 0.32 | 65.68 ± 0.23 | 64.85 ± 0.21 |
| Condensation | GDEM³ | 56.25 ± 1.21 | 61.41 ± 0.48 | 63.68 ± 0.40 |
| Condensation | GECC | 61.52 ± 0.22 | 64.95 ± 0.19 | 66.44 ± 0.19 |
| Condensation | CGC | 60.71 ± 0.61 | 63.52 ± 0.28 | 64.65 ± 0.28 |
| Condensation | ClustGDD | 62.11 ± 0.09 | 65.32 ± 0.08 | 66.38 ± 0.12 |

Whole graph: 70.70 ± 0.14.

#### Flickr — test accuracy (%)

| Family | Method | 0.10% | 0.50% | 1.00% |
| --- | --- | ---: | ---: | ---: |
| Node sparsification | Cent-D | 40.69 ± 0.39 | 42.88 ± 0.29 | 42.63 ± 0.23 |
| Node sparsification | Cent-P | 39.82 ± 1.34 | 41.77 ± 0.53 | 42.14 ± 0.35 |
| Node sparsification | Random | 41.22 ± 1.91 | 43.89 ± 0.45 | 44.04 ± 0.68 |
| Node sparsification | Herding | 40.05 ± 1.49 | 42.63 ± 0.95 | 44.44 ± 0.45 |
| Node sparsification | K-Center | 39.88 ± 2.36 | 42.35 ± 1.10 | 43.02 ± 0.69 |
| Edge sparsification | TSpanner³ | 47.18 ± 0.13 | 47.19 ± 0.16 | 47.18 ± 0.14 |
| Coarsening | Averaging | 37.93 ± 0.31 | 37.76 ± 0.35 | 37.66 ± 0.34 |
| Coarsening | VNG | 42.73 ± 0.76 | 41.86 ± 0.82 | 42.37 ± 1.03 |
| Condensation | GEOM² | 47.15 ± 0.1 | 46.71 ± 0.2 | 46.13 ± 0.2 |
| Condensation | SFGC² | 46.38 ± 0.2 | 46.38 ± 0.2 | 46.61 ± 0.1 |
| Condensation | GCDMX | 43.25 ± 0.26 | 45.25 ± 0.21 | 45.82 ± 0.30 |
| Condensation | DosCondX¹ | 46.68 ± 0.08 | 46.80 ± 0.11 | 46.83 ± 0.09 |
| Condensation | GCond | 46.64 ± 0.19 | 47.02 ± 0.23 | 46.91 ± 0.13 |
| Condensation | DosCond | 45.73 ± 0.34 | 46.07 ± 0.26 | 46.30 ± 0.20 |
| Condensation | MSGC | 45.71 ± 0.26 | 46.35 ± 0.31 | 46.43 ± 0.22 |
| Condensation | SGDD | 42.21 ± 0.31 | 44.32 ± 0.20 | 43.97 ± 0.21 |
| Condensation | GDEM³ | 44.61 ± 1.04 | 43.88 ± 1.45 | 45.30 ± 0.67 |
| Condensation | GECC | 46.34 ± 0.10 | 45.92 ± 0.20 | 45.83 ± 0.13 |
| Condensation | CGC | 46.01 ± 0.14 | 45.93 ± 0.18 | 45.38 ± 0.31 |
| Condensation | ClustGDD | 46.32 ± 0.05 | 46.63 ± 0.04 | 46.62 ± 0.08 |

Whole graph: 46.95 ± 0.19.

#### Reddit — test accuracy (%)

| Family | Method | 0.05% | 0.10% | 0.20% |
| --- | --- | ---: | ---: | ---: |
| Node sparsification | Cent-D | 39.87 ± 0.78 | 51.49 ± 1.15 | 56.83 ± 1.99 |
| Node sparsification | Cent-P | 46.18 ± 1.55 | 52.37 ± 0.82 | 62.00 ± 0.41 |
| Node sparsification | Herding | 49.88 ± 4.41 | 58.71 ± 4.57 | 69.33 ± 1.92 |
| Edge sparsification | TSpanner³ | 94.01 ± 0.04 | 93.99 ± 0.03 | 94.00 ± 0.05 |
| Coarsening | Averaging | 88.29 ± 0.06 | 88.43 ± 0.09 | 88.41 ± 0.07 |
| Condensation | GEOM² | 90.63 ± 0.2 | 91.33 ± 0.1 | 91.03 ± 0.3 |
| Condensation | SFGC² | 90.18 ± 0.2 | 89.84 ± 0.3 | 90.71 ± 0.1 |
| Condensation | GCDMX | 87.31 ± 0.17 | 89.09 ± 0.06 | 89.96 ± 0.04 |
| Condensation | DosCondX¹ | 87.31 ± 0.23 | 88.04 ± 0.24 | 88.67 ± 0.23 |
| Condensation | GCond | 84.48 ± 0.76 | 86.98 ± 0.13 | 90.18 ± 0.07 |
| Condensation | DosCond | 82.72 ± 0.10 | 86.67 ± 0.10 | 88.07 ± 0.26 |
| Condensation | GDEM³ | 87.95 ± 0.09 | 88.48 ± 0.62 | 90.76 ± 0.25 |
| Condensation | GECC | 89.95 ± 0.10 | 91.00 ± 0.03 | 91.58 ± 0.05 |
| Condensation | CGC | 89.39 ± 0.14 | 90.29 ± 0.06 | 90.92 ± 0.06 |
| Condensation | ClustGDD | 86.89 ± 0.14 | 90.39 ± 0.04 | 91.33 ± 0.07 |

Whole graph: 93.99 ± 0.05.

#### Yelp — F1-macro (%)

| Family | Method | 0.05% | 0.10% | 0.20% |
| --- | --- | ---: | ---: | ---: |
| Node sparsification | Cent-D | 50.23 ± 0.53 | 46.08 ± 0.00 | 46.08 ± 0.00 |
| Node sparsification | Cent-P | 47.40 ± 0.09 | 48.62 ± 0.54 | 46.11 ± 0.06 |
| Node sparsification | Random | 46.10 ± 0.05 | 47.74 ± 1.05 | 49.21 ± 2.60 |
| Node sparsification | Herding | 46.13 ± 0.07 | 46.43 ± 0.39 | 47.00 ± 0.35 |
| Node sparsification | K-Center | 46.45 ± 0.26 | 48.30 ± 0.30 | 48.42 ± 0.64 |
| Edge sparsification | TSpanner³ | 47.50 ± 0.26 | 47.50 ± 0.23 | 47.54 ± 0.13 |
| Coarsening | Averaging | 53.42 ± 0.86 | 53.79 ± 0.35 | 53.83 ± 0.35 |
| Coarsening | VNG | 46.08 ± 0.00 | 46.08 ± 0.00 | 46.08 ± 0.00 |
| Condensation | GEOM² | 52.80 ± 2.2 | 47.56 ± 0.2 | 49.48 ± 0.7 |
| Condensation | SFGC² | 46.20 ± 0.1 | 47.96 ± 0.0 | 46.70 ± 0.1 |
| Condensation | GCDMX | 52.15 ± 1.20 | 55.12 ± 0.78 | 53.63 ± 0.46 |
| Condensation | DosCondX¹ | 52.43 ± 0.13 | 52.34 ± 0.62 | 50.30 ± 0.26 |
| Condensation | GCond | 52.09 ± 0.13 | 51.74 ± 0.14 | 49.83 ± 0.16 |
| Condensation | DosCond | 46.74 ± 1.48 | 51.72 ± 0.58 | 53.41 ± 0.17 |
| Condensation | SGDD | 53.66 ± 0.25 | 51.81 ± 0.30 | 51.89 ± 0.12 |
| Condensation | GDEM³ | 46.08 ± 0.00 | 46.08 ± 0.00 | 46.08 ± 0.00 |

Whole graph: 47.43 ± 0.22.

¹ Archived DosCondX measurements use the archived experimental settings and corresponding reference row. The Reddit configuration differs from current DosCondX defaults; raw archived names and configurations are preserved.
² Published GC4NC values (mean ± standard error). The paper obtained GEOM and SFGC scores from the authors' implementations; their expert-trajectory training is too slow to rerun for the full matrix.
³ The published GC4NC benchmark reports no value for these cells; GraphSlim measurements under the same protocol are shown without a published comparison.

Rows more than 3 points from the paper on average are omitted: VN (Citeseer, Cora, Pubmed); GCSNTK (Citeseer, Pubmed); MSGC (Citeseer, Reddit, Yelp); GDEM (Citeseer, Pubmed); Cent-P (Cora); Random (Cora, Reddit); Herding (Cora); ClustGDD (Cora); K-Center (Pubmed, Reddit); TSpanner (Pubmed); VNG (Arxiv, Reddit); GCond (Arxiv); SGDD (Reddit). Runs that failed: GCSNTK (Arxiv, Flickr, Reddit, Yelp); SGDD (Arxiv); VN (Flickr, Yelp).

<!-- reproduction-results:end -->

</details>

## Install

Use Python 3.10 or 3.11. For the manuscript, check out the immutable Git commit
identified in the cover letter before following these source-installation instructions.
Install PyTorch first for your CPU/CUDA environment, then
install matching PyG extension wheels. For example, a CPU development environment:

```bash
git clone https://github.com/Emory-Melody/GraphSlim.git
cd GraphSlim
git checkout --detach 1a024c6bf2d3a73e40a3d1facb6dd6b9c1328bab
uv venv --python 3.11
source .venv/bin/activate
uv pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install torch-scatter torch-sparse --no-deps --no-index -f https://data.pyg.org/whl/torch-2.6.0+cpu.html
uv pip install -e ".[dev]" --constraint <(printf 'torch==2.6.0\ntorchvision==0.21.0\n')
```

### GPU setup and checks

For CUDA, choose matching PyTorch and extension wheels from
[PyTorch](https://pytorch.org/get-started/previous-versions/) and
[PyG](https://data.pyg.org/whl/). Keep the installed Torch version constrained while
installing the remaining dependencies. Install `.[dgl]` for the Yelp/Amazon fraud
datasets, `.[fuzzy]` for GECC's fuzzy clustering, and `.[viz]` for the web interface.
The extension-wheel step uses `--no-deps` because the PyG wheel directory does
not contain SciPy; the subsequent GraphSlim installation supplies dependencies.

The checked-in `uv.lock` records dependency resolution. `uv sync --locked` reads it;
`uv pip sync` requires a requirements file and does not consume `uv.lock`.
Compiled PyG extensions must still match the Torch/CUDA environment.

Use `graphslim-doctor` to check PyTorch, sparse operations and optional DGL.

```bash
python -m graphslim.doctor --device cuda:0 --dgl --json --output doctor.json
```

The published PyPI distribution is installed with `pip install graphslim`.
Use [the pinned Git source](https://github.com/Emory-Melody/GraphSlim/tree/1a024c6bf2d3a73e40a3d1facb6dd6b9c1328bab) for the manuscript's 49-method, eight-backbone feature set; PyPI 1.2.1 predates this revision.

## Quick start

Run from the repository root. Data defaults to `./data`; use `--load_path` and
`--save_path` to select dataset and checkpoint locations. `-G -1` selects CPU;
nonnegative GPU IDs index devices already visible through `CUDA_VISIBLE_DEVICES`.

```bash
python -m graphslim.train_all -D cora -M clustering -G -1 --save_path ./checkpoints
python -m graphslim.train_all -D cora -M gcond -G 0 --save_path ./checkpoints
python -m graphslim.run_eval -D cora -M gcond -G 0 --save_path ./checkpoints
python -m graphslim.run_eval -D cora -W -G -1
python -m graphslim.train_all --help
```

Installed commands: `graphslim-train`, `graphslim-eval`, `graphslim-doctor`, and `graphslim-visualize`.
`--run_reduction N` uses consecutive seeds for N reductions; `--run_eval N`
controls downstream evaluation repetitions. `--run_reduction 0` evaluates a saved
graph, and `-W` evaluates the whole graph. Method JSON defaults load before explicit
CLI overrides. `--agg` (also `--aggpreprocess`) selects an aggregated variant where
available.

### Hyperparameter optimization (HPO)

Search reduction-method settings using **mean validation accuracy only**.
For example, GDEM ships a search space:

```bash
python -m graphslim.train_all -D cora -M gdem -R 0.5 -G 0 --hpo --save_path ./checkpoints
```

Use `--hpo_space path/to/space.json` for a custom grid. Each key is a method
setting and each list contains its candidate values. For a method without a
shipped grid, the first invocation writes a one-point template to widen.
The command previews the grid and requests confirmation; `--hpo_yes` starts it
directly. Rerunning the same command resumes from `trials.jsonl`.

The winning settings are saved to `checkpoints/hpo/gdem/cora_r0.5/best.json`:

```bash
python -m graphslim.train_all -D cora -M gdem -R 0.5 -G 0 \
  --config ./checkpoints/hpo/gdem/cora_r0.5/best.json --save_path ./checkpoints
```

HPO computes validation scores during search; the replay command evaluates the
selected configuration. `--hpo_eval_runs` controls validation fits per trial
(default: 3). [Search-space format](graphslim/configs/hpo/README.md).

### Examples

Each example uses the same pipeline and loads defaults for its selected method.

```bash
python -m examples.train_node_sparsification -G -1
python -m examples.train_coarsen -G -1
python -m examples.train_gcond -G -1
```

Direct execution such as `python examples/train_gcond.py --help` also works from
any directory. Importing an example does not launch training.

### Package API

```python
from graphslim.config import get_args
from graphslim.train_all import run

args = get_args(
    [
        "-D",
        "cora",
        "-M",
        "gcond",
        "-G",
        "-1",
        "--run_reduction",
        "1",
        "--save_path",
        "./checkpoints",
    ]
)
results = run(args)  # one (mean, standard deviation) pair per reduction
```

For custom orchestration, use `graphslim.dataset.get_dataset`,
`graphslim.reduction.create_reducer`, and `graphslim.evaluation.Evaluator`.
Public method classes remain available, e.g. `from graphslim.condensation import GCond`.

## Methods

Discover canonical names with:

```python
from graphslim.reduction import list_methods

print(list_methods("condensation"))
```

| Family | Canonical method names |
| --- | --- |
| Node sparsification / coreset (5) | `random`, `kcenter`, `herding`, `cent_d`, `cent_p` |
| Edge sparsification (18) | `random_edge`, `spanning_forest`, `g_spar`, `local_degree`, `rank_degree`, `scan`, `t_spanner`, `simmelian_parametric`, `simmelian_non_parametric`, `quadrilateral_simmelian`, `simmelian_multiscale`, `degree_multiscale`, `local_similarity`, `multiscale`, `random_node_edge`, `forest_fire`, `triangle`, `algebraic_distance` |
| Coarsening / clustering (10) | `averaging`, `clustering`, `vng`, `variation_neighborhoods`, `variation_edges`, `variation_cliques`, `heavy_edge`, `algebraic_jc`, `affinity_gs`, `kron` |
| Graph condensation (16) | `gcond`, `doscond`, `doscondx`, `gcdm`, `gcdmx`, `sfgc`, `geom`, `msgc`, `sgdd`, `gcsntk`, `gdem`, `simgc`, `gecc`, `cgc`, `clustgdd`, `gcgp` |

See each method's configuration for its settings, including GECC's evolving-graph
workflow.

Node sparsifiers select existing nodes, with `-R` setting a fraction of labeled
training nodes for the reduced-node budget. All 18 edge sparsifiers keep every node. For them `args.edge_reduction`
is `True` and `--reduction_rate` is the fraction of edges kept; `t_spanner` and
`spanning_forest` ignore the rate (`--ts` sets the spanner stretch). The flag is
derived from the method and `--edge_reduction` / `--node_reduction` only assert
it; the inductive setting sparsifies the training subgraph so that the adjacency
matches the features the evaluator trains on.

```bash
# Select nodes: target 10% of labeled training nodes.
python -m graphslim.train_all -D cora -M kcenter -R 0.1 -G -1 --run_reduction 1
# Keep nodes and sparsify edges: target 10% of input edges.
python -m graphslim.train_all -D cora -M random_edge -R 0.1 -G -1 --run_reduction 1
```

All 15 concrete Python sparsifiers in NetworKit are available, alongside
`spanning_forest`, `rank_degree`, and `t_spanner`. Existing `g_spar` uses
NetworKit's Jaccard similarity backend (`jaccard_similarity` is a Python registry
alias). Constructor settings can be passed through `--sparsifier_options` or a
`"sparsifier_options"` object in `--config`. For example:

```bash
python -m graphslim.train_all -D cora -M simmelian_parametric -R 0.1 -G -1 --run_reduction 1 --sparsifier_options '{"maxRank": 20}'
```

Native threshold methods target the requested ratio but can retain more or fewer
edges when scores tie. `triangle` uses NetworKit triangle counts with an exact
`floor(R * edges)` budget and endpoint order to break ties, since upstream has
no size parameterization. The existing reduction workflow uses unweighted,
undirected connectivity. See the [full backend mapping and constructor defaults](docs/source/graphslim.sparsification.rst).
The fixed reproduction snapshot above does not include the 11 newly integrated methods.

CGC reduces graphs without GNN training. ClustGDD uses linear pretraining
and refinement; GCGP uses a Gaussian process to optimize reduced features and
targets. [Method configurations](graphslim/configs).

```bash
python -m examples.train_lightweight -D cora -M cgc -G -1 --save_path ./checkpoints
python -m graphslim.train_all -D cora -M gcgp -G -1 --save_path ./checkpoints
```

Use `--config method_overrides.json` for method-specific options; explicit CLI
options take precedence. Public imports include `CGC`, `ClustGDD`, `GCGP` from
`graphslim.condensation`.

## Search a method's settings

The fixed reproduction artifact records the settings adopted for each historical pair.
For a new dataset or method, `--hpo`
grid-searches the reduction settings themselves with validation-only selection:

```bash
python -m graphslim.train_all -D yelp -M gdem -R 0.001 -G 0 --hpo --save_path ./checkpoints
```

It prints the search space (from `--hpo_space`, else `graphslim/configs/hpo/<method>.json`,
else a one-point template written under `save_path/hpo/` for you to widen), the number of
trials and a time estimate, then asks for confirmation (`--hpo_yes` skips it). Every trial
reduces the graph and scores it by the mean validation accuracy of `--hpo_eval_runs` downstream
fits (`--eval_model`, GCN by default); no test score is computed. Finished trials are appended to `trials.jsonl`, so an
interrupted search resumes, and the winner lands in `best.json` for `--config`.

## Reproduce the benchmark

The runner plans jobs without importing Torch and runs jobs serially when
`--dry-run` is omitted. Use the dry run to inspect the requested matrix:

```bash
python -m benchmark.reproduce performance --dry-run
python -m benchmark.reproduce performance --datasets cora citeseer --methods clustering gcond --dry-run
python -m benchmark.reproduce performance --datasets cora --methods clustering --skip-whole -- -G -1
bash benchmark/scripts/performance.sh --dry-run
```

The performance matrix covers all seven published datasets and the benchmark's
19 retained methods, plus clustering and the four new lightweight extensions,
followed by whole-graph baselines. Published OOM or missing
cells remain visible in the table; reruns may also fail for resource or method-specific
requirements. GEOM/SFGC paper performance used the authors' implementations; the local
expert-buffer defaults differ, so this package runner alone does not establish exact
reproduction of those published scores.

Other suites: `scalability`, `data_initialization`, `transferability`, `nas`,
`graph_property_preservation`, and `robustness`. The shell scripts in
[benchmark/scripts](benchmark/scripts) forward arguments to the same runner and
resolve their repository path independently of the caller's directory. Transferability,
NAS, and graph-property suites require saved reduced graphs. See
[benchmark/README.md](benchmark/README.md) for the paper-to-command mapping.

### Reduction rates

Node-selection and condensation budgets for the **19 datasets** recognized by the
[dataset loader](graphslim/dataset/loader.py) are listed below. For these methods, **R**
(CLI `-R`) is the fraction of **labeled training nodes** used to set the node budget.

For edge sparsification, `-R` is the fraction of input edges kept;
`t_spanner` and `spanning_forest` do not use it.

| Dataset | CLI `-D` | Setting | R (`-R`) | Retained nodes (approx. %) |
| --- | --- | --- | --- | --- |
| Citeseer | `citeseer` | Transductive | 0.1, 0.25, 0.5 | 0.36, 0.90, 1.80 |
| Cora | `cora` | Transductive | 0.1, 0.25, 0.5 | 0.50, 1.30, 2.60 |
| Pubmed | `pubmed` | Transductive | 0.05, 0.1, 0.5 | 0.02, 0.03, 0.15 |
| Arxiv | `ogbn-arxiv` | Transductive | 0.001, 0.005, 0.01 | 0.05, 0.25, 0.50 |
| Flickr | `flickr` | Inductive | 0.001, 0.005, 0.01 | 0.10, 0.50, 1.00 |
| Reddit | `reddit` | Inductive | 0.0005, 0.001, 0.002 | 0.05, 0.10, 0.20 |
| Yelp | `yelp` | Inductive | 0.0005, 0.001, 0.002 | 0.05, 0.10, 0.20 |
| Roman-empire | `roman-empire` | Transductive | 0.005, 0.01, 0.02 | — |
| Amazon-ratings | `amazon-ratings` | Transductive | 0.005, 0.01, 0.02 | — |
| OGBN-Products | `ogbn-products` | Transductive | 0.0005, 0.001, 0.002 | — |
| Amazon Photo | `photo` | Transductive | 0.005, 0.01, 0.02 | — |
| Amazon Computers | `computers` | Transductive | 0.005, 0.01, 0.02 | — |
| DBLP | `dblp` | Transductive | 0.1 | — |
| Cora-ML | `cora_ml` | Transductive | 0.1 | — |
| Coauthor CS | `cs` | Transductive | 0.1 | — |
| Coauthor Physics | `physics` | Transductive | 0.1 | — |
| Amazon Fraud | `amazon` | Inductive | 0.002 | — |
| OGBN-Proteins | `ogbn-proteins` | Transductive | 0.1 | — |
| OGBN-Papers100M | `ogbn-papers100m` | Transductive | 0.1 | — |

For node selection and condensation, the target node count is approximately `R × N_train`, where
`N_train` is the number of labeled training nodes. The final column expresses the
same budget as a percentage: `100 × N_reduced / N_reference`. For Citeseer, Cora,
Pubmed and Arxiv, `N_reference` is the number of nodes in the **whole graph**;
for Flickr, Reddit and Yelp, it is the number of nodes in the **training graph**.

For example, Cora has 140 labeled training nodes and 2,708 nodes in total.
`-R 0.1` targets 14 retained nodes: 10% of the labeled training nodes, or about
0.5% of the whole graph. The percentage labels are rounded; per-class integer
allocation and minimum class budgets can change the final node count. Use the
saved reduced graph for the actual count. A dash means no percentage label is
provided here for that dataset. This column describes node budgets; edge
sparsification uses the separate edge-retention definition above.

Multilevel coarseners apply `-R` to their working graph before extracting labeled
training clusters. Their output node count can differ from the node-selection budget;
read the saved graph dimensions for the actual retained count.

Use `benchmark.reproduce performance` for the original seven datasets and the
[dataset expansion runner](benchmark/expand_datasets.py) for Roman-empire,
Amazon-ratings, OGBN-Products, Photo and Computers. Roman-empire and Amazon-ratings
use official split column 0 by default; OGBN-Products uses the official OGB split.
Photo and Computers use seeded classwise 80/10/10 splits.

`amazon` is the DGL fraud dataset; `photo` and `computers` are PyG co-purchase
datasets. Completed results are reported above; loader entries alone do not
establish end-to-end reducer compatibility or completed benchmarks, including
task handling for `ogbn-proteins` and resource feasibility for `ogbn-papers100m`.

<!-- package-results:start -->

## Package measurements completing published NR coverage

Complete local cells: 105/105 (7 datasets × 3 CLI rates × 5 methods).
Only cells with all 3 reduction seeds × 10 downstream test fits are numeric.
Values are test mean ± population standard deviation (%), not the published
table's reported standard errors. Yelp uses F1-macro; all other rows use accuracy.
Pending repaired SimGC cells do not fall back to superseded measurements.

This is an explicitly source-scoped coverage table, not a controlled ranking:
NR methods use the archived fixed-GCN protocol; GECC–Yelp uses paper-grid HPO;
SimGC uses a fresh repaired source and validation-only GCN HPO (citation raw
features/no-activation/3,001 updates; other datasets 300 updates). Source
code, preprocessing, training budgets and evaluation seed schedules differ.
Published results and earlier local batches remain archived under `benchmark/results`.

| Dataset | CLI rate | Clustering | DosCondX | GCDM | SimGC (repaired HPO) | GECC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| citeseer | 0.1 | 69.78 ± 0.48 | 70.45 ± 1.27 | 71.25 ± 0.94 | 42.13 ± 5.38 | 73.06 ± 0.31 |
| citeseer | 0.25 | 69.35 ± 0.30 | 70.06 ± 1.03 | 71.21 ± 0.80 | 51.92 ± 8.36 | 73.01 ± 0.31 |
| citeseer | 0.5 | 69.14 ± 0.24 | 70.86 ± 0.99 | 71.96 ± 0.57 | 50.87 ± 4.53 | 72.89 ± 0.27 |
| cora | 0.1 | 73.74 ± 0.65 | 78.98 ± 0.54 | 78.40 ± 1.01 | 57.80 ± 16.33 | 81.93 ± 0.73 |
| cora | 0.25 | 75.46 ± 0.71 | 78.38 ± 0.52 | 79.59 ± 0.62 | 77.97 ± 2.72 | 82.81 ± 0.22 |
| cora | 0.5 | 77.31 ± 0.43 | 79.25 ± 0.78 | 80.90 ± 0.45 | 78.13 ± 2.32 | 82.53 ± 0.41 |
| pubmed | 0.05 | 75.56 ± 0.40 | 72.53 ± 1.54 | 77.65 ± 0.31 | 72.19 ± 2.35 | 75.09 ± 0.19 |
| pubmed | 0.1 | 75.10 ± 0.75 | 71.26 ± 1.43 | 77.39 ± 1.44 | 68.03 ± 6.62 | 77.71 ± 0.23 |
| pubmed | 0.5 | 76.46 ± 0.47 | 71.84 ± 1.26 | 76.76 ± 0.58 | 72.44 ± 1.96 | 80.20 ± 0.40 |
| ogbn-arxiv | 0.001 | 60.84 ± 0.53 | 60.42 ± 0.59 | 59.97 ± 0.46 | 61.41 ± 0.63 | 61.38 ± 0.55 |
| ogbn-arxiv | 0.005 | 61.95 ± 0.20 | 62.21 ± 0.98 | 60.59 ± 0.42 | 62.91 ± 0.99 | 64.92 ± 0.26 |
| ogbn-arxiv | 0.01 | 63.13 ± 0.17 | 61.75 ± 0.54 | 60.67 ± 0.48 | 62.61 ± 0.90 | 66.38 ± 0.35 |
| flickr | 0.001 | 46.04 ± 0.25 | 46.63 ± 0.22 | 43.56 ± 1.06 | 46.21 ± 0.32 | 46.33 ± 0.25 |
| flickr | 0.005 | 46.27 ± 0.31 | 46.64 ± 0.19 | 45.07 ± 0.61 | 46.26 ± 0.34 | 45.88 ± 0.22 |
| flickr | 0.01 | 45.70 ± 0.26 | 46.62 ± 0.16 | 45.33 ± 0.65 | 46.29 ± 0.36 | 46.18 ± 0.24 |
| reddit | 0.0005 | 89.44 ± 0.11 | 54.07 ± 3.38 | 86.88 ± 0.37 | 87.12 ± 0.66 | 89.50 ± 0.09 |
| reddit | 0.001 | 90.03 ± 0.18 | 56.83 ± 1.50 | 88.48 ± 0.44 | 89.11 ± 0.31 | 90.89 ± 0.09 |
| reddit | 0.002 | 91.57 ± 0.06 | 58.02 ± 1.31 | 89.92 ± 0.12 | 90.33 ± 0.11 | 91.63 ± 0.11 |
| yelp | 0.0005 | 53.63 ± 0.36 | 51.15 ± 1.07 | 51.77 ± 1.47 | 47.85 ± 1.34 | 47.71 ± 2.86 |
| yelp | 0.001 | 53.69 ± 0.19 | 51.09 ± 1.20 | 53.43 ± 1.75 | 47.67 ± 1.69 | 50.69 ± 2.92 |
| yelp | 0.002 | 52.45 ± 0.22 | 50.53 ± 0.73 | 53.87 ± 0.55 | 47.71 ± 1.62 | 53.94 ± 0.41 |

Source assignment (no scores pooled across batches):

- `clustering, doscondx, gcdm, gecc` on `citeseer, cora, pubmed, ogbn-arxiv, flickr, reddit, yelp`: `runs/nr_20261009_v2` — NR batch: fixed GCN, 300 updates.
- `gecc` on `yelp`: `runs/paper_hpo_20261010_v2` — Completed GECC Yelp paper-grid search.
- `simgc` on `citeseer, cora, pubmed, ogbn-arxiv, flickr, reddit, yelp`: `runs/simgc_repaired_hpo_20261010` — Repaired SimGC: fresh reductions and validation-only GCN search.

[Machine-readable coverage and per-cell provenance](benchmark/results/package_local_results.json);
[CSV export](benchmark/results/package_local_results.csv). `PENDING`/`FAILED`/`CONFIG`
are unavailable results, not zero. Published `NR` means absent from the paper,
not absent from this local coverage table. Raw artifacts live in the listed run directories.

<!-- package-results:end -->

## Lightweight methods: downstream validation

The [dataset expansion runner](benchmark/expand_datasets.py) uses three reduction seeds and
ten GCN seeds per rate. [Archived Products results](benchmark/results/products_20261028.json) include
CGC, ClustGDD, GCGP and Random at all three rates.

Package regression tests cover eight downstream architectures, transductive and inductive
settings, noise, saved-graph reloads and validation-based model selection.
[Integration runner](benchmark/lightweight_downstream.py) · [Regression checks](tests/test_lightweight_methods.py).

```bash
python -m benchmark.lightweight_downstream --run-dir runs/lightweight_cora --dry-run
python -m benchmark.lightweight_downstream --run-dir runs/lightweight_cora --profile smoke
python -m benchmark.reproduce robustness --datasets cora --methods cgc clustgdd gcgp --dry-run
```

<!-- lightweight-checks:start -->

The short Cora checks use 20 reduction and downstream updates, hidden size 32,
and one reduction/evaluation seed. [Check results (CSV)](benchmark/results/lightweight_integration.csv) ·
[Settings and verification records (JSON)](benchmark/results/lightweight_integration.json).

<!-- lightweight-checks:end -->

## Development and tests

```bash
python -m pytest -q
ruff check graphslim examples benchmark tests interface
ruff format --check graphslim examples benchmark tests interface
python -m build
```

Regression tests use small in-memory graphs, CPU GNN training and mocked job
orchestration. Benchmark runs use the original datasets, method-specific settings
and repeated evaluation. [Historical development checks](efficiency.md).

## Repository layout

```text
graphslim/       package, methods, loaders, models, evaluation and CLI
examples/        small entry points using the package pipeline
tests/           regression tests
benchmark/       reproduction planner, research scripts and published results
docs/            Sphinx documentation
```

## Citation and acknowledgements

```bibtex
@inproceedings{gong2025gcnc,
  title={{GC}4{NC}: A Benchmark Framework for Graph Condensation on Node Classification with New Insights},
  author={Shengbo Gong and Juntong Ni and Noveen Sachdeva and Carl Yang and Wei Jin},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
  year={2025}
}
@inproceedings{hashemi2024comprehensive,
  title={A comprehensive survey on graph reduction: sparsification, coarsening, and condensation},
  author={Hashemi, Mohammad and Gong, Shengbo and Ni, Juntong and Fan, Wenqi and Prakash, B Aditya and Jin, Wei},
  booktitle={Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence},
  pages={8058--8066},
  year={2024}
}
```

Implementations build on [SCAL](https://github.com/szzhang17/Scaling-Up-Graph-Neural-Networks-Via-Graph-Coarsening),
[graph sparsification](https://github.com/yuhanchan/sparsification),
[GCond](https://github.com/ChandlerBang/GCond),
[GCSNTK](https://github.com/WANGLin0126/GCSNTK),
[SFGC](https://github.com/Amanda-Zheng/SFGC),
[GEOM](https://github.com/NUS-HPC-AI-Lab/GEOM), and
[DeepRobust](https://github.com/DSE-MSU/DeepRobust). See [LICENSE](LICENSE).

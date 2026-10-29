# Clustering operator optimizations

The shared operators live in `graphslim/reduction/cluster_ops.py`. The native
sklearn KMeans, MiniBatchKMeans and BisectingKMeans algorithms, initializers,
restart counts and stopping rules are retained. Optimizations change distance
working storage, propagation, assignment storage and aggregation rather than
replace these algorithms with a different clustering method.

| Method/path | Operator change | Numerical or interface boundary |
| --- | --- | --- |
| GECC hard k-means and k-means++ | Stream subtract/square/sum distances under a 64 MiB working-tensor budget; avoid repeated GPU-to-CPU copies | Exact distance arithmetic and first-center tie rule; original centroid update and empty-cluster RNG retained |
| GECC propagation | Accumulate weighted sparse hops while retaining only the current hop | Exact accumulation order for the established full-graph path |
| GECC diagnostic assignment | Sparse COO with original graph node IDs; skip unsaved dense partition matrices | `data.assignment` is now sparse; small consumers needing dense access must explicitly call `.to_dense()` |
| Clustering | Reuse final BisectingKMeans labels instead of predicting the same training input again | Native fitted labels verified against the previous prediction path |
| ClusteringAgg | Two sparse-dense propagations instead of forming the sparse matrix square; reuse fitted labels | Same two-hop operator, with floating-point reassociation; `pre_conv` retains the one-hop normalized matrix |
| VNG | Sparse weighted memberships E/M and vectorized degree repair; remove an unused graph copy | Same weighted k-means, linear algebra and label voting; floating-point summation can differ |
| CGC | Segmented stable softmax and indexed weighted-center aggregation | Same weights and cluster assignments; summation order can differ; uses additional O(ND) temporary weighted-feature storage |
| ClustGDD | Chunked cosine and streamed class-view compression; shared native fitted-label path | Global top-k selection and sampling ratio retained; see the earlier Products streaming validation |
| GCSNTK dataset batching | Reuse native KMeans fitted labels | Same fit, RNG consumption and partition budget |

Fuzzy GECC continues to use the existing skfuzzy solver. Its propagation and
assignment operators benefit from the changes, but the solver has not been
replaced or advertised as GPU-accelerated. Commented-out clustering examples in
MSGC are not active execution paths. NOPE's greedy contraction is a separate
algorithm and is not covered by these k-means/operator speed claims.

## GECC Products repair and experiment scope

The previous Products-only NeighborSampler branch wrote to all sampled context
nodes repeatedly across batches. Products now follows the same normalized
full-graph weighted propagation as the other transductive GECC datasets. This is
an implementation repair and an explicitly new experimental source, not numerical
equivalence with that legacy sampled branch. Diagnostic assignments now map
training positions back to their original graph node IDs and remain sparse.

`runs/products_cluster_ops_20261028` freezes the optimized implementation and
queues GECC, Clustering, CGC, ClustGDD, VNG, and a fresh full-graph GCN baseline:
46 jobs, with nine GECC reductions. Each reduced method has three training-relative
rates (0.0005, 0.001, 0.002), three reduction seeds, and ten evaluation seeds.
Official Products splits and 300 downstream updates are retained. Model selection
uses validation only. No hyperparameter search or historical-cell substitution.

```bash
.venv-gpu/bin/python -m benchmark.expand_datasets \
  --run-dir runs/products_cluster_ops_20261028 --datasets ogbn-products \
  --methods gecc clustering cgc clustgdd vng --gpus 2 4 5 \
  --timeout 14400 --max-rss-gib 100
```

The original expansion's default 470-job roster remains unchanged; the added
methods are explicit CLI choices. Each numeric cell requires all 30 fits. The
older GECC seven-dataset results and completed ClustGDD streaming batch retain
their original sources; they are not pooled into this new batch.

## Validation and measured scope

`tests/test_cluster_ops.py` covers distance tiling in both dimensions, ties,
fitted-label reuse, large-logit stable aggregation, sparse membership, original
node IDs, empty-cluster RNG and a real GECC reduction. The broader CPU regression
run passed 181 tests with 22 CUDA tests skipped; the GPU suite passed 61 tests.

`benchmark/verify_cluster_ops.py` compares six reduction implementations on a
fixed small graph against the sealed `products_streaming_20261028` snapshot.
VNG's paired check supplies identical fixed teacher embeddings to its compression
stage; its teacher-plus-compression path has a separate executable integration test.
All passed at absolute/relative tolerance 1e-4; five were exactly equal in the
observed CPU check, and ClusteringAgg's maximum feature difference was 1.79e-7.
This does not establish identical large-graph final scores, which require the
new complete seed grid.

The verifier records numerical differences, per-repeat timings and peak CUDA
allocations in its output directory.

```bash
.venv-gpu/bin/python -m benchmark.verify_cluster_ops \
  --device cuda:0 --output runs/cluster_ops_check \
  --reference runs/products_streaming_20261028
```

Reserve a GPU and bind its UUID before running the CUDA check. The formal queue
does this automatically and validates the visible device against its lock.

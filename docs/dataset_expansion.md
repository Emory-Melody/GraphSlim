# Ordered dataset expansion

This batch extends the four new lightweight methods in three execution stages:

1. Cora, Citeseer, Pubmed, ogbn-arxiv, Flickr, Reddit2 and FraudYelp.
2. Roman-empire and Amazon-ratings with official split column 0.
3. ogbn-products with the official OGB sales-rank split.

Later-stage experiments start only after every prior-stage job has reached a
terminal status. Complete, failed, timeout and resource-limit outcomes are
retained separately. Reaching a terminal status does not imply successful or
complete numeric benchmark coverage.

The implementation is [benchmark.expand_datasets](https://github.com/Emory-Melody/GraphSlim/blob/main/benchmark/expand_datasets.py).
It freezes a source ZIP and source digest before dispatch, honors the host's
physical-GPU and UUID locks, and checks that each selected GPU is idle. Workers
load data with fixed seed 1 and then reset the independent reduction seed.
Interrupted jobs can reuse hash-verified saved reductions and completed GCN fits.
Worker logs, dataset/split hashes, resolved configurations and checkpoint hashes
remain in each job directory.

## Protocol

Primary methods are CGC, ClustGDD, GCGP and the existing NOPE condensation adapter.
Random class-stratified sampling and whole-graph GCN are new, same-batch baselines.
There are three reduction seeds (1, 2, 3), ten GCN initialization seeds (0–9), and
three training-relative reduction rates per dataset. A reduced-graph numeric cell
requires all 30 test fits. Whole-graph cells contain ten fits. Uncertainty is the
population standard deviation across these fits, not a confidence interval; the
three identical reductions of deterministic NOPE are not three independent graph
samples. No incomplete cell receives a numeric aggregate.

GCN uses two layers, hidden size 256, ReLU, dropout 0, Adam learning rate 0.01 and
weight decay 0.0005. There are 300 training updates; validation runs at loop indices
0, 10, ..., 290, and only validation performance selects model weights. The
package's midpoint learning-rate decay is retained. Method-specific reduction
settings and feature preprocessing are retained, including GCGP's soft-target MSE
downstream loss. These are package-default experiments without HPO, not exact
author-paper reproductions or preprocessing-controlled algorithm comparisons.
Previous published and local result batches are not pooled with this batch.

The first stage retains the existing benchmark's reduction rates. Roman-empire
and Amazon-ratings use 0.005, 0.01 and 0.02 of labeled training nodes. ogbn-products
uses 0.0005, 0.001 and 0.002. These added rates are declared local choices, not
reported author optima. Yelp uses macro-F1; other datasets use accuracy.

The original package loads Reddit2 rather than the denser Reddit release and
constructs a fixed classwise 80/10/10 FraudYelp split with data seed 1. The new
heterophilous loader selects an explicit official mask column without resampling.
The ogbn-products loader calls `get_idx_split()` before the generic split helper;
the former random classwise fallback is not used.
OGB's processed PyG cache is loaded with an explicit safe-global context for
`DataEdgeAttr`, `DataTensorAttr` and `GlobalStorage`. Weights-only loading remains
enabled. This handles the older OGB cache format on Torch 2.6+.

Official new-dataset sources:

- [Heterophilous graph release](https://github.com/yandex-research/heterophilous-graphs)
- [PyG HeterophilousGraphDataset](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.HeterophilousGraphDataset.html)
- [OGB node-property datasets and splits](https://ogb.stanford.edu/docs/nodeprop/)

## Timing, resources and numerical equivalence

Every reduction records elapsed time and actual retained nodes. GCN fit times,
data-loading/hash time, worker wall time, CUDA allocator peaks and process peak
RSS are retained separately. Whole-graph reduction time is zero because no
reduction is performed. CUDA timings synchronize at phase boundaries.
Per-job limits and the failure policy are fixed in the manifest; failures retain
their actual logs and never become zero-valued scores.

[benchmark.gcgp_parity](https://github.com/Emory-Melody/GraphSlim/blob/main/benchmark/gcgp_parity.py) separately compares the pinned
author SGNK arithmetic in eager mode and native CUDA kernel capture at each
dataset's largest declared rate. GP solves, Adam and changing minibatch sampling
remain eager. Three seeds, three timing repeats and 200 GP updates are checked.
Every positive row requires exact loss histories, final feature/target hashes and
the same downstream validation/test scores at evaluation seed 0. Graph setup is
reported separately and included in an additional speed ratio. A speedup is only
supported for passing paired rows; slower or failed cases remain negative results.
Package reduction and GCN defaults run eagerly in the primary experiment.

The NOPE adapter retains its existing training-relative budget and removes
supernodes without training labels. It reports requested versus actual nodes,
component limits and missing classes in reduction metadata. It does not reproduce
the author's mapped-original-node classifier or whole-graph budget protocol; see
the existing `runs/nope_diagnosis_20261011/REPORT.md` evidence.

## Run and inspect

```bash
.venv-gpu/bin/python -m benchmark.expand_datasets \
  --run-dir runs/dataset_expansion_20261012_v3 \
  --gpus 3 4 5 --timeout 3600 --max-rss-gib 100

.venv-gpu/bin/python -m benchmark.expand_datasets \
  --run-dir runs/dataset_expansion_20261012_v3 --export
```

The run directory contains `manifest.json`, `source.zip`, `frozen_workspace`,
`heartbeat.json`, `summary.json`, `scores.csv` and `results.md`. `complete.json`
appears only when all jobs are terminal; its outcome counts must still be checked
for unavailable results. Re-running the first command resumes the same frozen
protocol. Changing the roster, seeds or limits requires a fresh directory.

The v3 snapshot adds OGB cache compatibility before the third stage. Complete
v2 cells retain their original source and raw artifacts through explicit
`inherited_sources` entries and linked job directories. All package science files
outside the OGB loader are copied byte-for-byte from the sealed v2 archive,
keeping concurrent development outside this experimental protocol. A cell never
mixes source versions. Incomplete v2 groups remain archived and are rerun as a
whole group in the new source; complete groups are not retrained.

## Continuing evaluation timeouts

The completed v3 queue retained 51 timeouts: 45 NOPE reductions and six
ogbn-products jobs interrupted during downstream evaluation. Those six jobs
already saved 42 of their 60 GCN fits. Continue only their missing fits with:

```bash
.venv-gpu/bin/python -m benchmark.resume_expansion \
  --parent runs/dataset_expansion_20261012_v3 \
  --run-dir runs/dataset_expansion_20261028_resume \
  --gpus 3 4 5 --timeout 14400 --max-rss-gib 100
```

This creates a separate continuation manifest, verifies the original source ZIP
and frozen workspace, copies verified checkpoints and completed evaluations,
and retains the original terminal outcomes. Workers use the original frozen
code, data, training budgets and seeds. Only execution resource limits change;
the manifest records both limits and each copied file hash. A numeric cell still
requires every declared seed. Successful compute totals reuse saved timings but
exclude time lost in interrupted attempts. Other terminal failures remain
unavailable. GECC is outside this expansion roster; its historical seven-dataset
results have their own provenance in `benchmark/results/package_local_results.json`.

## Products memory optimization

`runs/products_streaming_20261028` is a separate frozen-source ClustGDD run with
chunked edge similarities and streamed class-view compression. It retains the
original three rates, three reduction seeds, ten GCN seeds, official split and
80 pretraining / 100 refinement / 300 downstream updates. Original OOM records
remain in v3; source versions are not pooled into one result cell.

```bash
.venv-gpu/bin/python -m benchmark.expand_datasets \
  --run-dir runs/products_streaming_20261028 \
  --datasets ogbn-products --methods clustgdd --reductions-only \
  --gpus 2 --timeout 14400 --max-rss-gib 100
```

`--reductions-only` omits additional whole-graph and parity jobs. New dispatches
bind CUDA visibility to the locked physical GPU UUID, record that UUID in each
job's `launch.json`, and validate the visible UUID before executing the worker.
This avoids dependence on the CUDA driver's integer device ordering.

The subsequent `runs/products_cluster_ops_20261028` batch adds GECC, Clustering
and VNG and reruns the clustering methods against a fresh whole-graph baseline.
It has 46 jobs and a separate source snapshot. See
[clustering operators](clustering_operators.md) for all optimized paths, numerical
checks, the GECC Products propagation repair and the exact launch command.

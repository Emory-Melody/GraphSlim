# ogbn-products measurements

Snapshot: 2026-10-28. [Machine-readable results and per-job provenance](../benchmark/results/products_20261028.json).

![GCN accuracy across three training-node budgets and graph reduction time at 0.2%](../figures/products_results.svg)

The dataset contains 2,449,029 nodes, 100 features and 47 classes. The official
OGB split has 196,615 training, 39,323 validation and 2,213,091 test nodes.
Reduction rates use the training-node count as their denominator. The tables
report the actual graph size produced by each method, including its rounding
and class-allocation rules.

## Accuracy and reduction time

These local measurements use package defaults. Each reduced graph is evaluated
with a two-layer GCN: hidden size 256, 300 updates, learning rate 0.01, weight decay
0.0005, ReLU and dropout 0. The checkpoint is selected by validation accuracy,
checked every 10 updates. Reduction seeds are 1, 2 and 3; each reduction has GCN
seeds 0–9. The full-graph baseline uses GCN seeds 0–9.

Accuracy is the mean ± population standard deviation of all 30 test scores per
reduced cell, or 10 scores for the full graph. Reduction time is the median of
the three `timing.reduction_seconds` values and includes reduced-graph saving.
Data loading and downstream GCN fits have separate timers in the raw records.

| Training-node budget | Method | Actual nodes | Test accuracy (%) | Median graph reduction time (s) |
| --- | --- | ---: | ---: | ---: |
| Full graph | GCN | 2,449,029 | 73.78 ± 0.11 | — |
| 0.05% | Random | 114 | 40.84 ± 1.49 | 0.224 |
| 0.05% | CGC | 98 | 61.83 ± 0.39 | 39.857† |
| 0.05% | ClustGDD | 98 | 58.18 ± 1.34 | 219.074 |
| 0.05% | GCGP | 98 | 19.99 ± 9.64 | 52.906† |
| 0.1% | Random | 208 | 47.80 ± 0.76 | 0.172 |
| 0.1% | CGC | 196 | 62.23 ± 0.55 | 35.369 |
| 0.1% | ClustGDD | 196 | 61.77 ± 0.34 | 215.168 |
| 0.1% | GCGP | 197 | 17.13 ± 10.21 | 36.015† |
| 0.2% | Random | 401 | 53.92 ± 0.60 | 0.183 |
| 0.2% | CGC | 393 | 63.90 ± 0.42 | 78.734 |
| 0.2% | ClustGDD | 393 | 64.38 ± 0.14 | 220.030 |
| 0.2% | GCGP | 393 | 26.67 ± 0.56 | 54.181 |

Hardware: NVIDIA RTX PRO 6000 Blackwell Server Edition, Torch 2.8.0+cu128,
CUDA 12.8. In the three cells marked †, reduction seed 2 used a MIG 2g.48gb
partition; the other jobs used the full-GPU profile. Each job's GPU name is
included in the JSON export. The homepage's 0.2% table uses the full-GPU profile
throughout.

CGC uses clustering and reconstruction. ClustGDD uses 80 pretraining and 100
refinement updates, with edges processed in chunks of 1,000,000. GCGP uses 200
optimization updates and a training batch size of 1,024. The accuracy table uses
GCGP's package arithmetic. The JSON retains each reduction's settings,
checkpoint hashes and all validation/test scores.

## CUDA Graph timing

![GCGP eager versus CUDA Graph optimization time, including setup](../figures/products_cuda_graph.svg)

The paired Products run uses the author's SGNK kernel implementation at commit
`d81ed427cd52d5a0d2a74bdf4328d31db1dcc1eb`, float32, 393 reduced nodes, batch
size 1,024 and 200 optimization updates. Each seed has three timing repeats on
the full-GPU profile above. [Timing implementation](../benchmark/gcgp_parity.py).

The timed loop includes kernel computation, batch sampling, GP solves and Adam
updates. CUDA Graph captures the kernel's forward and backward computation;
sampling, solves and Adam use eager execution. Data preprocessing runs before
the timer, and downstream GCN evaluation runs after it.

| Reduction seed | Median eager loop (s) | Median captured loop (s) | Median graph setup (s) | Captured loop + setup (s) | Speedup including setup |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.758 | 0.783 | 0.539 | 1.321 | 1.33× |
| 2 | 1.720 | 0.649 | 0.588 | 1.237 | 1.39× |
| 3 | 1.515 | 0.521 | 0.576 | 1.097 | 1.38× |

The reported ratio is `median(eager loop) / (median(captured loop) + median(setup))`.
All three seeds have matching optimization losses, final parameter hashes and
downstream validation/test scores between the two execution modes. The paired
run evaluates one GCN seed per reduction seed. Its raw timings and checks are in
the JSON's `cuda_graph` field.

## Source records

The export contains 13 accuracy cells, 37 completed accuracy jobs with 370 GCN
evaluations, and one completed paired-timing job. Means, population standard
deviations and speed ratios were recomputed from those records. Dataset,
feature, label and split hashes match across all 38 jobs.

| Results | Local run directory | Source SHA-256 |
| --- | --- | --- |
| Full graph, Random, CGC, GCGP and paired timing | `runs/dataset_expansion_20261028_resume` | `66e175c052a27271d5dbda4adb1d587e02fa9ae69c3171d7c78f3f315481dda0` |
| ClustGDD, all three rates | `runs/products_streaming_20261028` | `1bd0fdfcaf12705f25a2afb5d7d271edcecd95f4b018d302ada65c20127a2d91` |

The two runs use the same dataset and GCN protocol. ClustGDD's source snapshot
adds chunked edge processing. Its table entries use the completed streaming run.
Some expansion jobs inherit completed files from
`runs/dataset_expansion_20261012_v3`; the export records both the requested path
and the resolved path, together with each result file's SHA-256. The source ZIPs
were checked against the archive hashes in their manifests.

The portable JSON includes per-seed scores, timings, GPU names, reduction
metadata and checkpoint hashes. Full local logs and source ZIPs remain in the
run directories above. [Expansion runner](../benchmark/expand_datasets.py) ·
[Dataset expansion settings](dataset_expansion.md).

## Regenerate the figures

The figures use the blue/gray palette, hatched bars and value labels of the
[DeepSeek-R1 benchmark chart](https://github.com/deepseek-ai/DeepSeek-R1/blob/main/figures/benchmark.jpg).
The plotting script recomputes accuracy statistics and timing ratios from the
per-seed records before drawing. Bar axes start at zero.

```bash
python -m benchmark.plot_products
```

Outputs are SVG, PNG and PDF in `figures/`. `products_figures.json` records the
input hash, script hash and output hashes. The full-graph baseline is repeated
across the three accuracy groups for comparison. Its ten evaluations are shared
across those groups.

[Results PDF](../figures/products_results.pdf) ·
[CUDA Graph PDF](../figures/products_cuda_graph.pdf) ·
[Plotting script](../benchmark/plot_products.py).

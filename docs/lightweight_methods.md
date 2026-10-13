# CGC, ClustGDD, GCGP and NOPE

All four methods use the standard `create_reducer(method, setting, data, args)`
and `reduce(data, verbose=False, save=True)` interface. They are lazy public
exports, support CPU and the package's device argument, and save the same three
feature/adjacency/label tensors as existing methods. No new algorithm dependency,
author checkout, FAISS installation, external LLM or API key is required to run them.
Inputs currently require single-class, one-dimensional training labels; multilabel
condensation is not implemented. GCGP's two-dimensional *synthetic* labels are
continuous regression targets, not a multilabel dataset extension.

| Method | Implemented computation | Reduction training |
| --- | --- | --- |
| CGC | Two-hop propagation; least-squares classifier; class-wise low-rank spectral partition; confidence-calibrated means; cosine graph and closed-form reconstruction | No iterative GNN/feature training |
| ClustGDD | Truncated PPR denoising; multi-layer linear classifier; clustering of logits; column-normalized graph compression; class-aware effective-resistance views; classification/consistency refinement | Linear pretraining and feature refinement, **not training-free** |
| GCGP | SGNK/NTK/SNTK kernels; trace-scaled GP ridge solve; prediction matching; optimized continuous features and targets; optional BinConcrete adjacency | No GNN inner loop; synthetic parameters still optimized |
| NOPE | Heap-based neighbor-interference contractions; size-weighted features; contracted graph; training-only majority labels | No learned model |

## Sources and implementation boundaries

Pinned author commits are recorded by `benchmark.verify_lightweight`:

| Method | Primary reference | Author repository and pinned commit |
| --- | --- | --- |
| CGC | [WWW 2025 paper](https://arxiv.org/abs/2405.13707) | [XYGaoG/CGC](https://github.com/XYGaoG/CGC), `08ff04dc0d23110ff53445438f30bcff887e7c31` |
| ClustGDD | [KDD 2025 paper](https://arxiv.org/abs/2505.20807) | [HKBU-LAGAS/ClustGDD](https://github.com/HKBU-LAGAS/ClustGDD), `60825c536f794e7a3bb0ff22ae4efc07ea2d3e3a` |
| GCGP | [GCGP paper](https://arxiv.org/abs/2501.02565) | [WANGLin0126/GCGP](https://github.com/WANGLin0126/GCGP), `d81ed427cd52d5a0d2a74bdf4328d31db1dcc1eb` |
| NOPE | [ICML 2026 paper](https://proceedings.mlr.press/v306/bai26g.html) | [dazonglian/NOPE-main](https://github.com/dazonglian/NOPE-main), `9b9ee38b3966ab550ca9c9e5dd6b393aff8f444a` |

These are package adaptations, not claims of bitwise end-to-end author parity:

- CGC uses seeded scikit-learn clustering instead of FAISS. Duplicate/empty
  clusters are repaired, and the reconstruction solve uses double precision and
  a tiny numerical ridge. The citation profiles preserve effective raw features.
  `--with_structure 0` selects CGC-X with the same partition/calibration pipeline;
  the author's independently tuned CGC-X profiles are not automatically substituted.
- ClustGDD keeps linear pretraining and class-aware refinement. The refinement
  graph is rebuilt each optimizer step, keeping forward features synchronized with
  the optimized offset instead of reusing stale features; best offsets are cloned. Validation uses denoised features,
  training mode is restored, negative cosine weights are clipped, and sampled
  views are symmetrized. Validation labels select snapshots; test labels never do.
  Effective pretraining weight decay defaults to zero, as enforced by the
  author's no-ReLU MLP (even when the author command declares nonzero decay).
- GCGP defaults to SGNK, as in the release, and saves **continuous learned targets**.
  Downstream `eval_loss=MSE` compares probabilities, not log probabilities, including
  SGFormer. NTK/SNTK preserve the released covariance recurrence. SGNK/NTK do not
  use adjacency, so requesting learned adjacency with either kernel is rejected;
  use SNTK. BinConcrete parameters are symmetrized. The default training batch cap
  is 1024; subsampling larger training sets is an explicit scalable extension.
- NOPE refreshes affected heap costs and never merges disconnected components
  through invented edges. The author coarsens the whole graph using `merge_ratio`;
  the package default follows its **training-relative** reduction rate. Set
  `nope_budget_base="graph"` to use the author's rate denominator. Supernodes with
  no training labels are excluded from the fully labeled saved training graph;
  validation/test labels are never used to label them. Thus requested and actual
  node counts can differ. `data.reduction_metadata` reports this and
  `data.nope_mapping` retains every original node's coarse assignment. The Python
  adapter does not establish the author's published large-graph runtime claims.

All reducers guard dense synthetic adjacency with `max_syn_nodes=4096` by default.
Raise that JSON setting only with enough memory; storage grows quadratically.
Common defaults on datasets without a released dataset profile are extensions,
not paper-optimal settings. Cora/Citeseer CGC and ClustGDD rate profiles are shipped.

## Run and customize

```bash
python -m examples.train_lightweight -D cora -M cgc -G -1 --save_path ./checkpoints
python -m graphslim.train_all -D cora -M clustgdd -G -1 --save_path ./checkpoints
python -m graphslim.train_all -D cora -M gcgp -G -1 --save_path ./checkpoints
python -m graphslim.train_all -D cora -M nope -G -1 --save_path ./checkpoints
```

Method JSON defaults load first, dataset/rate profiles second, `--config` overrides
third, and explicit CLI options last. `--epochs` means refinement steps for
ClustGDD and GP optimization steps for GCGP; CGC/NOPE ignore iterative epochs.
For a short ClustGDD check, supply a JSON file containing:

```json
{"clustgdd_pre_epochs": 20, "clustgdd_hidden": 32}
```

Then pass `--config your_config.json --epochs 20 --eval_epochs 20`. These are
integration settings, not the Cora release's 80 pretraining / 2000 refinement steps.
Other examples of method JSON overrides:

```json
{"gcgp_kernel": "SNTK", "gcgp_learn_adj": true, "gcgp_ridge": 0.5}
```

```json
{"nope_relaxed": true, "nope_budget_base": "graph"}
```

The latter enables NOPE*. The mathematical verification script requires separate
author clones at the pinned commits, but normal package execution does not:

```bash
python -m benchmark.verify_lightweight --references runs/reference --output runs/author_parity.json
```

## Downstream and robustness coverage

The shared model roster is GCN, SGC, APPNP, Cheby, GraphSage, GAT and SGFormer.
Cross-architecture searches use validation only, then fit/test the frozen winner.
The no-search path evaluates each actual architecture, including SGFormer.
Saved graph tensors are reloaded independently, so GAT sparsification does not
contaminate another model's graph.
GAT preserves every nonzero edge constructed by these four methods; in particular,
small ClustGDD compression weights are not discarded by the legacy 0.5 threshold.

```bash
python -m benchmark.run_cross_arch -D cora -M cgc -G -1 --save_path ./checkpoints
python -m benchmark.run_nas -D cora -M clustgdd -G -1 --save_path ./checkpoints
python -m benchmark.graph_property -D cora -M gcgp -G -1 --save_path ./checkpoints
python -m benchmark.reproduce robustness --datasets cora --methods cgc clustgdd gcgp nope --dry-run
```

`perturb_graph` returns an independent copy, is seed-controlled, and distinguishes
training corruption from fixed-model inference corruption:

```python
from graphslim.dataset import perturb_graph
noisy_train = perturb_graph(data, "gaussian", .1, 1, setting="ind", stage="training")
noisy_test = perturb_graph(data, "edge_drop", .1, 1, setting="ind", stage="inference")
```

| Corruption | Contract |
| --- | --- |
| `random_adj` | Add undirected non-self edges; budget is a fraction of existing undirected edges, capped by available nonedges |
| `edge_drop` | Remove that fraction of existing undirected non-self edges |
| `random_feat` | Mask a fraction of feature columns, retaining legacy semantics |
| `feature_mask` | Independently mask feature entries |
| `gaussian` | Gaussian noise with per-feature sigma = rate × training-feature standard deviation |
| `label_noise` | Change only a fraction of training labels to a different class; forbidden at inference |
| CLI `metattack` | Existing **PRBCD** adversarial path, not literal MetaAttack; no test-label surrogate training/selection |

Transductive feature/edge noise affects the visible full graph. Inductive training
noise leaves validation/test graphs unchanged; inductive inference noise affects
only the test graph. CLI noise caches include content/split/label identities;
fresh and cached runs must agree. Noise is not reapplied inside each evaluator fit.
The inductive PRBCD surrogate uses only the training graph, with an internal
training-label validation subset. Cheap noise does not train an extra baseline GCN.

The reproducible real-data integration runner freezes source **before** dispatch:

```bash
python -m benchmark.lightweight_downstream --run-dir runs/lightweight_cora --dry-run
python -m benchmark.lightweight_downstream --run-dir runs/lightweight_cora --profile smoke
```

Default smoke roster: 32 reductions = four methods × eight conditions, 224 real
downstream fits, 140 fixed-model inference-noise predictions, two-candidate NAS
on original/synthetic graphs for each clean reduction, graph properties and MIA.
It runs on CPU and does not preempt GPU experiments. Full method defaults can be
selected with `--profile full`; this still is not a paper HPO/repeated benchmark.
Multiple reduction/evaluation seeds are explicit `--seeds` / `--eval-seeds` rosters.

Each run retains a manifest, source ZIP, frozen execution directory, method configs,
data/checkpoint hashes, raw scores, perturbation metadata and errors. A complete
model/seed roster is required before a numeric table row appears. Undefined
properties (e.g. homophily on an empty graph) are `null`, not fake measurements.
MIA is the existing oracle-threshold confidence diagnostic, not an independently
calibrated privacy guarantee. Two-candidate NAS is an entrypoint check, not the
full 480-candidate research experiment. All smoke accuracy tables stay separate
from published and 3 × 10 repeated benchmark results.

```bash
python -m pytest -q tests/test_lightweight_methods.py tests/test_lightweight_runner.py
pip install -e ".[docs]"
sphinx-build -b html docs docs/_build/html
```

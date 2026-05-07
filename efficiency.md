# Efficiency Notes

This file records performance-related package changes and benchmark expectations.

## Baseline

No full benchmark was run in this pass because the repository does not include a small checked-in dataset fixture and the existing benchmark scripts require external graph datasets.

Known current bottlenecks from code inspection:

- Several coarsening methods convert between PyG, SciPy sparse matrices, PyGSP graphs, NumPy arrays, and Torch tensors.
- `graphslim/dataset/convertor.py` often materializes CPU NumPy arrays during graph conversion.
- Condensation training frequently moves sampled data to the target device inside class loops.
- Edge sparsification delegates to NetworKit, which is already compiled, but conversion back to PyG can dominate small runs.

## Changes In This Pass

- Added a lazy reduction registry. This avoids importing every reduction implementation when package users only need method discovery or construction metadata.
- Replaced the `train_all.py` method switch with `create_reducer`, reducing orchestration overhead and removing duplicated code paths.
- Reused the same registry for condensation initialization, so initialization methods and top-level reduction methods share one construction path.
- Added cheap graph summary tracking for WandB that avoids serializing full graphs by default.
- Added local visualization helpers that cap plotted nodes for small-graph inspection.

## Test Results

Environment:

- Helper tests: Python 3.13 from `/home/ubuntu/miniconda3/bin/python`; `pytest` 9.0.3 installed for this run.
- Method smoke tests: `pyg` conda environment, Python 3.11.14, Torch 2.8.0+cu128, Torch Geometric 2.7.0, CPU execution.
- Dataset and command shape: Citeseer, `graphslim.train_all`, `--run_reduction 1`, `--run_eval 1`, `--run_inter_eval 1`, `--eval_epochs 1`, `--reduction_rate 0.01`, one seed.
- Minimum viable reduction setting: `--epochs 10`. A literal `--epochs 1` currently fails before training because `setting_config()` computes `args.epochs // 10`, producing an invalid checkpoint step of `0`.

Interpretation:

- The repeated `23.10 +/- 0.00` accuracy is not a meaningful benchmark score. With `eval_epochs=1`, `run_eval=1`, one seed, and a 1% Citeseer reduction, the evaluator barely trains and mostly reports a deterministic smoke-test value.
- Treat the method sweep as a pipeline health check: dataset load, reducer construction, minimum reduction path, reduced graph handoff, and evaluator invocation.
- For quality comparisons, rerun with enough evaluation epochs, multiple seeds, and method-appropriate reduction epochs.

### Before Optimization

No before-optimization method-level run was recorded for this pass, so the baseline remains unmeasured.

| Scope | Command | Result | Duration | Warnings | Notes |
| --- | --- | --- | --- | --- | --- |
| Method minimum-epoch smoke sweep | Not recorded | Not measured | Not measured | Not measured | No saved pre-optimization `train_all` output is available. |
| Helper unit tests | Not recorded | Not measured | Not measured | Not measured | No saved pre-optimization pytest output is available. |
| Full benchmarks | Not recorded | Not measured | Not measured | Not measured | Benchmark scripts require external graph datasets and longer runs. |

### After Optimization

| Scope | Command | Result | Duration | Warnings | Notes |
| --- | --- | --- | --- | --- | --- |
| Method minimum-epoch smoke sweep | `conda run --no-capture-output -n pyg python ... graphslim.train_all` | 22 passed, 12 failed, 2 timed out | 840.14s total | PyTorch/PyG deprecation warnings from environment | Ran all 36 registered methods on Citeseer with CPU, `epochs=10`, `eval_epochs=1`, and a 180s per-method timeout. |
| Unit tests | `python -m pytest -q` | 8 passed | 7.78s | 2 deprecation warnings | Warnings came from `torch_geometric.distributed` deprecation and `importlib.metadata` implicit `None` behavior. |
| Test collection | `python -m pytest --collect-only -q` | 8 collected | 3.71s | 2 deprecation warnings | Confirms the active suite covers compatibility, registry, tracking, and visualization tests. |
| Benchmarks | Not run | Not measured | Not measured | Not measured | External graph datasets are still required for end-to-end benchmark scripts. |

| Method | Status | Runtime (s) | Eval Accuracy | Notes |
| --- | --- | ---: | --- | --- |
| `affinity_gs` | Failed | 7.86 | - | `TypeError: Graph.__init__() got an unexpected keyword argument 'W'` |
| `algebraic_jc` | Failed | 7.54 | - | `TypeError: Graph.__init__() got an unexpected keyword argument 'W'` |
| `averaging` | Failed | 7.66 | - | `IndexError: index 6 is out of bounds for dimension 0 with size 6` |
| `cent_d` | Passed | 7.07 | 23.10 +/- 0.00 | Completed `train_all` smoke path. |
| `cent_p` | Passed | 7.10 | 23.10 +/- 0.00 | Completed `train_all` smoke path. |
| `clustering` | Failed | 7.72 | - | `IndexError: index 6 is out of bounds for dimension 0 with size 6` |
| `doscond` | Passed | 10.39 | 23.10 +/- 0.00 | Completed `train_all` smoke path. |
| `doscondx` | Passed | 11.50 | 23.10 +/- 0.00 | Completed `train_all` smoke path. |
| `g_spar` | Passed | 7.26 | 23.10 +/- 0.00 | Completed `train_all` smoke path. |
| `gcdm` | Passed | 8.37 | 23.10 +/- 0.00 | Completed `train_all` smoke path. |
| `gcdmx` | Failed | 8.30 | - | `IndexError: mask [3327] does not match indexed tensor [6]` |
| `gcond` | Passed | 16.14 | 23.10 +/- 0.00 | Completed `train_all` smoke path. |
| `gcondx` | Passed | 13.01 | 23.10 +/- 0.00 | Completed `train_all` smoke path. |
| `gcsntk` | Passed | 8.31 | 23.10 +/- 0.00 | Completed `train_all` smoke path. |
| `gdem` | Failed | 36.72 | - | `PermissionError: [Errno 13] Permission denied: '../../data//'` |
| `gecc` | Passed | 8.07 | 23.10 +/- 0.00 | Completed `train_all` smoke path. |
| `geom` | Timed out | 181.68 | - | Exceeded 180s per-method timeout. |
| `heavy_edge` | Failed | 7.56 | - | `TypeError: Graph.__init__() got an unexpected keyword argument 'W'` |
| `herding` | Passed | 7.53 | 23.10 +/- 0.00 | Completed `train_all` smoke path. |
| `kcenter` | Passed | 7.79 | 23.10 +/- 0.00 | Completed `train_all` smoke path. |
| `kron` | Failed | 7.44 | - | `TypeError: Graph.__init__() got an unexpected keyword argument 'W'` |
| `local_degree` | Passed | 7.47 | 23.10 +/- 0.00 | Completed `train_all` smoke path. |
| `msgc` | Passed | 20.36 | 23.10 +/- 0.00 | Completed `train_all` smoke path. |
| `random` | Passed | 8.04 | 23.10 +/- 0.00 | Completed `train_all` smoke path. |
| `random_edge` | Passed | 7.46 | 23.10 +/- 0.00 | Completed `train_all` smoke path. |
| `rank_degree` | Passed | 7.81 | 23.10 +/- 0.00 | Completed `train_all` smoke path. |
| `scan` | Passed | 7.14 | 23.10 +/- 0.00 | Completed `train_all` smoke path. |
| `sfgc` | Timed out | 181.41 | - | Exceeded 180s per-method timeout. |
| `sgdd` | Passed | 16.54 | 23.10 +/- 0.00 | Completed `train_all` smoke path. |
| `simgc` | Passed | 151.86 | 23.10 +/- 0.00 | Completed `train_all` smoke path. |
| `spanning_forest` | Passed | 7.67 | 23.10 +/- 0.00 | Completed `train_all` smoke path. |
| `t_spanner` | Passed | 7.42 | 23.10 +/- 0.00 | Completed `train_all` smoke path. |
| `variation_cliques` | Failed | 7.46 | - | `TypeError: Graph.__init__() got an unexpected keyword argument 'W'` |
| `variation_edges` | Failed | 7.45 | - | `TypeError: Graph.__init__() got an unexpected keyword argument 'W'` |
| `variation_neighborhoods` | Failed | 7.45 | - | `TypeError: Graph.__init__() got an unexpected keyword argument 'W'` |
| `vng` | Failed | 7.43 | - | `NameError: name 'SGC' is not defined` |

### Fast Smoke Path For `sfgc` And `geom`

`sfgc` and `geom` are slow in the generic sweep because they generate teacher trajectory replay buffers before the actual condensation loop:

- `sfgc`: forces `num_experts=20`, then trains each expert for `teacher_epochs=800` on Citeseer.
- `geom`: forces `num_experts=20`, then trains each expert for `teacher_epochs=1000`; its config also sets `coreset_epochs=800` and `syn_steps=200`.

For a CI or smoke-test path, prebuild a tiny replay buffer and run the reducers with `--no_buff`, then override JSON-only knobs in a small harness: `epochs=1`, `teacher_epochs=10`, `expert_epochs=10`, `syn_steps=1`, `coreset_epochs=1`, `optim_lr=0`, `beta=0`, and empty checkpoints. This keeps the real reducer and evaluator path while skipping expensive teacher-buffer generation.

| Method | Smoke Strategy | Result | Runtime (s) | Eval Accuracy | Notes |
| --- | --- | --- | ---: | --- | --- |
| `sfgc` | Tiny replay buffer + `--no_buff` + one synthetic update | Passed | 0.34 | 23.10 +/- 0.00 | Runtime validates the fast-path harness, not model quality. |
| `geom` | Tiny replay buffer + `--no_buff` + one synthetic update | Passed | 0.33 | 23.10 +/- 0.00 | Runtime validates the fast-path harness, not model quality. |

### Cross-Dataset Loader Coverage

Representative datasets were selected to exercise different loader branches rather than only Planetoid/Citeseer.

| Dataset | Loader Path | Setting | Result | Runtime (s) | Loaded Shape |
| --- | --- | --- | --- | ---: | --- |
| `cora` | Planetoid | `trans` | Passed | 10.23 | 2,708 nodes, 140 train, 7 classes, 1,433 features |
| `photo` | PyG Amazon | `trans` | Passed | 8.50 | 7,650 nodes, 6,116 train, 8 classes, 745 features |
| `cs` | PyG Coauthor | `trans` | Passed | 10.03 | 18,333 nodes, 14,661 train, 15 classes, 6,805 features |
| `flickr` | PyG Flickr | `ind` | Passed | 18.83 | 89,250 nodes, 44,625 train, 7 classes, 500 features |
| `ogbn-arxiv` | GraphSAINT/OGB fallback | `trans` | Passed | 24.75 | 169,343 nodes, 90,941 train, 40 classes, 128 features |
| `yelp` | DGL `FraudDataset` conversion | `ind` | Passed | 10.76 | 45,954 nodes, 36,762 train, 2 classes, 32 features |

### All-Method Constructor Coverage

This checks method registry resolution, imports, method-specific config loading, and reducer construction across loader paths. It does not run reduction loops. The large-loader constructor sweep was stopped after `cs` because it became too slow without per-method process isolation.

| Dataset | Loader Path | Methods Checked | Passed | Failed | Runtime (s) | Failure Pattern |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `cora` | Planetoid | 36 | 35 | 1 | 17.14 | `gdem` hardcodes `../../data//` and hits `PermissionError`. |
| `photo` | PyG Amazon | 36 | 31 | 5 | 14.31 | Missing method-config attributes for methods such as `gcsntk`, `geom`, and `sfgc`; `gdem` path issue. |
| `cs` | PyG Coauthor | 36 | 31 | 5 | 196.16 | Same missing method-config attributes as `photo`; `gdem` path issue. |

### Module-By-Loader Smoke Matrix

This matrix runs one representative method per major reduction module across each loader path: node sparsification (`kcenter`), edge sparsification (`random_edge`), coarsening (`heavy_edge`), and condensation (`gcond`). Each run uses CPU, one seed, minimum smoke settings, and a 120s per-combo timeout.

| Dataset | Loader Path | Method | Module | Result | Runtime (s) | Eval Accuracy | Notes |
| --- | --- | --- | --- | --- | ---: | --- | --- |
| `cora` | Planetoid | `kcenter` | Node sparsification | Passed | 7.43 | 14.40 +/- 0.00 | Completed `train_all` smoke path. |
| `cora` | Planetoid | `random_edge` | Edge sparsification | Passed | 7.45 | 14.40 +/- 0.00 | Completed `train_all` smoke path. |
| `cora` | Planetoid | `heavy_edge` | Coarsening | Failed | 7.28 | - | `TypeError: Graph.__init__() got an unexpected keyword argument 'W'` |
| `cora` | Planetoid | `gcond` | Condensation | Passed | 14.85 | 14.40 +/- 0.00 | Completed `train_all` smoke path. |
| `photo` | PyG Amazon | `kcenter` | Node sparsification | Passed | 8.36 | 25.32 +/- 0.00 | Completed `train_all` smoke path. |
| `photo` | PyG Amazon | `random_edge` | Edge sparsification | Passed | 7.79 | 22.60 +/- 0.00 | Completed `train_all` smoke path. |
| `photo` | PyG Amazon | `heavy_edge` | Coarsening | Failed | 7.45 | - | `TypeError: Graph.__init__() got an unexpected keyword argument 'W'` |
| `photo` | PyG Amazon | `gcond` | Condensation | Passed | 22.52 | 30.26 +/- 0.00 | Completed `train_all` smoke path. |
| `cs` | PyG Coauthor | `kcenter` | Node sparsification | Passed | 8.64 | 33.37 +/- 0.00 | Completed `train_all` smoke path. |
| `cs` | PyG Coauthor | `random_edge` | Edge sparsification | Passed | 8.37 | 57.61 +/- 0.00 | Completed `train_all` smoke path. |
| `cs` | PyG Coauthor | `heavy_edge` | Coarsening | Failed | 8.24 | - | `TypeError: Graph.__init__() got an unexpected keyword argument 'W'` |
| `cs` | PyG Coauthor | `gcond` | Condensation | Passed | 47.50 | 22.50 +/- 0.00 | Completed `train_all` smoke path. |
| `flickr` | PyG Flickr | `kcenter` | Node sparsification | Passed | 12.10 | 13.52 +/- 0.00 | Completed `train_all` smoke path. |
| `flickr` | PyG Flickr | `random_edge` | Edge sparsification | Failed | 10.28 | - | Progress bar starts, then exits nonzero without a final Python exception line in captured output. |
| `flickr` | PyG Flickr | `heavy_edge` | Coarsening | Failed | 10.25 | - | `TypeError: Graph.__init__() got an unexpected keyword argument 'W'` |
| `flickr` | PyG Flickr | `gcond` | Condensation | Passed | 23.65 | 13.00 +/- 0.00 | Completed `train_all` smoke path. |
| `ogbn-arxiv` | GraphSAINT/OGB fallback | `kcenter` | Node sparsification | Passed | 18.10 | 0.58 +/- 0.00 | Completed `train_all` smoke path. |
| `ogbn-arxiv` | GraphSAINT/OGB fallback | `random_edge` | Edge sparsification | Passed | 15.06 | 2.58 +/- 0.00 | Completed `train_all` smoke path. |
| `ogbn-arxiv` | GraphSAINT/OGB fallback | `heavy_edge` | Coarsening | Failed | 11.40 | - | `TypeError: Graph.__init__() got an unexpected keyword argument 'W'` |
| `ogbn-arxiv` | GraphSAINT/OGB fallback | `gcond` | Condensation | Timed out | 121.43 | - | Exceeded 120s per-combo timeout. |
| `yelp` | DGL `FraudDataset` conversion | `kcenter` | Node sparsification | Passed | 16.95 | 46.08 +/- 0.00 | Completed `train_all` smoke path. |
| `yelp` | DGL `FraudDataset` conversion | `random_edge` | Edge sparsification | Failed | 10.49 | - | `SPMMSum` gradient shape mismatch: got `[45954, 256]`, expected `[36762, 256]`. |
| `yelp` | DGL `FraudDataset` conversion | `heavy_edge` | Coarsening | Failed | 9.35 | - | `TypeError: Graph.__init__() got an unexpected keyword argument 'W'` |
| `yelp` | DGL `FraudDataset` conversion | `gcond` | Condensation | Passed | 26.89 | 46.08 +/- 0.00 | Completed `train_all` smoke path. |

| Test File | Tests Collected | After-Optimization Result |
| --- | ---: | --- |
| `tests/test_compat.py` | 2 | Passed |
| `tests/test_reduction_registry.py` | 2 | Passed |
| `tests/test_tracking.py` | 2 | Passed |
| `tests/test_visualization.py` | 2 | Passed |

## C++ / Compiled Acceleration Plan

Do not introduce compiled code until a before/after benchmark exists. Candidate targets:

- Connected component extraction and contraction loops in coarsening.
- Sparse edge filtering for high-volume sparsification.
- CSR/COO conversion utilities used repeatedly across reduction families.

Recommended implementation path:

1. Add a Python benchmark fixture that creates synthetic sparse graphs with fixed seeds.
2. Record CPU time, peak memory, and output edge/node counts for pure-Python implementations.
3. Add optional extensions under `graphslim/_C` with Python fallbacks.
4. Gate use through feature detection, not mandatory imports, so PyPI wheels remain installable.
5. Record speedup here with environment, graph size, method, memory, and disk footprint.

## Benchmark Template

Record each run with:

- Date
- Method
- Dataset or synthetic fixture
- Device
- Before and after runtime
- Speedup
- Peak memory
- Disk usage
- Notes and environment details

Initial entry: 2026-05-06, registry construction, import-only CPU path. Runtime, memory, and disk usage were not measured because this pass was a structural refactor only.

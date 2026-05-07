# GraphSlim Package Structure

GraphSlim is a PyTorch/PyG graph reduction package. The current code already separates the three reduction families, but orchestration logic and method discovery were previously duplicated across entry points.

## Current Layout

- `graphslim/condensation/`: graph condensation methods and shared `GCondBase`.
- `graphslim/coarsening/`: graph coarsening methods, `Coarsen`, and coarsening utilities.
- `graphslim/sparsification/`: node/core-set and edge sparsification methods.
- `graphslim/models/`: GNN backbones, graph layers, parametrized adjacency, kernels, and attack models.
- `graphslim/layers/`: public PyGOD-style layer namespace that re-exports existing graph layer implementations.
- `graphslim/dataset/`: dataset loading, conversion, attack, and persistence helpers. Keep this layer stable for existing data loading behavior.
- `graphslim/evaluation/`: downstream evaluation, graph-property evaluation, NAS, timing, and memory helpers.
- `graphslim/reduction/`: shared method registry and factory for constructing reduction methods.
- `graphslim/compat.py`: PyG/DGL conversion helpers for package users.
- `graphslim/tracking.py`: optional WandB tracking wrapper and graph evolution summaries.
- `graphslim/visualization.py`: local visualization utilities for small before/after graph inspection.
- `benchmark/`: benchmark scripts for end-to-end experiments.
- `examples/`: small training examples.
- `interface/`: Streamlit visualization prototype.

## Target Modularization

Follow a PyGOD-like split where reusable pieces are explicit:

- `models`: GNN architectures and reduction models only.
- `layers`: graph neural network layers and low-level model components. The public namespace is now `graphslim.layers`, backed by the existing implementations in `graphslim/models/layers.py`.
- `utils`: tensor, sparse, metric, persistence, profiling, and conversion helpers.
- `reduction`: method registry, initialization, condensation, coarsening, sparsification, and validation pipelines.
- `evaluation`: cross-model validation and downstream model evaluation.
- `applications`: command-line, visualization, and benchmark entry points.

## Graph Reduction Flow

1. Initialization: `graphslim.reduction.create_reducer(args.init, ...)` constructs reusable initialization methods such as `random`, `kcenter`, `herding`, `clustering`, and `averaging`.
2. Reduction:
   - Sparsification methods select nodes or edges.
   - Coarsening methods contract or aggregate graph structure.
   - Condensation methods synthesize features and, when needed, structure.
3. Cross-model validation: `graphslim.evaluation.Evaluator` evaluates the reduced graph with GCN, SGC, APPNP, Cheby, GraphSage, GAT, or SGFormer.
4. Tracking: `graphslim.tracking` can log original/reduced graph summaries and evaluation metrics to WandB.
5. Visualization: `graphslim.visualization.draw_graph_pair` compares small original/reduced graphs locally.

## Reuse Boundaries

- Method construction should go through `graphslim.reduction.registry` instead of `eval()` or long `if/elif` chains.
- Dataset loading should stay in `graphslim.dataset` and continue to return the current PyG-like objects.
- DGL support should use conversion helpers at package boundaries, not replace the PyG-first internal data contract.
- C++ acceleration should be introduced behind Python APIs in `graphslim/coarsening` or `graphslim/sparsification`, with pure-Python fallbacks.

## Near-Term Refactor Checklist

- Move duplicate class-selection logic into the registry.
- Gradually move implementations from `graphslim/models/layers.py` into `graphslim.layers` once downstream imports are updated.
- Add a unified reduction pipeline for initialization, reduction, validation, tracking, and persistence.
- Add fast CPU tests for construction, conversion, tracking, and visualization helpers.
- Add optional acceleration benchmarks before introducing compiled extensions.

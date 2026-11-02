# Quick Start

After [installation](installation.md), run commands from the repository root.

## Select nodes

Reduce Cora with K-Center and evaluate the saved graph using a GCN:

```bash
python -m graphslim.train_all -D cora -M kcenter -R 0.1 -G -1 --run_reduction 1 --save_path ./checkpoints
```

`-R 0.1` sets a budget of 10% of labeled training nodes. Use `-G 0` for a GPU.

## <span class="edge-sparsification">Sparsify edges</span>

````{container} edge-sparsification
Keep the input nodes and target 10% of the edges:

```bash
python -m graphslim.train_all -D cora -M random_edge -R 0.1 -G -1 --run_reduction 1 --save_path ./checkpoints
```
````

## Evaluate a saved graph

Reuse the dataset, method, rate, seed, and checkpoint path from reduction:

```bash
python -m graphslim.run_eval -D cora -M kcenter -R 0.1 -G -1 --save_path ./checkpoints
```

See the [method API](source/graphslim.sparsification.rst) for method-specific settings,
or run `python -m graphslim.train_all --help` for all options.

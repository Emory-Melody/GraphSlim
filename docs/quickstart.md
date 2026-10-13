# Quick Start

Run examples from the repository root after installing the package:

```bash
python -m examples.train_node_sparsification -G -1
python -m examples.train_coarsen -G -1
python -m examples.train_gcond -G -1
```

The selected defaults are KCenter, VariationNeighborhoods, and GCond, respectively.
`-G -1` selects CPU. Override any CLI option explicitly.

```bash
python -m graphslim.train_all -D cora -M clustering -G -1 --save_path ./checkpoints
python -m graphslim.train_all --help
python -m benchmark.reproduce performance --dry-run
```

The package and all examples share the same configuration, reduction, and evaluation
pipeline. Benchmark reproduction instructions and full published performance tables
are in the repository README.

CGC, ClustGDD, GCGP and NOPE use the same pipeline:

```bash
python -m examples.train_lightweight -M cgc -G -1 --save_path ./checkpoints
```

See [lightweight methods](lightweight_methods.md) for algorithm boundaries,
method JSON overrides, all seven downstream architectures, noise/PRBCD tests,
and auditable real-data integration matrices.

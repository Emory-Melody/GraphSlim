# Performance profiling

GraphSlim uses native PyTorch operations and an optional [CUDA Graph backend](cuda_graph.md)
for repeated computation. The backend requires no GraphSlim-specific CUDA
extension. See [GPU setup](compatibility.md) for supported
hardware and installation checks.

## SGNK computation

The released SGNK computes full `X @ X.T` and `Y @ Y.T` matrices to obtain their
diagonals. GraphSlim obtains those squared row norms directly. The required
cross-kernel remains `X @ Y.T`; the erf recurrence and GP prediction matching
retain their mathematical definitions. This removes quadratic self-Gram
intermediates using existing Torch kernels.

The paired benchmark checks forward outputs, synthetic-feature gradients and
GP predictions before measuring either implementation. It can also replay a
fixed Cora reduction and downstream evaluation. Its reference is the pinned
[GCGP mathematical implementation](https://github.com/WANGLin0126/GCGP/blob/d81ed427cd52d5a0d2a74bdf4328d31db1dcc1eb/gcgp/sgnk.py).

## Run a profile

Measure loading, conversion, preprocessing, reduction, validation and downstream
fitting separately. Record the dataset split, reduced-node budget, seeds,
precision, hardware and evaluation settings with each run. Include setup and
compilation when reporting the time required for a complete job.

The optional reference checkout must contain `gcgp/sgnk.py` and `gcgp/gpr.py` at
commit `d81ed427cd52d5a0d2a74bdf4328d31db1dcc1eb`. The benchmark loads these
definitions without running author entrypoints. Normal GraphSlim execution
requires no reference checkout.

```bash
python -m benchmark.benchmark_kernels --device cpu --output runs/sgnk_cpu.json
python -m benchmark.benchmark_kernels --device cuda:0 --output runs/sgnk_cuda.json
python -m benchmark.benchmark_kernels --device cpu --dtype float64 --cora-replay --output runs/sgnk_cpu_fp64.json
```

Use distinct output paths for each device, precision and source revision. The
benchmark snapshots scientific source and author definitions, checks that
source remains unchanged, and retains raw timing and score records.

## Choose an implementation

Start by removing repeated normalization, unnecessary dense matrices, avoidable
transfers and redundant computations. Cache quantities that remain fixed;
changes to synthetic features, teacher parameters, graph noise or split identity
must invalidate the relevant cache.

Use built-in sparse operations, matrix multiplication and optional Torch
compilation where profiling identifies useful work. Evaluate a custom
CUDA/Triton operator on matched data, budgets and accuracy, with numerical and
gradient checks and a portable Torch fallback. Keep cold setup and warm
execution times in the experiment record.

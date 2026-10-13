# Efficiency evidence and acceleration policy

The development checkout includes a paired benchmark against pinned GCGP author
mathematics and a runtime compatibility probe. The measured improvements below
apply to SGNK kernels and a controlled prediction-matching replay, not to every
reducer or to the complete original repository. Negative timing and precision
results are retained. No GraphSlim-specific CUDA extension is required.

The optional [CUDA Graph backend](cuda_graph.md) now connects the shared GNN
training entry points and eligible reduction stages. Interface coverage spans
the complete method registry; its general-backend performance must be validated
separately from the historical SGNK measurements below.

## Removing unnecessary work before writing CUDA

The released SGNK computes full `X @ X.T` and `Y @ Y.T` matrices solely to obtain
their diagonals. GraphSlim obtains those squared row norms directly. The required
cross-kernel remains `X @ Y.T`; the erf recurrence and GP prediction matching are
unchanged mathematically. This avoids quadratic self-Gram intermediates and their
unnecessary matrix multiplication. It is an algorithmic/memory optimization using
existing Torch kernels, not a newly written CUDA operator.

Both implementations receive the same tensors, dtype and kernel parameters.
Forward outputs, synthetic-feature gradients and GP predictions are checked
before timing. Numerical equivalence of one call is not a guarantee that a long
nonconvex optimization follows the same finite-precision trajectory.

## Float32 kernel measurements

CPU: AMD EPYC 9115, four Torch/BLAS threads. CUDA: RTX PRO 6000 Blackwell SM120,
Torch 2.8.0+cu128. Inputs are seeded synthetic tensors with query nodes N,
synthetic nodes S and feature dimension D. They are shape checks, not a dataset
accuracy benchmark. The host was shared; only an authorized idle GPU was leased.

Times are medians of seven alternating paired repeats, three calls per repeat,
after three warmups. CUDA wall time includes host dispatch and device
synchronization. It is not a device-event-only measurement.

| Device | N × S × D | Author forward ms | GraphSlim forward ms | Forward speedup | Forward plus backward speedup |
| --- | --- | ---: | ---: | ---: | ---: |
| CPU | 140 × 70 × 1433 | 5.599 | 2.822 | 1.98× | 1.58× |
| CPU | 1024 × 256 × 128 | 19.075 | 6.996 | 2.73× | 1.50× |
| CPU | 4096 × 256 × 128 | 145.303 | 26.545 | 5.47× | 3.23× |
| CUDA 12.8 | 140 × 70 × 1433 | 2.306 | 1.633 | 1.41× | 1.27× |
| CUDA 12.8 | 1024 × 256 × 128 | 3.686 | 2.720 | 1.36× | 0.97× |
| CUDA 12.8 | 4096 × 256 × 128 | 2.342 | 1.972 | 1.19× | 1.24× |

The 0.97× result is a slight slowdown, not a speedup. Query/synthetic-feature
forward differences are at most 1.20e-7 and GP prediction differences at most
2.40e-7 in these Float32 cases. Raw repeats and measured additional CUDA peak
allocation are in the machine-readable records. The avoided self-Gram byte count
is a separate theoretical intermediate size; it is not presented as measured CPU
peak RSS or a complete-method memory saving.

## Controlled Cora replay and accuracy checks

This comparison replaces only the SGNK/GP mathematical implementation inside a
shared package workflow. It is **not an end-to-end run of the author's repository**.
Both sides share the Cora public split, raw BOW features, four-hop normalized
propagation, explicit CPU-generated initialization, 70 synthetic nodes, lr 0.01,
ridge 0.5 and 200 Adam updates. GCN evaluation uses hidden 256, 300 steps and
three fixed initialization seeds 0/1/2. Only validation selects GCN weights.
Accuracy is the mean and population standard deviation of these three fits,
not the benchmark's 3 reductions × 10 evaluations or a confidence interval.

| Device | GP precision | Author update loop s | GraphSlim update loop s | Author / GraphSlim | Author GCN accuracy % | GraphSlim GCN accuracy % |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| CPU | Float32 | 5.35 | 4.11 | 1.30× | 64.63±1.66 | 56.07±6.46 |
| CUDA 12.8 | Float32 | 4.93 | 3.99 | 1.23× | 68.67±1.48 | 63.57±0.54 |
| CPU | Float64 | 7.83 | 4.82 | 1.62× | 73.67±0.50 | 73.83±0.29 |
| CUDA 12.8 | Float64 | 2.23 | 3.72 | 0.60× | 64.13±0.26 | 64.30±0.50 |

The update-loop timing is one warmed optimization run per backend, excluding
loading, propagation, parameter/optimizer construction and downstream fitting.
It includes scalar loss recording and the release's extra training-accuracy
calculation. The seven-repeat kernel medians must not be confused with these
single-run optimization timings. Float64 optimization feeds Float32 downstream
evaluation on both sides; the reference's temporary default dtype is also set to
Float64 so its identity matrix does not silently downcast the regularizer.

Float32 has substantial long-trajectory/downstream differences despite small
single-call errors. Therefore it does **not** support a claim of faster complete
training with unchanged accuracy. Float64 substantially reduces final-parameter
differences (below 7.1e-6 in these replays), but the CUDA update loop is slower
than the author mathematics. It likewise does not establish a universal GPU
speedup. Float64 is a separate benchmark precision setting; the package's normal
GCGP reduction defaults and historical accuracy tables were not changed.

CPU/GPU GCN initializations and sparse propagation can differ numerically. The
required equality is within each paired device/precision experiment; these
results do not assert identical training trajectories across devices. No row is
silently substituted for a historical run or ranked against a different budget.

## Reproduce the measurements

The optional reference checkout must contain `gcgp/sgnk.py` and `gcgp/gpr.py` at
GCGP commit `d81ed427cd52d5a0d2a74bdf4328d31db1dcc1eb`. The benchmark AST-loads
mathematical definitions without running author entrypoints or changing GPUs.
Normal GraphSlim execution requires no reference checkout.

```bash
python -m benchmark.benchmark_kernels --device cpu --output runs/sgnk_cpu.json
python -m benchmark.benchmark_kernels --device cuda:0 --output runs/sgnk_cuda.json
python -m benchmark.benchmark_kernels --device cpu --dtype float64 --cora-replay --output runs/sgnk_cpu_fp64.json
```

Use distinct output paths for each device, precision or source revision. The
benchmark snapshots scientific source and author definitions, records hashes,
checks that source remains unchanged and retains raw timing/score records.

[Machine-readable results and compatibility probes](../benchmark/results/efficiency_compatibility.json)
include all four replay groups and the selected timing records. The original
local JSON files, source ZIPs and logs are in `runs/efficiency_compat_20261011`.
The baseline is pinned [GCGP author mathematics](https://github.com/WANGLin0126/GCGP/blob/d81ed427cd52d5a0d2a74bdf4328d31db1dcc1eb/gcgp/sgnk.py),
not a fabricated whole-GraphSlim baseline.

## When an optional compiled backend is justified

Profile loading/conversion, preprocessing, reduction, validation and downstream
fitting separately. First remove repeated normalization, unnecessary dense
matrices, avoidable transfers and redundant computations. Cache only quantities
that remain fixed; changing synthetic features, teacher parameters, graph noise
or split identity must invalidate the relevant cache.

Use built-in sparse/GEMM kernels and optional Torch compilation or fusion before
considering a new operator. A custom CUDA/Triton backend is justified only when
profiling identifies a shared dominant hotspot, numerical/gradient tests pass,
and complete matched-protocol runs improve without an unacceptable quality
tradeoff. Cold compilation and warm steady-state times must be reported
separately. Keep an explicit portable Torch fallback and no mandatory local
GraphSlim CUDA compilation.

Dynamic priority queues and edge contractions such as NOPE are CPU algorithm/data
structure candidates, not automatically good CUDA targets. Its current budget
and evaluation adaptation must be corrected before a fair end-to-end speed
comparison. The existing NOPE diagnosis must not be advertised as an acceleration
result. Full matched-data/budget/accuracy comparisons for GCond, SimGC, ClustGDD
and the other methods remain additional work; no whole-repository speedup is
inferred from the SGNK measurements above.

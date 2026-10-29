# CUDA Graph acceleration

GraphSlim has an optional native Torch CUDA Graph backend for fixed-shape,
first-order computation. All 40 registered reduction methods share its downstream
training entry points, covering GCN, SGC, APPNP, Cheby, GraphSage, GAT and SGFormer.
This is interface coverage, not a claim that all methods or models run faster.
GAT's dynamic attention structure currently stays eager before capture preflight.
The default remains eager execution. No GraphSlim CUDA extension or local NVCC
build is required.

```bash
python -m graphslim.train_all -D cora -M gcond -G 0 --cuda_graph
python -m graphslim.run_eval -D cora -M clustering -G 0 --cuda_graph
python -m examples.train_lightweight -D cora -M clustgdd -G 0 --cuda_graph
```

`--no_cuda_graph` explicitly restores the default. Stages with fewer than
`--cuda_graph_min_steps` updates remain eager; the default is 64. Native capture
uses `--cuda_graph_warmup` warmups, default 3. Short diagnostics may lower the
threshold, but their setup-inclusive timings are not an amortized speed claim.

## Captured stages

| Methods | Eligible computation | Computation retained outside capture |
| --- | --- | --- |
| All 40 registered methods | Shared first-order downstream fitting | Validation, checkpoint selection, testing and optimizer updates |
| GCGP | Dense SGNK and NTK kernel pairs | GP solve, synthetic-target updates, batching; sparse SNTK remains eager |
| ClustGDD | Classifier pretraining and refinement loss/backward | Clustering, view construction, validation and optimizer updates |
| SFGC | Expert training forward/backward | Trajectory persistence and second-order trajectory matching |
| GDEM | Spectral matching loss/backward and feature pretraining | Eigen preprocessing, optimizer scheduling and validation |
| SimGC, KCenter, Herding, VNG | Teacher or embedding fitting through the shared model API | Dynamic edge construction, selection and clustering |
| GCond, GCondX, DosCond, DosCondX, SGDD, MSGC | Shared downstream fitting | Second-order gradient matching |
| GEOM, GCDM, GCDMX | Shared downstream fitting | Dynamic curriculum, sampling and higher-order trajectories |
| CGC, NOPE, GECC and other CPU or noniterative reducers | Shared downstream fitting | CPU queues, eigensolvers, clustering and discrete graph operations |

Eligible stages are candidates, not unconditional captures. A runtime record
states whether each stage actually captured, remained eager or failed. The
complete machine-readable roster is returned by
`graphslim.acceleration.method_acceleration_support()`; new registry entries are
included automatically in downstream coverage.

## Numerical and state checks

Before accepting a first-order training capture, the backend compares one eager
forward/backward against replay using exact tensor equality for outputs,
gradients, weights and buffers. CPU and CUDA RNG states must also match. Warmups
and capture perform no optimizer updates. Weight, BatchNorm and RNG snapshots are
restored before verification, so setup does not add training steps or consume
the experiment's random sequence. The first gradient tape is initialized on a
side stream to avoid legacy-stream dependencies during capture.

CPU execution, short stages, untracked differentiable inputs and known unsafe
operations stay eager. A preflight rejects data-dependent shapes and CPU
synchronization such as `nonzero`, `item` and checked linear solves. Numerical
or state mismatches reject capture and reuse the original eager update. Parameter
storage or mode changes invalidate an accepted training graph.

A low-level CUDA capture failure may leave CUDA stream or generator state
unusable. The backend stops with an actionable error instead of continuing a
potentially corrupted trajectory; restart that process with `--no_cuda_graph`.
Sparse extension support depends on the installed Torch/PyG/CUDA combination.

GP kernels use a narrower, autograd-aware partial capture, with exact forward
and derivative checks. A fresh forward follows derivative verification because
native backward may overwrite shared forward workspace. The checked GP solve
remains eager. Changes to shape, stride, dtype, device or gradient contract
invalidate that kernel graph. These callbacks do not support double backward
and are never installed in gradient-matching or trajectory-matching paths.

```python
from graphslim.acceleration import acceleration_records

for stage in acceleration_records(args):
    print(stage["stage"], stage["status"], stage["reason"])
```

## GCGP arithmetic and historical results

Graph capture does not silently replace the historical GCGP row-norm arithmetic.
`gcgp_arithmetic="package"` remains the default. To use the released SGNK
operation order and tensor ridge division, select the separate arithmetic option:

```python
from graphslim.config import get_args

args = get_args(["-D", "cora", "-M", "gcgp", "-G", "0", "--cuda_graph"])
args.gcgp_arithmetic = "released"
```

The same override can be placed in a method JSON passed through `--config`.
This choice applies identically with graphs on or off and is recorded in reduced
graph metadata. Only detached original statistics from a fixed full training
batch are cached. Random minibatches do not reuse a full-graph diagonal cache.
Historical accuracy tables remain tied to their original source and arithmetic.

## Validation and timing scope

The integrated CUDA audit completed all 280 clean method/model pairs and 336
noise pairs (six corruptions, four lightweight method tags, seven models,
transductive and inductive). Loss histories, validation-selected parameter hashes
and validation/test scores matched exactly under the declared deterministic
Torch/CuBLAS setting. Six model architectures captured in every clean method tag;
GAT remained eager. The [machine-readable audit](https://github.com/Emory-Melody/GraphSlim/blob/main/benchmark/results/cuda_graph_integration.json)
retains source hashes, complete coverage and timing scope.

Capturing is not sufficient to make a short fit faster. In this 80-step,
48-node, 16-feature synthetic diagnostic, median setup-inclusive ratios were
0.49× for GCN, 0.35× SGC, 0.43× APPNP, 0.56× Cheby, 0.52× GraphSage and 0.95×
SGFormer. These are mostly slowdowns caused by setup/verification overhead in
small runs, not acceleration claims. Keep the backend opt-in and profile the
actual dataset, fitting budget and validation interval before enabling it.

Actual 200-step toy reduction pairs also retained reduced features, adjacency,
labels and real optimization loss histories. Including setup and preprocessing,
GCGP measured 1.87× for the historical arithmetic and 2.01× for released arithmetic;
ClustGDD measured 1.21× across pretraining and refinement. These are one-seed
integration measurements, not a Cora or whole-repository speed claim. The
[published audit record](https://github.com/Emory-Melody/GraphSlim/blob/main/benchmark/results/cuda_graph_integration.json) retains
these paired timings, tensor hashes and complete optimization loss histories.

The earlier Cora SGNK feasibility experiment retained exact 200-step trajectories
and downstream GCN scores. Its three Float32 reduction seeds measured 3.10–3.21×
warm-loop speedups, or 1.64–1.81× including graph/cache setup. Its one Float64
reduction seed measured 3.48× warm and 1.72× including setup. These are kernel
replacement experiments, not measurements of the new general training backend
or the complete original repository. See the [efficiency protocol](efficiency.md).

The integration verifier executes actual package fitting with explicit synthetic
graphs under every method tag and all seven downstream architectures. It compares
loss histories, validation-selected weights and validation/test scores. These
synthetic inputs are not outputs of the tagged reducers and their scores are not
scientific dataset results. Whole-fit timing includes setup, verification,
validation and diagnostic loss recording; retain slowdowns as well as speedups.
The parity audit enables deterministic Torch algorithms on both sides by default;
this is an explicit diagnostic setting, not a change to package/paper defaults.

```bash
python -m benchmark.verify_cuda_graph --device cpu --steps 20 --output runs/cuda_graph_cpu.json
python -m benchmark.verify_cuda_graph --device cuda:0 --steps 200 --output runs/cuda_graph_gpu.json
python -m benchmark.verify_cuda_graph --device cuda:0 --steps 200 --methods cgc clustgdd gcgp nope --noise clean gaussian label_noise random_adj --output runs/cuda_graph_noise.json
python -m benchmark.cuda_graph_queue --run-dir runs/cuda_graph_validation --gpus 3 4 5
```

The queue snapshots package, test and benchmark source, honors GPU UUID and
physical-device locks, and waits for idle devices without interrupting other
jobs. GPU unit tests must pass before the broad matrix starts. Its full roster
contains 280 clean method/model pairs and 336 noise pairs across transductive and
inductive settings. CPU parity and a passed generic GPU state test do not imply
that this complete GPU roster has finished.

Native constraints are described in [PyTorch CUDA semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html)
and [make graphed callables](https://docs.pytorch.org/docs/stable/generated/torch.cuda.make_graphed_callables.html).
Gradient-tape stream requirements are covered by
[NVIDIA capture troubleshooting](https://docs.nvidia.com/dl-cuda-graph/troubleshooting/capture-failures.html).

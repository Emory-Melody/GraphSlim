# Runtime compatibility

GraphSlim uses Torch/PyG for graph reduction and downstream training. Its
optimizations use standard PyTorch tensor operations and native CUDA Graphs.
DGL provides optional graph interchange and selected dataset loaders.

## GPU portability

The GPU backend targets NVIDIA A100 and newer GPUs through the same PyTorch
implementation. GraphSlim has no architecture-specific CUDA extension, SM120
dispatch branch or locally compiled kernel. PyTorch selects the device kernels;
install Torch and PyG extension wheels that support your GPU and CUDA runtime.

| Architecture | Example GPUs | Compute capability | GraphSlim backend |
| --- | --- | --- | --- |
| Ampere | A100 | 8.0 | Native PyTorch operations and CUDA Graphs |
| Ada | L40S, RTX 4090 | 8.9 | Same implementation |
| Hopper | H100, H200 | 9.0 | Same implementation |
| Blackwell | B200; RTX PRO 6000 Server Edition | 10.0; 12.0 | Same implementation |

The [NVIDIA device table](https://developer.nvidia.com/cuda/gpus) lists these
architecture identifiers. [PyTorch's CUDA Graph API](https://docs.pytorch.org/docs/2.8/notes/cuda.html#cuda-graphs)
provides capture and replay. The current Products timing figures use the
Blackwell device recorded below; A100 and Hopper timing records can be added
using the same experiment protocol.

On the selected GPU, check the installed operators and the acceleration backend:

```bash
python -m graphslim.doctor --device cuda:0 --json --output doctor.json
python -m pytest -q tests/test_acceleration.py
```

The acceleration tests exercise actual capture/replay, gradients, optimizer
updates, random state and kernel outputs on the visible GPU.

## Verified local profiles

The following profiles were checked on Linux with Python 3.11.15 and NumPy 1.26.4.
The CPU regression suite passed 260 tests in each of the Torch 2.6 and 2.8
environments after installing the existing optional `fuzzy` dependency for GECC.
CUDA checks exercised real operators, not merely `torch.cuda.is_available()`.

| Torch build | Device | PyG | Extension wheels | DGL | Verified scope |
| --- | --- | --- | --- | --- | --- |
| 2.6.0+cpu | CPU | 2.8.0.post1 | scatter 2.1.2+pt26cpu; sparse 0.6.18+pt26cpu | 1.1.3 | CPU suite; sparse/solve/scatter/GCN forward and backward; DGL CPU interchange |
| 2.8.0+cu128 | CPU | 2.8.0.post1 | scatter 2.1.2+pt28cu128; sparse 0.6.18+pt28cu128 | 1.1.3 | CPU suite and DGL CPU interchange inside a CUDA-enabled environment |
| 2.8.0+cu128 | RTX PRO 6000 Blackwell, SM120 | 2.8.0.post1 | same pt28cu128 wheels | 1.1.3 CPU | CUDA COO/CSR sparse operations, scatter, dense solve and PyG GCN forward/backward; DGL interchange remains a CPU test |

Native COO checks include feature and edge-value gradients. Native CSR and
torch-sparse checks include feature gradients; CSR edge-value gradients are not
certified. These operator probes do not certify the full method/noise roster on
GPU. DGL GPU training, GraphBolt, DGL 2.x, other GPU architectures and other CUDA
runtime versions are not covered by these local results.

CI is configured for Python 3.10/3.11 × Torch 2.6/2.8 CPU, with real DGL 1.1.3
interchange and the optional GECC dependency installed. Configuration alone is
not evidence that those remote jobs have passed. Local Python 3.10 validation
has not been substituted for the Python 3.11 results above.

## Check the installed environment

```bash
python -m graphslim.doctor --device cpu --dgl
python -m graphslim.doctor --device cuda:0 --dgl --json --output doctor.json
```

`graphslim-doctor` is the equivalent installed console command. The diagnostic
prints package versions, the selected runtime and GPU architecture, actual
forward/backward results and optional DGL interchange status. It never installs
or upgrades dependencies and does not silently replace a requested CUDA device
with CPU. Nonzero exit means a requested check failed. The JSON contains source
hashes and can be retained alongside an experiment manifest.

CUDA indices refer to devices visible through `CUDA_VISIBLE_DEVICES`; physical
GPU selection/leases belong to the caller or experiment scheduler. On a shared
machine, run probes only on an authorized idle device. CPU is the default.

## Installation profiles

Install Torch first, then extension wheels matching **that Torch build and its
CUDA runtime**, then GraphSlim. Do not use the maximum CUDA version reported by
`nvidia-smi` as the extension-wheel selector. See the primary
[PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)
and [Torch version-specific wheels](https://pytorch.org/get-started/previous-versions/).

Tested CUDA profile:

```bash
uv pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install torch-scatter torch-sparse --no-deps --no-index -f https://data.pyg.org/whl/torch-2.8.0+cu128.html
uv pip install -e '.[dev,docs,fuzzy]' --constraint <(printf 'torch==2.8.0\ntorchvision==0.23.0\n')
uv pip install 'dgl==1.1.3' 'numpy==1.26.4'
python -m graphslim.doctor --device cuda:0 --dgl
```

These commands use Bash process substitution. For CPU, use the existing README
CPU recipe with Torch 2.6.0 and `torch-2.6.0+cpu.html`. The DGL 1.1.3/NumPy 1.26.4
pins describe a verified optional profile, not a claim that all modern DGL builds
must be downgraded. Select other DGL wheels using the
[DGL installation guide](https://www.dgl.ai/pages/start.html), then rerun the probe.
GraphBolt/Torch binary compatibility and DGL's GPU backend must be verified
separately; successfully importing the CPU interchange path does not prove them.

Existing PyG extension wheels are still required by parts of GraphSlim. There
is no additional GraphSlim-specific CUDA extension or `nvcc` requirement.

## Graph interchange guarantees

`graphslim.compat.to_pyg_data` and `to_dgl_graph` preserve homogeneous graph node
features, labels, train/validation/test masks, scalar edge weights, edge
attributes and isolated-node counts. Reduced dense, COO, CSR, SciPy and
torch-sparse adjacency paths retain their actual weights. A reduced graph never
inherits masks from the larger original graph. Reset synthetic fields set to
`None` do not hide original features/labels.

Homogeneous interchange rejects a typed DGL graph instead of silently flattening
it. `dataset.convertor.from_dgl(..., hetero=True)` preserves typed node stores,
actual edge attribute tensors and isolated node counts as PyG `HeteroData`.
This is an I/O capability, not heterogeneous training support for every reducer.
Flattening graphs with multiple node types requires an explicit ID mapping and
is rejected. The legacy single-node-type dataset flattening path retains its
existing benchmark split behavior; these changes do not resample the historical
Yelp/Amazon experiments.

```python
from graphslim.compat import to_dgl_graph, to_pyg_data
restored = to_pyg_data(to_dgl_graph(pyg_data))
assert restored.num_nodes == pyg_data.num_nodes
```

Machine-readable local probe records are in
`runs/efficiency_compat_20261011/doctor_*_final.json`. They report DGL CPU
interchange separately from the Torch/PyG CUDA checks.

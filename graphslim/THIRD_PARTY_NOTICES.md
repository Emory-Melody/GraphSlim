# Third-party notices

CGC is independently implemented from its published algorithm.
The package does not redistribute their author source files; the verification
script reads separately cloned, pinned references supplied by the user.

ClustGDD algorithm components were adapted from
https://github.com/HKBU-LAGAS/ClustGDD,
commit `60825c536f794e7a3bb0ff22ae4efc07ea2d3e3a`.

GCGP kernel/GP algorithm components were adapted from
https://github.com/WANGLin0126/GCGP,
commit `d81ed427cd52d5a0d2a74bdf4328d31db1dcc1eb`.

Both releases use the following MIT license, with these respective notices:

Copyright (c) 2025 LAGAS Group, HKBU

Copyright (c) 2025 WANG Lin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Additional bundled adaptations

The GraphSlim project license is MIT. Adapted upstream components retain their
upstream terms; the complete license texts below are distributed in both source
and wheel archives. Importing a dependency does not make its entire source part
of GraphSlim.

| Component / source location | Upstream and license evidence | Retained terms |
| --- | --- | --- |
| Coarsening utilities, including the matching routine (`coarsening/utils.py`) and associated coarsening implementations | [graph-coarsening at 1d7aad02](https://github.com/loukasa/graph-coarsening/tree/1d7aad02ef294a37e79e009d52f215b70d3ef1e8) | [Apache-2.0](licenses/graph-coarsening-Apache-2.0.txt); modified and integrated into GraphSlim. The upstream matching module documents Galil/Edmonds/Gabow and its implementation history. |
| PyGSP-derived reduction utilities in `coarsening/utils.py` | [PyGSP](https://github.com/epfl-lts2/pygsp) | Copyright (c) 2014, EPFL LTS2; [BSD-3-Clause](licenses/PyGSP-BSD-3-Clause.txt). |
| PyG-derived GAT wrapper and graph-attention layer (`models/gat.py`, `models/layers.py`) | [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric) | Copyright (c) 2023 PyG Team; [MIT](licenses/PyG-MIT.txt). |
| pygcn-derived graph-convolution / linear layers (`models/layers.py`) | [pygcn at 1600b5b7](https://github.com/tkipf/pygcn/tree/1600b5b748b3976413d1e307540ccc62605b4d6d) | Copyright (c) 2017 Thomas Kipf; [MIT](licenses/pygcn-MIT.txt). |
| Modified PRBCD attack (`models/prbcd.py`) | [robustness_of_gnns_at_scale at d28a9a20](https://github.com/sigeisler/robustness_of_gnns_at_scale/tree/d28a9a20794d221bb25657db2c7caa2b53d1032f) | [MIT](licenses/PRBCD-MIT.txt), retaining the upstream copyright line verbatim. |
| DeepRobust-derived graph utilities and attack components | [DeepRobust](https://github.com/DSE-MSU/DeepRobust) | Copyright (c) 2020 Yaxin Li, Wei Jin, Han Xu and Jiliang Tang; [MIT](licenses/DeepRobust-MIT.txt). |

This inventory records the explicit source attributions checked during packaging
on 2026-11-05. The cited upstream revisions identify license evidence, not a claim
that every adaptation was originally copied from that revision. It is not an
exhaustive historical authorship audit of every algorithm implementation.

## Per-file license identifiers

Every Python module and shell script distributed under `graphslim/` states its
license in a source header. GraphSlim contributions use MIT. The graph-coarsening
adaptations additionally retain Apache-2.0 terms, and the PyGSP-derived portions
of `coarsening/utils.py` additionally retain BSD-3-Clause terms. Combined SPDX
expressions identify these retained terms; the upstream texts above remain in
source and wheel distributions. Headers do not replace the original notices.

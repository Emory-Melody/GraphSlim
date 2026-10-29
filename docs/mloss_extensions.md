# GraphSlim software extensions for the MLOSS description

This is a local development document prepared on 2026-10-11 (America/New_York).
It describes the working tree used for `mloss_graphslim_draft.tex`, not a newly
published release. The package metadata still says `1.2.1`; that number does not
identify the uncommitted changes described here. A public reviewed revision must
be selected before submission.

## Relation to prior publication

[GC4NC (NeurIPS 2025)](https://proceedings.neurips.cc/paper_files/paper/2025/hash/49caeae02008a3d2c8fc7f178b389c19-Abstract-Datasets_and_Benchmarks_Track.html)
already presents the benchmark and links the GraphSlim repository. Its
[version 3 software appendix](https://arxiv.org/html/2406.16715v3#A1.SS12)
describes PyG input, reduction, evaluation, and extension to new data and methods.
Those capabilities are prior work. The package paper should explain what users
can now do with the software and distinguish subsequent engineering and method
adapters from the original benchmark results.

The [MLOSS instructions](https://jmlr.org/mloss/mloss-info.html) request a
repository document explaining improvements when earlier peer reviewed work
already focused on the software. This document supplies that disclosure for the
local draft; it will need to be included in the reviewed public revision and
linked from the cover letter.

## Inspectable changes

| Development capability | Evidence in this working tree | Boundary |
| --- | --- | --- |
| Consistent command, example, and benchmark entry points | `graphslim/train_all.py`, `examples/train_*.py`, `benchmark/train_all.py`, `tests/test_entrypoints.py`, `tests/test_pipeline.py` | A shared interface does not guarantee equal scientific settings. |
| Explicit configuration precedence and short-run handling | `graphslim/config.py`, `tests/test_config.py` | Parameters must be resolved before constructing the dataset and reducer. |
| Four additional method adapters | `graphslim/condensation/{cgc,clustgdd,gcgp}.py`, `graphslim/coarsening/nope.py`, `graphslim/reduction/registry.py` | These integrate existing algorithms; they are not new algorithms proposed by the package paper. |
| Method-specific numerical and data contracts | [Lightweight method documentation](lightweight_methods.md), [SimGC documentation](simgc.md) | GCGP targets, NOPE budgets, feature preprocessing, and author-code differences are explicit. |
| Checkpoint and held-out selection checks | `tests/test_saved_graphs.py`, `tests/test_lightweight_methods.py`, `tests/test_paper_hpo.py` | Bounded regression tests do not validate every method/dataset combination. |
| Experiment planning and result provenance | `benchmark/reproduce.py`, `benchmark/results/provenance.json`, `benchmark/package_results.py` | Planning is distinct from executing or scientifically reproducing an experiment. |
| Real-data integration records for the added methods | `benchmark/results/lightweight_integration.json`, `.csv`; source run `runs/lightweight_cora_20261010_v5` | Historical 20-step smoke profile on Cora: 32 completed reductions and 224 downstream fits; no paper-performance claim. |
| Package and documentation checks defined in CI | `.github/workflows/tests.yml`, `docs/installation.md`, `docs/quickstart.md`, `pyproject.toml` | Ubuntu CPU jobs for Python 3.10/3.11 are defined locally; successful public CI is not asserted. |

The method registry already existed in the checked-in parent. Its mere existence
is not claimed as a new contribution. The present changes extend its roster and
route surrounding entry points through the shared training workflow. A historical
diff against the exact GC4NC release is still needed to establish which changes
postdate that publication.

## Verification and remaining evidence

On 2026-10-11, the following existing tests passed in the local Python 3.11 CPU
environment: `test_config`, `test_reduction_registry`, `test_pipeline`,
`test_saved_graphs`, `test_entrypoints`, and `test_lightweight_methods`.
There were 148 passing tests and one PyTorch sparse-CSR beta warning. This was a
focused verification, not a whole-repository coverage measurement.

The registry has 12 sparsification, 11 coarsening, and 17 condensation entries,
excluding aliases. The historical integration manifest uses source SHA-256
`ffc655aea569996bd8511033a5f1c6c8f268053037cdb16a56a2f55b26cd87d0`.
Its source snapshot is distinct from the current working tree.

Before submission, identify the exact public version, compare its changes against
the GC4NC source release, measure test coverage, verify the advertised platform
matrix, and document a comparison with relevant implementations. Community
evidence belongs in the cover letter. No adoption, speedup, memory reduction, or
complete benchmark reproduction is inferred from this document.

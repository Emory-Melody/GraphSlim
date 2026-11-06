# GraphSlim 1.3.2 release verification

This record belongs to Git release/tag `1.3.2`. `environment.json` identifies the
Python/PyTorch environment and confirms 49 canonical methods and eight backbones.
`pytest.log`, `pytest.xml`, `coverage.json` and `coverage.xml` contain the fresh CPU
run: 1,589 passed, 31 skipped, zero failures/errors; 98.79% statement coverage
(10,405/10,532) and 96.06% branch coverage (3,222/3,354) across 105 modules, with
no excluded statements. The 31 skips are CUDA-only checks and optional pinned
reference checkouts.

`archive_check.json` lists the wheel and sdist built from the release tree: 105
package modules, each with an SPDX identifier, no Mirage module and no stray files.
`twine_check.log`, `docs_build.log`, `wheel_import.json`, `wheel_doctor.json` and
`artifact_check.log` record the packaging, strict documentation, unpacked-wheel and
results-artifact checks; `checks.json` summarizes them.

The release comparison artifact is `benchmark/results/reproducibility_20261106`. It
embeds the unchanged 1.3.1 artifact, excludes CentP and GDEM, and substitutes 15
inductive K-Center and Herding settings rerun after a fix (see `CHANGELOG.md`).

These are local CPU checks; they do not verify GPU paths. CI results are identified
separately.

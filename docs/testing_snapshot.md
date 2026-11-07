# Version-specific software testing record

GraphSlim 1.3.5 retains 49 canonical methods and eight downstream GNN architectures.
It uses anomaly-class AP for Yelp/Amazon validation, HPO and final evaluation.
The frozen 110-pair comparison artifact retains its historical metric definitions.

## Current release: 1.3.5

Release 1.3.5 removes the retired GCondX and NOPE implementations, nine
configuration files, obsolete tests and active benchmark references.
The 49-method registry and eight downstream GNN architectures are unchanged.
A fresh Linux CPU run passes **1,591 tests**, with **22 CUDA checks skipped**,
zero failures/errors, **98.77% statement coverage** and **95.97% branch coverage**
across all **102 package modules**, with zero excluded statements.
Statements: 10,039 / 10,164;
branches: 3,121 / 3,252.
The [release record](https://github.com/Emory-Melody/GraphSlim/tree/1.3.5/validation/release_1_3_5)
contains the source/configuration hashes, complete test log, coverage reports
and distribution checks. Regression tests reject retired registry entries,
public exports, importable modules and configuration directories.

Historical result tables retain their original version, source hashes and
numerical records; they are not claimed as reruns of 1.3.5. The full CPU/CUDA
demo was rerun on the 1.3.5 tag after release; its records are in
[validation/demo_1_3_5_full_20261107](https://github.com/Emory-Melody/GraphSlim/tree/main/validation/demo_1_3_5_full_20261107),
and the earlier 1.3.4 records remain in `validation/demo_1_3_4_full_20261106`. To reproduce the current CPU checks, use the
[versioned installation](installation.md) and the coverage commands below.

## Release 1.3.4

Release 1.3.4 freezes the final README selection policy and extends per-file
licensing to auxiliary sources. Production Python code is unchanged from 1.3.3.
A fresh CPU run and archive checks are recorded in
[validation/release_1_3_4](https://github.com/Emory-Melody/GraphSlim/tree/1.3.4/validation/release_1_3_4).
The run passes **1,601 tests**, with **22 CUDA checks skipped**, and measures
**98.79% statement coverage** and **96.04% branch coverage** across 104 modules.
[License-only provenance](https://github.com/Emory-Melody/GraphSlim/blob/1.3.4/validation/release_1_3_4/license_changes.json)
records the verifier comment changes; measured result records are unchanged.

Use the versioned setup, then run the commands below. The original 1.3.3
record remains pinned to its own release.

## Release 1.3.3

A fresh CPU test and coverage record is stored in
[validation/release_1_3_3](https://github.com/Emory-Melody/GraphSlim/tree/1.3.3/validation/release_1_3_3).
The fresh Linux CPU run reports **1,601 passed, 22 skipped**, zero failures/errors,
**98.79% statement coverage** and **96.01% branch coverage** across all 104
package modules. Historical counts below describe their own source revisions.

Use the [versioned CPU setup](installation.md), then run:

```bash
python -m pip install -e '.[dev,fuzzy,dgl]' -c constraints/cpu-torch26.txt 'coverage==7.16.2'
python -m coverage run -m pytest -q -ra --junitxml=pytest.xml
python -m coverage combine
python -m coverage json -o coverage.json
python benchmark/results/reproducibility_20261106/verify.py --check
```

New regression checks cover incompatible HPO resumes and fraud AP ranking and
configuration. Small synthetic CPU checks verify reducer behavior and training;
they are separate from complete benchmark accuracy and GPU validation.

## Release 1.3.2

On Linux CPU with Python 3.11 and PyTorch 2.6.0+cpu, the repository suite reports
**1,589 passed, 31 skipped, zero failures/errors**. Skips are CUDA-only checks and
unavailable optional pinned reference checkouts. The full-package coverage run
includes 105 Python modules, with zero excluded statements:

| Measure | Covered / total | Percentage |
| --- | ---: | ---: |
| Statements | 10,405 / 10,532 | 98.79% |
| Branches | 3,222 / 3,354 | 96.06% |

Tests use small synthetic graphs on CPU and need no dataset downloads. They check
method behaviour (selection budgets, kernel symmetry, loss decrease, save and reload,
reference identities such as PageRank against NetworkX) rather than only calling
code. The defects they exposed are fixed in this release; see the changelog. The
remaining uncovered lines are mostly CUDA-only branches and defensive paths that
cannot be reached.

To reproduce, install the supported CPU dependencies from the README, then run:

```bash
python -m pip install -e '.[dev,fuzzy]' 'dgl==1.1.3' 'coverage==7.16.2'
python -m coverage run -m pytest -q -ra --junitxml=pytest.xml
python -m coverage combine
python -m coverage json -o coverage.json
python benchmark/results/reproducibility_20261106/verify.py --check
```

The checked-in `.coveragerc` measures all `graphslim` modules, enables branch
coverage, and combines multiprocessing/subprocess execution.

## Release 1.3.1

The 1.3.1 suite reported 479 passed and 31 skipped tests, 42.56% statement coverage
(4,522/10,625) and 34.00% branch coverage (1,145/3,368) across 106 modules. Its record
is in [validation/release_1_3_1](https://github.com/Emory-Melody/GraphSlim/blob/1.3.1/validation/release_1_3_1/README.md).

## Earlier development record

The 2026-10-31 development snapshot had 843 passed and 52 skipped tests,
9,378/10,397 covered statements (90.20%), and 2,622/3,266 covered branches
(80.28%) across 102 modules. It was a frozen working-tree snapshot, rather than
the clean Git revision `33dc43bcb6e1e413b9c8b5273cebd86b57ce43ee`.

Hash comparison against the source on which this cleanup is based found eight
identical modules, 93 changed modules, one removed module, and five added modules.
The later license-comment additions also change source hashes. The historical
test suite and production code are therefore not interchangeable with the
reviewed tree. The earlier percentages remain historical evidence only; they
must not be presented as coverage of the reviewed Git revision.

The source-only cleanup adds license headers without changing any package
module AST. This 1.3.1 record reruns the actual release suite rather than carrying
forward the historical coverage claim.

See the earlier historical summary and module comparison under
`validation/20261105`. They document the discrepancy; they do not assert that
the historic 843-test suite was rerun on this revision.

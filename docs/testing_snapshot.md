# Version-specific software testing record

The Git 1.3.2 release has **49 canonical methods** (5 node,
18 edge, 10 coarsening, 16 condensation) and **eight downstream GNN architectures**.
The environment, raw test log, JUnit results, full coverage data and packaging
checks are in [validation/release_1_3_2](https://github.com/Emory-Melody/GraphSlim/blob/1.3.2/validation/release_1_3_2/README.md).
These files are committed with the implementation they describe.

## Current reviewed release: 1.3.2

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

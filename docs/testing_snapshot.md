# Version-specific software testing record

The 2026-11-05 reviewed implementation has **49 canonical methods** (5 node,
18 edge, 10 coarsening, 16 condensation) and **eight downstream GNN architectures**.
The source fingerprint, environment, raw test log, JUnit results, and full
coverage data are in [validation/20261105](https://github.com/Emory-Melody/GraphSlim/blob/1a024c6bf2d3a73e40a3d1facb6dd6b9c1328bab/validation/20261105/README.md).
These files are committed with the implementation they describe.

## Current reviewed implementation

On Linux CPU with Python 3.11 and PyTorch 2.6.0+cpu, the repository suite reports
**479 passed, 31 skipped, zero failures/errors**. Skips include CUDA-only checks
and unavailable optional pinned reference checkouts. The full-package coverage
run includes 106 Python modules, with zero excluded statements:

| Measure | Covered / total | Percentage |
| --- | ---: | ---: |
| Statements | 4,522 / 10,625 | 42.56% |
| Branches | 1,145 / 3,368 | 34.00% |
| Combined coverage.py measure | 5,667 / 13,993 | 40.50% |

These tests establish the exercised software behavior in this environment;
they do not replace the numerical benchmark artifact or verify every GPU path.
No claim of 90% coverage is made for this revision. CI produces separate
coverage artifacts for each tested Python/PyTorch combination; local results
and remote CI results must be identified separately.

To reproduce, install the supported CPU dependencies from the README, then run:

```bash
python -m pip install -e '.[dev,fuzzy]' 'dgl==1.1.3' 'coverage==7.16.2'
python -m coverage run -m pytest -q -ra --junitxml=pytest.xml
python -m coverage combine
python -m coverage json -o coverage.json
python -m coverage xml -o coverage.xml
python benchmark/results/reproducibility_20261103/verify.py --check
```

The checked-in `.coveragerc` measures all `graphslim` modules, enables branch
coverage, and combines multiprocessing/subprocess execution. The record contains
exact dependency versions. Optional reference availability and CUDA availability
affect skips; keep those conditions explicit when comparing runs.

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

See the archived historical summary and module comparison under
`validation/20261105`. They document the discrepancy; they do not assert that
the historic 843-test suite was rerun on this revision.

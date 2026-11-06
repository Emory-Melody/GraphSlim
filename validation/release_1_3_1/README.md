# GraphSlim 1.3.1 release verification

This record belongs to Git release/tag `1.3.1`. `environment.json` identifies the
Python/PyTorch environment and confirms 49 canonical methods and eight backbones.
`pytest.log`, `pytest.xml`, `coverage.json`, and `coverage.xml` contain the fresh CPU
run: 479 passed, 31 skipped, zero failures/errors; 42.56% statement coverage and
34.00% branch coverage across 106 modules, with no excluded statements.

`license_header_audit.json` checks all 106 Python modules and 24 shell scripts.
All package Python ASTs match the pre-cleanup implementation. SPDX identifiers
retain the bundled Apache/BSD/MIT terms. The old manuscript draft is removed.
`source_manifest.json` fingerprints the tested package, configuration and tests.
`checks.json` records local packaging and documentation checks.

The release comparison artifact is `benchmark/results/reproducibility_20261105`.
Its numerical settings, pair results and counts match the earlier immutable
archive. Only correction status and public documentation were updated.

Installation checks reuse existing runtime dependencies in a fresh GraphSlim
environment; they do not claim a clean dependency resolution or GPU verification.
CI results and local results are identified separately.

# GraphSlim 1.3.4 release verification

The reviewed source tag is `1.3.4`. This is a source release; PyPI remains at 1.2.1.

- Fresh Linux CPU suite: 1,601 passed, 22 CUDA checks skipped, zero failures or errors.
- Statements: 10,211 / 10,336 (98.79%); branches: 3,175 / 3,306 (96.04%). All 104 package Python modules are measured, with zero excluded statements.
- HTML, EPUB and LaTeX documentation builds pass with warnings treated as errors.
- Wheel and source archive build and Twine checks pass. All 271 Python/shell source files in the sdist contain a licensing notice; 104 wheel Python modules match the checked source.
- All 104 production Python modules are byte-identical to release 1.3.3. The result-table blocks are byte-identical to commit `19ee5ff` and retain the requested selection policy.
- Historical numerical CSV/JSON records and measured source ZIP archives are unchanged. `license_changes.json` records the four verifier comment additions and corresponding manifest changes; original tags preserve the original archives.
- `scope_check.json` documents the boundaries; renderer validation refreshes source guards without altering numerical data or figure assets.
- `demo_output.txt`, `demo_summary.json` and `demo_provenance.json` record a fresh short CPU example (40.00% +/- 2.79%, three GAT fits). The source hashes and four-thread limits make its settings explicit.
- The separately labelled `validation/demo_records` preserves a 2026-11-01 historical GPU example. It is not 1.3.4 GPU validation.

Reproduce the test and coverage run with the versioned CPU installation recipe in `docs/installation.md`, then:

```bash
python -m coverage run -m pytest -q -ra --junitxml=pytest.xml
python -m coverage combine
python -m coverage json -o coverage.json
python benchmark/results/reproducibility_20261106/verify.py --check
```

No new complete GPU validation was performed. Funding and conflicts-of-interest disclosures remain author-confirmation items outside this software check.

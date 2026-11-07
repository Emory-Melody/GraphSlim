# GraphSlim 1.3.5 release verification

The reviewed source tag is `1.3.5`. PyPI remains at 1.2.1.

- Remove the GCondX and NOPE implementations and nine configuration files from the active source and built distributions. Both remain absent from the 49-method canonical registry; DosCondX remains supported.
- Fresh Linux CPU suite: **1,591 passed, 22 CUDA checks skipped**, zero failures/errors. Removing retired-method tests and adding two absence checks changes the test count from 1.3.4.
- Statements: **10,039 / 10,164 (98.77%)**; branches: **3,121 / 3,252 (95.97%)**. All **102** remaining package Python modules are measured, with zero excluded statements.
- `source_hashes.json` identifies the tested package/configuration files; `scope_check.json` verifies their hashes and the exact runtime deletion scope. The only changed surviving runtime module removes NOPE's dataset special case and dead comments.
- The complete test log, XML report, coverage JSON/XML and environment are included. HTML, EPUB and LaTeX documentation builds, lint, formatting and renderer checks pass.
- `archive_check.json` records source/wheel content, licensing and retired-file absence checks. The installed wheel's CPU/DGL doctor record is saved separately in the submission packet.
- New Table 9 job plans exclude GCondX; historical rendering still recognizes archived labels. Numerical result tables, figures, archived inputs and the requested README selection policy are unchanged.
- The complete CPU/CUDA demo in `validation/demo_1_3_4_full_20261106` retains its original 1.3.4 version, source/input hashes, raw logs, ten-fit scores and saved graphs. Its standalone verifier passes. These records are not relabelled as 1.3.5 executions.

Reproduce with the versioned CPU installation recipe, then:

```bash
python -m coverage run -m pytest -q -ra --junitxml=pytest.xml
python -m coverage combine
python -m coverage json -o coverage.json
python benchmark/results/reproducibility_20261106/verify.py --check
python validation/demo_1_3_4_full_20261106/verify.py
```

Funding and conflicts-of-interest fields remain pending author confirmation outside this software cleanup.

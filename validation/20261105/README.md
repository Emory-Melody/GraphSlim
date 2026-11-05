# Software verification, 2026-11-05

[Testing record](../../docs/testing_snapshot.md) defines the current and historical scopes.
`pytest.log`, `pytest.xml`, `coverage.json`, and `coverage.xml` record the current CPU run.
`source_manifest.json` pins the tested implementation, configuration, tests, and benchmark scripts by SHA-256.
`environment.json` records the runtime. `checks.json` records packaging, installation, registry, and artifact verification.
The source fingerprint excludes documentation and report files to avoid self-reference.

The historical coverage summary and module comparison are evidence about an older snapshot only.
The 843-test/90.20% statement/80.28% branch result does not describe the current tree.

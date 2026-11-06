# GraphSlim 1.3.3 verification

Fresh Linux CPU checks: **1,601 passed, 22 skipped**, no failures or errors.
Statement coverage: **10,211/10,336 (98.79%)**;
branch coverage: **3,174/3,306 (96.01%)**,
across all 104 package modules, with 0 excluded statements.

`environment.json`, `pytest.log`, `pytest.xml`, `coverage.json` and `coverage.xml`
record the exact environment, skips and coverage. New checks reject incompatible
HPO resumes and verify anomaly-class AP across all eight backbones in both graph
settings. `custom_reducer.log` records the runnable extension example.

`yelp_ap_verification.json` records checks of 55 restored AP cells, 228 validation
candidates (684 fits), and 730 final AP fits against their raw historical records.
The restored JSON differs from a fresh arithmetic reconstruction by at most
1.12e-16 in a standard deviation; the original frozen values are preserved.

The 110-pair reference artifact and all scores and metric definitions are unchanged.
Current Yelp/Amazon runs use AP for validation, HPO and final testing. These checks
do not establish full-dataset reproduction or GPU-path coverage. CI results are
identified separately for the exact commit.

# Validation-based hyperparameter search

HPO searches reduction settings and selects the highest mean validation score
over repeated downstream fits. Classification datasets use accuracy; Yelp and
Amazon Fraud use anomaly-class **average precision (AP)**, with fraud label 1 and
continuous positive-class log-odds. The same metric selects downstream epochs.
Test scores are not computed during HPO.

```bash
python -m graphslim.train_all -D cora -M gdem -R 0.5 -G 0 \
  --hpo --save_path ./checkpoints
```

`--hpo_space space.json` supplies a JSON parameter-to-list mapping. Otherwise a
bundled grid is used, or an editable one-point template is written. Each Cartesian
product combination is one trial. `--hpo_eval_runs` defaults to three;
`--eval_model` selects the downstream architecture, with GCN as the default.
`--hpo_yes` skips the interactive start prompt.

Search records are stored under `save_path/hpo/<method>/<dataset>_r<rate>/`:

- `search_identity.json`: grid, scientific configuration, seeds, model, graph and
  split hashes, package source hash and dependency versions.
- `trials.jsonl`: append-only completed or failed trial records.
- `best.json` and `summary.json`: selected parameters and validation results.

Resuming verifies the identity and recorded parameter dictionaries before reusing
any scores. Changed grids, configurations, seeds, models, graph contents, source
or dependencies require a new `--save_path`. Legacy histories without an identity
are rejected. Failed trials remain recorded; they do not enter winner selection.
Ties among complete trials use the earliest trial.

```bash
python -m graphslim.train_all -D cora -M gdem -R 0.5 -G 0 \
  --config ./checkpoints/hpo/gdem/cora_r0.5/best.json \
  --save_path ./checkpoints
```

From Python, call `graphslim.hpo.run_search(args, graph)`. Its successful return
record contains `params`, `validation` and `validation_mean`.

## Current metrics and historical records

Current Yelp/Amazon runs use AP throughout validation, HPO and final evaluation.
Archived macro-F1 measurements retain their original metric labels. The fixed
110-pair reproduction artifact combines documented historical macro-F1 and AP
comparisons; changing the current default does not convert those archived scores.
The README's separate Yelp AP table links its verified saved-graph evaluation
snapshot and keeps unavailable AP results as NR.

# Reduction-method search spaces

`python -m graphslim.train_all -D <dataset> -M <method> --hpo` loads `<method>.json` from this
directory when it exists. Each key is a setting of the method (the same names as in
`graphslim/configs/<method>/`), each value the list of values to try; every combination is one trial.
Keys starting with `_` are comments. Identity, device and seed settings cannot be searched.

Without a shipped file, the first `--hpo` run writes a one-point template to
`<save_path>/hpo/<method>/<dataset>_r<rate>/space.json`; widen it and rerun. `--hpo_space path.json`
points at any file. Trials are selected by mean validation accuracy only and resume from
`trials.jsonl`; the winner is written to `best.json` for `--config`.

# GC4NC benchmark reproduction

The [ordered dataset expansion](expand_datasets.py) adds repeated-seed
lightweight-method experiments on the original seven datasets, followed by
Roman-empire/Amazon-ratings, ogbn-products, and Amazon Photo/Computers.
Use `--datasets photo computers` for a separate co-purchase batch. Frozen-source new measurements,
resource failures and GCGP numerical-parity checks remain separate from published
results and historical local batches.

Source: [GC4NC, arXiv:2406.16715v3](https://arxiv.org/abs/2406.16715).
The main [README](../README.md#reproduced-results) presents reproduced results.
The full published table remains in the [frozen CSV](results/gc4nc_table9.csv);
[reduction-rate definitions](../README.md#reduction-rates) cover all dataset entries.

Run commands from the repository root after installing GraphSlim:

```bash
python -m benchmark.reproduce performance --dry-run
python -m benchmark.reproduce performance --datasets cora --methods clustering gcond --dry-run
python -m benchmark.reproduce performance --datasets cora --methods clustering --skip-whole -- -G -1
```

For paired kernel and fixed-budget replay timing, use
`python -m benchmark.benchmark_kernels --device cpu --output runs/sgnk.json`.
The [historical efficiency records](../efficiency.md) distinguish these measurements
from end-to-end author-code reproduction and retain accuracy/precision failures.
Check the environment first with `python -m graphslim.doctor --device cpu --dgl`;
see the [GPU setup and checks](../README.md#gpu-setup-and-checks).

The planner imports only standard-library modules and the method registry.
Without `--dry-run`, jobs run serially and stop on failure. Extra GraphSlim flags
follow `--`. Use `--skip-whole` to omit whole-graph baselines.

| Paper v3 experiment | Command / status |
| --- | --- |
| Tables 1 and 9: performance | `python -m benchmark.reproduce performance` |
| Scalability | `python -m benchmark.reproduce scalability` |
| Initialization | `python -m benchmark.reproduce data_initialization` |
| Transferability | `python -m benchmark.reproduce transferability` |
| Table 4: NAS | `python -m benchmark.reproduce nas` |
| Tables 5, 15, 16: graph properties | `python -m benchmark.reproduce graph_property_preservation` |
| Tables 3 and 18: denoising / robustness | `python -m benchmark.reproduce robustness` |
| Table 2: membership inference | Evaluator exposes `MIA_evaluate`; no dedicated reproduction suite is provided |
| Table 10 / 11: tuned hyperparameters | `python -m benchmark.paper_hpo` provides a paper-sourced GECC/SimGC local extension; see the protocol below |

The shell files in `benchmark/scripts/` are thin wrappers around the same planner.
Python files such as `benchmark/train_all.py` remain compatibility entry points.

GEOM and SFGC paper performance was obtained from the authors' implementations.
This package's expert-buffer defaults differ. GDEM has hardware and precomputed
eigendecomposition requirements. The reproduction planner specifies commands;
successfully planning them does not establish scientific reproduction.

Transferability, NAS, and graph-property evaluation need saved reduced graphs.
The performance matrix includes all seven datasets and clustering. Table 9 has no
clustering, DosCondX, GCDM, SimGC, or GECC scores. The README distinguishes archived package measurements, literature references, and the separate verified Yelp AP evaluations.

## Lightweight methods and complete downstream integration checks

CGC, ClustGDD and GCGP use the standard reducer/checkpoint/evaluator API.
The planner accepts their canonical names in every applicable suite:

```bash
python -m examples.train_lightweight -D cora -M cgc -G -1 --save_path ./checkpoints
python -m benchmark.reproduce robustness --datasets cora --methods cgc clustgdd gcgp --dry-run
python -m benchmark.lightweight_downstream --run-dir runs/lightweight_NEW --dry-run
python -m benchmark.lightweight_downstream --run-dir runs/lightweight_NEW --profile smoke
python -m benchmark.lightweight_downstream --run-dir runs/lightweight_full_NEW --profile full --conditions clean
```

The integration runner follows GC4NC's seven-model experiment protocol:
three methods × eight training conditions ×
seven downstream models: GCN, SGC, APPNP, Cheby, GraphSage, GAT and SGFormer.
The experiment protocol and the package's supported-architecture count are separate
specifications. Historical integration batches retain their recorded GC4NC rosters.
Conditions include clean, edge addition/deletion, column/entry feature masking,
Gaussian feature noise, training-label corruption and the legacy `metattack`
entrypoint (actually PRBCD). Clean jobs also evaluate five inference corruptions
without retraining, two-candidate original/synthetic NAS, graph properties and
oracle-threshold MIA diagnostics. Unit tests additionally cover both transductive
and inductive splits, validation-only selection, checkpoint reloads and variants.

Every batch freezes its scientific source, dataset/split identities, reduction
and evaluation seeds, checkpoints and raw records. A table row becomes numeric
only when its full architecture/seed roster is present. `results.md` contains the
complete matrix, with raw data in `scores.csv` and `jobs/*/result.json`. Smoke is
20-step integration validation, not paper accuracy. Full uses reduction defaults
and 300 downstream steps, not a paper-specific HPO/repeated benchmark. Use a new
run directory whenever source or protocol changes.

See the [integration runner](lightweight_downstream.py) and [regression tests](../tests/test_lightweight_methods.py) for
costs, soft-target handling, reduction budgets,
undefined property metrics, perturbation contracts and algorithmic adaptations.
The [README integration table](../README.md#lightweight-methods-downstream-validation)
is separate from the published and repeated local performance tables.

## Resumable local NR reruns

Use the GPU environment and select only physical GPUs you are authorized to use:

```bash
.venv-gpu/bin/python -m benchmark.rerun_nr plan --run-dir runs/nr_20261009_v2
.venv-gpu/bin/python -m benchmark.rerun_nr run --run-dir runs/nr_20261009_v2 --gpus 0 1 2 3 4 5 --update-readme
.venv-gpu/bin/python -m benchmark.rerun_nr status --run-dir runs/nr_20261009_v2
```

The scheduler checks live GPU occupancy, takes both UUID and physical-index locks,
and never kills another process. Each experiment has an isolated checkpoint path,
console log, resolved configuration, exact dataset hashes, raw evaluation scores,
environment record and checkpoint hashes. Restart the same command to resume
interrupted work; failed jobs retain their error record and are not silently retried.
Scientific source changes require a new run directory and source snapshot.

The default roster is five NR methods × seven datasets × three rates × three
reduction seeds (315 runs). Each reduction has ten downstream GCN fits; only cells
with all thirty scores are reported numerically, without ranking partial coverage.
GECC's missing Yelp configuration is recorded explicitly rather than borrowing
parameters without approval. Local results do not overwrite the frozen published
CSV. The local README section also discloses Reddit2 and FraudYelp split differences.

## Paper-sourced GECC / SimGC hyperparameter search

```bash
.venv-gpu/bin/python -m benchmark.paper_hpo plan --run-dir runs/paper_hpo_NEW
.venv-gpu/bin/python -m benchmark.paper_hpo run --run-dir runs/paper_hpo_NEW --gpus 0 1 2 3 4 5 --update-readme
.venv-gpu/bin/python -m benchmark.paper_hpo status --run-dir runs/paper_hpo_NEW
```

GECC–Yelp uses the complete grid from [GECC §5.1](https://arxiv.org/html/2502.17614):
three propagation weights in `{-0.3, -0.2, ..., 0.9}` and weight decay in
`{0.0005, 0.001}` (4,394 configurations). Yelp follows the paper's large-dataset
setting: hard k-means++, depth 2, one restart, 300 iterations and tolerance 1e-8;
fixed GCN hidden 256, depth 2, learning rate 0.01, dropout 0.5 and 300 epochs.
This is an extension: the GECC paper has no Yelp result or optimal configuration.

SimGC uses settings from [its paper §5.1](https://arxiv.org/html/2403.14951) and
[author code pinned at 1b3d1d5](https://github.com/BangHonor/SimGC/tree/1b3d1d59d9afaf2c9c0c7b652fb183db2a5ffa01).
The publication does not specify a condensation-parameter search grid. We do not
invent one: we repair the port (differentiable synthetic multi-hop alignment,
negative smoothness similarity, propagate-before-predict teacher and normalized
teacher validation) and use the author learning rates/epochs/feature weights.
Cora/Citeseer use the author's 2048-hidden nonlinear teacher; other datasets use
a linear two-hop teacher. Pubmed/Yelp were not in that paper and use common
transductive/inductive author defaults, explicitly not published optima.

Then, on each SimGC dataset, downstream GCN is searched over
[GC4NC Table 11](https://arxiv.org/html/2406.16715v3): hidden `{64,128,256}`,
learning rate `{0.01,0.001}`, layers `{2,3,4}`, weight decay `{0,0.0005,0.001}`
and dropout `{0,0.5}` (108 configurations). Evaluation uses `mode="cross"`
so legacy fixed-evaluation overrides cannot silently erase grid parameters.
Current Cora/Citeseer searches add the released hidden-1024 baseline (109
candidates each). They retain the author's no-activation GCN profile and
3,001-update downstream budget; graph selection separately uses hidden 2048,
3,000 updates and the released validation cadence. These are explicit local
search extensions, not GC4NC's original 300-epoch replay. See
[the implementation](../graphslim/condensation/simgc.py). Other datasets retain the existing
300-epoch local protocol.
Each training epoch explicitly restores training mode after validation, keeping
dropout active. GECC's inductive propagation now includes the requested second hop.

Both searches use the canonical seed-1 reduction at the highest CLI rate and
ten fixed validation fits (seeds 0–9); no test metric is computed during search
or intermediate condensation evaluation. Every candidate must finish before
selection is frozen. The selected configuration is reused for all three rates
and three reduction seeds. Final GCN weights are selected on validation and
tested once per fit. Each reported cell pools all 30 test scores and reports
population standard deviation. This seed-1 search protocol is a local extension,
not a claim of exact paper replay. Scientific repairs also mean the difference
from the old NR batch is not attributable to hyperparameter search alone.

New durable manifests contain 369 jobs: 63 SimGC reductions, 65 GCN-search chunks
(758 configurations), 169 GECC-search chunks (4,394 configurations), and
72 final evaluations. Each search chunk writes resumable per-candidate records.
The archived `paper_hpo_20261010_v2` retains its original 367-job roster and
pre-correction citation results; do not resume it under changed scientific code.
GPU UUID/index leases, source snapshots, fixed dataset hashes and complete-cell
publication follow the NR runner. Results appear in a separate README section;
the old local table and published CSV are untouched. Only authorized idle GPUs
are leased, including newly idle devices. Failed jobs are retained, never
automatically discarded or scored as completed.

## Repaired SimGC batch and complete local NR coverage

```bash
.venv-gpu/bin/python -m benchmark.paper_hpo plan --run-dir runs/simgc_repaired_hpo_20261010 --methods simgc --readme-section repaired-hpo --update-readme
.venv-gpu/bin/python -m benchmark.paper_hpo run --run-dir runs/simgc_repaired_hpo_20261010 --methods simgc --readme-section repaired-hpo --gpus 0 1 2 3 4 5 --update-readme
.venv-gpu/bin/python -m benchmark.paper_hpo status --run-dir runs/simgc_repaired_hpo_20261010
python -m benchmark.package_results --update-readme
python -m benchmark.package_results --check
```

The scoped batch contains 191 jobs: 63 fresh SimGC reductions, 65 GCN search
chunks (758 configurations × 10 validation fits), and 63 final evaluation jobs
(21 cells × 30 test fits). Condensation settings remain author-sourced; there
is no invented condensation HPO grid. The completed GECC–Yelp search is not
repeated. `--datasets` can restrict a new batch; changing scope on resume is
rejected. Source/configuration freezes and validation-only selection still apply.

The catalog in `benchmark/results/package_runs.json` explicitly assigns each
method/dataset to its source run. `benchmark.package_results` verifies frozen
source archives, raw job identities, source/split consistency, evaluation seed
coverage, and HPO selection hashes; it recomputes mean/population standard
deviation from all 30 test scores. It exports 105 local cells to JSON/CSV and
the README, keeping published and historical tables intact. Pending repaired
SimGC cells cannot fall back to earlier failed-reproduction scores. Cross-batch
source, preprocessing, budgets and seed schedules are disclosed; this coverage
table is not a single-protocol ranking. The running repaired batch refreshes
both its own progress table and the source-scoped coverage table automatically.

## SLURM and legacy scripts

Inspect a submission plan before submitting:

```bash
python -m benchmark.reproduce performance --datasets cora --methods clustering --slurm --dry-run
python -m benchmark.reproduce performance --datasets cora --methods clustering --slurm --slurm-arg=--gpus=1
```

`--slurm` uses `sbatch --wrap`. Choose account, partition, and resource options
through repeated `--slurm-arg` flags; activate the intended environment first.

The historical files under `graphslim/scripts/` and
`graphslim/evaluation/scripts/` now call the shared package commands. Sweep scripts
use the canonical benchmark matrices. Submission scripts use the same planner with
`--slurm`; add `--dry-run` to inspect them. Single-job positional adapters
(`main.sh`, `all.sh`, `nas.sh`, and `script_cross.sh`) retain their argument forms.
`transferbility.sh DATASET METHOD` accepts additional CLI flags, including `-R`.
No script activates a named Conda environment or assumes a personal filesystem path.

Set `PYTHON`, `GRAPHSLIM_LOAD_PATH`, and `GRAPHSLIM_SAVE_PATH` to customize
single-job adapters. Set planner data/checkpoint paths with GraphSlim flags after
`--`. The historical `sub_grid.sh` is a custom GCDM sweep, not Table 10's full
paper-matched tuning protocol.

## Package-paper reproducibility cohort

The package manuscript and both READMEs use the fixed 110-pair numerical-reference
cohort, with one-sided shortfall and two-sided relative error reported together.
GC4NC provides the primary reference under its unified evaluation protocol. Four
comparisons reuse observations underlying their GC4NC references; their sources
are explicit and they are not independent replications. CentP and GDEM are
excluded because defects fixed after 1.3.1 also affected the implementation that
generated their GC4NC references.

The [fixed artifact](results/reproducibility_20261106/README.md) contains all 313
matched-setting scores, reference versions/corrections, per-pair aggregates,
both formulas, sensitivity results and SHA-256 hashes. It embeds the unchanged
124-pair 1.3.1 artifact and records fresh CentP and GDEM scores outside the counts.
This offline calculation check requires only Python's standard library:

```bash
python benchmark/results/reproducibility_20261106/verify.py --check
python -m benchmark.plot_reproduction_coverage --check
python -m benchmark.render_readme_highlights --coverage-only --check
```

The calculation check reproduces reported statistics from fixed scores; training
reproduction uses the separate commands above and the original archived profiles.

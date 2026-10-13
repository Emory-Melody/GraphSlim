# SimGC citation implementation and verification

Cora and Citeseer have specialized profiles in the
[released SimGC code](https://github.com/BangHonor/SimGC/tree/1b3d1d59d9afaf2c9c0c7b652fb183db2a5ffa01).
The package must not replace these profiles with a generic GCN or silently
inherit downstream HPO settings into the teacher or graph-selection model.

| Stage | Model | Updates | Validation cadence |
| --- | --- | ---: | ---: |
| Teacher | Two-hop SGC_Multi, hidden 2048, ReLU, no BN/dropout | 1001 | 50 |
| Graph selection | Two-layer GCN, hidden 2048, no activation/BN/dropout | 3000 | 100 |
| Final downstream | Two-layer GCN, hidden 1024, no activation/BN/dropout | 3001 | 100 |

All three use constant Adam lr=0.001 and weight_decay=0. Condensation uses
lr_feat=0.005, lr_adj=0.001, feat_alpha=10, smoothness_alpha=0.1,
threshold=0.01, 1501 outer updates and graph selection every 100 outer steps.
The loop counts are deliberate: the author's teacher/final loops are inclusive,
whereas intermediate GCN training uses `range(3000)`. Synthetic graph selection
and model-weight selection use validation only.

Citation profiles set `pre_norm=false` to match the author's effective input:
`get_dataset()` attaches a `NormalizeFeatures` transform, but `Pyg2Dpr` reads
`dataset.data` instead of `dataset[0]`, bypassing that transform. Cora/Citeseer
therefore use raw features in the released implementation. This is a documented
method-specific preprocessing difference, not a global loader change or an
optimization based on test scores. `--pre_norm True` explicitly requests the
normalized variant and is not the released author input protocol.
SimGC also allocates `ceil(class_training_count * reduction_rate)` nodes per
class, matching the author; the total can exceed the generic floor budget.

The library's GCN now honors `args.activation=None`, `"none"` or `"linear"`.
Other supported activations use the PyG resolver; the legacy `with_relu=False`
switch still disables hidden activation. Glorot/zero-bias initialization is
enabled by the citation configuration, not imposed on other methods. Teacher
linear layers use the author's PyG initialization independently.

`validation_config` in the citation JSON is a separate graph-selection profile.
Explicit CLI overrides such as `--hidden`, `--activation` and `--eval_epochs`
take precedence in both selection and downstream stages. Such overrides must
be reported; a shortened check is not paper reproduction.

## Run the package

```bash
python -m graphslim.train_all -M simgc -D cora -R 0.5 --run_reduction 3 --run_eval 10
```

The author Cora script uses rates 0.25, 0.5, 1.0 and seeds 1–5; the repository's
benchmark roster uses 0.1, 0.25, 0.5 and seeds 1–3. CLI rates are fractions of
labeled training nodes. The upstream shell script still names `SimGC.py`, but
the pinned checkout contains `SimGC_transductive.py`; do not silently claim a
native script replay without resolving that entrypoint and its environment.

## Verify without accessing test metrics

```bash
python -m benchmark.verify_simgc --run-dir runs/simgc_author_check_NEW --gpu 3 \
  --dataset cora --rate 0.5 --seed 1 --reference runs/reference/SimGC
```

This leases only the explicitly selected idle physical GPU, masks by UUID,
compares GCN/SGC_Multi outputs and feature/edge gradients with pinned author
operators, checks features/adjacency/split metadata against the actual author
converter, freezes source/configuration/dataset/checkpoint hashes, and performs
one fresh full-budget reduction followed by validation of the saved graph.
It never calls test evaluation and refuses a nonempty output directory.
One-seed validation is an integration check, not a new mean/std benchmark cell
or an end-to-end native-author reproduction.

The 2026-10-10 integration checks completed at rate 0.5 (reduction seed 1,
downstream seed 0): Cora validation 0.726 and Citeseer validation 0.518. Records
are in `runs/simgc_fix_cora_20261010_v2` and
`runs/simgc_fix_citeseer_20261010_v2`, both frozen at scientific source SHA256
`9e9337f1181a0579eaf9a39c7fa71ce00e50b6a57b09775424581b5c96d22e09`.
For both datasets, features, adjacency and split indices matched the pinned
author converter; GCN/SGC_Multi forward, input gradients and edge gradients had
zero maximum absolute error in the supplied-weight check. No test metric was
computed. The earlier normalized-input Cora check in
`runs/simgc_fix_cora_20261010_v1` is retained, not overwritten.

Operator/input parity is not whole-training bitwise parity: package RNG
consumption differs, and intermediate validation starts a fresh Adam optimizer
instead of retaining the author's optimizer state across model resets. Keep
these boundaries explicit until a native end-to-end comparison is available.

Future citation HPO uses the author's activation/budget profile and includes
the hidden-1024 baseline in addition to the 108 GC4NC candidates. This is an
explicit local search extension. Historic `nr_20261009_v2` and
`paper_hpo_20261010_v2` artifacts retain their original frozen code and scores;
they cannot be resumed under modified scientific source.

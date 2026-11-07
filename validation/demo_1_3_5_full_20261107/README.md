# GraphSlim 1.3.5 complete demo records

Both devices run the immutable source tag `1.3.5` at
`0acbde0feff27e43b2b9141ec867fbb06fc5cc35`, in the same PyTorch 2.8.0+cu128
environment with four CPU threads. Package versions match the 1.3.4 records in
`validation/demo_1_3_4_full_20261106`; the provenance files describe the
environment. They use the full default Cora configuration:
1,000 GCond epochs, 20 outer steps, 15 inner steps, 300 downstream epochs and
ten GAT fits. One seed-1 reduction is validation-selected on each device.

CPU and GPU use the same input file hashes, source hashes and scientific
configuration. GPU execution uses an existing RTX PRO 6000 Blackwell MIG
2g.48gb instance; the physical and visible device UUIDs are in its provenance.
No other GPU job was stopped or reconfigured.

The CUDA run reproduces the 1.3.4 GPU record bit for bit: the same ten
scores and identical saved graph tensors. The CPU scores differ from the 1.3.4
CPU record (76.68% +/- 1.28%) because four-thread CPU execution is not bitwise
deterministic. `cpu_determinism_check.json` records three repeated four-thread
runs of the short configuration, which give three different GCond graphs, and
two single-thread runs, which are identical. K-Center initialization is
identical in every run.

These are integration examples, not a benchmark of reduction methods or an
isolated CPU/GPU speed experiment. The files preserve all ten validation/test
scores, original console/training logs, environment and source/input hashes,
and the saved validation-selected reduced graph for each device. Training logs
are gzip-compressed without changing the original log bytes.

Recompute counts, population summaries, configuration agreement and file hashes
with the Python standard library:

```bash
python validation/demo_1_3_5_full_20261107/verify.py
```

The runner records actual absolute paths used during these runs. For a new run,
adjust only the input/output locations and visible device, keeping the source tag
and scientific defaults fixed. The paper's console excerpts omit the trailing
saved-artifact path list; the original complete console files are retained here.

Reload a saved graph without rerunning reduction (run from the repository root,
using the same environment, with Cora under the chosen `--load_path`):

```bash
python -m graphslim.run_eval -D cora -M gcond -R 0.5 -G -1   --eval_model GAT --save_path validation/demo_1_3_5_full_20261107/cpu
```

# Add a reducer or GNN

The data, reduction and evaluation layers can be composed independently. Run a
complete example on an in-memory graph, with no dataset download:

```bash
python -m examples.custom_reducer
```

The [example source](https://github.com/Emory-Melody/GraphSlim/blob/1.3.4/examples/custom_reducer.py)
implements `FirstPerClass`, saves and reloads its reduced graph, and evaluates a
GCN for five epochs. The resulting accuracy is an integration check.

## Implement a node selector

Subclass `graphslim.sparsification.model_free_coreset_base.MFCoreSet` and implement
`select(embedds=None)`. The inherited `reduce` slices the appropriate graph,
attaches `feat_syn`, `adj_syn` and `labels_syn`, and saves the result.

```python
import numpy as np
from graphslim.sparsification.model_free_coreset_base import MFCoreSet

class FirstPerClass(MFCoreSet):
    def select(self, embedds=None):
        ids = np.asarray(self.idx_train)
        labels = np.asarray(self.labels_train)
        return np.concatenate([
            ids[labels == label][:count]
            for label, count in self.num_class_dict.items()
        ])
```

`num_class_dict` contains the inherited per-class budgets. In the transductive
setting, selected IDs refer to original graph nodes; in the inductive setting,
they refer to training-subgraph rows. Selection must use training labels only.
The example instantiates the class directly and does not alter the 49-method
registry.

For a permanent CLI extension, put the class in
`graphslim/sparsification/first_per_class.py` and add this entry to `_METHODS` in
`graphslim/reduction/registry.py` before importing the CLI:

```python
"first_per_class": MethodSpec(
    "first_per_class", "sparsification",
    "graphslim.sparsification.first_per_class", "FirstPerClass",
),
```

The shared `list_methods()` and `create_reducer()` functions then expose it through
the Python API and CLI. Put optional dataset defaults in
`graphslim/configs/first_per_class/<dataset>.json`. For other reduction families,
use `EdgeSparsifier`, `Coarsen` or `GCondBase` and implement their `reduce` contract.
Declare `edge_reduction=True` in an edge method's `MethodSpec`.

## Add a downstream GNN

Implement a model under `graphslim/models/` with the constructor
`(nfeat, nhid, nclass, args, mode="train")`. Subclassing
`graphslim.models.base.BaseGNN` reuses training, validation checkpoint selection
and prediction. `forward(x, adj, output_layer_features=False)` returns class log
probabilities and must support the adjacency representations used by the model.
`graphslim/models/gcn.py` is a complete reference implementation.

In `graphslim/models/__init__.py`, add the new name to **both**
`DOWNSTREAM_MODELS` and the mapping passed to `lazy_exports`, for example
`"MyGNN": (".my_gnn", "MyGNN")`. The evaluator and `--eval_model MyGNN`
then use the same class resolver. Verify training, saved-graph reloads,
transductive/inductive views and validation selection on a small graph before
running larger experiments. New tests belong in `tests/`.

"""Extend the node selector and run a complete CPU pipeline without downloads.

Run ``python -m examples.custom_reducer`` from the source checkout. The example
class is deliberately outside the canonical method registry.
"""

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch_geometric.data import Data

from graphslim.config import get_args
from graphslim.dataset.loader import TransAndInd
from graphslim.dataset.utils import load_reduced, splits
from graphslim.evaluation import Evaluator
from graphslim.sparsification.model_free_coreset_base import MFCoreSet
from graphslim.utils import seed_everything


class FirstPerClass(MFCoreSet):
    """Select the first training nodes in each class, using the inherited budgets."""

    def select(self, embedds=None):
        """Return training-only IDs; inductive IDs are local to the training graph.

        :param embedds: Unused optional embeddings from the selector interface.
        :returns: One-dimensional array of selected node IDs.
        """
        ids = np.asarray(self.idx_train)
        labels = np.asarray(self.labels_train)
        return np.concatenate(
            [ids[labels == label][:count] for label, count in self.num_class_dict.items()]
        )


def main(argv=None):
    """Reduce an in-memory graph, reload the saved tensors, and evaluate a GCN.

    :param argv: Optional GraphSlim command-line overrides.
    :returns: Mean and population standard deviation of two test fits.
    """
    args = get_args(
        [
            "-D",
            "cora",
            "-M",
            "random",
            "-R",
            "0.5",
            "-G",
            "-1",
            "--save_path",
            "./checkpoints/custom_reducer",
            *(sys.argv[1:] if argv is None else argv),
        ]
    )
    args.method = "first_per_class"
    args.eval_epochs, args.run_eval = 5, 2
    seed_everything(args.seed)
    nodes = torch.arange(24)
    edges = torch.stack([nodes.repeat(2), torch.cat([nodes.roll(2), nodes.roll(-2)])])
    raw = Data(
        x=torch.rand(24, 4) + (nodes % 2)[:, None].float(),
        y=nodes % 2,
        edge_index=edges,
        train_mask=nodes < 16,
        val_mask=(nodes >= 16) & (nodes < 20),
        test_mask=nodes >= 20,
    )
    graph = TransAndInd(splits(raw), "custom", norm=False)
    graph.nclass = 2
    reduced = FirstPerClass("trans", graph, args).reduce(graph)
    adjacency, features, labels = load_reduced(args, reduced)
    print(
        f"Saved and reloaded: features={tuple(features.shape)}, adjacency={tuple(adjacency.shape)}, labels={tuple(labels.shape)}"
    )
    mean, std = Evaluator(args).evaluate(reduced, model_type="GCN", verbose=False)
    print(f"Illustrative five-epoch GCN test accuracy: {100 * mean:.2f}% +/- {100 * std:.2f}%")
    return mean, std


if __name__ == "__main__":
    main()

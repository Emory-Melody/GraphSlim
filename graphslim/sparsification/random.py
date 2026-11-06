# SPDX-License-Identifier: MIT
# See LICENSE and graphslim/THIRD_PARTY_NOTICES.md for retained upstream terms.
import numpy as np

from graphslim.sparsification.model_free_coreset_base import MFCoreSet


class Random(MFCoreSet):
    """Sample training nodes uniformly without replacement within each class."""

    def __init__(self, setting, data, args, **kwargs):
        """Initialize Random and its configuration.

        :param setting: Graph learning setting: ``trans`` for the full graph or ``ind`` for split subgraphs.
        :param data: Graph data containing features, adjacency, labels, and split indices.
        :param args: Configuration namespace containing the options used by this operation.
        :param kwargs: Additional keyword options passed to the underlying implementation.
        """
        super(Random, self).__init__(setting, data, args, **kwargs)

    def select(self, embedds=None):
        """Sample training nodes uniformly without replacement within each class.

        :param embedds: Unused argument retained for the common selector interface. Defaults to ``None``.
        :returns: One-dimensional NumPy array of selected node IDs; IDs are local to the training subgraph in the inductive setting.
        """
        idx_selected = []

        for class_id, cnt in self.num_class_dict.items():
            idx = self.idx_train[self.labels_train == class_id]
            selected = np.random.permutation(idx)
            idx_selected.append(selected[:cnt])

        return np.hstack(idx_selected)

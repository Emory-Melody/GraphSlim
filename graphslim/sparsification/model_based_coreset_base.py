# SPDX-License-Identifier: MIT
# See LICENSE and graphslim/THIRD_PARTY_NOTICES.md for retained upstream terms.
from pathlib import Path

from graphslim.models import get_model_class
import numpy as np

from graphslim.dataset.utils import save_reduced
from graphslim.evaluation import *
from graphslim.models import *
from graphslim.sparsification.coreset_base import CoreSet
from graphslim.utils import to_tensor

# Precomputed SFGC k-center selections ship with the package.
FIXED_IDX_DIR = Path(__file__).resolve().parent / "fixed_idx"


def fixed_kcenter_path(args):
    """Path of the packaged SFGC k-center selection for a dataset and rate."""
    return FIXED_IDX_DIR / f"idx_{args.dataset}_{args.reduction_rate}_kcenter_15.npy"


def fixed_kcenter_indices(args):
    """Load the packaged SFGC k-center node selection for a dataset and rate.

    :param args: Configuration with ``dataset`` and ``reduction_rate``.
    :returns: One-dimensional NumPy array of selected node IDs.
    """
    return np.load(fixed_kcenter_path(args))


def uses_fixed_sfgc_selection(args):
    """Whether SFGC's released k-center selection applies to this run.

    SFGC initializes from the authors' fixed k-center nodes. The selection is used
    when SFGC itself or its k-center initializer runs and a packaged file exists for
    the dataset and rate; otherwise k-center is trained as usual.
    """
    serves_sfgc = "sfgc" in (args.method, getattr(args, "init_for", None))
    return serves_sfgc and fixed_kcenter_path(args).is_file()


class MBCoreSet(CoreSet):
    """Node coreset reduction using embeddings from a trained graph model."""

    def __init__(self, setting, data, args, **kwarg):
        """Initialize MBCoreSet and its configuration.

        :param setting: Graph learning setting: ``trans`` for the full graph or ``ind`` for split subgraphs.
        :param data: Graph data containing features, adjacency, labels, and split indices.
        :param args: Configuration namespace containing the options used by this operation.
        :param kwarg: Additional keyword options passed to the underlying implementation.
        """
        super(MBCoreSet, self).__init__(setting, data, args, **kwarg)

    @verbose_time_memory
    def reduce(self, data, verbose=False, save=True):
        """Train an embedding model, select a coreset, and build its induced graph.

        Synthetic tensors are attached to ``data`` and converted to CPU tensors.
        The aggregation path uses an identity synthetic adjacency.

        :param data: Graph data containing features, adjacency, labels, and split indices.
        :param verbose: Whether to print progress and diagnostic information. Defaults to ``False``.
        :param save: Whether to write the reduced graph checkpoint to ``args.save_path``. Defaults to ``True``.
        :returns: The input data with ``adj_syn``, ``feat_syn``, and ``labels_syn`` set.
        """
        args = self.args
        model = get_model_class(self.condense_model)(
            data.feat_full.shape[1], args.hidden, data.nclass, args
        ).to(self.device)
        if self.setting == "trans":
            if uses_fixed_sfgc_selection(args):
                # model.fit_with_val(data, train_iters=1200, normadj=True, verbose=verbose,
                #                    setting=args.setting, reduced=False, final_output=True)
                # embeds = model.predict(data.feat_full, data.adj_full, output_layer_features=True)[0].detach()
                idx_selected = fixed_kcenter_indices(args)
            else:
                model.fit_with_val(
                    data,
                    train_iters=args.eval_epochs,
                    normadj=True,
                    verbose=verbose,
                    setting=args.setting,
                    reduced=False,
                )
                embeds = model.predict(data.feat_full, data.adj_full).detach()
                idx_selected = self.select(embeds)

            data.adj_syn = data.adj_full[np.ix_(idx_selected, idx_selected)]
            data.feat_syn = data.feat_full[idx_selected]
            data.labels_syn = data.labels_full[idx_selected]

        if self.setting == "ind":
            if uses_fixed_sfgc_selection(args):
                idx_selected = fixed_kcenter_indices(args)
            else:
                model.fit_with_val(
                    data,
                    train_iters=args.eval_epochs,
                    normadj=True,
                    verbose=verbose,
                    setting=args.setting,
                    reduced=False,
                )
                # Inductive selection uses local training IDs, so embed the training graph.
                embeds = model.predict(data.feat_train, data.adj_train).detach()

                idx_selected = self.select(embeds)
            data.feat_syn = data.feat_train[idx_selected]
            data.adj_syn = data.adj_train[np.ix_(idx_selected, idx_selected)]
            data.labels_syn = data.labels_train[idx_selected]

        if verbose:
            print("selected nodes:", idx_selected.shape[0])
            print("induced edges:", data.adj_syn.sum())
        data.adj_syn, data.feat_syn, data.labels_syn = to_tensor(
            data.adj_syn, data.feat_syn, label=data.labels_syn, device="cpu"
        )
        if save:
            save_reduced(data.adj_syn, data.feat_syn, data.labels_syn, args)

        # if args.method in ['sfgc', 'geom']:
        #     # recover args
        #     args.eval_epochs = epoch
        #     args.weight_decay = wd
        #     args.lr = lr

        return data

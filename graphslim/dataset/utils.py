# SPDX-License-Identifier: MIT
# See LICENSE and graphslim/THIRD_PARTY_NOTICES.md for retained upstream terms.
import os
import sys
from graphslim.dataset.convertor import *
from graphslim.utils import is_sparse_tensor
import logging


def sparsify(model_type, adj_syn, args, verbose=False):
    """
    Applies sparsification to the adjacency matrix based on the model type and given arguments.

    This function modifies the adjacency matrix to make it sparser according to the model type and method specified.
    For specific methods and datasets, it adjusts the threshold used for sparsification.

    Parameters
    ----------
    model_type : str
        The type of model used, which determines the sparsification strategy. Can be 'MLP', 'GAT', or other.
    adj_syn : torch.Tensor
        The adjacency matrix to be sparsified.
    args : argparse.Namespace
        Command-line arguments and configuration parameters which may include method-specific settings.
    verbose : bool, optional
        If True, prints information about the sparsity of the adjacency matrix before and after sparsification.
        Default is False.

    Returns
    -------
    adj_syn : torch.Tensor
        The sparsified adjacency matrix.
    """
    threshold = 0
    if model_type == "MLP":
        adj_syn = adj_syn - adj_syn
        torch.diagonal(adj_syn).fill_(1)
    elif model_type == "GAT":
        if args.method in ("cgc", "clustgdd", "gcgp", "nope"):
            # These methods construct their topology explicitly. Compression can
            # produce valid weights below 0.5; preserve all nonzero edges.
            threshold = 0
        elif args.method in ["gcond", "doscond"]:
            if args.dataset in ["cora", "citeseer"]:
                threshold = 0.5  # Make the graph sparser as GAT does not work well on dense graph
            else:
                threshold = 0.1
        elif args.method in ["msgc"]:
            threshold = args.threshold
        else:
            threshold = 0.5
    else:
        if args.method in ["gcond", "doscond"]:
            threshold = args.threshold
        elif args.method in ["msgc"]:
            threshold = 0
        else:
            threshold = 0

    # if verbose and args.method not in ['gcondx', 'doscondx', 'sfgc', 'geom', 'gcsntk']:
    # print('Sum:', adj_syn.sum().item())
    # print(adj_syn)
    # print('Sparsity:', adj_syn.nonzero().shape[0] / adj_syn.numel())
    # if args.method in ['sgdd']:
    #     threshold = 0.5
    if threshold > 0:
        adj_syn[adj_syn < threshold] = 0
        if verbose:
            print("Sparsity after truncating:", adj_syn.nonzero().shape[0] / adj_syn.numel())
        # else:
        #     print("structure free methods do not need to truncate the adjacency matrix")
    return adj_syn


def index2mask(index, size):
    """
    Convert an index list to a boolean mask.

    Parameters
    ----------
    index : list or tensor
        List or tensor of indices to be set to True.
    size : int or tuple of int
        Shape of the mask. If an integer, the mask is 1-dimensional.

    Returns
    -------
    mask : tensor
        A boolean tensor of the specified size, with True at the given `index` positions and False elsewhere.

    Examples
    --------
    >>> index = [0, 2, 4]
    >>> size = 5
    >>> index2mask(index, size)
    tensor([True, False, True, False, True], dtype=torch.bool)
    """
    mask = torch.zeros(size, dtype=torch.bool)
    mask[index] = 1
    return mask


def splits(data, exp="default"):
    # customize your split here
    """Keep existing train/validation/test masks or create classwise random splits.

    ``random`` uses 20/30 training/validation nodes per class, ``few`` uses 5/5,
    and other values use classwise 80/10/10 proportions. Split indices are attached
    to the returned data.

    :param data: Graph data containing features, adjacency, labels, and split indices.
    :param exp: Split protocol: ``random``, ``few``, or the default classwise proportions. Defaults to ``'default'``.
    :returns: The input data with masks and ``idx_train``, ``idx_val``, and ``idx_test`` set.
    """
    if hasattr(data, "y"):
        num_classes = max(data.y) + 1
    else:
        num_classes = max(data.labels_full).item() + 1
    # data.nclass = num_classes
    if not hasattr(data, "train_mask"):
        indices = []
        for i in range(num_classes):
            data.y = data.y.reshape(-1)
            index = (data.y == i).nonzero().view(-1)
            index = index[torch.randperm(index.size(0))]
            indices.append(index)

        if exp == "random":
            train_index = torch.cat([i[:20] for i in indices], dim=0)
            val_index = torch.cat([i[20:50] for i in indices], dim=0)
            test_index = torch.cat([i[50:] for i in indices], dim=0)
        elif exp == "few":
            train_index = torch.cat([i[:5] for i in indices], dim=0)
            val_index = torch.cat([i[5:10] for i in indices], dim=0)
            test_index = torch.cat([i[10:] for i in indices], dim=0)
        else:
            # if fixed but no split is provided, use the default 8/1/1 split classwise
            train_index = torch.cat([i[: int(i.shape[0] * 0.8)] for i in indices], dim=0)
            val_index = torch.cat(
                [i[int(i.shape[0] * 0.8) : int(i.shape[0] * 0.9)] for i in indices], dim=0
            )
            test_index = torch.cat([i[int(i.shape[0] * 0.9) :] for i in indices], dim=0)
            # raise NotImplementedError('Unknown split type')
        data.train_mask = index2mask(train_index, size=data.num_nodes)
        data.val_mask = index2mask(val_index, size=data.num_nodes)
        data.test_mask = index2mask(test_index, size=data.num_nodes)
    data.idx_train = data.train_mask.nonzero().view(-1)
    data.idx_val = data.val_mask.nonzero().view(-1)
    data.idx_test = data.test_mask.nonzero().view(-1)

    return data


def save_reduced(adj_syn=None, feat_syn=None, labels_syn=None, args=None):
    """Save available synthetic graph tensors under the method checkpoint directory.

    Files are named by dataset, reduction rate, and seed beneath
    ``args.save_path/reduced_graph/args.method``; attacked Flickr graphs use
    the corresponding ``corrupt_graph`` subtree.

    :param adj_syn: Synthetic adjacency, features, or labels to save; ``None`` skips this component. Defaults to ``None``.
    :param feat_syn: Synthetic adjacency, features, or labels to save; ``None`` skips this component. Defaults to ``None``.
    :param labels_syn: Synthetic adjacency, features, or labels to save; ``None`` skips this component. Defaults to ``None``.
    :param args: Configuration namespace containing the options used by this operation. Defaults to ``None``.
    """
    base_path = os.path.abspath(os.path.expanduser(args.save_path))
    save_path = os.path.join(base_path, "reduced_graph", args.method)
    if args.attack is not None and args.dataset in ["flickr"]:
        save_path = os.path.join(
            base_path, "corrupt_graph", args.attack, "reduced_graph", args.method
        )
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if adj_syn is not None:
        torch.save(
            adj_syn,
            os.path.join(save_path, f"adj_{args.dataset}_{args.reduction_rate}_{args.seed}.pt"),
        )
    if feat_syn is not None:
        torch.save(
            feat_syn,
            os.path.join(save_path, f"feat_{args.dataset}_{args.reduction_rate}_{args.seed}.pt"),
        )
    if labels_syn is not None:
        torch.save(
            labels_syn,
            os.path.join(save_path, f"label_{args.dataset}_{args.reduction_rate}_{args.seed}.pt"),
        )
    args.logger.info(
        f"Saved {os.path.join(save_path, f'adj_{args.dataset}_{args.reduction_rate}_{args.seed}.pt')}"
    )


def load_reduced(args, data=None):
    """Load synthetic graph checkpoints for a dataset, method, rate, and seed.

    Raises ``FileNotFoundError`` when no matching component exists. For partial
    checkpoints, missing features or labels can use the supplied original data,
    and a missing adjacency uses an identity graph. Missing labels always fall
    back to ``data.labels_train``: in the transductive setting an adjacency-only
    checkpoint (edge sparsifiers) therefore yields full-graph features with
    training labels, and downstream training indexes its outputs by
    ``data.idx_train`` so that validation and test labels are never trained on.

    :param args: Configuration namespace containing the options used by this operation.
    :param data: Graph data containing features, adjacency, labels, and split indices. Defaults to ``None``.
    :returns: Tuple ``(adjacency, features, labels)`` on the resolved device.
    """
    flag = 0
    base_path = os.path.abspath(os.path.expanduser(args.save_path))
    save_path = os.path.join(base_path, "reduced_graph", args.method)
    if args.attack is not None and args.dataset in ["flickr"]:
        save_path = os.path.join(
            base_path, "corrupt_graph", args.attack, "reduced_graph", args.method
        )

    suffix = f"{args.dataset}_{args.reduction_rate}_{args.seed}.pt"
    if not any(
        os.path.isfile(os.path.join(save_path, f"{kind}_{suffix}"))
        for kind in ("feat", "adj", "label")
    ):
        raise FileNotFoundError(
            f"No reduced graph checkpoints for {args.method} at {save_path} ({suffix}). "
            "Run reduction first, or use --eval_whole for the whole-graph baseline."
        )

    def _resolve_device(device_str):
        """Resolve checkpoint loading device and fall back to CPU when CUDA is unavailable.

        :param device_str: Requested checkpoint loading device.
        :returns: Resolved device string.
        """
        if device_str is None:
            return "cpu"
        if isinstance(device_str, str) and device_str.lower() == "cpu":
            return "cpu"
        if isinstance(device_str, str) and device_str.startswith("cuda"):
            if not torch.cuda.is_available():
                if hasattr(args, "logger"):
                    args.logger.warning(
                        "CUDA requested but not available. Falling back to CPU for reduced graph loading."
                    )
                return "cpu"
            try:
                parts = device_str.split(":")
                idx = int(parts[1]) if len(parts) > 1 else 0
            except ValueError:
                idx = 0
            if idx < torch.cuda.device_count():
                return f"cuda:{idx}"
            fallback = "cuda:0" if torch.cuda.device_count() > 0 else "cpu"
            if hasattr(args, "logger"):
                args.logger.warning(
                    f"Requested device {device_str} unavailable. Using {fallback} for reduced graph loading."
                )
            return fallback
        return device_str

    target_device = _resolve_device(getattr(args, "device", "cpu"))
    if hasattr(args, "device") and args.device != target_device:
        args.device = target_device

    def _safe_load_tensor(file_path):
        """Load a tensor checkpoint with compatibility handling for PyTorch loading options.

        :param file_path: Filesystem location used by this operation.
        :returns: Deserialized tensor on the resolved device.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(file_path)
        load_locations = [target_device, "cpu"] if target_device != "cpu" else ["cpu"]
        last_exc = None
        for loc in load_locations:
            try:
                tensor = torch.load(file_path, map_location=loc)
                if loc != target_device and target_device != "cpu" and hasattr(tensor, "to"):
                    try:
                        tensor = tensor.to(target_device)
                    except Exception as move_err:
                        if hasattr(args, "logger"):
                            args.logger.warning(
                                f"Loaded {file_path} on {loc} but failed to move to {target_device}: {move_err}"
                            )
                return tensor
            except Exception as e:
                last_exc = e
        # The CPU attempt always runs, so reaching here means every attempt failed.
        raise last_exc

    try:
        feat_path = os.path.join(
            save_path, f"feat_{args.dataset}_{args.reduction_rate}_{args.seed}.pt"
        )
        feat_syn = _safe_load_tensor(feat_path)
    except Exception as e:
        file_path = os.path.join(
            save_path, f"feat_{args.dataset}_{args.reduction_rate}_{args.seed}.pt"
        )
        print(f"find no feat at {file_path}, use original feature matrix instead. Error: {e}")
        if hasattr(args, "logger"):
            args.logger.warning(f"Failed to load synthetic features from {file_path}: {e}")
        flag += 1
        if args.setting == "trans":
            feat_syn = data.feat_full
        else:
            feat_syn = data.feat_train
    try:
        label_path = os.path.join(
            save_path, f"label_{args.dataset}_{args.reduction_rate}_{args.seed}.pt"
        )
        labels_syn = _safe_load_tensor(label_path)
    except Exception as e:
        file_path = os.path.join(
            save_path, f"label_{args.dataset}_{args.reduction_rate}_{args.seed}.pt"
        )
        print(f"find no label at {file_path}, use original label matrix instead. Error: {e}")
        if hasattr(args, "logger"):
            args.logger.warning(f"Failed to load synthetic labels from {file_path}: {e}")
        flag += 1
        labels_syn = data.labels_train

    try:
        adj_path = os.path.join(
            save_path, f"adj_{args.dataset}_{args.reduction_rate}_{args.seed}.pt"
        )
        adj_syn = _safe_load_tensor(adj_path)
    except Exception as e:
        file_path = os.path.join(
            save_path, f"adj_{args.dataset}_{args.reduction_rate}_{args.seed}.pt"
        )
        print(f"find no adj at {file_path}, use identity matrix instead. Error: {e}")
        if hasattr(args, "logger"):
            args.logger.warning(f"Failed to load synthetic adjacency from {file_path}: {e}")
        flag += 1
        adj_syn = torch.eye(feat_syn.size(0), device=args.device)
    # args.logger.info(f"Load {save_path}/adj_{args.dataset}_{args.reduction_rate}_{args.seed}.pt")
    if flag == 3:
        args.logger.info("no file found, use original graph instead")

    return adj_syn, feat_syn, labels_syn


def get_syn_data(data, args, model_type, verbose=False):
    """Loads or computes synthetic data for evaluation.

    Parameters
    ----------
    data : Dataset
        The dataset containing the graph data.
    args : Namespace
        Configuration containing the checkpoint paths, reduction method and
        adjacency sparsification options.
    model_type : str
        Downstream model name used to select adjacency sparsification.
    verbose : bool, optional, default=False
        Whether to print detailed logs.

    Returns
    -------
    feat_syn : torch.Tensor
        Synthetic feature matrix.
    adj_syn : torch.Tensor
        Synthetic adjacency matrix.
    labels_syn : torch.Tensor
        Synthetic labels.
    """
    adj_syn, feat_syn, labels_syn = load_reduced(args, data)

    if labels_syn.shape[0] == data.labels_train.shape[0]:
        return feat_syn, adj_syn, labels_syn

    if isinstance(adj_syn, torch.sparse.FloatTensor):
        adj_syn = adj_syn.to_dense()
    else:
        adj_syn = adj_syn

    adj_syn = sparsify(model_type, adj_syn, args, verbose=verbose)
    return feat_syn, adj_syn, labels_syn


# =============from graphsaint================#
# import networkx as nx
# import numpy as np
# import os
# import scipy as sp
# import tempfile
# import zipfile
# from pygsp import graphs
# from scipy import sparse
# from urllib import request
#
# _YEAST_URL = "http://nrvis.com/download/data/bio/bio-yeast.zip"
# _MOZILLA_HEADERS = [("User-Agent", "Mozilla/5.0")]
#
# def download_yeast():
#     r"""
#     A convenience method for loading a network of protein-to-protein interactions in budding yeast.
#
#     http://networkrepository.com/bio-yeast.php
#     """
#     with tempfile.TemporaryDirectory() as tempdir:
#         zip_filename = os.path.join(tempdir, "bio-yeast.zip")
#         with open(zip_filename, "wb") as zip_handle:
#             opener = request.build_opener()
#             opener.addheaders = _MOZILLA_HEADERS
#             request.install_opener(opener)
#             with request.urlopen(_YEAST_URL) as url_handle:
#                 zip_handle.write(url_handle.read())
#         with zipfile.ZipFile(zip_filename) as zip_handle:
#             zip_handle.extractall(tempdir)
#         mtx_filename = os.path.join(tempdir, "bio-yeast.mtx")
#         with open(mtx_filename, "r") as mtx_handle:
#             _ = next(mtx_handle)  # header
#             n_rows, n_cols, _ = next(mtx_handle).split(" ")
#             E = np.loadtxt(mtx_handle)
#     E = E.astype(int) - 1
#     W = sparse.lil_matrix((int(n_rows), int(n_cols)))
#     W[(E[:, 0], E[:, 1])] = 1
#     W = W.tocsr()
#     W += W.T
#     return W
#
# def real(N, graph_name, connected=True):
#     r"""
#     A convenience method for loading toy graphs that have been collected from the internet.
#
#     Parameters:
#     ----------
#     N : int
#         The number of nodes. Set N=-1 to return the entire graph.
#
#     graph_name : a string
#         Use to select which graph is returned. Choices include
#             * airfoil
#                 Graph from airflow simulation
#                 http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.50.9217&rep=rep1&type=pdf
#                 http://networkrepository.com/airfoil1.php
#             * yeast
#                 Network of protein-to-protein interactions in budding yeast.
#                 http://networkrepository.com/bio-yeast.php
#             * minnesota
#                 Minnesota road network.
#                 I am using the version provided by the PyGSP software package (initially taken from the MatlabBGL library.)
#             * bunny
#                 The Stanford bunny is a computer graphics 3D test model developed by Greg Turk and Marc Levoy in 1994 at Stanford University
#                 I am using the version provided by the PyGSP software package.
#     connected : Boolean
#         Set to True if only the giant component is to be returned.
#     """
#
#     directory = os.path.join(
#         os.path.dirname(os.path.dirname(__file__)), "data"
#     )
#
#     tries = 0
#     while True:
#         tries = tries + 1
#
#         if graph_name == "airfoil":
#             G = graphs.Airfoil()
#             G = graphs.Graph(W=G.W[0:N, 0:N], coords=G.coords[0:N, :])
#
#         elif graph_name == "yeast":
#             W = download_yeast()
#             G = graphs.Graph(W=W[0:N, 0:N])
#
#         elif graph_name == "minnesota":
#             G = graphs.Minnesota()
#             W = G.W.astype(np.float)
#             G = graphs.Graph(W=W[0:N, 0:N], coords=G.coords[0:N, :])
#
#         elif graph_name == "bunny":
#             G = graphs.Bunny()
#             W = G.W.astype(np.float)
#             G = graphs.Graph(W=W[0:N, 0:N], coords=G.coords[0:N, :])
#
#         if connected == False or G.is_connected():
#             break
#         if tries > 1:
#             print("WARNING: Disconnected graph. Using the giant component.")
#             G, _ = get_giant_component(G)
#             break
#
#     if not hasattr(G, 'coords'):
#         try:
#             import networkx as nx
#             graph = nx.from_scipy_sparse_matrix(G.W)
#             pos = nx.nx_agraph.graphviz_layout(graph, prog='neato')
#             G.set_coordinates(np.array(list(pos.values())))
#         except ImportError:
#             G.set_coordinates()
#
#     return G
#
# def models(N, graph_name, connected=True, default_params=False, k=12, sigma=0.5):
#     tries = 0
#     while True:
#         tries = tries + 1
#         if graph_name == "regular":
#             if default_params:
#                 k = 10
#             offsets = []
#             for i in range(1, int(k / 2) + 1):
#                 offsets.append(i)
#                 offsets.append(-(N - i))
#
#             offsets = np.array(offsets)
#             vals = np.ones_like(offsets)
#             W = sp.sparse.diags(
#                 vals, offsets, shape=(N, N), format="csc", dtype=np.float
#             )
#             W = (W + W.T) / 2
#             G = graphs.Graph(W=W)
#
#         else:
#             print("ERROR: uknown model")
#             return
#
#         if connected == False or G.is_connected():
#             break
#         if tries > 1:
#             print("WARNING: disconnected graph.. trying to use the giant component")
#             G = get_giant_component(G)
#             break
#     return G
#
# def to_networkx(G):
#     return nx.from_scipy_sparse_matrix(G.W)
#
# def get_neighbors(G, i):
#     return G.A[i, :].indices
#     # return np.arange(G.N)[np.array((G.W[i,:] > 0).todense())[0]]
#
# def get_giant_component(G):
#     [ncomp, labels] = sp.sparse.connected_components(G.W, directed=False, return_labels=True)
#
#     W_g = np.array((0, 0))
#     coords_g = np.array((0, 2))
#     keep = np.array(0)
#
#     for i in range(0, ncomp):
#
#         idx = np.where(labels != i)
#         idx = idx[0]
#
#         if G.N - len(idx) > W_g.shape[0]:
#             W_g = G.W.toarray()
#             W_g = np.delete(W_g, idx, axis=0)
#             W_g = np.delete(W_g, idx, axis=1)
#             if hasattr(G, 'coords'):
#                 coords_g = np.delete(G.coords, idx, axis=0)
#             keep = np.delete(np.arange(G.N), idx)
#
#     if not hasattr(G, 'coords'):
#         # print(W_g.shape)
#         G_g = graphs.Graph(W=W_g)
#     else:
#         G_g = graphs.Graph(W=W_g, coords=coords_g)
#
#     return (G_g, keep)
#
# def get_S(G):
#     """
#     Construct the N x |E| gradient matrix S
#     """
#     # the edge set
#     edges = G.get_edge_list()
#     weights = np.array(edges[2])
#     edges = np.array(edges[0:2])
#     M = edges.shape[1]
#
#     # Construct the N x |E| gradient matrix S
#     S = np.zeros((G.N, M))
#     for e in np.arange(M):
#         S[edges[0, e], e] = np.sqrt(weights[e])
#         S[edges[1, e], e] = -np.sqrt(weights[e])
#
#     return S
#
# # Compare the spectum of L and Lc

#

#
# def is_symmetric(As):
#     """Check if a sparse matrix is symmetric
#
#     Parameters
#     ----------
#     As : array or sparse matrix
#         A square matrix.
#
#     Returns
#     -------
#     check : bool
#         The check result.
#
#     """
#     from scipy import sparse
#
#     if As.shape[0] != As.shape[1]:
#         return False
#
#     if not isinstance(As, sparse.coo_matrix):
#         As = sparse.coo_matrix(As)
#
#     r, c, v = As.row, As.col, As.data
#     tril_no_diag = r > c
#     triu_no_diag = c > r
#
#     if triu_no_diag.sum() != tril_no_diag.sum():
#         return False
#
#     rl = r[tril_no_diag]
#     cl = c[tril_no_diag]
#     vl = v[tril_no_diag]
#     ru = r[triu_no_diag]
#     cu = c[triu_no_diag]
#     vu = v[triu_no_diag]
#
#     sortl = np.lexsort((cl, rl))
#     sortu = np.lexsort((ru, cu))
#     vl = vl[sortl]
#     vu = vu[sortu]
#
#     check = np.allclose(vl, vu)
#
#     return check
# if transform is not None and normalize_features:
#     dataset.transform = T.Compose([T.NormalizeFeatures(), transform])
# elif normalize_features:
#     dataset.transform = T.NormalizeFeatures()
# elif transform is not None:
#     dataset.transform = transform

# def get_mnist(data_path):
#     channel = 1
#     im_size = (28, 28)
#     num_classes = 10
#     mean = [0.1307]
#     std = [0.3081]
#     transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
#     dst_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)  # no        augmentation
#     dst_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)
#     class_names = [str(c) for c in range(num_classes)]
#
#     labels = []
#     feat = []
#     for x, y in dst_train:
#         feat.append(x.view(1, -1))
#         labels.append(y)
#     feat = torch.cat(feat, axis=0).numpy()
#     adj = sp.eye(len(feat))
#     idx = np.arange(len(feat))
#     dpr_data = GraphData(adj - adj, feat, labels, idx, idx, idx)
#     return Dpr2Pyg(dpr_data)
# ====mirage utils====#

from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader

import torch


def disjointed_union(tree_list, class_=None, device=None):
    """Combine nontrivial PyG trees without adding edges between trees.

    :param tree_list: PyG graphs sharing a class, feature width and device.
        Single-node trees are omitted; edge attributes are preserved when present.
    :param class_: Label assigned to the combined graph's ``y`` attribute.
        Defaults to ``None``.
    :param device: Output device, which must match every retained input tree.
        ``None`` uses the first retained tree's device.
    :returns: PyG ``Data`` with offset edge indices and no ``roots_to_embed``
        attribute, or ``None`` when at most one node remains after filtering.
    """
    tree_list = [
        tree for tree in tree_list if tree.x.shape[0] > 1
    ]  # skipping single node trees for now because it causes issues
    total_nodes = sum(tree.x.shape[0] for tree in tree_list)
    total_edges = sum(tree.edge_index.shape[1] for tree in tree_list)
    if total_nodes <= 1:
        return None  # if the whole graph has just one node, then we return none
    if device is None:
        device = tree_list[0].x.device
    assert all(device == tree.x.device for tree in tree_list), "Trees must be on same device."
    assert all(tree.x.shape[1] == tree_list[0].x.shape[1] for tree in tree_list), (
        "Number of node features are different."
    )
    assert all(tree.y == tree_list[0].y for tree in tree_list), "Tree must be in same class."
    # PyG ``Data`` reports absent attributes as ``None``, so test the value.
    if getattr(tree_list[0], "edge_attr", None) is not None:
        assert all(getattr(tree, "edge_attr", None) is not None for tree in tree_list), (
            "Not all trees have edge features."
        )
        assert all(
            tree.edge_attr.shape[1:] == tree_list[0].edge_attr.shape[1:] for tree in tree_list
        ), "Number of edge features are different."
        hasedgeattr = True
    else:
        hasedgeattr = False
    x = torch.zeros(
        (total_nodes, tree_list[0].x.shape[1]),
        device=device,
        dtype=tree_list[0].x.dtype,
    )
    if hasedgeattr:
        edge_attr = torch.zeros(
            (total_edges, *tree_list[0].edge_attr.shape[1:]),
            device=device,
            dtype=tree_list[0].edge_attr.dtype,
        )
    edge_index = torch.zeros((2, total_edges), device=device, dtype=torch.long)
    node_start = 0
    edge_start = 0
    for tree in tree_list:
        num_nodes = tree.x.shape[0]
        idxs = torch.arange(num_nodes) + node_start
        x[idxs, :] = tree.x
        num_edges = tree.edge_index.shape[1]
        idxs = torch.arange(num_edges) + edge_start
        if hasedgeattr:
            edge_attr[idxs] = tree.edge_attr
        edge_index_extract = tree.edge_index
        edge_index_extract = edge_index_extract + node_start
        edge_index[:, idxs] = edge_index_extract
        node_start += num_nodes
        edge_start += num_edges
    if hasedgeattr:
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=class_)
    else:
        data = Data(x=x, edge_index=edge_index, y=class_)
    return data


def roots_to_embed(data):
    """Mark child-to-parent tree roots that occur as targets but never as sources.

    :param data: PyG graph containing node features and directed ``edge_index``.
    :returns: The input graph with a ``roots_to_embed`` mask. Isolated nodes are
        not marked by this implementation.
    """
    nn = data.x.shape[0]
    root = torch.zeros((nn,), device=data.x.device)
    root_indices = torch.tensor(
        sorted(set(data.edge_index[1].tolist()).difference(set(data.edge_index[0].tolist()))),
        dtype=torch.long,
    ).to(data.x.device)
    root[root_indices] = 1
    data.roots_to_embed = root
    return data


class myDataset(Dataset):
    """PyG dataset wrapper around an in-memory sequence of graph samples."""

    def __init__(this, dataset_list, dataset_len=0, **kwargs):
        """Initialize myDataset and its configuration.

        :param dataset_list: Sequence of graph samples stored by the dataset wrapper.
        :param dataset_len: Number of graphs to draw with replacement in proportion to each
            entry's ``freq`` (at least the list length); zero keeps the stored list as is.
            Defaults to ``0``.
        :param kwargs: Additional keyword options passed to the underlying implementation.
        """
        super().__init__()
        this.dataset_list = [list(d.values())[0] for d in dataset_list]
        this.rng = np.random.RandomState(seed=kwargs.get("seed", 0))
        freq, data = list(zip(*((d["freq"], d["data"]) for d in this.dataset_list)))
        if dataset_len:
            # Draw max(len, dataset_len) graphs in proportion to pattern frequency.
            this.dataset_len = max(len(this.dataset_list), dataset_len)
            probs = np.asarray(freq) / np.sum(freq)
            this.data = [
                data[this.rng.choice(np.arange(len(data)), p=probs)]
                for _ in range(this.dataset_len)
            ]
        else:
            this.dataset_len = len(this.dataset_list)
            this.data = list(data)

    def len(this):
        """Report the configured dataset length.

        :returns: Number of graph samples exposed by this wrapper.
        """
        return this.dataset_len

    def get(this, idx):
        """Retrieve a graph from the stored sample list.

        :param idx: Index or indices to access.
        :returns: PyG graph at the requested index.
        """
        return this.data[idx]


def get_dataloader(dataset_classwise, **kwargs):
    """Build a PyG DataLoader from graphs or a two-class graph dictionary.

    :param dataset_classwise: Graph collection or dictionary with graph lists for classes 0 and 1.
    :param kwargs: Additional keyword options passed to the underlying implementation.
    :returns: Configured PyG DataLoader.
    """
    if isinstance(dataset_classwise, dict):
        dataset = (
            dataset_classwise[0] + dataset_classwise[1]
        )  # for now assumes only two classes named 0 and 1 exist
        mydataset = myDataset(dataset, **kwargs)
        if "dataset_len" in kwargs:
            del kwargs["dataset_len"]
        dataloader = DataLoader(mydataset, **kwargs)
    else:
        dataloader = DataLoader(dataset_classwise, **kwargs)
    return dataloader


# ---------------  PROCESS DATASET (LABELS)  ------------------
def get_label_maps(dataset):
    """Enumerate node labels and complete feature tuples for canonical tree encoding.

    :param dataset: Dataset name or dataset collection, as required by this operation.
    :returns: Tuple ``(node_label_map, node_feature_tuple_map)``.
    """
    node_labels = set()
    node_labels_full = set()
    for d in dataset:
        node_label = d.y
        node_label_full = d.x
        [node_labels.add(n.item()) for n in node_label]
        [
            node_labels_full.add(tuple(node_label_full[n, :].tolist()))
            for n in range(len(node_label_full))
        ]
    node_label_map = {n: idx for idx, n in enumerate(node_labels)}
    node_label_map_full = {n: idx for idx, n in enumerate(node_labels_full)}
    return node_label_map, node_label_map_full


def process_labels(dataset, node_label_map, node_label_map_full):
    """Encode node labels and feature tuples as categorical IDs for tree mining.

    :param dataset: Dataset name or dataset collection, as required by this operation.
    :param node_label_map: Mapping between node labels and categorical IDs, or parsed node IDs and labels.
    :param node_label_map_full: Dictionary mapping original node feature tuples to canonical IDs.
    :returns: List of PyG Data objects retaining original features and edge indices.
    """
    dataset_ = []
    for d in dataset:
        data = Data(
            x=torch.tensor(
                [node_label_map[n.item()] for n in d.y]
            ),  # since this goes into C++ and out comes just a string, the node features are lost
            original_x=d.x,
            edge_index=d.edge_index,
            y=d.y,
            node_attr=torch.tensor(
                [node_label_map_full[tuple(d.x[n, :].tolist())] for n in range(d.x.shape[0])]
            ),
        )
        dataset_.append(data)
    return dataset_


def preprocess_dataset(dataset, full_dataset):
    """Create canonical node dictionaries from a full dataset and encode a subset.

    :param dataset: Dataset name or dataset collection, as required by this operation.
    :param full_dataset: Reference dataset used to enumerate all supported labels and feature tuples.
    :returns: Tuple ``(encoded_dataset, node_label_map, node_feature_tuple_map)``.
    """
    node_label_map, node_label_map_full = get_label_maps(full_dataset)
    label_processed_dataset = process_labels(
        dataset,
        node_label_map,
        node_label_map_full,
    )
    return label_processed_dataset, node_label_map, node_label_map_full


def preprocess_dataset_test(dataset, node_label_map, edge_label_map, node_label_map_full):
    """Encode held-out graphs using the existing training label dictionaries.

    :param dataset: Dataset name or dataset collection, as required by this operation.
    :param node_label_map: Mapping between node labels and categorical IDs, or parsed node IDs and labels.
    :param edge_label_map: Parsed or original dictionary of categorical edge labels.
    :param node_label_map_full: Dictionary mapping original node feature tuples to canonical IDs.
    :returns: List of encoded PyG graphs; ``edge_label_map`` is retained for interface compatibility.
    """
    label_processed_dataset = process_labels(dataset, node_label_map, node_label_map_full)
    return label_processed_dataset


# ----------------  PROCESS CANONICAL LABELS  -----------------
def prettify_canonical_label(label):
    """Convert whitespace-separated tree tokens to bracketed canonical notation.

    :param label: Canonical tree string to parse or format.
    :returns: Formatted canonical tree string.
    """
    assert isinstance(label, str)
    tokens = label.split()
    label2 = f"<{tokens[0]},{tokens[1]}>"
    ptr = 2
    while ptr < len(tokens):
        if tokens[ptr] == "$":
            label2 += tokens[ptr]
            ptr += 1
        else:
            node_label = tokens[ptr]
            node_original_label = tokens[ptr + 1]
            edge_label = tokens[ptr + 2]
            ptr += 3
            label2 += f"<{node_label},{node_original_label},{edge_label}>"
    return label2


def canonical_label_to_naturals(dataset_labels):
    """Assign contiguous integer IDs to unique canonical tree labels.

    :param dataset_labels: Canonical label lists, one list per graph.
    :returns: Mapping from canonical label strings to integer IDs in encounter order.
    """
    mapping = {}
    ctr = 0
    for graph_labels in dataset_labels:
        for label in graph_labels:
            if label not in mapping:
                mapping[label] = ctr
                ctr += 1
    return mapping


# ---------------  COMPUTE FREQUENT PATTERNS  -----------------
def tree_class_ctr(classes, dataset):
    """Count occurrences of each canonical tree within each graph class.

    :param classes: Graph class IDs aligned with the graph transaction lists.
    :param dataset: Dataset name or dataset collection, as required by this operation.
    :returns: Nested mapping ``tree -> class -> occurrence_count``.
    """
    tree_class_count = {}
    for class_, graph in zip(classes, dataset):
        for tree in graph:
            if tree not in tree_class_count:
                tree_class_count[tree] = {}
            if class_ not in tree_class_count[tree]:
                tree_class_count[tree][class_] = 0
            tree_class_count[tree][class_] += 1
    return tree_class_count


def get_invalid_trees(tree_class_count):
    """Find tree patterns that occur in more than one class.

    :param tree_class_count: Nested tree-to-class occurrence counts.
    :returns: List of tree labels with non-exclusive class membership.
    """
    trees = []
    for tree, ctr in tree_class_count.items():
        if len(ctr) != 1:
            trees.append(tree)
    return trees


def pyfpgrowth_wrapper(classwise, freq_thresholds):
    """Mine frequent tree transactions separately for each class.

    :param classwise: Mapping from class IDs to tree transaction lists.
    :param freq_thresholds: Minimum pattern frequency for each class.
    :returns: Mapping from class IDs to the frequent patterns returned by pyfpgrowth.
    """
    import pyfpgrowth

    patterns = {}
    for class_, dataset in classwise.items():
        thresh = freq_thresholds[class_]
        class_patterns = pyfpgrowth.find_frequent_patterns(dataset, thresh)
        patterns[class_] = class_patterns
    return patterns


# -------------  </COMPUTE FREQUENT PATTERNS>  ----------------


def parse_canonical_label_bak(label):
    """Parse the legacy bracketed tree representation into nodes and child-to-parent edges.

    :param label: Canonical tree string to parse or format.
    :returns: Tuple ``(node_label_map, edge_label_map)``.
    """
    q = 0
    parsed = []
    idx = 0
    tmp = ""
    etmp = ""
    stack = []
    node_label_map = {}
    edge_label_map = {}
    nid = 0
    for idx in range(len(label) - 1):
        if q == 0:
            if label[idx].isnumeric():
                tmp += label[idx]
            elif label[idx] == "<":
                # create root here
                stack.append(nid)
                node_label_map[nid] = tmp
                nid += 1
                parsed.append({"node_label": tmp})
                q = 1
                tmp = ""
            elif label[idx] == "$":
                # create root here
                stack.append(nid)
                node_label_map[nid] = tmp
                break
        if q == 1:
            if label[idx] == "$":
                # this will be used to pop parent for creating actual edges later
                stack.pop()
                pass
            if label[idx] == "<":
                q = 2
        if q == 2:
            if label[idx].isnumeric():
                tmp += label[idx]
            elif label[idx] == ",":
                q = 3
        if q == 3:
            if label[idx].isnumeric():
                etmp += label[idx]
            if label[idx] == ">":
                parsed.append({"node_label": tmp, "edge_label": etmp})
                # create node here
                par = stack[-1]
                stack.append(nid)
                node_label_map[nid] = tmp
                # create edges here
                # (to, from)
                edge_label_map[(nid, par)] = etmp
                nid += 1
                tmp = ""
                etmp = ""
                q = 1
    assert q == 1, f"Parsing failed for input {label}"
    return node_label_map, edge_label_map


def parse_canonical_label(label):
    """Parse canonical tree labels including encoded original feature IDs.

    :param label: Canonical tree string to parse or format.
    :returns: Tuple ``(node_label_map, edge_label_map, original_node_label_map)``.
    """
    q = 0
    idx = 0
    tmp = ""
    otmp = ""
    etmp = ""
    stack = []
    node_label_map = {}
    node_label_map_original = {}
    edge_label_map = {}
    nid = 0
    for idx in range(len(label) - 1):
        if q == 0:
            if label[idx] == "<":
                q = 1
        elif q == 1:
            if label[idx].isnumeric():
                tmp += label[idx]
            elif label[idx] == ",":
                q = 2
        elif q == 2:
            if label[idx].isnumeric():
                otmp += label[idx]
            elif label[idx] == ">":
                # create root here
                stack.append(nid)
                node_label_map[nid] = tmp
                node_label_map_original[nid] = otmp
                tmp = ""
                otmp = ""
                nid += 1
                q = 3
        elif q == 3:
            if label[idx] == "<":
                q = 4
            elif label[idx] == "$":
                # pop stack here
                stack.pop()
        elif q == 4:
            if label[idx].isnumeric():
                etmp += label[idx]
            elif label[idx] == ",":
                q = 5
        elif q == 5:
            if label[idx].isnumeric():
                tmp += label[idx]
            elif label[idx] == ",":
                q = 6
        elif q == 6:
            if label[idx].isnumeric():
                otmp += label[idx]
            elif label[idx] == ">":
                # create new node and edge here
                par = stack[-1]
                stack.append(nid)
                node_label_map[nid] = tmp
                node_label_map_original[nid] = otmp
                edge_label_map[(nid, par)] = etmp
                # https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/utils/convert.html#from_networkx
                # in the source, edge_index[0,:] is sources and edge_index[1,:] is destinations
                nid += 1
                tmp = ""
                otmp = ""
                etmp = ""
                q = 3
    assert q == 3, f"Parsing failed for input {label}"
    return node_label_map, edge_label_map, node_label_map_original


def get_data(
    node_label_map,
    edge_label_map,
    node_label_map_original,
    node_label_map_orig,
    edge_label_map_orig,
    node_label_map_full,
    class_,
):
    """Reconstruct a PyG tree graph from canonical IDs and inverse label dictionaries.

    :param node_label_map: Mapping between node labels and categorical IDs, or parsed node IDs and labels.
    :param edge_label_map: Parsed or original dictionary of categorical edge labels.
    :param node_label_map_original: Parsed mapping from tree node IDs to original feature tuple IDs.
    :param node_label_map_orig: Original mapping from node labels to canonical IDs.
    :param edge_label_map_orig: Parsed or original dictionary of categorical edge labels.
    :param node_label_map_full: Dictionary mapping original node feature tuples to canonical IDs.
    :param class_: Class identifier.
    :returns: PyG Data with original features, node labels, and directed labeled edges; ``class_`` is unused.
    """
    i2n = {v: k for k, v in node_label_map_orig.items()}
    i2n_o = {v: k for k, v in node_label_map_full.items()}
    i2e = {v: k for k, v in edge_label_map_orig.items()}
    # features = [list(i2n_o[int(n)]) for idx, n in node_label_map_original.items()]
    # node_labels = [i2n[int(n)] for idx, n in node_label_map.items()]
    features, node_labels = [], []
    for idx, n in node_label_map_original.items():
        feature = list(i2n_o[int(n)])
        features.append(feature)
        #
        node_label = i2n[int(node_label_map[idx])]
        node_labels.append(node_label)
    #
    edge_index_row = []
    edge_index_col = []
    edge_attr = []
    for k, v in edge_label_map.items():
        edge_index_row.append(k[0])  # k[0] is child (source)
        edge_index_col.append(k[1])  # k[1] is parent (destination)
        edge_attr.append(i2e[int(v)])
    edge_index = [edge_index_row, edge_index_col]
    data = Data(
        x=torch.tensor(features),
        edge_index=torch.tensor(edge_index),
        edge_attr=torch.tensor(edge_attr),
        node_labels=torch.tensor(node_labels),
    )
    return data

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
    if model_type == 'MLP':
        adj_syn = adj_syn - adj_syn
        torch.diagonal(adj_syn).fill_(1)
    elif model_type == 'GAT':
        if args.method in ['gcond', 'doscond']:
            if args.dataset in ['cora', 'citeseer']:
                threshold = 0.5  # Make the graph sparser as GAT does not work well on dense graph
            else:
                threshold = 0.1
        elif args.method in ['msgc']:
            threshold = args.threshold
        else:
            threshold = 0.5
    else:
        if args.method in ['gcond', 'doscond']:
            threshold = args.threshold
        elif args.method in ['msgc']:
            threshold = 0
        else:
            threshold = 0
    if verbose and args.method not in ['gcondx', 'doscondx', 'sfgc', 'geom', 'gcsntk']:
        # print('Sum:', adj_syn.sum().item())
        print('Sparsity:', adj_syn.nonzero().shape[0] / adj_syn.numel())
    # if args.method in ['sgdd']:
    #     threshold = 0.5
    if threshold > 0:
        adj_syn[adj_syn < threshold] = 0
        if verbose:
            print('Sparsity after truncating:', adj_syn.nonzero().shape[0] / adj_syn.numel())
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


def splits(data, exp):
    # customize your split here
    if hasattr(data, 'y'):
        num_classes = max(data.y) + 1
    else:
        num_classes = max(data.labels_full).item() + 1
    if not hasattr(data, 'train_mask'):
        indices = []
        for i in range(num_classes):
            data.y = data.y.reshape(-1)
            index = (data.y == i).nonzero().view(-1)
            index = index[torch.randperm(index.size(0))]
            indices.append(index)

        if exp == 'random':
            train_index = torch.cat([i[:20] for i in indices], dim=0)
            val_index = torch.cat([i[20:50] for i in indices], dim=0)
            test_index = torch.cat([i[50:] for i in indices], dim=0)
        elif exp == 'few':
            train_index = torch.cat([i[:5] for i in indices], dim=0)
            val_index = torch.cat([i[5:10] for i in indices], dim=0)
            test_index = torch.cat([i[10:] for i in indices], dim=0)
        else:
            # if fixed but no split is provided, use the default 6/2/2 split classwise
            train_index = torch.cat([i[:int(i.shape[0] * 0.6)] for i in indices], dim=0)
            val_index = torch.cat([i[int(i.shape[0] * 0.6):int(i.shape[0] * 0.8)] for i in indices], dim=0)
            test_index = torch.cat([i[int(i.shape[0] * 0.8):] for i in indices], dim=0)
            # raise NotImplementedError('Unknown split type')

        data.train_mask = index2mask(train_index, size=data.num_nodes)
        data.val_mask = index2mask(val_index, size=data.num_nodes)
        data.test_mask = index2mask(test_index, size=data.num_nodes)
    data.idx_train = data.train_mask.nonzero().view(-1)
    data.idx_val = data.val_mask.nonzero().view(-1)
    data.idx_test = data.test_mask.nonzero().view(-1)

    return data


def save_reduced(adj_syn=None, feat_syn=None, labels_syn=None, args=None):
    save_path = f'{args.save_path}/reduced_graph/{args.method}'
    if args.attack is not None and args.dataset in ['flickr']:
        save_path = f'{args.save_path}/corrupt_graph/{args.attack}/reduced_graph/{args.method}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if adj_syn is not None:
        torch.save(adj_syn,
                   f'{save_path}/adj_{args.dataset}_{args.reduction_rate}_{args.seed}.pt')
    if feat_syn is not None:
        torch.save(feat_syn,
                   f'{save_path}/feat_{args.dataset}_{args.reduction_rate}_{args.seed}.pt')
    if labels_syn is not None:
        torch.save(labels_syn,
                   f'{save_path}/label_{args.dataset}_{args.reduction_rate}_{args.seed}.pt')
    args.logger.info(f"Saved {save_path}/adj_{args.dataset}_{args.reduction_rate}_{args.seed}.pt")


def load_reduced(args, data=None):
    flag = 0
    save_path = f'{args.save_path}/reduced_graph/{args.method}'
    if args.attack is not None and args.dataset in ['flickr']:
        save_path = f'{args.save_path}/corrupt_graph/{args.attack}/reduced_graph/{args.method}'
    try:
        feat_syn = torch.load(
            f'{save_path}/feat_{args.dataset}_{args.reduction_rate}_{args.seed}.pt', map_location=args.device)
    except:
        print("find no feat, use original feature matrix instead")
        flag += 1
        if args.setting == 'trans':
            feat_syn = data.feat_full
        else:
            feat_syn = data.feat_train
    try:
        labels_syn = torch.load(
            f'{save_path}/label_{args.dataset}_{args.reduction_rate}_{args.seed}.pt', map_location=args.device)
    except:
        print("find no label, use original label matrix instead")
        flag += 1
        labels_syn = data.labels_train

    try:
        adj_syn = torch.load(
            f'{save_path}/adj_{args.dataset}_{args.reduction_rate}_{args.seed}.pt', map_location=args.device)
    except:
        print("find no adj, use identity matrix instead")
        flag += 1
        adj_syn = torch.eye(feat_syn.size(0), device=args.device)
    # args.logger.info(f"Load {save_path}/adj_{args.dataset}_{args.reduction_rate}_{args.seed}.pt")
    assert flag < 3, "no file found, please run the reduction method first"

    return adj_syn, feat_syn, labels_syn


def get_syn_data(data, args, model_type, verbose=False):
    """
    Loads or computes synthetic data for evaluation.

    Parameters
    ----------
    data : Dataset
        The dataset containing the graph data.
    model_type : str
        The type of model used for generating synthetic data.
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

    if type(adj_syn) == torch.tensor and is_sparse_tensor(adj_syn):
        adj_syn = adj_syn.to_dense()
    elif isinstance(adj_syn, torch.sparse.FloatTensor):
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

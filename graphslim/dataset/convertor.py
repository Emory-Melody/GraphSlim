# from deeprobust.graph.data import Dataset
from typing import Optional

import numpy as np
import torch
from pygsp import graphs
from scipy.sparse import coo_matrix
from torch_geometric.utils import to_undirected, to_dense_adj, remove_self_loops, add_self_loops
from torch_sparse import SparseTensor
import networkit as nk
from torch_geometric.data import Data, HeteroData
import dgl


def from_dgl(g, name, hetero=True):
    if g.is_homogeneous:
        data = Data()
        data.edge_index = torch.stack(g.edges(), dim=0)

        for attr, value in g.ndata.items():
            data[attr] = value
        for attr, value in g.edata.items():
            data[attr] = value

        return data

    data = HeteroData()
    data.name = name

    for node_type in g.ntypes:
        for attr, value in g.nodes[node_type].data.items():
            data[node_type][attr] = value

    for edge_type in g.canonical_etypes:
        row, col = g.edges(form="uv", etype=edge_type)
        data[edge_type].edge_index = torch.stack([row, col], dim=0)
        for attr, value in g.edge_attr_schemes(edge_type).items():
            data[edge_type][attr] = value

    data_out = Data()
    if not hetero:
        edge_index_list = []
        for edge_type in g.canonical_etypes:
            edge_index_list.append(data[edge_type].edge_index)
        data_out.edge_index = add_self_loops(torch.cat(edge_index_list, dim=1))[0]
        data_out.x = data.node_stores[0]['feature']  # Features for each node
        # Assigning labels to data_out
        data_out.y = data.node_stores[0]['label']  # Labels for each node

        # Assuming the train, validation, and test masks are also in node_stores[0]
        #data_out.train_mask = data.node_stores[0]['train_mask']  # Training mask
        #data_out.val_mask = data.node_stores[0].get('val_mask', None)  # Optional: Validation mask (if exists)
        #data_out.test_mask = data.node_stores[0]['test_mask']  # Test mask

    data_out.num_nodes = len(data_out.x)
    data_out.num_classes = max(data_out.y).item() + 1

    return data_out
def pyg2gsp(edge_index):
    G = graphs.Graph(W=to_dense_adj(to_undirected(edge_index))[0])
    return G


def csr2ei(adjacency_matrix_csr):
    adjacency_matrix_coo = adjacency_matrix_csr.tocoo()
    # Convert numpy arrays directly to a tensor
    edge_index = torch.tensor(np.vstack([adjacency_matrix_coo.row, adjacency_matrix_coo.col]), dtype=torch.long)
    return edge_index


def ei2csr(edge_index, num_nodes):
    edge_index = edge_index.numpy()
    scoo = coo_matrix((np.ones_like(edge_index[0]), (edge_index[0], edge_index[1])), shape=(num_nodes, num_nodes))
    adjacency_matrix_csr = scoo.tocsr()
    return adjacency_matrix_csr


def dense2sparsetensor(mat: torch.Tensor, has_value: bool = True):
    if mat.dim() > 2:
        index = mat.abs().sum([i for i in range(2, mat.dim())]).nonzero()
    else:
        index = mat.nonzero()
    index = index.t()

    row = index[0]
    col = index[1]

    value: Optional[torch.Tensor] = None
    if has_value:
        value = mat[row, col]

    return SparseTensor(
        row=row,
        rowptr=None,
        col=col,
        value=value,
        sparse_sizes=(mat.size(0), mat.size(1)),
        is_sorted=True,
        trust_data=True,
    )


def networkit_to_pyg(graph):
    # Extract edges from Networkit graph
    edges = list(graph.edges())
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    # Check if the graph is weighted
    if graph.isWeighted():
        edge_attr = torch.tensor([graph.weight(u, v) for u, v in edges], dtype=torch.float)
    else:
        edge_attr = None

    pyg_graph = Data(edge_index=edge_index, edge_attr=edge_attr)
    return pyg_graph


def pyg_to_networkit(pyg_graph):
    # Create an empty Networkit graph
    # if hasattr(pyg_graph, 'edge_attr') and pyg_graph.edge_attr is not None:
    #     graph = nk.Graph(weighted=True, directed=False)
    # else:
    #     graph = nk.Graph(weighted=False, directed=False)

    # Add edges to the Networkit graph
    edge_index = pyg_graph.edge_index.numpy()
    if hasattr(pyg_graph, 'edge_attr') and pyg_graph.edge_attr is not None:
        edge_attr = pyg_graph.edge_attr.numpy()
        graph = nk.GraphFromCoo(inputData=(edge_attr, (edge_index[0], edge_index[1])), n=pyg_graph.num_nodes,
                                weighted=True, directed=False)
    else:
        graph = nk.GraphFromCoo(inputData=((edge_index[0], edge_index[1])), n=pyg_graph.num_nodes,
                                weighted=False, directed=False)

    graph.indexEdges()

    return graph


def loadSparseGraph(dataset_name):
    """Load original graph from file from paper
    CHEN Y, YE H, VEDULA S, et al. Demystifying graph sparsification algorithms in graph properties preservation[M/OL].

    GraphSlim package only supports undirected graph and we do not distinguish the weighted and unweighted
    pyg->nt->save sparsified nt->pyg->evaluation

    Args:
        dataset_name (str): dataset name
        config (dict): config loaded from json
        undirected_only (bool, optional): Set to True to override graph directness in config file and load undirected graph only.
                                          Defaults to False. This is used for sparsifiers that only support undirected graph.

    Returns:
        nk graph: original graph
    """

    # else:
    #     if config[dataset_name]["directed"] and config[dataset_name]["weighted"]:
    #         originalGraph = nk.readGraph(f"../data/{dataset_name}/raw/dw.wel", nk.Format.EdgeListSpaceZero,
    #                                      directed=True)
    #     elif config[dataset_name]["directed"]:
    #         originalGraph = nk.readGraph(f"../data/{dataset_name}/raw/duw.el", nk.Format.EdgeListSpaceZero,
    #                                      directed=True)
    #     elif config[dataset_name]["weighted"]:
    #         originalGraph = nk.readGraph(f"../data/{dataset_name}/raw/udw.wel", nk.Format.EdgeListSpaceZero,
    #                                      directed=False)
    #     else:
    #         originalGraph = nk.readGraph(f"../data/{dataset_name}/raw/uduw.el", nk.Format.EdgeListSpaceZero,
    #                                      directed=False)

    nk.overview(originalGraph)
    nk.graph.Graph.indexEdges(originalGraph)
    return graph

# class Pyg2Dpr(Dataset):
# def __init__(self, pyg_data, **kwargs):
#         try:
#             splits = pyg_data.get_idx_split()
#         except:
#             pass
#
#         dataset_name = pyg_data.name
#         pyg_data = pyg_data[0]
#         n = pyg_data.num_nodes
#
#         if dataset_name == 'ogbn-arxiv':  # symmetrization
#             pyg_data.edge_index = to_undirected(pyg_data.edge_index, pyg_data.num_nodes)
#
#         self.adj = sp.csr_matrix((np.ones(pyg_data.edge_index.shape[1]),
#                                   (pyg_data.edge_index[0], pyg_data.edge_index[1])), shape=(n, n))
#
#         self.features = pyg_data.x.numpy()
#         self.labels = pyg_data.y.numpy()
#
#         if len(self.labels.shape) == 2 and self.labels.shape[1] == 1:
#             self.labels = self.labels.reshape(-1)  # ogb-arxiv needs to reshape
#
#         if hasattr(pyg_data, 'train_mask'):
#             # for fixed split
#             self.idx_train = mask_to_index(pyg_data.train_mask, n)
#             self.idx_val = mask_to_index(pyg_data.val_mask, n)
#             self.idx_test = mask_to_index(pyg_data.test_mask, n)
#             self.name = 'Pyg2Dpr'
#         else:
#             try:
#                 # for ogb
#                 self.idx_train = splits['train']
#                 self.idx_val = splits['valid']
#                 self.idx_test = splits['test']
#                 self.name = 'Pyg2Dpr'
#             except:
#                 # for other datasets
#                 self.idx_train, self.idx_val, self.idx_test = get_train_val_test(
#                     nnodes=n, val_size=0.1, test_size=0.8, stratify=self.labels)
# class Dpr2Pyg(InMemoryDataset):
#
#     def __init__(self, dpr_data, transform=None, **kwargs):
#         root = 'data/'  # dummy root; does not mean anything
#         self.dpr_data = dpr_data
#         super(Dpr2Pyg, self).__init__(root, transform)
#         pyg_data = self.process()
#         self.data, self.slices = self.collate([pyg_data])
#         self.transform = transform
#
#     def process(self):
#         dpr_data = self.dpr_data
#
#         edge_index = torch.LongTensor(dpr_data.adj.nonzero())
#         # if type(dpr_data.adj) == torch.Tensor:
#         #     adj_selfloop = dpr_data.adj + torch.eye(dpr_data.adj.shape[0]).cuda()
#         #     edge_index_selfloop = adj_selfloop.nonzero().T
#         #     edge_index = edge_index_selfloop
#         #     edge_weight = adj_selfloop[edge_index_selfloop[0], edge_index_selfloop[1]]
#         # else:
#         #     adj_selfloop = dpr_data.adj + sp.eye(dpr_data.adj.shape[0])
#         #     edge_index = torch.LongTensor(adj_selfloop.nonzero()).cuda()
#         #     edge_weight = torch.FloatTensor(adj_selfloop[adj_selfloop.nonzero()]).cuda()
#
#         # by default, the features in pyg data is dense
#         if scipy.sparse.issparse(dpr_data.features):
#             x = torch.FloatTensor(dpr_data.features.todense()).float()
#         else:
#             x = torch.FloatTensor(dpr_data.features).float()
#         y = torch.LongTensor(dpr_data.labels)
#
#         # try:
#         #     x = torch.FloatTensor(dpr_data.features.cpu()).float().cuda()
#         # except:
#         #     x = torch.FloatTensor(dpr_data.features).float().cuda()
#         # try:
#         #     y = torch.LongTensor(dpr_data.labels.cpu()).cuda()
#         # except:
#         #     y = dpr_data.labels
#
#         data = Data(x=x, edge_index=edge_index, y=y)
#         data.train_mask = None
#         data.val_mask = None
#         data.test_mask = None
#         return data
#
#     def get(self, idx):
#         data = self.data.__class__()
#
#         if hasattr(self.data, '__num_nodes__'):
#             data.num_nodes = self.data.__num_nodes__[idx]
#
#         for key in self.data.keys:
#             item, slices = self.data[key], self.slices[key]
#             s = list(repeat(slice(None), item.dim()))
#             s[self.data.__cat_dim__(key, item)] = slice(slices[idx],
#                                                         slices[idx + 1])
#             data[key] = item[s]
#         return data
#
#     @property
#     def raw_file_names(self):
#         return ['some_file_1', 'some_file_2', ...]
#
#     @property
#     def processed_file_names(self):
#         return ['data.pt']
#
#     def _download(self):
#         pass
# class Data2Pyg:
#
#     def __init__(self, data, device='cuda', transform=None, **kwargs):
#         self.data_train = Dpr2Pyg(data.data_train, transform=transform)[0].to(device)
#         self.data_val = Dpr2Pyg(data.data_val, transform=transform)[0].to(device)
#         self.data_test = Dpr2Pyg(data.data_test, transform=transform)[0].to(device)
#         self.nclass = data.nclass
#         self.nfeat = data.nfeat
#         self.class_dict = None
#
#     def retrieve_class(self, c, num=256):
#         if self.class_dict is None:
#             self.class_dict = {}
#             for i in range(self.nclass):
#                 self.class_dict['class_%s' % i] = (self.data_train.y == i).cpu().numpy()
#         idx = np.arange(len(self.data_train.y))
#         idx = idx[self.class_dict['class_%s' % c]]
#         return np.random.permutation(idx)[:num]

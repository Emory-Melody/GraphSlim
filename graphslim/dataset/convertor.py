# from deeprobust.graph.data import Dataset
from typing import Optional

import numpy as np
import torch
from pygsp import graphs
from scipy.sparse import coo_matrix
from torch_geometric.utils import to_undirected, to_dense_adj
from torch_sparse import SparseTensor


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

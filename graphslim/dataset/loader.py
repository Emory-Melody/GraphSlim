import os.path as osp

import numpy as np
import torch
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.datasets import Planetoid, Coauthor, CitationFull, Amazon, Flickr, Reddit
from torch_geometric.loader import NeighborSampler
from torch_sparse import SparseTensor

from graphslim.dataset.convertor import ei2csr, csr2ei
from graphslim.dataset.utils import splits


def get_dataset(name, args):
    path = osp.join('../../data')
    # Create a dictionary that maps standard names to normalized names
    standard_names = ['flickr', 'reddit', 'dblp', 'cora_ml', 'physics', 'cs', 'cora', 'citeseer', 'pubmed', 'photo',
                      'computers', 'ogbn-products', 'ogbn-proteins', 'ogbn-papers100m', 'ogbn-arxiv']
    normalized_names = [name.lower().replace('-', '').replace('_', '') for name in standard_names]
    name_dict = dict(zip(normalized_names, standard_names))

    # Normalize the name input
    normalized_name = name.lower().replace('-', '').replace('_', '')

    if normalized_name in name_dict:
        name = name_dict[normalized_name]  # Transfer to standard name
        if name in ['flickr']:
            dataset = Flickr(root=path + '/flickr')
        elif name in ['reddit']:
            dataset = Reddit(root=path + '/reddit')
        elif name in ['dblp', 'cora_ml']:
            dataset = CitationFull(root=path, name=name)
        elif name in ['physics', 'cs']:
            dataset = Coauthor(root=path, name=name)
        elif name in ['cora', 'citeseer', 'pubmed']:
            dataset = Planetoid(root=path, name=name)
        elif name in ['photo', 'computers']:
            dataset = Amazon(root=path, name=name)
        elif name in ['ogbn-products', 'ogbn-proteins', 'ogbn-papers100m', 'ogbn-arxiv']:
            dataset = PygNodePropPredDataset(name, root=path)
        else:
            raise ValueError("Dataset name not recognized.")
    data = dataset[0]
    data = splits(data, args.split)

    data = TransAndInd(data)
    return data


class TransAndInd:

    def __init__(self, data):
        self.class_dict = None  # sample the training data per class when initializing synthetic graph
        self.samplers = None
        self.class_dict2 = None  # sample from the same class when training
        self.sparse_adj = None
        self.adj_full = None
        self.feat_full = None
        self.labels_full = None
        self.train_mask, self.val_mask, self.test_mask = data.train_mask, data.val_mask, data.test_mask
        self.pyg_saint(data)
        self.idx_train, self.idx_val, self.idx_test = data.idx_train, data.idx_val, data.idx_test
        self.nclass = max(self.labels_full) + 1

        self.adj_train = self.adj_full[np.ix_(self.idx_train, self.idx_train)]
        self.adj_val = self.adj_full[np.ix_(self.idx_val, self.idx_val)]
        self.adj_test = self.adj_full[np.ix_(self.idx_test, self.idx_test)]
        # if inductive
        # print('size of adj_train:', self.adj_train.shape)
        # print('#edges in adj_train:', self.adj_train.sum())

        self.labels_train = self.labels_full[self.idx_train]
        self.labels_val = self.labels_full[self.idx_val]
        self.labels_test = self.labels_full[self.idx_test]

        self.feat_train = self.feat_full[self.idx_train]
        self.feat_val = self.feat_full[self.idx_val]
        self.feat_test = self.feat_full[self.idx_test]

    def pyg_saint(self, data):
        # reference type
        # pyg format use x,y,edge_index
        if hasattr(data, 'x'):
            self.x = data.x
            self.y = data.y
            self.feat_full = data.x
            self.labels_full = data.y
            self.adj_full = ei2csr(data.edge_index, data.x.shape[0])
            self.edge_index = data.edge_index
            self.sparse_adj = SparseTensor.from_edge_index(data.edge_index)
        # saint format use feat,labels,adj
        elif hasattr(data, 'feat_full'):
            self.adj_full = data.adj_full
            self.feat_full = data.feat_full
            self.labels_full = data.labels_full
            self.edge_index = csr2ei(data.adj_full)
            self.sparse_adj = SparseTensor.from_edge_index(data.edge_index)
            self.x = data.feat_full
            self.y = data.labels_full
        return data

    def retrieve_class(self, c, num=256):
        # change the initialization strategy here
        if self.class_dict is None:
            self.class_dict = {}
            for i in range(self.nclass):
                self.class_dict['class_%s' % i] = (self.labels_train == i)
        idx = np.arange(len(self.labels_train))
        idx = idx[self.class_dict['class_%s' % c]]
        return np.random.permutation(idx)[:num]

    def retrieve_class_sampler(self, c, adj, args, num=256):
        if self.class_dict2 is None:
            self.class_dict2 = {}
            for i in range(self.nclass):
                if args.setting == 'ind':
                    idx = self.idx_train[self.labels_train == i]
                else:
                    idx = np.arange(len(self.labels_train))[self.labels_train == i]
                self.class_dict2[i] = idx

        if args.nlayers == 1:
            sizes = [15]
        if args.nlayers == 2:
            sizes = [10, 5]
            # sizes = [-1, -1]
        if args.nlayers == 3:
            sizes = [15, 10, 5]
        if args.nlayers == 4:
            sizes = [15, 10, 5, 5]
        if args.nlayers == 5:
            sizes = [15, 10, 5, 5, 5]

        if self.samplers is None:
            self.samplers = []
            for i in range(self.nclass):
                node_idx = torch.LongTensor(self.class_dict2[i])
                self.samplers.append(NeighborSampler(adj,
                                                     node_idx=node_idx,
                                                     sizes=sizes, batch_size=num,
                                                     num_workers=8, return_e_id=False,
                                                     num_nodes=adj.size(0),
                                                     shuffle=True))
        batch = np.random.permutation(self.class_dict2[c])[:num]
        out = self.samplers[c].sample(batch)
        return out

    def retrieve_class_multi_sampler(self, c, adj, transductive, num=256, args=None):
        if self.class_dict2 is None:
            self.class_dict2 = {}
            for i in range(self.nclass):
                if transductive:
                    idx = self.idx_train[self.labels_train == i]
                else:
                    idx = np.arange(len(self.labels_train))[self.labels_train == i]
                self.class_dict2[i] = idx

        if self.samplers is None:
            self.samplers = []
            for l in range(args.nlayers):
                layer_samplers = []
                if l == 0:
                    sizes = [15]
                elif l == 1:
                    sizes = [10, 5]
                else:
                    sizes = [10, 5, 5]
                for i in range(self.nclass):
                    node_idx = torch.LongTensor(self.class_dict2[i])
                    layer_samplers.append(NeighborSampler(adj,
                                                          node_idx=node_idx,
                                                          sizes=sizes, batch_size=num,
                                                          num_workers=12, return_e_id=False,
                                                          num_nodes=adj.size(0),
                                                          shuffle=True))
                self.samplers.append(layer_samplers)
        batch = np.random.permutation(self.class_dict2[c])[:num]
        out = self.samplers[args.nlayers - 1][c].sample(batch)
        return out

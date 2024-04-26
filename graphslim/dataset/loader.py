import os.path as osp

import numpy as np
import torch
import torch.nn as nn
from ogb.nodeproppred import PygNodePropPredDataset
from sklearn.preprocessing import StandardScaler
from torch_geometric.datasets import Planetoid, Coauthor, CitationFull, Amazon, Flickr, Reddit
from torch_geometric.loader import NeighborSampler
from torch_geometric.utils import to_undirected
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
    if name in ['ogbn-arxiv']:
        data.edge_index = to_undirected(data.edge_index, data.num_nodes)
        feat_train = data.x[data.idx_train]
        scaler = StandardScaler()
        scaler.fit(feat_train)
        data.feat = scaler.transform(data.x)

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
        out = self.samplers[c].sample(batch.astype(np.int64))
        return out

    # def retrieve_class_multi_sampler(self, c, adj, transductive, num=256, args=None):
    #     if self.class_dict2 is None:
    #         self.class_dict2 = {}
    #         for i in range(self.nclass):
    #             if transductive:
    #                 idx = self.idx_train[self.labels_train == i]
    #             else:
    #                 idx = np.arange(len(self.labels_train))[self.labels_train == i]
    #             self.class_dict2[i] = idx
    #
    #     if self.samplers is None:
    #         self.samplers = []
    #         for l in range(args.nlayers):
    #             layer_samplers = []
    #             if l == 0:
    #                 sizes = [15]
    #             elif l == 1:
    #                 sizes = [10, 5]
    #             else:
    #                 sizes = [10, 5, 5]
    #             for i in range(self.nclass):
    #                 node_idx = torch.LongTensor(self.class_dict2[i])
    #                 layer_samplers.append(NeighborSampler(adj,
    #                                                       node_idx=node_idx,
    #                                                       sizes=sizes, batch_size=num,
    #                                                       num_workers=12, return_e_id=False,
    #                                                       num_nodes=adj.size(0),
    #                                                       shuffle=True))
    #             self.samplers.append(layer_samplers)
    #     batch = np.random.permutation(self.class_dict2[c])[:num]
    #     out = self.samplers[args.nlayers - 1][c].sample(batch)
    #     return out


from sklearn.cluster import KMeans


class FlickrDataLoader(nn.Module):
    def __init__(self, name='Flickr', split='train', batch_size=3000, split_method='kmeans'):
        super(FlickrDataLoader, self).__init__()
        if name == 'flickr':
            from torch_geometric.datasets import Flickr as DataSet
        elif name == 'reddit':
            from torch_geometric.datasets import Reddit as DataSet
        path = osp.join('../../data')

        Dataset = DataSet(root=path + f'/{name}')
        self.n, self.dim = Dataset[0].x.shape
        mask = split + '_mask'
        features = Dataset[0].x
        labels = Dataset[0].y
        edge_index = Dataset[0].edge_index

        values = torch.ones(edge_index.shape[1])
        Adj = torch.sparse_coo_tensor(edge_index, values, torch.Size([self.n, self.n]))
        sparse_eye = torch.sparse_coo_tensor(torch.arange(self.n).repeat(2, 1), torch.ones(self.n), (self.n, self.n))
        self.Adj = Adj + sparse_eye
        features = self.normalize_data(features)
        # features      = self.GCF(self.Adj, features, k=2)
        self.split_idx = torch.where(Dataset[0][mask])[0]
        self.n_split = len(self.split_idx)
        self.k = torch.round(torch.tensor(self.n_split / batch_size)).to(torch.int)

        # Masked Adjacency Matrix
        optor_index = torch.cat(
            (self.split_idx.reshape(1, self.n_split), torch.tensor(range(self.n_split)).reshape(1, self.n_split)),
            dim=0)
        optor_value = torch.ones(self.n_split)
        optor_shape = torch.Size([self.n, self.n_split])
        optor = torch.sparse_coo_tensor(optor_index, optor_value, optor_shape)
        self.Adj_mask = torch.sparse.mm(torch.sparse.mm(optor.t(), self.Adj), optor)
        self.split_feat = features[self.split_idx]
        # self.split_feat   = self.GCF(self.Adj_mask, self.split_feat, k = 2)

        self.split_label = labels[self.split_idx]
        self.split_method = split_method
        self.n_classes = Dataset.num_classes

    def normalize_data(self, data):
        """
        normalize data
        parameters:
            data: torch.Tensor, data need to be normalized
        return:
            torch.Tensor, normalized data
        """
        mean = data.mean(dim=0)  # 沿第0维（即样本维）求均值
        std = data.std(dim=0)  # 沿第0维（即样本维）求标准差
        std[std == 0] = 1  # 将std中的0值替换为1，以避免分母为0的情况
        normalized_data = (data - mean) / std  # 对数据进行归一化处理
        return normalized_data

    def GCF(self, adj, x, k=2):
        """
        Graph convolution filter
        parameters:
            adj: torch.Tensor, adjacency matrix, must be self-looped
            x: torch.Tensor, features
            k: int, number of hops
        return:
            torch.Tensor, filtered features
        """
        n = adj.shape[0]
        ind = torch.tensor(range(n)).repeat(2, 1)
        adj = adj + torch.sparse_coo_tensor(ind, torch.ones(n), (n, n))

        D = torch.pow(torch.sparse.sum(adj, 1).to_dense(), -0.5)
        D = torch.sparse_coo_tensor(ind, D, (n, n))

        filter = torch.sparse.mm(torch.sparse.mm(D, adj), D)
        for i in range(k):
            x = torch.sparse.mm(filter, x)
        return x

    def properties(self):
        return self.k, self.n_split, self.n_classes, self.dim, self.n

    def split_batch(self):
        """
        split data into batches
        parameters:
            split_method: str, method to split data, default is 'kmeans'
        """
        if self.split_method == 'kmeans':
            kmeans = KMeans(n_clusters=self.k.item())
            kmeans.fit(self.split_feat.numpy())
            self.batch_labels = kmeans.predict(self.split_feat.numpy())

    def getitem(self, idx):
        """
        对于给定的 idx 输出对应的 node_features, labels, sub Ajacency matrix
        """
        # idx   = [idx]
        n_idx = len(idx)
        idx_raw = self.split_idx[idx]
        feat = self.split_feat[idx]
        label = self.split_label[idx]
        # idx   = idx.tolist()

        optor_index = torch.cat((idx_raw.reshape(1, n_idx), torch.tensor(range(n_idx)).reshape(1, n_idx)), dim=0)
        optor_value = torch.ones(n_idx)
        optor_shape = torch.Size([self.n, n_idx])
        optor = torch.sparse_coo_tensor(optor_index, optor_value, optor_shape)
        sub_A = torch.sparse.mm(torch.sparse.mm(optor.t(), self.Adj), optor)

        return (feat, label, sub_A)

    def get_batch(self, i):
        idx = torch.where(torch.tensor(self.batch_labels) == i)[0]
        batch_i = self.getitem(idx)
        return batch_i


class OgbDataLoader(nn.Module):
    def __init__(self, dataset_name='ogbn-arxiv', split='train', batch_size=5000, split_method='kmeans'):
        super(OgbDataLoader, self).__init__()

        path = osp.join('../../data')
        Dataset = PygNodePropPredDataset(dataset_name, root=path)
        self.n, self.dim = Dataset.graph['node_feat'].shape
        split_set = Dataset.get_idx_split()
        graph, labels = Dataset[0]
        features = torch.tensor(graph['node_feat'])
        edge_index = torch.tensor(graph['edge_index'])
        values = torch.ones(edge_index.shape[1])
        Adj = torch.sparse_coo_tensor(edge_index, values, torch.Size([self.n, self.n]))
        sparse_eye = torch.sparse_coo_tensor(torch.arange(self.n).repeat(2, 1), torch.ones(self.n), (self.n, self.n))
        self.Adj = Adj + sparse_eye

        features = self.normalize_data(features)
        features = self.GCF(self.Adj, features, k=1)
        labels = torch.tensor(labels)

        self.split_idx = torch.tensor(split_set[split])
        self.n_split = len(self.split_idx)
        self.k = torch.round(torch.tensor(self.n_split / batch_size)).to(torch.int)
        self.split_feat = features[self.split_idx]
        self.split_label = labels[self.split_idx]

        self.split_method = split_method
        self.n_classes = Dataset.num_classes

    def normalize_data(self, data):
        """
        normalize data
        parameters:
            data: torch.Tensor, data need to be normalized
        return:
            torch.Tensor, normalized data
        """
        mean = data.mean(dim=0)
        std = data.std(dim=0)
        std[std == 0] = 1
        normalized_data = (data - mean) / std
        return normalized_data

    def GCF(self, adj, x, k=2):
        """
        Graph convolution filter
        parameters:
            adj: torch.Tensor, adjacency matrix, must be self-looped
            x: torch.Tensor, features
            k: int, number of hops
        return:
            torch.Tensor, filtered features
        """
        n = adj.shape[0]
        ind = torch.tensor(range(n)).repeat(2, 1)
        adj = adj + torch.sparse_coo_tensor(ind, torch.ones(n), (n, n))

        D = torch.pow(torch.sparse.sum(adj, 1).to_dense(), -0.5)
        D = torch.sparse_coo_tensor(ind, D, (n, n))

        filter = torch.sparse.mm(torch.sparse.mm(D, adj), D)
        for i in range(k):
            x = torch.sparse.mm(filter, x)
        return x

    def properties(self):
        return self.k, self.n_split, self.n_classes, self.dim, self.n

    def split_batch(self):
        """
        split data into batches
        parameters:
            split_method: str, method to split data, default is 'kmeans'
        """
        if self.split_method == 'kmeans':
            kmeans = KMeans(n_clusters=self.k)
            kmeans.fit(self.split_feat.numpy())
            self.batch_labels = kmeans.predict(self.split_feat.numpy())

        # save batch labels
        torch.save(self.batch_labels, './{}_{}_batch_labels.pt'.format(self.split_method, self.k))

    def getitem(self, idx):
        # idx   = [idx]
        n_idx = len(idx)
        idx_raw = self.split_idx[idx]
        feat = self.split_feat[idx]
        label = self.split_label[idx]
        # idx   = idx.tolist()

        optor_index = torch.cat((torch.tensor(idx_raw).reshape(1, n_idx), torch.tensor(range(n_idx)).reshape(1, n_idx)),
                                dim=0)
        optor_value = torch.ones(n_idx)
        optor_shape = torch.Size([self.n, n_idx])

        optor = torch.sparse_coo_tensor(optor_index, optor_value, optor_shape)
        sub_A = torch.sparse.mm(torch.sparse.mm(optor.t(), self.Adj), optor)

        return (feat, label, sub_A)

    def get_batch(self, i):
        idx = torch.where(torch.tensor(self.batch_labels) == i)[0]
        batch_i = self.getitem(idx)
        return batch_i

import json

import numpy as np
import scipy as sp
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.loader import NeighborSampler


class DataGraphSAINT:
    '''datasets used in GraphSAINT paper'''

    def __init__(self, dataset, **kwargs):
        dataset_str = '../../data/' + dataset + '/'
        adj_full = sp.sparse.load_npz(dataset_str + 'adj_full.npz')
        self.nnodes = adj_full.shape[0]
        if dataset == 'ogbn-arxiv':
            adj_full = adj_full + adj_full.T
            adj_full[adj_full > 1] = 1

        role = json.load(open(dataset_str + 'role.json', 'r'))
        idx_train = role['tr']
        idx_test = role['te']
        idx_val = role['va']

        # if 'label_rate' in kwargs:
        #     label_rate = kwargs['label_rate']
        #     if label_rate < 1:
        #         idx_train = idx_train[:int(label_rate * len(idx_train))]
        #
        # self.adj_train = adj_full[np.ix_(idx_train, idx_train)]
        # self.adj_val = adj_full[np.ix_(idx_val, idx_val)]
        # self.adj_test = adj_full[np.ix_(idx_test, idx_test)]

        feat = np.load(dataset_str + 'feats.npy')
        # ---- normalize feat ----
        feat_train = feat[idx_train]
        scaler = StandardScaler()
        scaler.fit(feat_train)
        feat = scaler.transform(feat)

        self.feat_train = feat[idx_train]
        self.feat_val = feat[idx_val]
        self.feat_test = feat[idx_test]

        class_map = json.load(open(dataset_str + 'class_map.json', 'r'))
        labels = self.process_labels(class_map)

        self.labels_train = labels[idx_train]
        self.labels_val = labels[idx_val]
        self.labels_test = labels[idx_test]

        self.data_full = GraphData(adj_full, feat, labels, idx_train, idx_val, idx_test)
        self.class_dict = None
        self.class_dict2 = None

        self.adj_full = adj_full
        self.feat_full = feat
        self.labels_full = labels
        self.idx_train = np.array(idx_train)
        self.idx_val = np.array(idx_val)
        self.idx_test = np.array(idx_test)
        self.samplers = None

    def process_labels(self, class_map):
        """
        setup vertex property map for output classests
        """
        num_vertices = self.nnodes
        if isinstance(list(class_map.values())[0], list):
            num_classes = len(list(class_map.values())[0])
            self.nclass = num_classes
            class_arr = np.zeros((num_vertices, num_classes))
            for k, v in class_map.items():
                class_arr[int(k)] = v
        else:
            class_arr = np.zeros(num_vertices, dtype=np.int)
            for k, v in class_map.items():
                class_arr[int(k)] = v
            class_arr = class_arr - class_arr.min()
            self.nclass = max(class_arr) + 1
        return class_arr

    def retrieve_class(self, c, num=256):
        if self.class_dict is None:
            self.class_dict = {}
            for i in range(self.nclass):
                self.class_dict['class_%s' % i] = (self.labels_train == i)
        idx = np.arange(len(self.labels_train))
        idx = idx[self.class_dict['class_%s' % c]]
        return np.random.permutation(idx)[:num]

    def retrieve_class_sampler(self, c, adj, transductive, num=256, args=None):
        if args.nlayers == 1:
            sizes = [30]
        if args.nlayers == 2:
            if args.dataset in ['reddit', 'flickr']:
                if args.option == 0:
                    sizes = [15, 8]
                if args.option == 1:
                    sizes = [20, 10]
                if args.option == 2:
                    sizes = [25, 10]
            else:
                sizes = [10, 5]

        if self.class_dict2 is None:
            # print(f'sampling size per hop{sizes}')
            self.class_dict2 = {}
            for i in range(self.nclass):
                if transductive:
                    idx_train = np.array(self.idx_train)
                    idx = idx_train[self.labels_train == i]
                else:
                    idx = np.arange(len(self.labels_train))[self.labels_train == i]
                self.class_dict2[i] = idx

        if self.samplers is None:
            self.samplers = []
            for i in range(self.nclass):
                node_idx = torch.LongTensor(self.class_dict2[i])
                if len(node_idx) == 0:
                    continue

                self.samplers.append(NeighborSampler(adj,
                                                     node_idx=node_idx,
                                                     sizes=sizes, batch_size=num,
                                                     num_workers=8, return_e_id=False,
                                                     num_nodes=adj.size(0),
                                                     shuffle=True))
        batch = np.random.permutation(self.class_dict2[c])[:num]
        out = self.samplers[c].sample(batch)
        return out


class GraphData:

    def __init__(self, adj, features, labels, idx_train, idx_val, idx_test):
        self.adj = adj
        self.features = features
        self.labels = labels
        self.idx_train = idx_train
        self.idx_val = idx_val
        self.idx_test = idx_test

from utils import *
import numpy as np
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from utils import index_to_mask
from torch_geometric.datasets import Planetoid, Coauthor, CitationFull
from pygsp import graphs
from torch_geometric.utils import to_undirected, to_dense_adj
from torch_sparse import SparseTensor
from sklearn.preprocessing import StandardScaler
import torch_geometric.transforms as T
from torchvision import datasets, transforms
from deeprobust.graph.data import Dataset
from torch_geometric.loader import NeighborSampler


def get_mnist(data_path):
    channel = 1
    im_size = (28, 28)
    num_classes = 10
    mean = [0.1307]
    std = [0.3081]
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    dst_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)  # no        augmentation
    dst_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)
    class_names = [str(c) for c in range(num_classes)]

    labels = []
    feat = []
    for x, y in dst_train:
        feat.append(x.view(1, -1))
        labels.append(y)
    feat = torch.cat(feat, axis=0).numpy()
    from dataset.utils_graphsaint import GraphData
    adj = sp.eye(len(feat))
    idx = np.arange(len(feat))
    dpr_data = GraphData(adj - adj, feat, labels, idx, idx, idx)
    from deeprobust.graph.data import Dpr2Pyg
    return Dpr2Pyg(dpr_data)


def get_dataset(name, normalize_features=False, transform=None, if_dpr=True):
    path = osp.join('../data')
    if name in ['cora', 'citeseer', 'pubmed']:
        dataset = Planetoid(path, name)
    elif name in ['ogbn-arxiv']:
        dataset = PygNodePropPredDataset(name='ogbn-arxiv')
    else:
        raise NotImplementedError

    if transform is not None and normalize_features:
        dataset.transform = T.Compose([T.NormalizeFeatures(), transform])
    elif normalize_features:
        dataset.transform = T.NormalizeFeatures()
    elif transform is not None:
        dataset.transform = transform

    dpr_data = Pyg2Dpr(dataset)
    if name in ['ogbn-arxiv']:
        # the features are different from the features provided by GraphSAINT
        # normalize features, following graphsaint
        feat, idx_train = dpr_data.features, dpr_data.idx_train
        feat_train = feat[idx_train]
        scaler = StandardScaler()
        scaler.fit(feat_train)
        feat = scaler.transform(feat)
        dpr_data.features = feat

    return dpr_data


class Pyg2Dpr(Dataset):
    def __init__(self, pyg_data, **kwargs):
        try:
            splits = pyg_data.get_idx_split()
        except:
            pass

        dataset_name = pyg_data.name
        pyg_data = pyg_data[0]
        n = pyg_data.num_nodes

        if dataset_name == 'ogbn-arxiv':  # symmetrization
            pyg_data.edge_index = to_undirected(pyg_data.edge_index, pyg_data.num_nodes)

        self.adj = sp.csr_matrix((np.ones(pyg_data.edge_index.shape[1]),
                                  (pyg_data.edge_index[0], pyg_data.edge_index[1])), shape=(n, n))

        self.features = pyg_data.x.numpy()
        self.labels = pyg_data.y.numpy()

        if len(self.labels.shape) == 2 and self.labels.shape[1] == 1:
            self.labels = self.labels.reshape(-1)  # ogb-arxiv needs to reshape

        if hasattr(pyg_data, 'train_mask'):
            # for fixed split
            self.idx_train = mask_to_index(pyg_data.train_mask, n)
            self.idx_val = mask_to_index(pyg_data.val_mask, n)
            self.idx_test = mask_to_index(pyg_data.test_mask, n)
            self.name = 'Pyg2Dpr'
        else:
            try:
                # for ogb
                self.idx_train = splits['train']
                self.idx_val = splits['valid']
                self.idx_test = splits['test']
                self.name = 'Pyg2Dpr'
            except:
                # for other datasets
                self.idx_train, self.idx_val, self.idx_test = get_train_val_test(
                    nnodes=n, val_size=0.1, test_size=0.8, stratify=self.labels)


class TransAndInd:
    # transductive setting to inductive setting

    def __init__(self, dpr_data, keep_ratio):
        idx_train, idx_val, idx_test = dpr_data.idx_train, dpr_data.idx_val, dpr_data.idx_test
        adj, features, labels = dpr_data.adj, dpr_data.features, dpr_data.labels
        self.nclass = labels.max() + 1
        self.adj_full, self.feat_full, self.labels_full = adj, features, labels
        self.idx_train = np.array(idx_train)
        self.idx_val = np.array(idx_val)
        self.idx_test = np.array(idx_test)

        if keep_ratio < 1:
            idx_train, _ = train_test_split(idx_train,
                                            random_state=None,
                                            train_size=keep_ratio,
                                            test_size=1 - keep_ratio,
                                            stratify=labels[idx_train])

        self.adj_train = adj[np.ix_(idx_train, idx_train)]
        self.adj_val = adj[np.ix_(idx_val, idx_val)]
        self.adj_test = adj[np.ix_(idx_test, idx_test)]
        # if inductive
        #print('size of adj_train:', self.adj_train.shape)
        #print('#edges in adj_train:', self.adj_train.sum())

        self.labels_train = labels[idx_train]
        self.labels_val = labels[idx_val]
        self.labels_test = labels[idx_test]

        self.feat_train = features[idx_train]
        self.feat_val = features[idx_val]
        self.feat_test = features[idx_test]

        self.class_dict = None
        self.samplers = None
        self.class_dict2 = None

    def retrieve_class(self, c, num=256):
        if self.class_dict is None:
            self.class_dict = {}
            for i in range(self.nclass):
                self.class_dict['class_%s' % i] = (self.labels_train == i)
        idx = np.arange(len(self.labels_train))
        idx = idx[self.class_dict['class_%s' % c]]
        return np.random.permutation(idx)[:num]

    def retrieve_class_sampler(self, c, adj, transductive, num=256, args=None):
        if self.class_dict2 is None:
            self.class_dict2 = {}
            for i in range(self.nclass):
                if transductive:
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
                                                     num_workers=12, return_e_id=False,
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
            for l in range(2):
                layer_samplers = []
                sizes = [15] if l == 0 else [10, 5]
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


def splits(data, num_classes, exp):
    if exp != 'fixed':
        indices = []
        for i in range(num_classes):
            index = (data.y == i).nonzero().view(-1)
            index = index[torch.randperm(index.size(0))]
            indices.append(index)

        if exp == 'random':
            train_index = torch.cat([i[:20] for i in indices], dim=0)
            val_index = torch.cat([i[20:50] for i in indices], dim=0)
            test_index = torch.cat([i[50:] for i in indices], dim=0)
        else:
            train_index = torch.cat([i[:5] for i in indices], dim=0)
            val_index = torch.cat([i[5:10] for i in indices], dim=0)
            test_index = torch.cat([i[10:] for i in indices], dim=0)

        data.train_mask = index_to_mask(train_index, size=data.num_nodes)
        data.val_mask = index_to_mask(val_index, size=data.num_nodes)
        data.test_mask = index_to_mask(test_index, size=data.num_nodes)

    return data


def load_data(dataset):
    if dataset == 'dblp':
        dataset = CitationFull(root='../data', name=dataset)
    elif dataset == 'Physics':
        dataset = Coauthor(root='../data', name=dataset)
    elif dataset == 'ogbn-arxiv':
        dataset = PygNodePropPredDataset(root='../data', name=dataset)
    else:
        dataset = Planetoid(root='../data', name=dataset)
    data = dataset[0]
    # default format: torch_sparse.SparseTensor
    data.sparse_adj = SparseTensor(row=data.edge_index[1], col=data.edge_index[0])
    return data, dataset


def pyg2gsp(data):
    G = graphs.Graph(W=to_dense_adj(to_undirected(data.edge_index))[0])
    return G

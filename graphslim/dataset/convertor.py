from itertools import repeat

from deeprobust.graph.data import Dataset
from pygsp import graphs
from scipy.sparse import csr_matrix
import scipy.sparse
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.loader import NeighborSampler
from torch_geometric.utils import to_undirected, to_dense_adj

from graphslim.utils import *


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


# prepare transductive setting by xxx_full and inductive setting by xxx_train/val/test
class TransAndInd:

    def __init__(self, data):
        idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
        self.adj_full, self.feat_full, self.labels_full = data.adj_full, data.feat_full, data.labels_full
        self.nclass = self.labels_full.max() + 1
        self.idx_train = np.array(idx_train).reshape(-1)
        self.idx_val = np.array(idx_val).reshape(-1)
        self.idx_test = np.array(idx_test).reshape(-1)

        self.adj_train = self.adj_full[np.ix_(self.idx_train, self.idx_train)]
        self.adj_val = self.adj_full[np.ix_(self.idx_val, self.idx_val)]
        self.adj_test = self.adj_full[np.ix_(self.idx_test, self.idx_test)]
        # if inductive
        # print('size of adj_train:', self.adj_train.shape)
        # print('#edges in adj_train:', self.adj_train.sum())

        self.labels_train = self.labels_full[idx_train]
        self.labels_val = self.labels_full[idx_val]
        self.labels_test = self.labels_full[idx_test]

        self.feat_train = self.feat_full[idx_train]
        self.feat_val = self.feat_full[idx_val]
        self.feat_test = self.feat_full[idx_test]

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


class Data2Pyg:

    def __init__(self, data, device='cuda', transform=None, **kwargs):
        self.data_train = Dpr2Pyg(data.data_train, transform=transform)[0].to(device)
        self.data_val = Dpr2Pyg(data.data_val, transform=transform)[0].to(device)
        self.data_test = Dpr2Pyg(data.data_test, transform=transform)[0].to(device)
        self.nclass = data.nclass
        self.nfeat = data.nfeat
        self.class_dict = None

    def retrieve_class(self, c, num=256):
        if self.class_dict is None:
            self.class_dict = {}
            for i in range(self.nclass):
                self.class_dict['class_%s' % i] = (self.data_train.y == i).cpu().numpy()
        idx = np.arange(len(self.data_train.y))
        idx = idx[self.class_dict['class_%s' % c]]
        return np.random.permutation(idx)[:num]


class Dpr2Pyg(InMemoryDataset):

    def __init__(self, dpr_data, transform=None, **kwargs):
        root = 'data/'  # dummy root; does not mean anything
        self.dpr_data = dpr_data
        super(Dpr2Pyg, self).__init__(root, transform)
        pyg_data = self.process()
        self.data, self.slices = self.collate([pyg_data])
        self.transform = transform

    def process(self):
        dpr_data = self.dpr_data

        edge_index = torch.LongTensor(dpr_data.adj.nonzero())
        # if type(dpr_data.adj) == torch.Tensor:
        #     adj_selfloop = dpr_data.adj + torch.eye(dpr_data.adj.shape[0]).cuda()
        #     edge_index_selfloop = adj_selfloop.nonzero().T
        #     edge_index = edge_index_selfloop
        #     edge_weight = adj_selfloop[edge_index_selfloop[0], edge_index_selfloop[1]]
        # else:
        #     adj_selfloop = dpr_data.adj + sp.eye(dpr_data.adj.shape[0])
        #     edge_index = torch.LongTensor(adj_selfloop.nonzero()).cuda()
        #     edge_weight = torch.FloatTensor(adj_selfloop[adj_selfloop.nonzero()]).cuda()

        # by default, the features in pyg data is dense
        if scipy.sparse.issparse(dpr_data.features):
            x = torch.FloatTensor(dpr_data.features.todense()).float()
        else:
            x = torch.FloatTensor(dpr_data.features).float()
        y = torch.LongTensor(dpr_data.labels)

        # try:
        #     x = torch.FloatTensor(dpr_data.features.cpu()).float().cuda()
        # except:
        #     x = torch.FloatTensor(dpr_data.features).float().cuda()
        # try:
        #     y = torch.LongTensor(dpr_data.labels.cpu()).cuda()
        # except:
        #     y = dpr_data.labels


        data = Data(x=x, edge_index=edge_index, y=y)
        data.train_mask = None
        data.val_mask = None
        data.test_mask = None
        return data

    def get(self, idx):
        data = self.data.__class__()

        if hasattr(self.data, '__num_nodes__'):
            data.num_nodes = self.data.__num_nodes__[idx]

        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[self.data.__cat_dim__(key, item)] = slice(slices[idx],
                                                        slices[idx + 1])
            data[key] = item[s]
        return data

    @property
    def raw_file_names(self):
        return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def _download(self):
        pass


def pyg2gsp(data):
    G = graphs.Graph(W=to_dense_adj(to_undirected(data.edge_index))[0])
    return G


def pyg2saint(data):
    data.idx_train = data.train_mask.nonzero().view(-1)
    data.idx_val = data.val_mask.nonzero().view(-1)
    data.idx_test = data.test_mask.nonzero().view(-1)
    # reference type
    data.feat_full = data.x
    data.labels_full = data.y
    data.adj_full = ei2csr(data.edge_index, data.x.shape[0])
    return data


def ei2csr(edge_index, num_nodes):
    edge_index = edge_index
    edge_index = edge_index.t()
    edge_index_np = edge_index.numpy()
    adjacency_matrix_csr = csr_matrix((np.ones(edge_index_np.shape[1]), (edge_index_np[0], edge_index_np[1])),
                                      shape=(num_nodes, num_nodes))
    return adjacency_matrix_csr

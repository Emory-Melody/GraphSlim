import os.path as osp

from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.datasets import Planetoid, Coauthor, CitationFull, Amazon

from graphslim.dataset.convertor import TransAndInd, pyg_saint
from graphslim.dataset.utils_graphsaint import DataGraphSAINT
from graphslim.utils import *


def get_dataset(name, normalize_features=False, transform=None, return_pyg=False):
    path = osp.join('../../../data')
    if name in ['flickr', 'reddit']:
        data = DataGraphSAINT(name)
    else:
        if name in ['dblp', 'cora_ml']:
            dataset = CitationFull(root=path, name=name)
        elif name in ['physics', 'cs']:
            dataset = Coauthor(root=path, name=name)
        elif name in ['cora', 'citeseer', 'pubmed']:
            dataset = Planetoid(root=path, name=name)
        elif name in ['photo', 'computers']:
            dataset = Amazon(root=path, name=name)
        elif name in ['ogbn-products', 'ogbn-proteins', 'ogbn-papers100M', 'ogbn-arxiv']:
            dataset = PygNodePropPredDataset(name, root=path)
        else:
            pass
        data = dataset[0]
    # support both pyg and saint format
    data = pyg_saint(data)

    return TransAndInd(data)

    # if transform is not None and normalize_features:
    #     dataset.transform = T.Compose([T.NormalizeFeatures(), transform])
    # elif normalize_features:
    #     dataset.transform = T.NormalizeFeatures()
    # elif transform is not None:
    #     dataset.transform = transform


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

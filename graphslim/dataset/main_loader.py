import os.path as osp

from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.datasets import Planetoid, Coauthor, CitationFull, Amazon, Flickr, Reddit

from graphslim.dataset.convertor import pyg_saint
from graphslim.dataset.utils import TransAndInd, merge_attributes


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
    # support both pyg and saint format
    data = pyg_saint(data, args)
    # support both transductive and inductive tasks
    return merge_attributes(TransAndInd(data), data)

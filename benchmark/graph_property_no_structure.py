import os
import sys
import scipy

from sklearn.metrics import pairwise_distances
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
if os.path.abspath('../..') not in sys.path:
    sys.path.append(os.path.abspath('../..'))
from graphslim.configs import *
from graphslim.dataset import *
from graphslim.evaluation.utils import sparsify

import numpy as np
import torch
from sklearn.metrics import davies_bouldin_score


def graph_property(adj, feat, label):
    db_list = []
    if len(adj.shape) == 3:
        for i in range(adj.shape[0]):
            db_index = davies_bouldin_score(feat, label)
            db_list.append(db_index)

        db_list = np.array(db_list)
        args.logger.info(f"Average Davies-Bouldin Index: {np.mean(db_list)}")


    else:
        db_index = davies_bouldin_score(feat, label)
        args.logger.info(f"Davies-Bouldin Index: {db_index}")


if __name__ == '__main__':
    args = cli(standalone_mode=False)
    args.device = 'cpu'

    if args.eval_whole:
        graph = get_dataset(args.dataset, args)
        if args.setting == 'ind':
            adj, feat, label = graph.adj_train, graph.feat_train, graph.labels_train
        else:
            adj, feat, label = graph.adj_full, graph.feat_full, graph.labels_full
        feat = feat.numpy()
        label = label.numpy()
        graph_property(adj, feat, label)
    method_list = ['vng', 'gcond', 'msgc', 'sgdd']
    method_list2 = method_list + ['gcondx', 'geom']
    for args.method in method_list2:
        print(f'========{args.method}========')
        adj_syn, feat, label = load_reduced(args)
        if args.method not in ['msgc']:
            label = label[:adj_syn.shape[0]]
        else:
            label = label[:adj_syn.shape[1]]
        if args.method in ['geom', 'gcsntk'] and len(label.shape) > 1:
            label = torch.argmax(label, dim=1)
        adj_syn = sparsify('GCN', adj_syn, args, verbose=args.verbose)
        adj = adj_syn.numpy()
        label = label.numpy()
        feat = feat.numpy()
        graph_property(adj, feat, label)

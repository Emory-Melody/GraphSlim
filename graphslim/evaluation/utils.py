import time
from functools import wraps

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch_sparse import SparseTensor

from graphslim.dataset.utils import csr2ei


def sparsify(model_type, adj_syn, args, verbose=False):
    threshold = 0
    if model_type == 'MLP':
        adj_syn = adj_syn - adj_syn
        torch.diagonal(adj_syn).fill_(1)
    elif model_type == 'GAT':
        if args.method in ['gcond', 'doscond']:
            if args.dataset in ['cora', 'citeseer']:
                threshold = 0.5  # Make the graph sparser as GAT does not work well on dense graph
            else:
                threshold = 0.1
        elif args.method in ['msgc']:
            threshold = args.threshold
        else:
            threshold = 0.5
    else:
        if args.method in ['gcond', 'doscond']:
            threshold = args.threshold
        elif args.method in ['msgc']:
            threshold = 0
        else:
            threshold = 0
    if verbose and args.method not in ['gcondx', 'doscondx', 'sfgc', 'geom', 'gcsntk']:
        # print('Sum:', adj_syn.sum().item())
        print('Sparsity:', adj_syn.nonzero().shape[0] / adj_syn.numel())
    # if args.method in ['sgdd']:
    #     threshold = 0.5
    if threshold > 0:
        adj_syn[adj_syn < threshold] = 0
        if verbose:
            print('Sparsity after truncating:', adj_syn.nonzero().shape[0] / adj_syn.numel())
        # else:
        #     print("structure free methods do not need to truncate the adjacency matrix")
    return adj_syn


def getsize_mb(elements):
    size = 0
    for e in elements:
        if type(e) == SparseTensor:
            row, col, value = e.coo()
            size += row.element_size() * row.nelement()
            size += col.element_size() * col.nelement()
            size += value.element_size() * value.nelement()
        elif isinstance(e, sp.csr_matrix):
            e = csr2ei(e)
            size += e.element_size() * e.nelement()
        else:
            try:
                size += e.element_size() * e.nelement()
            except:
                e = torch.from_numpy(e)
                size += e.element_size() * e.nelement()
    return size / 1024 / 1024


def verbose_time_memory(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        verbose = kwargs.get('verbose', False)
        if verbose:
            start = time.perf_counter()

        result = func(*args, **kwargs)

        if verbose:
            end = time.perf_counter()
            runTime = end - start
            runTime_ms = runTime * 1000
            print("Function Time:", runTime, "s")
            print("Function Time:", runTime_ms, "ms")

            data = kwargs.get('data', None)
            if data is None:
                for arg in args:
                    if hasattr(arg, 'feat_train') or hasattr(arg, 'x'):
                        data = arg
                        break
                if data is None:
                    raise ValueError("The function must be called with 'data' as an argument.")

            if 'setting' in kwargs and kwargs['setting'] == 'trans':
                origin_storage = getsize_mb([data.x, data.edge_index, data.y])
            else:
                origin_storage = getsize_mb([data.feat_train, data.adj_train, data.labels_train])
            if not hasattr(data, 'feat_syn'):
                if 'setting' in kwargs and kwargs['setting'] == 'trans':
                    data.feat_syn = data.feat_full
                else:
                    data.feat_syn = data.feat_train
            if not hasattr(data, 'adj_syn'):
                data.adj_syn = torch.eye(data.labels_train.shape[0])
            if not hasattr(data, 'labels_syn'):
                data.labels_syn = data.labels_train
            condensed_storage = getsize_mb([data.feat_syn, data.adj_syn, data.labels_syn])
            print(f'Original graph:{origin_storage:.2f} Mb  Condensed graph:{condensed_storage:.2f} Mb')

        return result

    return wrapper


# from deeprobust.graph.utils import accuracy


def calc_f1(y_true, y_pred, is_sigmoid):
    if not is_sigmoid:
        y_pred = np.argmax(y_pred, axis=1)
    else:
        y_pred[y_pred > 0.5] = 1
        y_pred[y_pred <= 0.5] = 0
    return f1_score(y_true, y_pred, average="micro"), f1_score(y_true, y_pred, average="macro")


def evaluate(output, labels, args):
    data_graphsaint = ['yelp', 'ppi', 'ppi-large', 'flickr', 'reddit', 'amazon']
    if args.dataset in data_graphsaint:
        labels = labels.cpu().numpy()
        output = output.cpu().numpy()
        if len(labels.shape) > 1:
            micro, macro = calc_f1(labels, output, is_sigmoid=True)
        else:
            micro, macro = calc_f1(labels, output, is_sigmoid=False)
        print("Test set results:", "F1-micro= {:.4f}".format(micro),
              "F1-macro= {:.4f}".format(macro))
    else:
        loss_test = F.nll_loss(output, labels)
        acc_test = accuracy_score(output, labels)
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
    return

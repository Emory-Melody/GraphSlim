import os
import sys

import numpy as np

if os.path.abspath('..') not in sys.path:
    sys.path.append(os.path.abspath('..'))
from graphslim.configs import *
from graphslim.dataset import *
import logging
from deeprobust.graph.defense import GCN
from deeprobust.graph.global_attack import DICE, Random, Metattack, PRBCD
from graphslim.models.gcn import GCN
import scipy.sparse as sp


def attack(data, args):
    if not os.path.exists(args.save_path + '/corrupt_graph'):
        os.makedirs(args.save_path + '/corrupt_graph')
    args.save_path = f'{args.save_path}/corrupt_graph'

    if os.path.exists(f'{args.save_path}/adj_{args.dataset}_{args.attack}_{args.ptb_r}.npz'):
        if args.setting == 'ind':
            data.adj_train = sp.load_npz(f'{args.save_path}/adj_{args.dataset}_{args.attack}_{args.ptb_r}.npz')
        else:
            data.adj_full = sp.load_npz(f'{args.save_path}/adj_{args.dataset}_{args.attack}_{args.ptb_r}.npz')

    else:
        gcn_model = GCN(nfeat=data.x.shape[1], nhid=args.hidden, nclass=data.nclass, args=args, mode='eval').to(
            args.device)
        if args.setting == 'ind':
            adj = data.adj_train
            args.ptb_n = int(args.ptb_r * (adj.sum() // 2))
        else:
            adj = data.adj_full
            args.ptb_n = int(args.ptb_r * (adj.sum() // 2))
        if args.attack == 'metattack':
            if args.setting == 'ind':
                model = PRBCD(data, device=args.device)
                # ignore the test results!
                edge_index, _ = model.attack(ptb_rate=args.ptb_r)
                data.adj_train = ei2csr(edge_index.cpu(), data.num_nodes)[np.ix_(data.idx_train, data.idx_train)]
                gcn_model.fit_with_val(data, train_iters=args.eval_epochs, verbose=args.verbose, setting=args.setting)
                gcn_model.test(data, setting=args.setting, verbose=True)
                sp.save_npz(f'{args.save_path}/adj_{args.dataset}_{args.attack}_{args.ptb_r}.npz', data.adj_train)
                print(f'save corrupt graph at adj_{args.dataset}_{args.attack}_{args.ptb_r}.npz')
            else:
                model = PRBCD(data, device=args.device)
                data.edge_index, _ = model.attack(data.edge_index, ptb_rate=args.ptb_r)
                data.adj_full = ei2csr(data.edge_index.cpu(), data.num_nodes)
                gcn_model.fit_with_val(data, train_iters=args.eval_epochs, verbose=args.verbose, setting=args.setting)
                gcn_model.test(data, setting=args.setting, verbose=True)
                sp.save_npz(f'{args.save_path}/adj_{args.dataset}_{args.attack}_{args.ptb_r}.npz', data.adj_full)
                print(f'save corrupt graph at {args.save_path}/adj_{args.dataset}_{args.attack}_{args.ptb_r}.npz')
        if args.attack == 'random':
            if args.setting == 'ind':
                model = Random()
                model.attack(data.adj_train, n_perturbations=args.ptb_n)
                data.adj_train = model.modified_adj.tocsr()
                gcn_model.fit_with_val(data, train_iters=args.eval_epochs, verbose=args.verbose, setting=args.setting)
                gcn_model.test(data, setting=args.setting, verbose=True)
                sp.save_npz(f'{args.save_path}/adj_{args.dataset}_{args.attack}_{args.ptb_r}.npz', data.adj_train)
            else:
                model = Random()
                model.attack(data.adj_full, n_perturbations=args.ptb_n)
                data.adj_full = model.modified_adj.tocsr()
                gcn_model.fit_with_val(data, train_iters=args.eval_epochs, verbose=args.verbose, setting=args.setting)
                gcn_model.test(data, setting=args.setting, verbose=True)
                sp.save_npz(f'{args.save_path}/adj_{args.dataset}_{args.attack}_{args.ptb_r}.npz', data.adj_full)

    return data

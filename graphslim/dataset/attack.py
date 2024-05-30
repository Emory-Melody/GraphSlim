import os
import sys
import torch

import numpy as np

if os.path.abspath('..') not in sys.path:
    sys.path.append(os.path.abspath('..'))
from graphslim.configs import *
from graphslim.dataset import *
import logging
from graphslim.models import *
import scipy.sparse as sp


def attack(data, args):
    seed_everything(args.seed)
    save_path = f'{args.save_path}/corrupt_graph/{args.attack}'
    gcn_model = GCN(nfeat=data.x.shape[1], nhid=args.hidden, nclass=data.nclass, args=args, mode='attack').to(
        args.device)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if args.attack in ['metattack', 'random_adj']:
        if os.path.exists(f'{save_path}/adj_{args.dataset}_{args.attack}_{args.ptb_r}_{args.seed}.npz'):
            if args.setting == 'ind':
                data.adj_train = sp.load_npz(
                    f'{save_path}/adj_{args.dataset}_{args.attack}_{args.ptb_r}_{args.seed}.npz')
            else:
                data.adj_full = sp.load_npz(
                    f'{save_path}/adj_{args.dataset}_{args.attack}_{args.ptb_r}_{args.seed}.npz')
            print(f'load corrupt graph at {save_path}/adj_{args.dataset}_{args.attack}_{args.ptb_r}_{args.seed}.npz')

        else:
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
                else:
                    model = PRBCD(data, device=args.device)
                    data.edge_index, _ = model.attack(data.edge_index, ptb_rate=args.ptb_r)
                    data.adj_full = ei2csr(data.edge_index.cpu(), data.num_nodes)
            elif args.attack == 'random_adj':
                model = RandomAttack()
                if args.setting == 'ind':
                    model.attack(data.adj_train, n_perturbations=args.ptb_n, type='add')
                    data.adj_train = model.modified_adj.tocsr()
                    sp.save_npz(f'{save_path}/adj_{args.dataset}_{args.attack}_{args.ptb_r}_{args.seed}.npz',
                                data.adj_train)
                else:
                    model.attack(data.adj_full, n_perturbations=args.ptb_n, type='add')
                    data.adj_full = model.modified_adj.tocsr()
                    sp.save_npz(f'{save_path}/adj_{args.dataset}_{args.attack}_{args.ptb_r}_{args.seed}.npz',
                                data.adj_full)
            gcn_model.fit_with_val(data, train_iters=args.eval_epochs, verbose=args.verbose, setting=args.setting)
            test_acc = gcn_model.test(data, setting=args.setting, verbose=True)
            args.logger.info(f'attack {args.attack}_{args.ptb_r} test acc: {test_acc}')
            print(f'save corrupt graph at {save_path}/adj_{args.dataset}_{args.attack}_{args.ptb_r}_{args.seed}.npz')
    elif args.attack == 'random_feat':
        if os.path.exists(f'{save_path}/feat_{args.dataset}_{args.attack}_{args.ptb_r}_{args.seed}.pt'):
            if args.setting == 'ind':
                data.feat_train = torch.load(f'{args.save_path}/feat_{args.dataset}_{args.attack}_{args.ptb_r}.pt')
            else:
                data.feat_full = torch.load(f'{args.save_path}/feat_{args.dataset}_{args.attack}_{args.ptb_r}.pt')
            print(f'load corrupt graph at {args.save_path}/feat_{args.dataset}_{args.attack}_{args.ptb_r}.pt')
        else:
            model = RandomAttack(attack_structure=False, attack_features=True)
            args.ptb_n = int(args.ptb_r * data.x.shape[1])
            if args.setting == 'ind':
                model.attack(data.feat_train, n_perturbations=args.ptb_n)
                data.feat_train = model.modified_features
            else:
                model.attack(data.feat_full, n_perturbations=args.ptb_n)
                data.feat_full = model.modified_features
            gcn_model.fit_with_val(data, train_iters=args.eval_epochs, verbose=args.verbose, setting=args.setting)
            test_acc = gcn_model.test(data, setting=args.setting, verbose=True)
            args.logger.info(f'attack {args.attack}_{args.ptb_r} test acc: {test_acc}')
            sp.save_npz(f'{args.save_path}/feat_{args.dataset}_{args.attack}_{args.ptb_r}_{args.seed}.pt',
                        data.adj_train)
            print(
                f'save corrupt graph at {args.save_path}/feat_{args.dataset}_{args.attack}_{args.ptb_r}_{args.seed}.pt')

    return data

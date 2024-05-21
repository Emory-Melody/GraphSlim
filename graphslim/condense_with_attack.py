import os
import sys

import numpy as np

if os.path.abspath('..') not in sys.path:
    sys.path.append(os.path.abspath('..'))
from configs import *
from evaluation.eval_agent import Evaluator
from graphslim.condensation import *
from graphslim.dataset import *
import logging
from deeprobust.graph.defense import GCN
from deeprobust.graph.global_attack import DICE, random_attack, Metattack, PRBCD
from graphslim.models.gcn import GCN
import scipy.sparse as sp
import time

args = cli(standalone_mode=False)
args.ptb_r = 0.25

data = get_dataset(args.dataset, args)
if not os.path.exists(args.save_path + '/corrupt_graph'):
    os.makedirs(args.save_path + '/corrupt_graph')
args.save_path = f'{args.save_path}/corrupt_graph'

if os.path.exists(f'{args.save_path}/adj_{args.dataset}_{args.attack}_{args.ptb_r}.npz'):
    if args.setting == 'ind':
        data.adj_train = sp.load_npz(f'{args.save_path}/adj_{args.dataset}_{args.attack}_{args.ptb_r}.npz')
    else:
        data.adj_full = sp.load_npz(f'{args.save_path}/adj_{args.dataset}_{args.attack}_{args.ptb_r}.npz')

else:
    gcn_model = GCN(nfeat=data.x.shape[1], nhid=args.hidden, nclass=data.nclass, args=args, mode='eval')
    if args.attack == 'metattack':
        if args.setting == 'ind':
            adj = data.adj_train
            args.ptb_n = int(args.ptb_r * (adj.sum() // 2))
            model = PRBCD(data, device=args.device)
            # ignore the test results!
            edge_index, _ = model.attack(ptb_rate=args.ptb_r)
            data.adj_train = ei2csr(edge_index.cpu(), data.num_nodes)[np.ix_(data.idx_train, data.idx_train)]
            gcn_model.fit_with_val(data, train_iters=args.train_iters, verbose=args.verbose, setting=args.setting)
            gcn_model.test(data, setting=args.setting, verbose=True)
            sp.save_npz(f'{args.save_path}/adj_{args.dataset}_{args.attack}_{args.ptb_r}.npz', data.adj_train)
            print(f'save corrupt graph at adj_{args.dataset}_{args.attack}_{args.ptb_r}.npz')
        else:
            adj, features, labels = data.adj_full, data.feat_full, data.labels_full
            args.ptb_n = int(args.ptb_r * (adj.sum() // 2))
            model = PRBCD(data, device=args.device)
            data.edge_index, _ = model.attack(data.edge_index, ptb_rate=args.ptb_r)
            gcn_model.fit_with_val(data, train_iters=args.train_iters, verbose=args.verbose, setting=args.setting)
            gcn_model.test(data, setting=args.setting, verbose=True)
            data.adj_full = ei2csr(data.edge_index.cpu(), data.num_nodes)
            sp.save_npz(f'{args.save_path}/adj_{args.dataset}_{args.attack}_{args.ptb_r}.npz', data.adj_full)
            print(f'save corrupt graph at {args.save_path}/adj_{args.dataset}_{args.attack}_{args.ptb_r}.npz')
    if args.attack == 'random_attack':
        if args.setting == 'ind':
            model = random_attack()
            data.adj_train = model.attack(data.adj_train, args.ptb_r)
            np.save(f'{args.save_path}/adj_{args.dataset}_{args.attack}_{args.ptb_r}.npy', data.adj_train)
        else:
            data.adj_full = random_attack(data.adj_full, args.ptb_r)
            np.save(f'{args.save_path}/adj_{args.dataset}_{args.attack}_{args.ptb_r}.npy', data.adj_full)
if args.method == 'gcond':
    agent = GCond(setting=args.setting, data=data, args=args)
elif args.method == 'doscond':
    agent = DosCond(setting=args.setting, data=data, args=args)
elif args.method in ['doscondx', 'gcondx']:
    agent = DosCondX(setting=args.setting, data=data, args=args)
elif args.method == 'sfgc':
    agent = SFGC(setting=args.setting, data=data, args=args)
elif args.method == 'sgdd':
    agent = SGDD(setting=args.setting, data=data, args=args)
elif args.method == 'gcsntk':
    agent = GCSNTK(setting=args.setting, data=data, args=args)
elif args.method == 'msgc':
    agent = MSGC(setting=args.setting, data=data, args=args)
elif args.method == 'geom':
    agent = GEOM(setting=args.setting, data=data, args=args)
start = time.perf_counter()
reduced_graph = agent.reduce(data, verbose=args.verbose)
end = time.perf_counter()
args.logger.info(f'Function Time: {end - start}s')
# reduced_graph = graph
evaluator = Evaluator(args)
evaluator.evaluate(reduced_graph, model_type=args.eval_model)

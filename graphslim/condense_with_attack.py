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
import time

args = cli(standalone_mode=False)
args.ptb_r = 0.25

data = get_dataset(args.dataset, args)
if args.attack == 'random_attack':
    pass

if args.attack == 'metattack':
    adj, features, labels = data.adj_full, data.feat_full, data.labels_full
    args.ptb_n = int(args.ptb_r * (adj.sum() // 2))
    if args.setting == 'ind':
        model = PRBCD(data, device=args.device)
        data.edge_index, _ = model.attack(csr2ei(data.adj_train), ptb_rate=args.ptb_r)
        data.adj_train = ei2csr(data.edge_index.cpu(), data.num_nodes)
    else:
        model = PRBCD(data, device=args.device)
        data.edge_index, _ = model.attack(data.edge_index, ptb_rate=args.ptb_r)
        data.adj_full = ei2csr(data.edge_index.cpu(), data.num_nodes)
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

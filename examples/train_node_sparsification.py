import numpy as np
import os, sys

if os.path.abspath('../graphslim') not in sys.path:
    sys.path.append(os.path.abspath('../graphslim'))

from graphslim.configs import cli
from graphslim.dataset import *
from graphslim.evaluation import *
from graphslim.sparsification import KCenter

args = cli(standalone_mode=False)
graph = get_dataset(args.dataset, args)
all_res = []
for i in range(args.run_reduction):
    args.seed += i
    agent = KCenter(setting=args.setting, data=graph, args=args)
    reduced_graph = agent.reduce(graph)
    evaluator = Evaluator(args)
    res = evaluator.evaluate(reduced_graph, 'GCN')
    all_res.append(res)
    graph.reset()
all_res = np.array(all_res)
print('Test Mean Result:', all_res[:, 0].mean(), '+/-', all_res[:, 1].mean())

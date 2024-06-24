import os, sys

if os.path.abspath('../graphslim') not in sys.path:
    sys.path.append(os.path.abspath('../graphslim'))
from graphslim.configs import *

from graphslim.evaluation.eval_agent import Evaluator
from graphslim.coarsening.coarsening_base import Coarsen
from graphslim.dataset import *

args = cli(standalone_mode=False)

graph = get_dataset(args.dataset, args)
agent = Coarsen(setting=args.setting, data=graph, args=args)
reduced_graph = agent.reduce(graph)

# print("num of synthetic node", reduced_graph.feat_syn.shape[0])
if args.setting == 'trans':
    print("real reduction rate", reduced_graph.feat_syn.shape[0] / graph.x.shape[0] * 100, "%")
else:
    print("real reduction rate", reduced_graph.feat_syn.shape[0] / sum(graph.train_mask) * 100, "%")

evaluator = Evaluator(args)
evaluator.evaluate(reduced_graph, 'GCN')

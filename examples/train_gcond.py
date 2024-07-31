import os, sys

if os.path.abspath('../graphslim') not in sys.path:
    sys.path.append(os.path.abspath('../graphslim'))
from graphslim.config import *
from graphslim.evaluation.eval_agent import Evaluator

from graphslim.condensation import GCond
from graphslim.dataset import *

args = cli(standalone_mode=False)

graph = get_dataset(args.dataset, args)
agent = GCond(setting=args.setting, data=graph, args=args)
reduced_graph = agent.reduce(graph, verbose=args.verbose)
evaluator = Evaluator(args)
evaluator.evaluate(reduced_graph, model_type='GAT')

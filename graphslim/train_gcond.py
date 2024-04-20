from configs import *
from evaluation.eval_agent import Evaluator
from graphslim.condensation import GCondX, GCond, DosCond
from graphslim.dataset import *

args = cli(standalone_mode=False)

graph = get_dataset(args.dataset, args)
if args.method == 'gcond':
    agent = GCond(setting=args.setting, data=graph, args=args)
elif args.method == 'gcondx':
    agent = GCondX(setting=args.setting, data=graph, args=args)
elif args.method == 'doscond':
    agent = DosCond(setting=args.setting, data=graph, args=args)
reduced_graph = agent.reduce(graph, verbose=args.verbose)
evaluator = Evaluator(args)
evaluator.evaluate(reduced_graph, model_type='GCN')
# python -m coarserning.kcenter

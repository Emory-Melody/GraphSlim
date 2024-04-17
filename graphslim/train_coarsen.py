from configs import *
from evaluation.eval_agent import Evaluator
from graphslim.coarsening.coarsening_agent import Coarsen
from graphslim.dataset import *

args = cli(standalone_mode=False)

graph = get_dataset(args.dataset, args)
agent = Coarsen(setting=args.setting, data=graph, args=args)
reduced_graph = agent.reduce(graph)
evaluator = Evaluator(args)
print(evaluator.evaluate(reduced_graph, 'GCN'))

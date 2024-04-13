from configs import *
from evaluation.eval_agent import Evaluator
from graphslim.coarsening import Coarsen
from graphslim.dataset import *

args = cli(standalone_mode=False)

graph = get_dataset(args.dataset, args)
agent = Coarsen(args)
reduced_graph = agent.train(graph)
evaluator = Evaluator(args)
print(evaluator.train(reduced_graph, 'GCN'))

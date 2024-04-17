from configs import cli
from graphslim.dataset import *
from graphslim.evaluation import *
from graphslim.sparsification import CoreSet

if __name__ == '__main__':
    args = cli(standalone_mode=False)
    graph = get_dataset(args.dataset, args)
    agent = CoreSet(setting=args.setting, data=graph, args=args)
    reduced_graph = agent.reduce(graph)
    evaluator = Evaluator(args)
    evaluator.evaluate(reduced_graph, 'GCN')

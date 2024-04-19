import numpy as np

from configs import cli
from graphslim.dataset import *
from graphslim.evaluation import *
from graphslim.sparsification import CoreSet

if __name__ == '__main__':
    args = cli(standalone_mode=False)
    graph = get_dataset(args.dataset, args)
    all_res = []
    for i in range(args.run_reduction):
        args.seed += i
        agent = CoreSet(setting=args.setting, data=graph, args=args)
        reduced_graph = agent.reduce(graph)
        evaluator = Evaluator(args)
        res = evaluator.evaluate(reduced_graph, 'GCN')
        all_res.append(res)
    all_res = np.array(all_res).reshape(-1)
    print('Test Mean Result:', all_res.mean(), '+/-', all_res.std())

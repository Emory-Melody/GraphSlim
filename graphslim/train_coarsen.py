import numpy as np

from configs import *
from evaluation.eval_agent import Evaluator
from graphslim.coarsening.coarsening_agent import Coarsen
from graphslim.dataset import *

if __name__ == '__main__':
    args = cli(standalone_mode=False)
    graph = get_dataset(args.dataset, args)
    all_res = []
    for i in range(args.run_reduction):
        args.seed += i
        agent = Coarsen(setting=args.setting, data=graph, args=args)
        reduced_graph = agent.reduce(graph)
        if args.setting == 'trans':
            print("real reduction rate", reduced_graph.feat_syn.shape[0] / graph.x.shape[0] * 100, "%")
        else:
            print("real reduction rate", reduced_graph.feat_syn.shape[0] / sum(graph.train_mask) * 100, "%")
        evaluator = Evaluator(args)
        res = evaluator.evaluate(reduced_graph, 'GCN')
        all_res.append(res)
    all_res = np.array(all_res).reshape(-1)
    print('Final Mean:', all_res.mean(), '+/-', all_res.std())

import numpy as np

from configs import *
from evaluation.eval_agent import Evaluator
from graphslim.coarsening import Cluster, Coarsen, VNG
from graphslim.dataset import *

if __name__ == '__main__':
    args = cli(standalone_mode=False)
    graph = get_dataset(args.dataset, args)
    all_res = []
    args.run_reduction = 1
    for i in range(args.run_reduction):
        seed_everything(args.seed + i)
        if args.method == 'vng':
            agent = VNG(setting=args.setting, data=graph, args=args)
        elif args.method == 'coarsen':
            agent = Coarsen(setting=args.setting, data=graph, args=args)
        elif args.method == 'clustering':
            agent = Cluster(setting=args.setting, data=graph, args=args)
        reduced_graph = agent.reduce(graph, verbose=True)
        if args.setting == 'trans':
            print("real reduction rate", reduced_graph.feat_syn.shape[0] / graph.x.shape[0] * 100, "%")
        else:
            print("real reduction rate", reduced_graph.feat_syn.shape[0] / sum(graph.train_mask) * 100, "%")
        evaluator = Evaluator(args)
        res_mean, res_std = evaluator.evaluate(reduced_graph, 'GCN')
        all_res.append([res_mean, res_std])
    all_res = np.array(all_res)
    print(f'Test Mean Result: {100 * all_res[:, 0].mean():.2f} +/- {100 * all_res[:, 1].mean():.2f}')

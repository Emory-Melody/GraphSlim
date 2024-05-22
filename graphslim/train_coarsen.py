import os
import sys

if os.path.abspath('..') not in sys.path:
    sys.path.append(os.path.abspath('..'))
import numpy as np

from graphslim.configs import *
from graphslim.evaluation.eval_agent import Evaluator
from graphslim.coarsening import *
from graphslim.dataset import *

if __name__ == '__main__':
    args = cli(standalone_mode=False)
    graph = get_dataset(args.dataset, args)

    all_res = []
    for i in range(args.run_reduction):
        args.seed = i + 1
        seed_everything(args.seed)
        if args.method == 'vng':
            agent = VNG(setting=args.setting, data=graph, args=args)
        elif args.method == 'variation_neighborhoods':
            agent = Coarsen(setting=args.setting, data=graph, args=args)
        elif args.method == 'clustering':
            agent = Cluster(setting=args.setting, data=graph, args=args)
        elif args.method == 'averaging':
            agent = Average(setting=args.setting, data=graph, args=args)
        reduced_graph = agent.reduce(graph, verbose=True)
        if args.setting == 'trans':
            print("real reduction rate", reduced_graph.feat_syn.shape[0] / graph.x.shape[0] * 100, "%")
        else:
            print("real reduction rate", reduced_graph.feat_syn.shape[0] / sum(graph.train_mask).item() * 100, "%")
        evaluator = Evaluator(args)
        res_mean, res_std = evaluator.evaluate(reduced_graph, model_type='GCN')
        all_res.append([res_mean, res_std])
    all_res = np.array(all_res)
    args.logger.info(f'Test Mean Accuracy: {100 * all_res[:, 0].mean():.2f} +/- {100 * all_res[:, 1].mean():.2f}')

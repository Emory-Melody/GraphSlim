import os
import sys

if os.path.abspath('..') not in sys.path:
    sys.path.append(os.path.abspath('..'))
import numpy as np

from configs import cli
from graphslim.dataset import *
from graphslim.evaluation import *
from graphslim.sparsification import *
from graphslim.utils import seed_everything

if __name__ == '__main__':
    args = cli(standalone_mode=False)
    graph = get_dataset(args.dataset, args)

    all_res = []
    for i in range(args.run_reduction):
        seed_everything(args.seed + i)
        if args.method == 'kcenter' and not args.aggpreprocess:
            agent = KCenter(setting=args.setting, data=graph, args=args)
        elif args.method == 'kcenter' and args.aggpreprocess:
            agent = KCenterAgg(setting=args.setting, data=graph, args=args)
        elif args.method == 'herding' and not args.aggpreprocess:
            agent = Herding(setting=args.setting, data=graph, args=args)
        elif args.method == 'herding' and args.aggpreprocess:
            agent = HerdingAgg(setting=args.setting, data=graph, args=args)
        elif args.method == 'random':
            agent = Random(setting=args.setting, data=graph, args=args)
        elif args.method == 'cent_p':
            agent = CentP(setting=args.setting, data=graph, args=args)
        elif args.method == 'cent_d':
            agent = CentD(setting=args.setting, data=graph, args=args)
        reduced_graph = agent.reduce(graph, verbose=args.verbose)
        evaluator = Evaluator(args)
        res_mean, res_std = evaluator.evaluate(reduced_graph, model_type='GCN')
        all_res.append([res_mean, res_std])
    all_res = np.array(all_res)
    print(f'Test Mean Result: {100 * all_res[:, 0].mean():.2f} +/- {100 * all_res[:, 1].mean():.2f}')

import numpy as np

from configs import cli
from graphslim.dataset import *
from graphslim.evaluation import *
from graphslim.sparsification import CoreSet
from graphslim.utils import seed_everything

if __name__ == '__main__':
    args = cli(standalone_mode=False)
    graph = get_dataset(args.dataset, args)
    all_res = []
    for i in range(args.run_reduction):
        seed_everything(args.seed + i)
        agent = CoreSet(setting=args.setting, data=graph, args=args)
        reduced_graph = agent.reduce(graph, verbose=args.verbose)
        evaluator = Evaluator(args)
        res_mean, res_std = evaluator.evaluate(reduced_graph, model_type='GCN')
        all_res.append([res_mean, res_std])
    all_res = np.array(all_res)
    print(f'Test Mean Result: {100 * all_res[:, 0].mean():.2f} +/- {100 * all_res[:, 1].mean():.2f}')

import os
import sys

if os.path.abspath('..') not in sys.path:
    sys.path.append(os.path.abspath('..'))

from graphslim.configs import cli
from graphslim.dataset import *
from graphslim.evaluation import NasEvaluator

if __name__ == '__main__':
    args = cli(standalone_mode=False)
    data = get_dataset(args.dataset, args)

    NasEvaluator = NasEvaluator(args)

    if not os.path.exists(f'./output/{args.dataset}_results_ori.csv'):
        args.logger.info("No original results found. Run evaluate_ori and test_params_ori.")
        NasEvaluator.evaluate_ori(data)
        NasEvaluator.test_params_ori(data)
    else:
        args.logger.info("Find original results. Run evaluate_ori and test_params_ori.")

    NasEvaluator.evaluate_syn(data)
    NasEvaluator.test_params_syn(data)
    NasEvaluator.cal_pearson()

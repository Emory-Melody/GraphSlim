import os
import sys

if os.path.abspath('../..') not in sys.path:
    sys.path.append(os.path.abspath('../..'))

from graphslim.configs import cli
from graphslim.dataset import *
from graphslim.evaluation import NasEvaluator

if __name__ == '__main__':
    args = cli(standalone_mode=False)

    data = get_dataset(args.dataset, args)

    NasEval = NasEvaluator(args)

    save_path = f'checkpoints/nas/{args.dataset}'
    if not os.path.exists(f'{save_path}/results_ori.csv'):
        args.logger.info("No original results found. Run evaluate_ori and test_params_ori.")
        NasEvaluator.evaluate_ori(data)
        NasEvaluator.test_params_ori(data)
    else:
        args.logger.info("Find original results. Run evaluate_syn and test_params_syn.")

    if args.method in ['random', 'kcenter', 'averaging', 'vng']:
        res_accs, res_person_accs, res_person_ranks = [], [], []
        for seed in [1, 2, 3]:
            NasEval = NasEvaluator(args)
            args.seed = seed
            NasEval.evaluate_syn(data)
            res_acc = NasEval.test_params_syn(data)
            res_person_acc, res_person_rank = NasEval.cal_pearson()
            res_accs.append(res_acc)
            res_person_accs.append(res_person_acc)
            res_person_ranks.append(res_person_rank)
        args.logger.info(f"{res_accs.mean()}")
        args.logger.info(f"{res_person_accs.mean()}")
        args.logger.info(f"{res_person_ranks.mean()}")
    else:
        NasEval = NasEvaluator(args)
        NasEval.evaluate_syn(data)
        res_acc = NasEval.test_params_syn(data)
        res_person_acc, res_person_rank = NasEval.cal_pearson()
        args.logger.info(f"{res_acc}")
        args.logger.info(f"{res_person_acc}")
        args.logger.info(f"{res_person_rank}")

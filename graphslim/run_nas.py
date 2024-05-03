from graphslim.configs import cli
from graphslim.dataset import *
from graphslim.evaluation import NasEvaluator

if __name__ == '__main__':
    args = cli(standalone_mode=False)
    data = get_dataset(args.dataset, args)

    NasEvaluator = NasEvaluator(args)

    # NasEvaluator.evaluate_ori(data)
    # NasEvaluator.test_params_ori(data)

    NasEvaluator.evaluate_syn(data)
    NasEvaluator.test_params_syn(data)
    NasEvaluator.cal_pearson()

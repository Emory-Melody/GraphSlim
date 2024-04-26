from graphslim.configs import cli
from graphslim.dataset import *
from graphslim.evaluation import NasEvaluator

if __name__ == '__main__':
    args = cli(standalone_mode=False)
    data = get_dataset(args.dataset, args)

    if args.dataset in ['cora', 'citeseer']:
        args.epsilon = 0.05
    else:
        args.epsilon = 0.01

    NasEvaluator = NasEvaluator(args)

    NasEvaluator.evaluate(data)

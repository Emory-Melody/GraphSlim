from graphslim.configs import cli
from graphslim.dataset import *
from graphslim.evaluation import Evaluator

if __name__ == '__main__':
    args = cli(standalone_mode=False)
    data = get_dataset(args.dataset, args)

    evaluator = Evaluator(args)
    evaluator.train_cross(data)

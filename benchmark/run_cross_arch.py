import os
import sys

if os.path.abspath('..') not in sys.path:
    sys.path.append(os.path.abspath('..'))

from graphslim.config import cli
from graphslim.dataset import *
from graphslim.evaluation import Evaluator

if __name__ == '__main__':
    args = cli(standalone_mode=False)
    data = get_dataset(args.dataset, args)
    evaluator = Evaluator(args)
    if args.eval_whole:
        evaluator.train_cross(data, reduced=False)

    else:
        evaluator.train_cross(data)

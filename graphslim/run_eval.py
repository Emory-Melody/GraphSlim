import os
import sys

if os.path.abspath('..') not in sys.path:
    sys.path.append(os.path.abspath('..'))

from graphslim.configs import cli
from graphslim.dataset import *
from graphslim.evaluation import Evaluator

if __name__ == '__main__':
    args = cli(standalone_mode=False)
    data = get_dataset(args.dataset, args)
    evaluator = Evaluator(args)
    if args.eval_whole:
        evaluator.evaluate(data, model_type=args.eval_model, reduced=False)

    else:
        evaluator.evaluate(data, model_type=args.eval_model)

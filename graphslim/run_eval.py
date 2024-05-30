import os
import sys

if os.path.abspath('..') not in sys.path:
    sys.path.append(os.path.abspath('..'))

from graphslim.configs import cli
from graphslim.dataset import *
from graphslim.evaluation import Evaluator
from graphslim.utils import seed_everything

if __name__ == '__main__':
    args = cli(standalone_mode=False)
    data = get_dataset(args.dataset, args)
    evaluator = Evaluator(args)
    if args.eval_whole:
        evaluator.evaluate(data, model_type=args.eval_model, reduced=False)

    else:
        if args.attack is not None:
            data = attack(data, args)
            args.save_path = f'checkpoints'
        evaluator.evaluate(data, model_type=args.eval_model)

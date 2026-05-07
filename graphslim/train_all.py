import os
import sys
import warnings

if os.path.abspath('..') not in sys.path:
    sys.path.append(os.path.abspath('..'))

warnings.filterwarnings("ignore", category=FutureWarning, message=".*pynvml package is deprecated.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*pkg_resources is deprecated as an API.*")

from graphslim.config import get_args
from graphslim.dataset import *
from graphslim.evaluation import *
from graphslim.reduction import create_reducer
from graphslim.tracking import build_tracker
from graphslim.utils import seed_everything


def main():
    args = get_args()
    graph = get_dataset(args.dataset, args, args.load_path)
    seed_everything(args.seed)
    if args.attack is not None:
        graph = attack(graph, args)
    agent = create_reducer(args.method, setting=args.setting, data=graph, args=args)
    tracker = build_tracker(args)
    tracker.log_graph("original", graph, step=0)

    if args.run_reduction > 0:
        reduced_graph = agent.reduce(graph, verbose=args.verbose)
    else:
        reduced_graph = graph
    tracker.log_graph("reduced", reduced_graph, step=max(args.run_reduction, 1))

    evaluator = Evaluator(args)
    res_mean, res_std = evaluator.evaluate(reduced_graph, model_type=args.eval_model)
    tracker.log({"eval/mean": float(res_mean), "eval/std": float(res_std)})
    tracker.finish()


if __name__ == '__main__':
    main()

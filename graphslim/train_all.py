"""Shared training command for the package, examples and benchmark scripts."""

from copy import copy
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graphslim.config import get_args
from graphslim.reduction import create_reducer
from graphslim.tracking import build_tracker


def run(args, *, load_dataset=None, on_result=None):
    """Return summary pairs, with optional dataset loading and result-recording hooks."""
    from graphslim.dataset import attack, get_dataset
    from graphslim.evaluation import Evaluator
    from graphslim.utils import seed_everything

    if args.device.startswith("cuda"):
        import torch

        torch.cuda.set_device(args.device)

    seed_everything(args.seed)
    loader = load_dataset or get_dataset
    graph = loader(args.dataset, args, args.load_path)
    if args.attack is not None:
        graph = attack(graph, args)
    tracker = build_tracker(args)
    results = []
    try:
        tracker.log_graph("original", graph, step=0)
        repetitions = 1 if args.eval_whole else max(args.run_reduction, 1)
        for index in range(repetitions):
            run_args = copy(args)
            run_args.seed = args.seed + index
            seed_everything(run_args.seed)
            if index:
                graph.reset()
            reduced_graph = graph
            if args.run_reduction and not args.eval_whole:
                agent = create_reducer(args.method, setting=args.setting, data=graph, args=run_args)
                reduced_graph = agent.reduce(graph, verbose=args.verbose)
                tracker.log_graph("reduced", reduced_graph, step=index + 1)
            evaluator = Evaluator(run_args)
            result = evaluator.evaluate(
                reduced_graph, model_type=args.eval_model, reduced=not args.eval_whole
            )
            results.append(result)
            if on_result is not None:
                on_result(run_args, reduced_graph, evaluator, result)
            tracker.log(
                {
                    "eval/mean": float(result[0]),
                    "eval/std": float(result[1]),
                    "reduction/seed": run_args.seed,
                },
                step=index + 1,
            )
        return results
    finally:
        tracker.finish()


def main(argv=None, *, default_method=None):
    args = get_args(argv, default_method=default_method)
    return run(args) if hasattr(args, "method") else None


def entrypoint():
    """Keep console exit codes independent from the returned results."""
    main()


if __name__ == "__main__":
    entrypoint()

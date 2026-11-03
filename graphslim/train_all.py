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
    """Execute the configured reduction and downstream evaluation with optional caller hooks.

    :param args: Configuration namespace containing the options used by this operation.
    :param load_dataset: Optional callable ``(dataset, args, load_path)`` used to load the graph. Defaults to ``None``.
    :param on_result: Optional callback ``(run_args, reduced_graph, evaluator, result)`` invoked after each evaluation. Defaults to ``None``.
    :returns: List of evaluation summary pairs, or the selected trial record (possibly None) in HPO mode.
    """
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
    if getattr(args, "hpo", False):
        from graphslim import hpo

        return hpo.run_search(args, graph)
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
    """Parse CLI arguments and execute reduction, evaluation, or hyperparameter search.

    :param argv: Command line arguments; ``None`` reads the process arguments. Defaults to ``None``.
    :param default_method: Method used when the caller does not supply ``--method``. Defaults to ``None``.
    :returns: Result of the shared ``run`` workflow, or ``None`` when no run configuration is produced.
    """
    args = get_args(argv, default_method=default_method)
    return run(args) if hasattr(args, "method") else None


def entrypoint():
    """Invoke the training CLI without returning evaluation data as a process exit code."""
    main()


if __name__ == "__main__":
    entrypoint()

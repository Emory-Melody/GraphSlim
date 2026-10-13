"""Graph-property diagnostics for the explicitly selected method."""

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graphslim.config import get_args


def main(argv=None, *, structure=True):
    args = get_args(argv)
    if not hasattr(args, "method"):
        return None
    from graphslim.dataset import get_dataset
    from graphslim.evaluation import PropertyEvaluator
    from graphslim.utils import seed_everything

    seed_everything(args.seed)
    args.with_structure = int(structure)
    graph = get_dataset(args.dataset, args, args.load_path)
    return PropertyEvaluator(args).evaluate(graph, reduced=not args.eval_whole)


if __name__ == "__main__":
    main()

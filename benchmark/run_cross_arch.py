"""Evaluate saved graphs across downstream architectures."""

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graphslim.config import get_args


def main(argv=None):
    args = get_args(argv)
    if not hasattr(args, "method"):
        return None
    from graphslim.dataset import get_dataset
    from graphslim.evaluation import Evaluator
    from graphslim.utils import seed_everything

    seed_everything(args.seed)
    data = get_dataset(args.dataset, args, args.load_path)
    return Evaluator(args).train_cross(data, reduced=not args.eval_whole)


if __name__ == "__main__":
    main()

"""Evaluate a saved reduced graph or the whole graph."""

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graphslim.config import get_args
from graphslim.train_all import run


def main(argv=None):
    """Run evaluation with reduction disabled unless explicitly requested.

    :param argv: Command line arguments; ``None`` reads the process arguments. Defaults to ``None``.
    :returns: Return value of the shared training pipeline.
    """
    args = get_args(argv)
    if not hasattr(args, "method"):
        return None
    args.run_reduction = 0
    return run(args)


def entrypoint():
    """Invoke the evaluation CLI from the installed console command.

    :returns: Return value of ``main``.
    """
    main()


if __name__ == "__main__":
    entrypoint()

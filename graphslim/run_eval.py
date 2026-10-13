"""Evaluate a saved reduced graph or the whole graph."""

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graphslim.config import get_args
from graphslim.train_all import run


def main(argv=None):
    args = get_args(argv)
    if not hasattr(args, "method"):
        return None
    args.run_reduction = 0
    return run(args)


def entrypoint():
    main()


if __name__ == "__main__":
    entrypoint()

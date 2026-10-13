"""Run the kcenter example through the shared GraphSlim pipeline."""

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graphslim.train_all import main


if __name__ == "__main__":
    main(default_method="kcenter")

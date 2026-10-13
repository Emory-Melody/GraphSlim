"""Structure-free graph-property diagnostics for the selected method."""

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark.graph_property import main

if __name__ == "__main__":
    main(structure=False)

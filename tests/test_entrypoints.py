import importlib
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "name", ["train_gcond", "train_coarsen", "train_node_sparsification", "train_lightweight"]
)
def test_examples_import_without_running_training(name):
    importlib.import_module("examples." + name)


@pytest.mark.parametrize(
    "script",
    [
        "examples/train_gcond.py",
        "examples/train_lightweight.py",
        "examples/train_coarsen.py",
        "examples/train_node_sparsification.py",
        "benchmark/train_all.py",
        "benchmark/run_eval.py",
        "benchmark/run_cross_arch.py",
        "benchmark/graph_property.py",
        "benchmark/graph_property_no_structure.py",
        "graphslim/train_all.py",
        "benchmark/run_nas.py",
    ],
)
def test_help_works_outside_repo_without_loading_data(script, tmp_path):
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(root / script), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert "--dataset" in result.stdout
    assert "Downloading" not in result.stdout


def test_method_discovery_does_not_import_torch_or_optional_methods():
    code = (
        "import sys; from graphslim.reduction import list_methods; "
        "import graphslim.condensation; import graphslim.coarsening; "
        'assert "gcond" in list_methods(); '
        'assert "torch" not in sys.modules; assert "skfuzzy" not in sys.modules'
    )
    subprocess.run([sys.executable, "-c", code], check=True, timeout=20)

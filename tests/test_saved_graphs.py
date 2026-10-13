from types import SimpleNamespace
import pytest

from graphslim.dataset.utils import load_reduced


def test_missing_checkpoints_do_not_silently_evaluate_original_graph(tmp_path):
    args = SimpleNamespace(
        save_path=str(tmp_path),
        method="gcond",
        dataset="cora",
        reduction_rate=0.1,
        seed=1,
        attack=None,
        device="cpu",
    )
    with pytest.raises(FileNotFoundError, match="Run reduction first"):
        load_reduced(args)

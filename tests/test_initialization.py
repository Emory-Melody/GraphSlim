from types import SimpleNamespace
import torch

import graphslim.condensation.gcond_base as base


def test_cached_initialization_loads_features_and_adjacency(tmp_path):
    args = SimpleNamespace(
        init="random",
        save_path=str(tmp_path),
        dataset="cora",
        reduction_rate=0.1,
        seed=1,
        device="cpu",
        method="gcond",
    )
    folder = tmp_path / "reduced_graph/random"
    folder.mkdir(parents=True)
    features = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    adjacency = torch.eye(2)
    torch.save(features, folder / "feat_cora_0.1_1.pt")
    torch.save(adjacency, folder / "adj_cora_0.1_1.pt")
    reducer = base.GCondBase.__new__(base.GCondBase)
    reducer.args = args
    reducer.data = SimpleNamespace()
    loaded_features, loaded_adjacency = reducer.init(with_adj=True, reuse_init=True)
    assert torch.equal(loaded_features, features)
    assert torch.equal(loaded_adjacency, adjacency)
    assert args.method == "gcond"


def test_initialization_cannot_change_condensation_method(monkeypatch):
    args = SimpleNamespace(init="clustering", setting="trans", method="gcond")
    data = SimpleNamespace(feat_syn=torch.ones(2, 2))
    reducer = base.GCondBase.__new__(base.GCondBase)
    reducer.args, reducer.data = args, data

    def create(method, setting, data, args):
        args.method = "clustering"
        return SimpleNamespace(reduce=lambda *a, **kw: data)

    monkeypatch.setattr(base, "create_reducer", create)
    assert torch.equal(reducer.init(), data.feat_syn)
    assert args.method == "gcond"

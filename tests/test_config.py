import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from graphslim.config import cli, get_args, method_config


def test_explicit_short_run_has_valid_checkpoints(tmp_path):
    args = get_args(
        [
            "-M",
            "gcond",
            "-G",
            "-1",
            "--epochs",
            "1",
            "--eval_epochs",
            "2",
            "--run_inter_eval",
            "1",
            "--save_path",
            str(tmp_path),
        ]
    )
    assert args.device == "cpu"
    assert args.epochs == args.eval_interval == 1
    assert args.checkpoints == [-1, 0, 1]
    assert args.eval_epochs == 2
    assert args.run_inter_eval == 1


def test_explicit_interval_rebuilds_checkpoints(tmp_path):
    args = get_args(
        ["-G", "-1", "--epochs", "20", "--eval_interval", "3", "--save_path", str(tmp_path)]
    )
    assert args.checkpoints == list(range(-1, 21, 3))


def test_example_default_loads_selected_config(tmp_path):
    args = get_args(["-G", "-1", "--save_path", str(tmp_path)], default_method="gcond")
    config = json.loads(
        (Path(__file__).resolve().parents[1] / "graphslim/configs/gcond/cora.json").read_text()
    )
    assert args.method == "gcond"
    assert args.lr_feat == config["lr_feat"]
    assert args._method_config_keys == set(config)


@pytest.mark.parametrize(
    "arguments",
    [["-M", "unknown"], ["--epochs", "0"], ["--eval_interval", "0"], ["--run_eval", "0"]],
)
def test_invalid_options_fail_before_loading_data(arguments):
    result = CliRunner().invoke(cli, arguments)
    assert result.exit_code == 2


def test_malformed_method_config_is_not_silently_ignored(tmp_path, monkeypatch):
    import graphslim.config as config
    from types import SimpleNamespace

    monkeypatch.setattr(config, "__file__", str(tmp_path / "config.py"))
    target = tmp_path / "configs/gcond/cora.json"
    target.parent.mkdir(parents=True)
    target.write_text("{broken")
    with pytest.raises(json.JSONDecodeError):
        method_config(SimpleNamespace(method="gcond", dataset="cora"))


def test_recommended_initialization_and_explicit_override(tmp_path, monkeypatch):
    import graphslim.config as config

    monkeypatch.setattr(config, "__file__", str(tmp_path / "config.py"))
    target = tmp_path / "configs/gcond/cora.json"
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps({"init": "clustering"}))
    args = get_args(["-G", "-1", "-M", "gcond", "--save_path", str(tmp_path)])
    assert args.init == "clustering"
    args = get_args(["-G", "-1", "-M", "gcond", "--init", "random", "--save_path", str(tmp_path)])
    assert args.init == "random"


def test_all_shipped_method_configs_are_valid_json():
    root = Path(__file__).resolve().parents[1] / "graphslim/configs"
    for path in root.rglob("*.json"):
        assert isinstance(json.loads(path.read_text()), dict), path


def test_lightweight_json_overrides_profile_and_cli_has_final_priority(tmp_path):
    config_path = tmp_path / "method.json"
    config_path.write_text(
        json.dumps({"epochs": 9, "clustgdd_pre_epochs": 3, "clustgdd_prop_steps": 7})
    )
    args = get_args(
        [
            "-M",
            "clustgdd",
            "-R",
            "0.25",
            "-G",
            "-1",
            "--config",
            str(config_path),
            "--epochs",
            "2",
            "--save_path",
            str(tmp_path),
        ]
    )
    assert args.epochs == 2 and args.clustgdd_pre_epochs == 3
    assert args.clustgdd_prop_steps == 7 and args.clustgdd_sampling_ratio == 0.06


@pytest.mark.parametrize("payload", [[1, 2], {"method": "gcgp"}])
def test_method_json_cannot_replace_identity_or_be_nonobject(tmp_path, payload):
    config_path = tmp_path / "invalid.json"
    config_path.write_text(json.dumps(payload))
    result = CliRunner().invoke(cli, ["-M", "cgc", "-G", "-1", "--config", str(config_path)])
    assert result.exit_code == 2 and "--config" in result.output

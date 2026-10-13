import json

import pytest

from graphslim.doctor import diagnose, main


def test_cpu_runtime_probes_actual_forward_backward():
    report = diagnose()
    assert report["device_requested"] == "cpu"
    assert all(
        value["status"] == "passed"
        for name, value in report["checks"].items()
        if name != "dgl_roundtrip_cpu"
    )
    assert report["checks"]["dgl_roundtrip_cpu"]["status"] == "not_requested"


def test_json_cli_and_file_agree(tmp_path, capsys):
    target = tmp_path / "doctor.json"
    main(["--json", "--output", str(target)])
    assert json.loads(capsys.readouterr().out) == json.loads(target.read_text())


def test_bad_device_is_reported_and_not_silently_replaced(capsys):
    with pytest.raises(SystemExit, match="1"):
        main(["--device", "not_a_device", "--json"])
    assert json.loads(capsys.readouterr().out)["checks"]["torch_import"]["status"] == "failed"


def test_optional_library_stdout_does_not_corrupt_json(monkeypatch, capsys):
    pytest.importorskip("dgl")
    import graphslim.compat

    original = graphslim.compat.to_dgl_graph

    def noisy_converter(*args, **kwargs):
        print("optional backend initialization notice")
        return original(*args, **kwargs)

    monkeypatch.setattr(graphslim.compat, "to_dgl_graph", noisy_converter)
    main(["--dgl", "--json"])
    report = json.loads(capsys.readouterr().out)
    assert report["checks"]["dgl_roundtrip_cpu"]["stdout_messages_suppressed"]

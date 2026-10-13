import json
from copy import deepcopy

import pytest

from benchmark.render_cuda_graph_results import validated_reductions


def pair_record():
    eager = {
        "hashes": {"feat_syn": "same", "adj_syn": "same", "labels_syn": "same"},
        "GP_loss_trace": [0.3, 0.2],
        "stages": [],
        "reduction_seconds_including_setup": 2.0,
    }
    graphed = deepcopy(eager)
    graphed["reduction_seconds_including_setup"] = 1.0
    return {
        "status": "complete",
        "cases": [
            {
                "method": "gcgp",
                "arithmetic": "package",
                "seed": 1,
                "status": "pass",
                "tensors_equal": True,
                "losses_equal": True,
                "speedup_including_setup": 2.0,
                "eager": eager,
                "cuda_graph": graphed,
            }
        ],
    }


def test_reduction_export_retains_exact_raw_record(tmp_path):
    path = tmp_path / "pairs.json"
    record = pair_record()
    path.write_text(json.dumps(record))
    result = validated_reductions(path)
    assert result["record"] == record
    assert len(result["raw_result_sha256"]) == 64


@pytest.mark.parametrize("corruption", ["partial", "duplicate", "hash", "trace", "ratio", "time"])
def test_reduction_export_rejects_inconsistent_pairs(tmp_path, corruption):
    record = pair_record()
    case = record["cases"][0]
    if corruption == "partial":
        record["status"] = "running"
    elif corruption == "duplicate":
        record["cases"].append(deepcopy(case))
    elif corruption == "hash":
        case["cuda_graph"]["hashes"]["feat_syn"] = "different"
    elif corruption == "trace":
        case["cuda_graph"]["GP_loss_trace"][0] = 0.4
    elif corruption == "ratio":
        case["speedup_including_setup"] = 3.0
    else:
        case["cuda_graph"]["reduction_seconds_including_setup"] = 0.0
    path = tmp_path / "pairs.json"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        validated_reductions(path)

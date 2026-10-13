"""Probe the selected runtime and optional DGL graph interchange, without installing anything."""

import argparse
from contextlib import redirect_stdout
import hashlib
import importlib.metadata
import io
import json
import platform
from pathlib import Path
import sys
import tempfile
from datetime import datetime, timezone


def version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def diagnose(device="cpu", *, dgl=False):
    report = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "device_requested": device,
        "versions": {
            name: version(name)
            for name in (
                "graphslim",
                "torch",
                "torch-geometric",
                "torch-scatter",
                "torch-sparse",
                "numpy",
                "scipy",
                "dgl",
            )
        },
        "checks": {},
        "scope": "runtime/operator probes, not certification of every method or CUDA/DGL combination",
    }
    package_root = Path(__file__).resolve().parent
    report["source_sha256"] = {
        name: hashlib.sha256((package_root / name).read_bytes()).hexdigest()
        for name in ("doctor.py", "compat.py", "dataset/convertor.py")
    }

    def check(name, action):
        captured = io.StringIO()
        try:
            with redirect_stdout(captured):
                detail = action()
            report["checks"][name] = {"status": "passed", "detail": detail}
        except Exception as exc:
            report["checks"][name] = {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc).replace(str(Path.home()), "$USER_HOME"),
            }
        if captured.getvalue():
            report["checks"][name]["stdout_messages_suppressed"] = True

    def interpreter():
        if sys.version_info[:2] not in ((3, 10), (3, 11)):
            raise RuntimeError("This package currently declares Python 3.10/3.11 support.")
        return "supported Python minor"

    check("python_support", interpreter)
    try:
        import torch

        selected = torch.device(device)
        report["torch_cuda_runtime"] = torch.version.cuda
        if selected.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "Requested CUDA, but Torch cannot access CUDA. Check wheel/runtime, driver and visible devices."
                )
            report["gpu"] = {
                "name": torch.cuda.get_device_name(selected),
                "capability": list(torch.cuda.get_device_capability(selected)),
                "torch_architectures": torch.cuda.get_arch_list(),
            }
        report["checks"]["torch_import"] = {"status": "passed"}
    except Exception as exc:
        report["checks"]["torch_import"] = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        return report

    def native_sparse():
        index = torch.tensor([[0, 1, 2], [1, 2, 0]], device=selected)
        values = torch.tensor([0.2, 0.7, 1.0], device=selected, requires_grad=True)
        sparse = torch.sparse_coo_tensor(index, values, (4, 4)).coalesce()
        features = torch.ones(4, 3, device=selected, requires_grad=True)
        output = torch.sparse.mm(sparse, features)
        output.sum().backward()
        assert torch.isfinite(features.grad).all() and torch.isfinite(values.grad).all()
        return "COO sparse matmul and value/feature backward; isolated node retained"

    def sparse_extension():
        from torch_sparse import SparseTensor

        index = torch.tensor([[0, 1, 2], [1, 2, 0]], device=selected)
        features = torch.ones(4, 3, device=selected, requires_grad=True)
        matrix = SparseTensor(
            row=index[0], col=index[1], value=torch.ones(3, device=selected), sparse_sizes=(4, 4)
        )
        matrix.matmul(features).sum().backward()
        assert torch.isfinite(features.grad).all()
        return "torch-sparse matmul and feature backward"

    def native_csr():
        index = torch.tensor([[0, 1, 2], [1, 2, 0]], device=selected)
        matrix = (
            torch.sparse_coo_tensor(index, torch.ones(3, device=selected), (4, 4))
            .coalesce()
            .to_sparse_csr()
        )
        features = torch.ones(4, 3, device=selected, requires_grad=True)
        torch.sparse.mm(matrix, features).sum().backward()
        assert torch.isfinite(features.grad).all()
        return "CSR matmul and feature backward; edge-value backward not probed"

    def solve_backward():
        matrix = torch.eye(3, device=selected).requires_grad_()
        target = torch.ones(3, 2, device=selected)
        torch.linalg.solve(matrix, target).square().mean().backward()
        assert torch.isfinite(matrix.grad).all()
        return "dense linear solve and matrix backward"

    def scatter_extension():
        from torch_scatter import scatter_add

        values = torch.ones(4, device=selected, requires_grad=True)
        groups = torch.tensor([0, 0, 1, 2], device=selected)
        scatter_add(values, groups, dim_size=4).sum().backward()
        assert torch.equal(values.grad, torch.ones_like(values))
        return "torch-scatter grouped sum and backward"

    def pyg_model():
        from torch_geometric.nn import GCNConv

        model = GCNConv(3, 2).to(selected)
        features = torch.ones(4, 3, device=selected, requires_grad=True)
        edges = torch.tensor([[0, 1, 2], [1, 2, 0]], device=selected)
        output = model(features, edges)
        output.square().sum().backward()
        assert output.shape == (4, 2) and torch.isfinite(features.grad).all()
        return "PyG GCN forward/backward on requested device"

    check("torch_native_sparse", native_sparse)
    check("torch_native_csr", native_csr)
    check("torch_solve_backward", solve_backward)
    check("torch_sparse_extension", sparse_extension)
    check("torch_scatter_extension", scatter_extension)
    check("pyg_gcn", pyg_model)
    if dgl:

        def dgl_roundtrip():
            from torch_geometric.data import Data
            from graphslim.compat import to_dgl_graph, to_pyg_data

            data = Data(
                x=torch.ones(4, 3),
                y=torch.tensor([0, 1, 0, 1]),
                edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]]),
                edge_weight=torch.tensor([0.1, 0.5, 0.9]),
                train_mask=torch.tensor([True, False, False, False]),
                val_mask=torch.tensor([False, True, False, False]),
                test_mask=torch.tensor([False, False, True, True]),
                num_nodes=4,
            )
            restored = to_pyg_data(to_dgl_graph(data))
            for key in (
                "x",
                "y",
                "edge_index",
                "edge_weight",
                "train_mask",
                "val_mask",
                "test_mask",
            ):
                assert torch.equal(restored[key], data[key]), key
            assert restored.num_nodes == 4
            return "actual DGL CPU interchange with weights, splits and isolated nodes; not a DGL GPU/GraphBolt test"

        check("dgl_roundtrip_cpu", dgl_roundtrip)
    else:
        report["checks"]["dgl_roundtrip_cpu"] = {
            "status": "not_requested",
            "detail": "DGL is optional; pass --dgl to test actual graph interchange.",
        }
    if selected.type == "cuda":
        check("cuda_synchronize", lambda: torch.cuda.synchronize(selected))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="cpu", help="explicit cpu or visible CUDA device, e.g. cuda:0"
    )
    parser.add_argument(
        "--dgl", action="store_true", help="test actual optional DGL CPU interchange"
    )
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument("--output", type=Path, help="save the diagnostic JSON")
    options = parser.parse_args(argv)
    report = diagnose(options.device, dgl=options.dgl)
    serialized = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if options.output:
        options.output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", dir=options.output.parent, prefix=options.output.name, delete=False
        ) as handle:
            handle.write(serialized)
        Path(handle.name).replace(options.output)
    if options.json:
        print(serialized, end="")
    else:
        print(f"GraphSlim runtime probe: Python {report['python']}, device {options.device}")
        for name, check in report["checks"].items():
            print(f"{name}: {check['status']} {check.get('error', '')}")
    if any(check["status"] == "failed" for check in report["checks"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

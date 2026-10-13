"""Compare package mathematics against pinned author source, without training.

Reference clones are inputs, not dependencies of the installed package. Only
explicit AST-selected mathematical definitions are loaded; author main scripts,
GPU environment changes, dataset downloads and LLM/API calls are never executed.
"""

import argparse
import ast
import importlib.util
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

from benchmark.rerun_nr import write_json, utcnow
from graphslim.condensation.cgc import reconstruct_features
from graphslim.condensation.clustgdd import compress_graph, consistency_loss
from graphslim.condensation.gcgp import (
    neural_kernel,
    tangent_kernel,
    gp_predict,
    concrete_adjacency,
)
from graphslim.coarsening.nope import interference_loss
from graphslim.reduction.lightweight import sparse_tensor

REFERENCES = {
    "CGC": ("08ff04dc0d23110ff53445438f30bcff887e7c31", "https://github.com/XYGaoG/CGC"),
    "ClustGDD": (
        "60825c536f794e7a3bb0ff22ae4efc07ea2d3e3a",
        "https://github.com/HKBU-LAGAS/ClustGDD",
    ),
    "GCGP": ("d81ed427cd52d5a0d2a74bdf4328d31db1dcc1eb", "https://github.com/WANGLin0126/GCGP"),
    "NOPE": ("9b9ee38b3966ab550ca9c9e5dd6b393aff8f444a", "https://github.com/dazonglian/NOPE-main"),
}


def load_definitions(path, names, *, methods=None):
    tree = ast.parse(path.read_text())
    selected = [
        node
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names
    ]
    if methods:
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                selected.extend(
                    method
                    for method in node.body
                    if isinstance(method, ast.FunctionDef) and method.name in methods
                )
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future] + selected, type_ignores=[]))
    namespace = {"torch": torch, "nn": torch.nn, "np": np, "F": F, "math": __import__("math")}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


def verify(reference_root):
    reference_root = Path(reference_root)
    for name, (commit, _) in REFERENCES.items():
        actual = subprocess.check_output(
            ["git", "-C", str(reference_root / name), "rev-parse", "HEAD"], text=True
        ).strip()
        if actual != commit:
            raise RuntimeError(f"{name} source is not pinned: {actual}")
        subprocess.run(
            ["git", "-C", str(reference_root / name), "diff", "--exit-code", "HEAD"],
            check=True,
            capture_output=True,
        )
    torch.manual_seed(36)
    x = torch.randn(6, 5, dtype=torch.float64)
    y = torch.randn(4, 5, dtype=torch.float64)
    results = {}

    def compare(name, actual, expected, tolerance=1e-8):
        error = float((actual - expected).abs().max())
        if error > tolerance:
            raise AssertionError(f"{name} max absolute error {error} > {tolerance}")
        results[name] = {"max_absolute_error": error, "tolerance": tolerance, "status": "passed"}

    cgc = load_definitions(
        reference_root / "CGC/scr/utils.py",
        {"get_feature", "laplacian_tensor", "normalize_adj_tensor"},
    )
    adjacency = torch.ones(6, 6, dtype=torch.float64) - torch.eye(6, dtype=torch.float64)
    compare(
        "CGC reconstruction",
        reconstruct_features(adjacency, x, alpha=3, ridge=0),
        cgc["get_feature"](adjacency, x, 3),
        1e-8,
    )
    neural = load_definitions(
        reference_root / "GCGP/gcgp/sgnk.py", {"SimplifyingGraphNeuralKernel"}
    )["SimplifyingGraphNeuralKernel"](L=2)
    compare("GCGP SGNK", neural_kernel(x, y), neural.nodes_gram(x, y, None, None))
    input_x = x.clone().requires_grad_()
    ours = neural_kernel(input_x, y).sum()
    author = neural.nodes_gram(input_x, y, None, None).sum()
    compare(
        "GCGP SGNK input gradient",
        torch.autograd.grad(ours, input_x)[0],
        torch.autograd.grad(author, input_x)[0],
    )
    tangent = load_definitions(reference_root / "GCGP/gcgp/ntk.py", {"NeuralTangentKernel"})[
        "NeuralTangentKernel"
    ](L=2)
    compare("GCGP NTK", tangent_kernel(x, y), tangent.nodes_gram(x, y, None, None))
    structural = load_definitions(reference_root / "GCGP/gcgp/sntk.py", {"StructureBasedNTK"})[
        "StructureBasedNTK"
    ](K=2, L=2)
    ax, ay = torch.eye(6, dtype=torch.float64) + 0.1, torch.eye(4, dtype=torch.float64) + 0.1
    compare("GCGP SNTK", tangent_kernel(x, y, ax, ay, 2, 2), structural.nodes_gram(x, y, ax, ay))
    gpr = load_definitions(reference_root / "GCGP/gcgp/gpr.py", {"GaussianProcessRegression"})[
        "GaussianProcessRegression"
    ](neural.nodes_gram, 0.5, 4)
    targets = torch.rand(4, 3, dtype=torch.float64)
    target_labels = torch.rand(6, 3, dtype=torch.float64)
    pred, _ = gpr(x, y, target_labels, targets, ax, ay, ay, learnA=False)
    compare(
        "GCGP GP prediction",
        gp_predict(neural_kernel(x, y), neural_kernel(y, y), targets, 0.5),
        pred,
    )
    alpha = torch.rand(4, 4)
    alpha = alpha + alpha.T
    compare(
        "GCGP adjacency threshold",
        concrete_adjacency(alpha, training=False),
        gpr.discretize(alpha.clone(), train=False),
    )
    clust = load_definitions(
        reference_root / "ClustGDD/clustgdd_agent_transduct.py",
        set(),
        methods={"graph_compress", "consistency_loss"},
    )
    assignment = np.array([0, 0, 1, 1, 2, 2])
    sparse = sparse_tensor(sp.csr_matrix(adjacency.float().numpy()))
    _, condensed = clust["graph_compress"](None, torch.from_numpy(assignment), sparse, [sparse])
    compare(
        "ClustGDD graph compression",
        torch.from_numpy(
            compress_graph(sp.csr_matrix(adjacency.numpy()), assignment, 3).toarray()
        ).float(),
        condensed.to_dense(),
        1e-6,
    )
    outputs = [torch.randn(5, 3, dtype=torch.float64).log_softmax(1) for _ in range(3)]
    compare(
        "ClustGDD consistency",
        consistency_loss(outputs),
        clust["consistency_loss"](None, outputs, 0.5),
    )
    nope = load_definitions(reference_root / "NOPE/model/merge.py", {"calculate_loss_m1_fast"})
    relaxed = load_definitions(
        reference_root / "NOPE/model/merge_relax.py", {"calculate_lm2_loss_fast"}
    )
    vectors = x.float().numpy()
    neighbors = {0: {1, 2}, 1: {0, 2, 3}, 2: {0, 1, 4}, 3: {1, 4}, 4: {2, 3, 5}, 5: {4}}
    sizes = np.array([1, 2, 1, 3, 1, 2], dtype=np.float32)
    sums = np.array([np.square(vectors[sorted(neighbors[i])] @ vectors[i]).sum() for i in range(6)])
    graph = SimpleNamespace(
        vectors=vectors,
        sizes=sizes,
        sum_sq=sums,
        adj=neighbors,
        degrees=np.array([len(neighbors[i]) for i in range(6)]),
    )
    compare(
        "NOPE interference",
        torch.tensor(interference_loss(0, 1, vectors, sizes, neighbors, sum_sq=sums)),
        torch.tensor(nope["calculate_loss_m1_fast"](0, 1, graph)),
        1e-5,
    )
    compare(
        "NOPE star interference",
        torch.tensor(interference_loss(0, 1, vectors, sizes, neighbors, relaxed=True)),
        torch.tensor(relaxed["calculate_lm2_loss_fast"]([0], [1], graph)[0]),
        1e-5,
    )
    return {
        "utc": utcnow(),
        "scope": "core mathematics only; not end-to-end paper training parity",
        "references": {
            name: {"commit": commit, "url": url} for name, (commit, url) in REFERENCES.items()
        },
        "checks": results,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, default=Path("runs/reference"))
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args(argv)
    result = verify(options.references)
    write_json(options.output, result)
    print(f"{len(result['checks'])} author-core checks passed: {options.output}")


if __name__ == "__main__":
    main()

"""Cross-dataset paired SGNK CUDA-capture experiment, separate from defaults.

Released SGNK arithmetic and tensor ridge division are retained. Sampling, GP
solve, Adam and evaluation stay eager. Captures use fixed minibatch shapes but
receive the actual new training batch on every update. No full N-by-N cache.
"""

from copy import deepcopy
import statistics
import time

import torch
import torch.nn.functional as F

from benchmark.rerun_nr import array_hash, write_json
from benchmark.verify_lightweight import load_definitions, REFERENCES
from graphslim.evaluation import Evaluator
from graphslim.reduction.lightweight import normalized_adjacency, propagate
from graphslim.utils import seed_everything


def check(data, args, seeds, folder, repeats=3):
    from pathlib import Path
    import subprocess

    root = Path(__file__).resolve().parents[1]
    reference = root / "runs/reference/GCGP"
    commit = subprocess.check_output(
        ["git", "-C", str(reference), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != REFERENCES["GCGP"][0]:
        raise RuntimeError("GCGP author reference changed")
    subprocess.run(
        ["git", "-C", str(reference), "diff", "--exit-code", "HEAD"],
        check=True,
        capture_output=True,
    )
    kernel = load_definitions(reference / "gcgp/sgnk.py", {"SimplifyingGraphNeuralKernel"})[
        "SimplifyingGraphNeuralKernel"
    ](L=args.gcgp_layers)
    prefix = "full" if args.setting == "trans" else "train"
    features = getattr(data, f"feat_{prefix}").to(args.device)
    train = (
        data.idx_train.to(args.device)
        if args.setting == "trans"
        else torch.arange(len(features), device=args.device)
    )
    query = propagate(
        normalized_adjacency(getattr(data, f"adj_{prefix}"), args.device), features, args.gcgp_k
    )[train].detach()
    targets = F.one_hot(data.labels_train.to(args.device), data.nclass).float()
    size = max(1, round(len(train) * args.reduction_rate))
    batch_size = min(args.gcgp_batch_size, len(train))
    identity = torch.eye(size, device=args.device)
    n_tensor = torch.tensor(size, device=args.device)

    def pair(original, synthetic):
        return (
            kernel.nodes_gram(synthetic, synthetic, None, None),
            kernel.nodes_gram(original, synthetic, None, None),
        )

    rows = []
    for seed in seeds:
        trials = {"eager": [], "capture": []}
        setup_seconds = []
        parameters = {}
        histories = {}
        for repeat in range(repeats):
            for backend in ("eager", "capture"):
                seed_everything(seed)
                synthetic = torch.nn.Parameter(torch.rand(size, query.shape[1], device=args.device))
                labels = torch.nn.Parameter(torch.rand(size, data.nclass, device=args.device))
                optimizer = torch.optim.Adam([synthetic, labels], lr=args.gcgp_lr)
                torch.cuda.synchronize()
                started = time.monotonic()
                calculate = pair
                if backend == "capture":
                    calculate = torch.cuda.make_graphed_callables(
                        pair, (query[:batch_size].clone(), synthetic)
                    )
                torch.cuda.synchronize()
                setup = time.monotonic() - started
                if backend == "capture":
                    setup_seconds.append(setup)
                losses = []
                started = time.monotonic()
                for epoch in range(args.epochs):
                    chosen = (
                        torch.randperm(len(train), device=args.device)[:batch_size]
                        if len(train) > batch_size
                        else torch.arange(len(train), device=args.device)
                    )
                    ss, ts = calculate(query[chosen], synthetic)
                    coefficients = torch.linalg.solve(
                        ss + args.gcgp_ridge * ss.trace() * identity / n_tensor, labels
                    )
                    loss = F.mse_loss((ts @ coefficients).softmax(1), targets[chosen])
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Non-finite parity loss")
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    losses.append(float(loss.detach()))
                torch.cuda.synchronize()
                trials[backend].append(time.monotonic() - started)
                hash_pair = (array_hash(synthetic), array_hash(labels))
                if backend in parameters and parameters[backend]["hashes"] != hash_pair:
                    raise RuntimeError("Non-deterministic repeat in paired experiment")
                parameters[backend] = dict(
                    hashes=hash_pair,
                    features=synthetic.detach().clone(),
                    labels=labels.detach().clone(),
                )
                histories.setdefault(backend, []).append(losses)
                write_json(
                    folder / "parity_progress.json",
                    dict(
                        seed=seed,
                        repeat=repeat,
                        backend=backend,
                        steps=len(losses),
                        seconds=trials[backend][-1],
                        hashes=hash_pair,
                    ),
                )
                del calculate, optimizer, synthetic, labels
                torch.cuda.empty_cache()
        loss_match = all(a == b for a, b in zip(histories["eager"], histories["capture"]))
        hash_match = parameters["eager"]["hashes"] == parameters["capture"]["hashes"]
        downstream = {}
        if loss_match and hash_match:
            for backend in ("eager", "capture"):
                evaluated = deepcopy(data)
                evaluated.feat_syn = parameters[backend]["features"]
                evaluated.labels_syn = parameters[backend]["labels"]
                evaluated.adj_syn = identity
                seed_everything(0)
                downstream[backend] = list(
                    map(
                        float,
                        Evaluator(args).test(
                            evaluated, "GCN", verbose=False, reduced=True, mode="cross"
                        ),
                    )
                )
        eager, captured = (statistics.median(trials[key]) for key in ("eager", "capture"))
        row = dict(
            seed=seed,
            warm_seconds=trials,
            graph_setup_seconds=setup_seconds,
            warm_speedup=eager / captured,
            speedup_including_setup=eager / (captured + statistics.median(setup_seconds)),
            loss_histories_exact=loss_match,
            parameter_hashes_exact=hash_match,
            downstream_exact=bool(downstream) and downstream["eager"] == downstream["capture"],
            downstream=downstream,
            parameter_hashes={key: value["hashes"] for key, value in parameters.items()},
            loss_history_sha256={
                key: [array_hash(torch.tensor(value, dtype=torch.float64)) for value in values]
                for key, values in histories.items()
            },
        )
        rows.append(row)
        write_json(folder / "parity_rows.json", rows)
    return dict(
        passed=all(
            row["loss_histories_exact"]
            and row["parameter_hashes_exact"]
            and row["downstream_exact"]
            for row in rows
        ),
        rows=rows,
        author_commit=commit,
        steps=args.epochs,
        repeats=repeats,
        batch_size=batch_size,
        synthetic_nodes=size,
        precision="float32",
        default_changed=False,
        scope="paired author SGNK/GP arithmetic in shared package preprocessing and GCN evaluation; not author-repository end-to-end or package-default speedup",
    )

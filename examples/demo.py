"""Run the paper's GCond-to-GAT demo with a readable console transcript.

Run from the repository root: ``python examples/demo.py``. GraphSlim options
can be appended, for example ``--epochs 20 --run_eval 3`` for a shorter run.
Full configuration and progress messages are saved in ``demo-details.log``.
"""

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graphslim.config import get_args


def section(title, rows=()):
    """Use the same compact, plain-text layout in terminals and the appendix."""
    print(f"\n{title}\n{'-' * 60}", flush=True)
    for label, value in rows:
        print(f"  {label:<22} {value}", flush=True)


def main(argv=None):
    config_output = StringIO()
    with redirect_stdout(config_output):
        args = get_args([
            "-D", "cora", "-M", "gcond", "--init", "kcenter", "-R", "0.5",
            "--save_path", "./checkpoints/demo",
            *(sys.argv[1:] if argv is None else argv),
        ])
    if not hasattr(args, "method"):  # Preserve Click's --help output.
        print(config_output.getvalue(), end="")
        return

    output_dir = Path(args.save_path)
    details_path = output_dir / "demo-details.log"
    print("GraphSlim | Composable API demo", flush=True)
    section("[1/4] Configuration", [
        ("Dataset / reducer", f"{args.dataset} / {args.method}"),
        ("Initializer / model", f"{args.init} / GAT"),
        ("Device / seed", f"{args.device} / {args.seed}"),
        ("Training-node budget", f"{100 * args.reduction_rate:g}%"),
        ("Reduction epochs", args.epochs),
        ("Outer / inner loops", f"{args.outer_loop} / {args.inner_loop}"),
        ("Eval epochs / fits", f"{args.eval_epochs} / {args.run_eval}"),
    ])

    with details_path.open("w", encoding="utf-8") as details:
        details.write(config_output.getvalue())
        with redirect_stdout(details), redirect_stderr(details):
            import torch
            from graphslim.dataset import get_dataset
            from graphslim.dataset.utils import load_reduced
            from graphslim.evaluation import Evaluator
            from graphslim.reduction import create_reducer
            from graphslim.utils import seed_everything

            if args.device.startswith("cuda"):
                torch.cuda.set_device(args.device)
            seed_everything(args.seed)
            graph = get_dataset(args.dataset, args, args.load_path)

        section("[2/4] Original graph", [
            ("Nodes / features", f"{graph.num_nodes:,} / {graph.feat_full.shape[1]:,}"),
            ("Train / val / test", f"{len(graph.idx_train):,} / "
             f"{len(graph.idx_val):,} / {len(graph.idx_test):,}"),
            ("Classes", graph.nclass),
        ])
        section("[3/4] Reduced graph")
        with redirect_stdout(details), redirect_stderr(details):
            reducer = create_reducer(args.method, setting=args.setting, data=graph, args=args)
            reduced = reducer.reduce(graph, verbose=True)
            # Report the saved, validation-selected graph used by Evaluator.
            reduced.adj_syn, reduced.feat_syn, reduced.labels_syn = load_reduced(args, reduced)

        shapes = {
            name: list(getattr(reduced, name).shape)
            for name in ("feat_syn", "adj_syn", "labels_syn")
        }
        for name, shape in shapes.items():
            print(f"  {name:<22} {tuple(shape)}", flush=True)
        section("[4/4] GAT evaluation")
        with redirect_stdout(details), redirect_stderr(details):
            evaluator = Evaluator(args)
            mean, std = evaluator.evaluate(reduced, model_type="GAT", verbose=False)

        for index, result in enumerate(evaluator.last_results, start=1):
            label = f"Fit {index} (seed {result['seed']})"
            print(f"  {label:<22} test {100 * result['test']:.2f}%", flush=True)
        print(f"  {'Test mean +/- std':<22} {100 * mean:.2f}% +/- {100 * std:.2f}%", flush=True)

        summary = {
            "config": {
                key: value for key, value in vars(args).items()
                if isinstance(value, (str, int, float, bool, type(None)))
            },
            "shapes": shapes,
            "evaluation_model": "GAT",
            "fits": evaluator.last_results,
            "test_mean": float(mean),
            "test_population_std": float(std),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "hardware": torch.cuda.get_device_name(args.device)
            if args.device.startswith("cuda") else "CPU",
        }
        (output_dir / "demo-summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )

    section("Saved artifacts", [
        ("Checkpoints", output_dir / "reduced_graph" / args.method),
        ("Detailed log", details_path),
        ("Run summary", output_dir / "demo-summary.json"),
    ])
    return float(mean), float(std)


if __name__ == "__main__":
    main()

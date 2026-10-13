"""Run the published APPNP architecture-search evaluation."""

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from graphslim.config import get_args


def main(argv=None):
    args = get_args(argv)
    if not hasattr(args, "method"):
        return None
    from graphslim.dataset import get_dataset
    from graphslim.evaluation import NasEvaluator
    from graphslim.utils import seed_everything

    seed_everything(args.seed)
    data = get_dataset(args.dataset, args, args.load_path)
    evaluator = NasEvaluator(args)
    save_path = Path(evaluator.save_path)
    if not (save_path / "results_ori.csv").exists():
        args.logger.info("No original results found; evaluating the original architectures.")
        evaluator.evaluate_ori(data)
        evaluator.test_params_ori(data)

    results = []
    seeds = (1, 2, 3) if args.method in ("random", "kcenter", "averaging", "vng") else (args.seed,)
    for seed in seeds:
        args.seed = seed
        seed_everything(seed)
        evaluator = NasEvaluator(args)
        evaluator.evaluate_syn(data)
        accuracy = evaluator.test_params_syn(data)
        correlation, rank_correlation = evaluator.cal_pearson()
        results.append((accuracy, correlation, rank_correlation))
    summary = np.asarray(results).mean(axis=0)
    args.logger.info("NAS accuracy=%s, accuracy correlation=%s, rank correlation=%s", *summary)
    return summary


if __name__ == "__main__":
    main()

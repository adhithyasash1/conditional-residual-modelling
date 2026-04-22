#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from conditional_residual_modelling.config import PROCESSED_DIR, RANDOM_SEED
from conditional_residual_modelling.pipeline import rebuild_processed_inputs, run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the SPX lower-tail uncertainty pipeline.")
    parser.add_argument("--download", action="store_true", help="Refresh raw inputs before training.")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Training random seed.")
    parser.add_argument("--walk-forward-folds", type=int, default=3, help="Number of walk-forward folds.")
    parser.add_argument("--skip-plots", action="store_true", help="Skip figure generation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.download:
        rebuild_processed_inputs(download=True)
    results = run_training(
        seed=args.seed,
        walk_forward_folds=args.walk_forward_folds,
        make_plots=not args.skip_plots,
    )
    out_path = PROCESSED_DIR / "training_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()

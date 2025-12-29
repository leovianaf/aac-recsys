"""
Main entry point for AAC pictogram recommendation system.

Example usage (with preprocessing):
python -m aac_recsys.main --model baseline --preprocess --preprocess-force

Example usage (without preprocessing):
python -m aac_recsys.main \
  --model baseline \
  --min-train-days 60 \
  --max-train-days 180 \
  --test-days 7 \
  --step-days 7 \
  --rank-k 60 \
  --ks 1,3,5 \
  --half-life-days 30
"""

from __future__ import annotations

import argparse
from typing import Callable

from aac_recsys.config import PROCESSED_DATA_DIR, logger
from aac_recsys.pre_processing import run_preprocess
from aac_recsys.predict import FoldConfig, run_predict
from aac_recsys.models.ranker_base import Ranker
from aac_recsys.models.baseline import BaselineRanker, BaselineParams


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    # preprocess
    p.add_argument("--preprocess", action="store_true", help="Run preprocessing before evaluation")
    p.add_argument("--preprocess-force", action="store_true", help="Force preprocessing overwrite")
    p.add_argument("--preprocess-plots", action="store_true", help="Generate plots during preprocessing")
    p.add_argument("--preprocess-user-idx", type=int, default=None, help="Preprocess only user_{idx}")

    p.add_argument("--model", type=str, default="baseline")
    p.add_argument("--min-train-days", type=int, default=60)
    p.add_argument("--max-train-days", type=int, default=180)
    p.add_argument("--test-days", type=int, default=7)
    p.add_argument("--step-days", type=int, default=7)
    p.add_argument("--rank-k", type=int, default=60)
    p.add_argument("--ks", type=str, default="1,3,5")

    # baseline params
    p.add_argument("--half-life-days", type=int, default=30)

    return p.parse_args()


def make_ranker_factory(args) -> Callable[[], Ranker]:
  if args.model == "baseline":
    params = BaselineParams(
      half_life_days=args.half_life_days,
      rank_k_default=args.rank_k,
    )
    return lambda: BaselineRanker(params)
  raise ValueError(f"Unknown model: {args.model}")

def main() -> None:
    args = parse_args()

    if args.preprocess:
      run_preprocess(
        user_idx=args.preprocess_user_idx,
        force=args.preprocess_force,
        plots=args.preprocess_plots,
      )

    if args.model == "baseline":
      baseline_path = PROCESSED_DATA_DIR / "df_baseline.parquet"
      if not baseline_path.exists():
        logger.warning("df_baseline.parquet not found. Run with --preprocess to generate it.")

    ks = tuple(int(x.strip()) for x in args.ks.split(",") if x.strip())

    fold_cfg = FoldConfig(
      min_train_days=args.min_train_days,
      max_train_days=args.max_train_days,
      test_days=args.test_days,
      step_days=args.step_days,

      ks=ks,
      rank_k=args.rank_k,
    )

    ranker_factory = make_ranker_factory(args)
    run_predict(fold_cfg=fold_cfg, ranker_factory=ranker_factory)


if __name__ == "__main__":
    main()
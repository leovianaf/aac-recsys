"""
Rolling evaluation for the AAC pictogram ranking baseline.

This script:
- Loads per-user timelines from data/processed/user_*/processed.parquet.
- Sorts interactions by timestamp.
- Builds ranking artifacts using only TRAIN history in each fold:
  - label click frequencies
  - per-label fuzzy time-of-day profiles
- Evaluates on the next TEST window without using future data.
- Uses an expanding window:
  - train_days = 60
  - test_days  = 7
  - step_days  = 7 (history updated every week)
- For each test interaction, ranks labels using:
    score(label) = freq_rel(label) * prob_temporal(label)
  where prob_temporal is the dot product between:
    - current fuzzy memberships (now_ts.hour)
    - label's learned fuzzy period profile (from train)
- Computes metrics per fold:
  - accuracy@1 (top-1)
  - macro F1@1 (top-1)
  - recall@K for K in {1,3,5}
  - mrr@K for K in {1,3,5}
- Saves per-user, per-fold metrics to reports/predict_metrics_{model}.csv
- Aggregates metrics by real test window (test_start, test_end), computing
  weighted averages across users and saving to:
  reports/predict_window_summary_{model}.csv
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Callable, Dict, Tuple

import json
import pandas as pd

from aac_recsys.config import PROCESSED_DATA_DIR, REPORTS_DIR, logger
from aac_recsys.metrics import calculate_metrics
from aac_recsys.models.ranker_base import Ranker


@dataclass
class FoldConfig:
  min_train_days: int = 60
  max_train_days: int = 180
  test_days: int = 7
  step_days: int = 7
  ks: tuple[int, ...] = (1, 3, 5)
  rank_k: int = 60  # ranks top-60 and after get @1/@3/@5


def generate_folds(df: pd.DataFrame, ts_col: str, cfg: FoldConfig) -> list[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
  """
  Return a list of folds with (train_start, train_end, test_start, test_end).
  Window of training and testing based on time.
  """
  if df.empty:
    return []

  t_min = df[ts_col].min()
  t_max = df[ts_col].max()

  min_train = timedelta(days=cfg.min_train_days)
  max_train = timedelta(days=cfg.max_train_days)
  test_len = timedelta(days=cfg.test_days)
  step = timedelta(days=cfg.step_days)

  folds = []
  test_start = t_min + min_train
  test_end = test_start + test_len

  while test_end <= t_max:
    train_end = test_start
    train_start = max(t_min, train_end - max_train)

    if (train_end - train_start) >= min_train:
      folds.append((train_start, train_end, test_start, test_end))

    test_start = test_start + step
    test_end = test_start + test_len

  return folds


def load_user_timelines(processed_dir: Path) -> Dict[str, Path]:
  paths = list(processed_dir.glob("user_*/processed.parquet"))
  out: Dict[str, Path] = {}
  for p in paths:
    user_id = p.parent.name.replace("user_", "")
    out[user_id] = p
  return out


# def _append_user_metrics_csv(*, csv_path: Path, new_df: pd.DataFrame) -> None:
#   """
#   Append-safe write:
#   - If csv exists, read it and append new_df
#   - Deduplicate rows by a stable key to avoid doubling when re-running
#   """
#   if new_df.empty:
#     return

#   # Ensure params is JSON string for diffing / storage
#   if "params" in new_df.columns and isinstance(new_df["params"].iloc[0], dict):
#     new_df = new_df.copy()
#     new_df["params"] = new_df["params"].apply(lambda d: json.dumps(d, ensure_ascii=False, sort_keys=True))

#   if csv_path.exists():
#     old_df = pd.read_csv(csv_path)

#     combined = pd.concat([old_df, new_df], ignore_index=True)

#     # Dedup key: identifies a fold evaluation uniquely for a user+model+window
#     dedup_cols = ["user_id", "model", "fold", "train_start", "train_end", "test_start", "test_end"]
#     existing = [c for c in dedup_cols if c in combined.columns]
#     if existing:
#       combined = combined.drop_duplicates(subset=existing, keep="last")
#   else:
#     combined = new_df

#   csv_path.parent.mkdir(parents=True, exist_ok=True)
#   combined.to_csv(csv_path, index=False)


def evaluate_user_timeline(
  user_id: str,
  df: pd.DataFrame,
  fold_cfg: FoldConfig,
  ranker_factory: Callable[[], Ranker],
  label_col: str = "card_enc",
  ts_col: str = "timestamp",
) -> pd.DataFrame:
  """
  Evaluates a single user using rolling evaluation.
  Returns a DataFrame with metrics per fold.
  """
  df = df.sort_values(ts_col).reset_index(drop=True)

  folds = generate_folds(df, ts_col=ts_col, cfg=fold_cfg)
  if not folds:
    logger.warning(f"[user={user_id}] With insufficient folds to evaluate (data too short).")
    return pd.DataFrame()

  logger.info(f"[user={user_id}] {len(folds)} folds for evaluation.")
  rows: list[dict] = []

  for fold_idx, (train_start, train_end, test_start, test_end) in enumerate(folds):

    train_df = df[(df[ts_col] >= train_start) & (df[ts_col] < train_end)].copy()
    test_df = df[(df[ts_col] >= test_start) & (df[ts_col] < test_end)].copy()

    if train_df.empty or test_df.empty:
      logger.warning(f"[user={user_id} fold={fold_idx}] No training or testing data; skipping fold.")
      continue

    logger.info(
      f"[user={user_id} fold={fold_idx}] "
      f"train={train_start.date()}..{(train_end - pd.Timedelta(seconds=1)).date()} (n={len(train_df)}) | "
      f"test={test_start.date()}..{(test_end - pd.Timedelta(seconds=1)).date()} (n={len(test_df)})"
    )

    ranker = ranker_factory()
    ranker.fit(train_df, label_col=label_col, ts_col=ts_col, train_end=train_end.to_pydatetime())

    y_true: list[int] = []
    y_pred_topk: list[list[int]] = []
    y_pred_top1: list[int] = []

    # Event-by-event evaluation (no test future leakage)
    for _, row in test_df.iterrows():
      ts = row[ts_col].to_pydatetime()
      true_label = int(row[label_col])

      preds = ranker.rank(now_ts=ts, k=fold_cfg.rank_k)

      y_true.append(true_label)
      y_pred_topk.append(preds[: fold_cfg.rank_k])
      y_pred_top1.append(preds[0] if preds else -1)

    metrics = calculate_metrics(
      ks=fold_cfg.ks,
      y_true=y_true,
      y_pred_top1=y_pred_top1,
      y_pred_topk=y_pred_topk,
    )

    rows.append(
      {
        "user_id": user_id,
        "model": ranker.name,
        "params": {
          **ranker.params_dict(),
          **fold_cfg.__dict__,
        },
        "fold": fold_idx,
        "is_final_fold": fold_idx == (len(folds) - 1),
        "train_start": train_start,
        "train_end": train_end,
        "test_start": test_start,
        "test_end": test_end,
        "n_train": len(train_df),
        "n_test": len(test_df),
        **metrics,
      }
    )

    logger.info(
      f"[user={user_id} fold={fold_idx}] "
      f"acc@1={metrics['accuracy_top1']:.4f} "
      f"f1_macro@1={metrics['f1_macro_top1']:.4f} "
      + ", ".join(f"R@{k}={metrics[f'recall@{k}']:.4f}" for k in fold_cfg.ks)
    )

  metrics_df = pd.DataFrame(rows)

  if not metrics_df.empty:
    metrics_df["params"] = metrics_df["params"].apply(lambda d: json.dumps(d, ensure_ascii=False, sort_keys=True))

  return metrics_df


def _weighted_mean(x: pd.Series, w: pd.Series) -> float:
  x = pd.to_numeric(x, errors="coerce")
  w = pd.to_numeric(w, errors="coerce").fillna(0.0)
  s = float(w.sum())
  if s <= 0:
    return float("nan")

  return float((x * w).sum() / s)


def run_predict(
  *,
  fold_cfg: FoldConfig,
  ranker_factory: Callable[[], Ranker],
  processed_dir: Path = PROCESSED_DATA_DIR,
) -> None:
  model_name = ranker_factory().name

  user_files = load_user_timelines(processed_dir)
  if not user_files:
    logger.error(f"No user timelines found in: {processed_dir}")
    return

  all_metrics: list[pd.DataFrame] = []

  for user_id, path in user_files.items():
    logger.info(f"Loading timeline for user={user_id} from {path}")
    df = pd.read_parquet(path)

    if "timestamp" not in df.columns or "card_enc" not in df.columns:
      logger.error(f"[user={user_id}] Missing required columns: timestamp/card_enc")
      continue

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=False, errors="coerce")
    df = df.dropna(subset=["timestamp", "card_enc"]).copy()
    df["card_enc"] = df["card_enc"].astype(int)

    mdf = evaluate_user_timeline(
      user_id=user_id,
      df=df,
      fold_cfg=fold_cfg,
      ranker_factory=ranker_factory,
    )
    if not mdf.empty:
      all_metrics.append(mdf)

  if not all_metrics:
    logger.error("No metrics generated.")
    return

  final_df = pd.concat(all_metrics, ignore_index=True)

  out_csv = REPORTS_DIR / f"predict_metrics_{model_name}.csv"
  final_df.to_csv(out_csv, index=False)
  logger.success(f"Metrics saved to: {out_csv}")

  metric_cols = (
    ["accuracy_top1", "f1_macro_top1"]
    + [f"recall@{k}" for k in fold_cfg.ks]
    + [f"mrr@{k}" for k in fold_cfg.ks]
  )

  final_df = final_df.copy()
  final_df["test_start"] = pd.to_datetime(final_df["test_start"], errors="coerce")
  final_df["test_end"] = pd.to_datetime(final_df["test_end"], errors="coerce")
  final_df["n_test"] = pd.to_numeric(final_df["n_test"], errors="coerce").fillna(0).astype(int)

  group_cols = ["model", "test_start", "test_end"]

  rows = []
  for (model, test_start, test_end), g in final_df.groupby(group_cols, dropna=False):
    w = g["n_test"]

    row = {
      "model": model,
      "test_start": test_start,
      "test_end": test_end,
      "users_in_window": int(g["user_id"].nunique()),
      "total_n_test": int(w.sum()),
      "total_n_train": int(pd.to_numeric(g["n_train"], errors="coerce").fillna(0).sum()),
    }

    # mean metrics weighted by n_test
    for m in metric_cols:
      row[m] = _weighted_mean(g[m], w)

    rows.append(row)

  window_summary_df = (
    pd.DataFrame(rows)
    .sort_values(["model", "test_start", "test_end"])
    .reset_index(drop=True)
  )

  out_window_csv = REPORTS_DIR / f"predict_window_summary_{model_name}.csv"
  window_summary_df.to_csv(out_window_csv, index=False)
  logger.success(f"Window summary saved to: {out_window_csv}")

  summary_cols = ["accuracy_top1", "f1_macro_top1"] + [f"recall@{k}" for k in fold_cfg.ks] + [f"mrr@{k}" for k in fold_cfg.ks]
  summary = final_df.groupby("user_id")[summary_cols].mean().reset_index()
  logger.success("Summary (average per user):\n" + summary.to_string(index=False))

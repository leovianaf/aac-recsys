"""
Rolling evaluation (walk-forward / expanding window) for the AAC pictogram ranking baseline.

This script:
- Loads per-user timelines from data/processed/user_*/processed.parquet.
- Sorts interactions by timestamp.
- Builds ranking artifacts using only TRAIN history in each fold:
  - label click frequencies
  - per-label fuzzy time-of-day profiles
- Evaluates on the next TEST window without using future data.
- Uses an expanding window:
  - train_days = 30
  - test_days  = 14
  - step_days  = 14 (history updated every 2 weeks)
- For each test interaction, ranks labels using:
    score(label) = freq_rel(label) * prob_temporal(label)
  where prob_temporal is the dot product between:
    - current fuzzy memberships (now_ts.hour)
    - label's learned fuzzy period profile (from train)
- Computes metrics per fold:
  - accuracy@1 (top-1)
  - macro F1@1 (top-1)
  - recall@K for K in {1,3,5}
- Saves per-fold metrics to reports/rolling_eval_metrics.csv and prints a per-user summary.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from sklearn.metrics import f1_score, accuracy_score

from config import PROCESSED_DATA_DIR, REPORTS_DIR, logger
from ranking import build_artifacts, rank_topk


@dataclass
class FoldConfig:
  train_days: int = 30
  test_days: int = 14
  step_days: int = 14
  ks: tuple[int, ...] = (1, 3, 5)
  rank_k: int = 60  # ranks top-60 and after get @1/@3/@5


def recall_at_k(y_true: List[int], y_pred_topk: List[List[int]], k: int) -> float:
  """
  y_pred_topk: top-k predictions for each example
  """
  hits = 0
  n = len(y_true)
  if n == 0:
    return 0.0
  for yt, preds in zip(y_true, y_pred_topk):
    if yt in preds[:k]:
      hits += 1
  return hits / n


def generate_folds(df: pd.DataFrame, ts_col: str, cfg: FoldConfig) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
  """
  Return a list of folds with (train_start, train_end, test_start, test_end).
  Window of training and testing based on time.
  """
  if df.empty:
    return []

  t_min = df[ts_col].min()
  t_max = df[ts_col].max()

  train_len = timedelta(days=cfg.train_days)
  test_len = timedelta(days=cfg.test_days)
  step = timedelta(days=cfg.step_days)

  folds = []
  train_start = t_min
  train_end = train_start + train_len
  test_start = train_end
  test_end = test_start + test_len

  while test_end <= t_max:
    folds.append((train_start, train_end, test_start, test_end))
    # expanding window
    train_end = train_end + step
    test_start = train_end
    test_end = test_start + test_len

  return folds


def evaluate_user_timeline(
  user_id: str,
  df: pd.DataFrame,
  cfg: FoldConfig,
  label_col: str = "card_enc",
  ts_col: str = "timestamp",
) -> pd.DataFrame:
  """
  Evaluates a single user using rolling/walk-forward evaluation.
  Returns a DataFrame with metrics per fold.
  """
  df = df.sort_values(ts_col).reset_index(drop=True)

  folds = generate_folds(df, ts_col=ts_col, cfg=cfg)
  if not folds:
    logger.warning(f"[user={user_id}] Sem folds suficientes para avaliar (dados curtos demais).")
    return pd.DataFrame()

  rows = []
  logger.info(f"[user={user_id}] {len(folds)} folds para avaliação.")

  for fold_idx, (train_start, train_end, test_start, test_end) in enumerate(folds):

    train_df = df[(df[ts_col] >= train_start) & (df[ts_col] < train_end)].copy()
    test_df = df[(df[ts_col] >= test_start) & (df[ts_col] < test_end)].copy()

    if train_df.empty or test_df.empty:
      logger.warning(f"[user={user_id} fold={fold_idx}] No training or testing data; skipping fold.")
      continue

    if len(test_df) < 30:
      logger.warning(f"[user={user_id} fold={fold_idx}] Less than 30 test events; skipping fold.")
      continue

    logger.info(
      f"[user={user_id} fold={fold_idx}] "
      f"train={train_start.date()}..{(train_end - pd.Timedelta(seconds=1)).date()} (n={len(train_df)}) | "
      f"test={test_start.date()}..{(test_end - pd.Timedelta(seconds=1)).date()} (n={len(test_df)})"
    )

    artifacts = build_artifacts(train_df, label_col=label_col, ts_col=ts_col)

    y_true: List[int] = []
    y_pred_topk: List[List[int]] = []
    y_pred_top1: List[int] = []

    # Event-by-event evaluation (no test future leakage)
    for _, row in test_df.iterrows():
      ts = row[ts_col].to_pydatetime()
      true_label = int(row[label_col])

      preds = rank_topk(artifacts, now_ts=ts, k=max(cfg.rank_k, 1))

      # if no predictions, use empty list (users with no training clicks)
      if not preds:
        preds = []

      y_true.append(true_label)
      y_pred_topk.append(preds[:cfg.rank_k])
      y_pred_top1.append(preds[0] if preds else -1)

    # Metrics
    # accuracy (top-1)
    acc = accuracy_score(y_true, y_pred_top1) if y_true else 0.0

    # f1 (top-1)
    valid = [p != -1 for p in y_pred_top1]
    y_true_v = [yt for yt, ok in zip(y_true, valid) if ok]
    y_pred_v = [yp for yp, ok in zip(y_pred_top1, valid) if ok]

    if y_true_v:
        f1 = f1_score(y_true_v, y_pred_v, average="macro")
    else:
        f1 = 0.0

    # recall@k
    recalls = {f"recall@{k}": recall_at_k(y_true, y_pred_topk, k=k) for k in cfg.ks}

    rows.append(
      {
        "user_id": user_id,
        "fold": fold_idx,
        "train_start": train_start,
        "train_end": train_end,
        "test_start": test_start,
        "test_end": test_end,
        "n_train": len(train_df),
        "n_test": len(test_df),
        "accuracy_top1": float(acc),
        "f1_macro_top1": float(f1),
        **recalls,
      }
    )

    logger.info(
      f"[user={user_id} fold={fold_idx}] "
      f"acc@1={acc:.4f} f1_macro@1={f1:.4f} "
      + ", ".join(f"R@{k}={recalls[f'recall@{k}']:.4f}" for k in cfg.ks)
    )

  return pd.DataFrame(rows)


def load_user_timelines(processed_dir: Path) -> Dict[str, Path]:
  paths = list(processed_dir.glob("user_*/processed.parquet"))
  out = {}
  for p in paths:
    user_id = p.parent.name.replace("user_", "")
    out[user_id] = p
  return out


def main():
  cfg = FoldConfig(train_days=30, test_days=14, step_days=14, ks=(1, 3, 5), rank_k=60)

  REPORTS_DIR.mkdir(parents=True, exist_ok=True)
  out_csv = REPORTS_DIR / "rolling_eval_metrics.csv"

  user_files = load_user_timelines(PROCESSED_DATA_DIR)
  if not user_files:
    logger.error(f"No user timelines found in: {PROCESSED_DATA_DIR}")
    return

  all_metrics = []
  for user_id, path in user_files.items():
    logger.info(f"Loading timeline for user={user_id} from {path}")
    df = pd.read_parquet(path)

    if "timestamp" not in df.columns or "card_enc" not in df.columns:
      logger.error(f"[user={user_id}] timeline missing required columns: timestamp/card_enc")
      continue

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=False, errors="coerce")
    df = df.dropna(subset=["timestamp", "card_enc"]).copy()
    df["card_enc"] = df["card_enc"].astype(int)

    metrics_df = evaluate_user_timeline(user_id=user_id, df=df, cfg=cfg)
    if not metrics_df.empty:
      all_metrics.append(metrics_df)

  if not all_metrics:
    logger.error("No metrics generated (all users without sufficient folds?).")
    return

  final_df = pd.concat(all_metrics, ignore_index=True)
  final_df.to_csv(out_csv, index=False)
  logger.info(f"Metrics saved to: {out_csv}")

  summary_cols = ["accuracy_top1", "f1_macro_top1"] + [f"recall@{k}" for k in cfg.ks]
  summary = final_df.groupby("user_id")[summary_cols].mean().reset_index()
  logger.info("Summary (average per user):\n" + summary.to_string(index=False))


if __name__ == "__main__":
  main()
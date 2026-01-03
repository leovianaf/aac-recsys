"""
Metric utilities for rolling evaluation in AAC pictogram ranking.

This module contains lightweight metric functions for top-k recommendation/ranking:

- accuracy@1: whether the top-1 prediction equals the ground-truth label.
- macro F1@1: macro-averaged F1 considering only the top-1 prediction per event.
- recall@k: fraction of events where the true label appears in the top-k list.
- mrr@k: Mean Reciprocal Rank at k.

Expected inputs
---------------
y_true: list[int]
  Ground-truth labels for each interaction/event.

y_pred_top1: list[int]
  The top-1 predicted label per event. Use -1 to indicate "no prediction".

y_pred_topk: list[list[int]]
  The ranked list of predicted labels per event (top-k or top-N).

Notes
-----
- In recommendation settings, the number of unique labels can be large relative to the
  number of test samples in a fold. This is normal and can trigger sklearn warnings for
  classification metrics. We configure F1 computation to be stable and deterministic.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


def recall_at_k(y_true: list[int], y_pred_topk: list[list[int]], k: int) -> float:
  """Recall@k for ranking: hit if true label appears in the first k predictions."""
  n = len(y_true)
  if n == 0:
    return 0.0

  hits = 0
  for yt, preds in zip(y_true, y_pred_topk):
    if yt in preds[:k]:
      hits += 1
  return hits / n


def mrr_at_k(y_true: list[int], y_pred_topk: list[list[int]], k: int) -> float:
  """MRR@k: average reciprocal rank of the true label within top-k predictions."""
  n = len(y_true)
  if n == 0:
    return 0.0

  s = 0.0
  for yt, preds in zip(y_true, y_pred_topk):
    top = preds[:k]
    if yt in top:
      rank = top.index(yt) + 1  # 1-based
      s += 1.0 / rank
  return s / n


def _union_labels(y_true: list[int], y_pred: list[int]) -> list[int]:
  """Deterministic union of labels present in y_true/y_pred."""
  return sorted(set(y_true).union(set(y_pred)))


def calculate_metrics(
  *,
  ks: Iterable[int],
  y_true: list[int],
  y_pred_top1: list[int],
  y_pred_topk: list[list[int]],
) -> dict[str, float]:
  """
  Compute ranking metrics for a fold.

  Returns a flat dict with keys:
    - accuracy_top1
    - f1_macro_top1
    - recall@K
    - mrr@K
  """
  if not (len(y_true) == len(y_pred_top1) == len(y_pred_topk)):
    raise ValueError(
      "Input lengths mismatch: "
      f"len(y_true)={len(y_true)}, len(y_pred_top1={len(y_pred_top1)}, len(y_pred_topk)={len(y_pred_topk)}"
    )

  valid_mask = [p != -1 for p in y_pred_top1]
  y_true_v = [yt for yt, ok in zip(y_true, valid_mask) if ok]
  y_pred_v = [yp for yp, ok in zip(y_pred_top1, valid_mask) if ok]

  if y_true_v:
    acc = accuracy_score(y_true_v, y_pred_v)
  else:
    acc = 0.0

  if y_true_v:
    labels = _union_labels(y_true_v, y_pred_v)
    f1 = f1_score(
      y_true_v,
      y_pred_v,
      labels=labels,
      average="macro",
      zero_division=0,
    )
  else:
    f1 = 0.0

  metrics: dict[str, float] = {
    "accuracy_top1": float(acc),
    "f1_macro_top1": float(f1),
  }

  for k in ks:
    kk = int(k)
    metrics[f"recall@{kk}"] = float(recall_at_k(y_true, y_pred_topk, k=kk))
    metrics[f"mrr@{kk}"] = float(mrr_at_k(y_true, y_pred_topk, k=kk))

  return metrics


def weighted_mean(x: pd.Series, w: pd.Series) -> float:
  """Safe weighted mean. Returns NaN if total weight <= 0."""
  x = pd.to_numeric(x, errors="coerce")
  w = pd.to_numeric(w, errors="coerce").fillna(0.0)
  s = float(w.sum())
  if s <= 0:
    return float("nan")

  return float((x * w).sum() / s)


def summarize_per_user(
  final_df: pd.DataFrame,
  *,
  metric_cols: Iterable[str],
  group_cols: tuple[str, str] = ("model", "user_id"),
  fold_col: str = "fold",
  weight_col: str = "n_test",
  n_train_col: str = "n_train",
  n_test_col: str = "n_test",
) -> pd.DataFrame:
  """
  Aggregate per-fold rows into per-user summary.

  Produces:
    - n_folds
    - total/avg n_test, n_train
    - simple mean of each metric across folds
    - weighted mean of each metric across folds (weight = n_test by default)

  Expected input columns (at least):
    group_cols + [fold_col, n_train_col, n_test_col] + metric_cols
  """
  df = final_df.copy()

  needed_num = [n_train_col, n_test_col] + list(metric_cols)
  for c in needed_num:
    if c in df.columns:
      df[c] = pd.to_numeric(df[c], errors="coerce")

  df = df.dropna(subset=[n_train_col, n_test_col]).copy()

  agg = {
    fold_col: "count",
    n_test_col: ["sum", "mean"],
    n_train_col: ["sum", "mean"],
  }
  for m in metric_cols:
    agg[m] = "mean"

  out = df.groupby(list(group_cols), dropna=False).agg(agg)

  out.columns = [
    "_".join([x for x in col if x]).strip("_") if isinstance(col, tuple) else str(col)
    for col in out.columns
  ]
  out = out.reset_index()

  out = out.rename(
    columns={
      f"{fold_col}_count": "n_folds",
      f"{n_test_col}_sum": "total_n_test",
      f"{n_test_col}_mean": "avg_n_test",
      f"{n_train_col}_sum": "total_n_train",
      f"{n_train_col}_mean": "avg_n_train",
    }
  )

  w_rows = []
  for keys, g in df.groupby(list(group_cols), dropna=False):
    model, user_id = keys
    w = g[weight_col]

    row = {"model": model, "user_id": user_id}
    for m in metric_cols:
      row[f"{m}_w"] = weighted_mean(g[m], w)
    w_rows.append(row)

  w_df = pd.DataFrame(w_rows)

  rename_metrics = {}
  for m in metric_cols:
    m_mean = f"{m}_mean"
    if m_mean in out.columns:
      rename_metrics[m_mean] = m
  out = out.rename(columns=rename_metrics)

  out = out.merge(w_df, on=list(group_cols), how="left")

  return out


def summarize_overall(
  user_summary_df: pd.DataFrame,
  *,
  metric_cols: Iterable[str],
  user_weight_col: str = "total_n_test",
  include_weighted_metrics: bool = True,
  weighted_suffix: str = "_w",
) -> dict:
  """
  Compute overall summaries from per-user summary.

  Returns a dict with:
    - macro_mean_* : mean across users (each user weight=1)
    - micro_weighted_* : weighted mean across users (weight=total_n_test)
    - avg_total_n_test_per_user, avg_n_folds_per_user

  If include_weighted_metrics=True, it will prefer using the per-user weighted
  metric columns (e.g. 'accuracy_top1_w') for micro aggregation when available.
  """
  df = user_summary_df.copy()

  missing = [m for m in metric_cols if m not in df.columns and f"{m}_mean" not in df.columns]
  if missing:
    raise KeyError(f"Missing metric columns in user_summary_df: {missing}")

  w_user = pd.to_numeric(df[user_weight_col], errors="coerce").fillna(0.0)

  out: dict = {}
  for m in metric_cols:
    out[f"macro_mean_{m}"] = float(pd.to_numeric(df[m], errors="coerce").mean())

  for m in metric_cols:
    m_w_col = f"{m}{weighted_suffix}"
    if include_weighted_metrics and m_w_col in df.columns:
      out[f"micro_weighted_{m}"] = weighted_mean(df[m_w_col], w_user)
    else:
      out[f"micro_weighted_{m}"] = weighted_mean(df[m], w_user)

  if "total_n_test" in df.columns:
    out["avg_total_n_test_per_user"] = float(pd.to_numeric(df["total_n_test"], errors="coerce").mean())
  if "n_folds" in df.columns:
    out["avg_n_folds_per_user"] = float(pd.to_numeric(df["n_folds"], errors="coerce").mean())

  return out

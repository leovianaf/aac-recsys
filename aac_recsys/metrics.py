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

  acc = accuracy_score(y_true, y_pred_top1) if y_true else 0.0

  # F1 macro@1 (ignore rows where no prediction exists)
  valid_mask = [p != -1 for p in y_pred_top1]
  y_true_v = [yt for yt, ok in zip(y_true, valid_mask) if ok]
  y_pred_v = [yp for yp, ok in zip(y_pred_top1, valid_mask) if ok]

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

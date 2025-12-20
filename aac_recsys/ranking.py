from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


PERIOD_KEYS = ("dawn", "morn", "noon", "aftn", "even", "nght")


def gaussian_membership(x: float, center: float, sigma: float) -> float:
  return float(np.exp(-((x - center) ** 2) / (2 * (sigma ** 2))))


def compute_fuzzy_time_memberships(hour: int) -> Dict[str, float]:
  """
  Return fuzzy memberships for each PERIOD_KEY given the hour of day.
  Note: gaussian can return values >0; does not necessarily sum to 1.
  We can normalize when using as a distribution.
  """
  h = int(hour)
  return {
    "dawn": gaussian_membership(h, 2.5, 2.0),
    "morn": gaussian_membership(h, 7.5, 2.0),
    "noon": gaussian_membership(h, 11.5, 2.0),
    "aftn": gaussian_membership(h, 13.5, 2.0),
    "even": gaussian_membership(h, 17.5, 2.0),
    "nght": gaussian_membership(h, 22.5, 2.0),
  }


def _normalize_dist(d: Dict[str, float], eps: float = 1e-12) -> Dict[str, float]:
  s = float(sum(d.values()))
  if s <= eps:
    u = 1.0 / len(d)
    return {k: u for k in d}
  return {k: float(v) / s for k, v in d.items()}


# Artifacts (computed on TRAIN only)
@dataclass
class RankingArtifacts:
  total_clicks: int
  count_by_label: Dict[int, int]
  # typical temporal profile of the label: distribution over PERIOD_KEYS (sum ~1)
  period_profile_by_label: Dict[int, Dict[str, float]]


def build_artifacts(
  train_df: pd.DataFrame,
  label_col: str = "label_id",
  ts_col: str = "timestamp",
) -> RankingArtifacts:
  """
  Builds:
    - count_by_label (frequency)
    - period_profile_by_label: fuzzy distribution of periods for each label
  Use ONLY the TRAIN part of the fold.
  """
  if train_df.empty:
    return RankingArtifacts(total_clicks=0, count_by_label={}, period_profile_by_label={})

  # Frequency by label
  count_by_label = train_df[label_col].value_counts().to_dict()
  count_by_label = {int(k): int(v) for k, v in count_by_label.items()}
  total_clicks = int(len(train_df))

  # Fuzzy temporal profile by label
  # We will sum Gaussian weights per period for each click.
  # w[label][period] += membership(period | hour_of_click)
  period_weights: Dict[int, Dict[str, float]] = {}

  hours = train_df[ts_col].dt.hour.to_numpy() # type: ignore
  labels = train_df[label_col].to_numpy()

  for lbl, hr in zip(labels, hours):
    lbl = int(lbl)
    mem = compute_fuzzy_time_memberships(int(hr))  # gaussian memberships
    w = period_weights.get(lbl)
    if w is None:
      w = {k: 0.0 for k in PERIOD_KEYS}
      period_weights[lbl] = w
    for k in PERIOD_KEYS:
      w[k] += float(mem[k])

  # Normalize to distribution per label
  period_profile_by_label: Dict[int, Dict[str, float]] = {}
  for lbl, w in period_weights.items():
    period_profile_by_label[lbl] = _normalize_dist(w)

  return RankingArtifacts(
    total_clicks=total_clicks,
    count_by_label=count_by_label,
    period_profile_by_label=period_profile_by_label,
  )


# Ranking (freq_rel * prob_temporal)
def rank_topk(
  artifacts: RankingArtifacts,
  now_ts: datetime,
  k: int = 60,
  candidate_labels: Sequence[int] | None = None,
) -> List[int]:
  """
  Final score = freq_rel(label) * prob_temporal(label)

  Where:
    freq_rel(label) = count(label)/total_clicks
    prob_temporal(label) = sum_p membership_now(p) * profile_label(p)

  Returns top-k label_ids.
  """
  if artifacts.total_clicks == 0:
    return []

  mem_now_raw = compute_fuzzy_time_memberships(now_ts.hour)
  # normalize memberships of "now" to make it a proper distribution
  mem_now = _normalize_dist(mem_now_raw)

  labels = candidate_labels if candidate_labels is not None else list(artifacts.count_by_label.keys())

  scored: List[Tuple[int, float]] = []
  for lbl in labels:
    lbl = int(lbl)
    cnt = artifacts.count_by_label.get(lbl, 0)
    if cnt <= 0:
      continue

    freq_rel = cnt / artifacts.total_clicks

    profile = artifacts.period_profile_by_label.get(lbl)
    if profile is None:
      # if no profile, assume uniform distribution
      profile = {k: 1.0 / len(PERIOD_KEYS) for k in PERIOD_KEYS}

    prob_temporal = 0.0
    for p in PERIOD_KEYS:
      prob_temporal += float(mem_now[p]) * float(profile.get(p, 0.0))

    score = float(freq_rel * prob_temporal)
    scored.append((lbl, score))

  scored.sort(key=lambda x: x[1], reverse=True)
  return [lbl for lbl, _ in scored[:k]]

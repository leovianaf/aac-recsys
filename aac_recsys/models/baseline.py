"""
Baseline ranking model for AAC pictogram recommendation.

This module implements a simple, per-user ranking baseline that reorders pictogram labels
based on historical usage and time-of-day context.

Core idea
---------
Given a TRAIN history of user clicks, the baseline learns, for each label (card_enc):

1) Decayed frequency (recency-aware)
   - Each past click contributes a weight w that decays exponentially with time:
      w = 0.5 ** (delta_days / half_life_days)
   - Newer interactions contribute more weight than older ones.
   - The relative frequency term is:
       freq_rel(label) = weight(label) / total_weight

2) Fuzzy time-of-day profile (temporal relevance)
   - Each click is mapped to fuzzy memberships over time periods:
     PERIOD_KEYS = ("dawn", "morn", "noon", "aftn", "even", "nght")
   - Memberships are computed via Gaussian functions over the hour-of-day.
   - For each label, memberships are accumulated (also weighted by w) and normalized,
     producing a per-label distribution over PERIOD_KEYS:
       profile_label(period) ~= P(period | label)

Ranking / inference
-------------------
At inference time for a given timestamp `now_ts`, the baseline computes the current fuzzy
distribution over periods (normalized), and scores each candidate label as:
  score(label) = freq_rel(label) * prob_temporal(label)

where:
  prob_temporal(label) = sum_{period} mem_now(period) * profile_label(period)

The output is a list of label IDs sorted by descending score, optionally truncated to top-k.

Main components
---------------
- compute_fuzzy_time_memberships(hour): Gaussian fuzzy memberships for each period key.
- build_artifacts(train_df, now_ts, ...): builds RankingArtifacts from TRAIN-only data.
- rank_topk(artifacts, now_ts, ...): returns top-k labels by score.
- BaselineRanker: a small OO wrapper with .fit() and .rank() using BaselineParams.

Notes
-----
- This module does not train a machine learning model; it computes statistics/artifacts from
  the training history and uses them to rank labels.
- Intended to be used inside walk-forward / rolling evaluation pipelines (e.g., predict.py),
  ensuring no future data leakage by building artifacts only from each fold's TRAIN window.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Sequence, Tuple, Optional, Any

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
  total_weight: float
  weight_by_label: Dict[int, float]
  period_profile_by_label: Dict[int, Dict[str, float]]

def _exp_decay_weight(delta_days: float, half_life_days: int) -> float:
  # w = 0.5 ** (delta/half_life)
  return float(0.5 ** (delta_days / max(half_life_days, 1e-9)))

def build_artifacts(
  train_df: pd.DataFrame,
  now_ts: datetime,
  label_col: str = "label_id",
  ts_col: str = "timestamp",
  half_life_days: int = 30,
) -> RankingArtifacts:
  """
  Builds:
    - weight_by_label: decayed frequency weights
    - period_profile_by_label: fuzzy distribution of periods for each label
  Use ONLY the TRAIN part of the fold.
  """
  if train_df.empty:
    return RankingArtifacts(total_weight=0, weight_by_label={}, period_profile_by_label={})

  ts = pd.to_datetime(train_df[ts_col], errors="coerce")
  labels = train_df[label_col].astype(int).to_numpy()
  hours = ts.dt.hour.to_numpy()

  # Fuzzy temporal profile by label
  weight_by_label: dict[int, float] = {}
  period_weights: dict[int, dict[str, float]] = {}
  total_weight = 0.0

  now = pd.Timestamp(now_ts)

  for lbl, hr, t in zip(labels, hours, ts):
    lbl = int(lbl)

    if pd.isna(t):
      continue


    delta_days = (now - t).total_seconds() / 86400.0
    if delta_days < 0:
      delta_days = 0.0
    w = _exp_decay_weight(delta_days, half_life_days)

    total_weight += w
    weight_by_label[lbl] = weight_by_label.get(lbl, 0.0) + w

    mem = compute_fuzzy_time_memberships(int(hr))
    pw = period_weights.get(lbl)
    if pw is None:
      pw = {k: 0.0 for k in PERIOD_KEYS}
      period_weights[lbl] = pw
    for k in PERIOD_KEYS:
      pw[k] += w * float(mem[k])

  period_profile_by_label = {lbl: _normalize_dist(wdict) for lbl, wdict in period_weights.items()}

  return RankingArtifacts(
    total_weight=float(total_weight),
    weight_by_label=weight_by_label,
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
    freq_rel = weight / artifacts.total_weight
    prob_temporal(label) = sum_p membership_now(p) * profile_label(p)

  Returns top-k label_ids.
  """
  if artifacts.total_weight == 0.0:
    return []

  mem_now_raw = compute_fuzzy_time_memberships(now_ts.hour)
  # normalize memberships of "now" to make it a proper distribution
  mem_now = _normalize_dist(mem_now_raw)

  labels = candidate_labels if candidate_labels is not None else list(artifacts.weight_by_label.keys())

  scored: List[Tuple[int, float]] = []
  for lbl in labels:
    lbl = int(lbl)
    weight = artifacts.weight_by_label.get(lbl, 0)
    if weight <= 0:
      continue

    freq_rel = weight / artifacts.total_weight

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


@dataclass
class BaselineParams:
  half_life_days: int = 30
  rank_k_default: int = 60


class BaselineRanker:
  """
  Ranker baseline: (decayed) frequency * fuzzy-time profile

  Use:
    ranker = BaselineRanker(BaselineParams(half_life_days=30, rank_k_default=60))
    ranker.fit(train_df, label_col="card_enc", ts_col="timestamp", train_end=train_end)
    topk = ranker.rank(now_ts=some_ts, k=60)
  """
  name = "baseline"

  def __init__(self, params: BaselineParams | None = None):
    self.params = params or BaselineParams()
    self.artifacts: RankingArtifacts | None = None
    self._label_col: str | None = None
    self._ts_col: str | None = None
    self._train_end: datetime | None = None

  def fit(
    self,
    train_df: pd.DataFrame,
    *,
    label_col: str,
    ts_col: str,
    train_end: datetime,
  ) -> None:
    self._label_col = label_col
    self._ts_col = ts_col
    self._train_end = train_end

    self.artifacts = build_artifacts(
      train_df,
      now_ts=train_end,
      label_col=label_col,
      ts_col=ts_col,
      half_life_days=self.params.half_life_days,
    )

  def rank(
    self,
    *,
    now_ts: datetime,
    k: int | None = None,
    candidate_labels: Sequence[int] | None = None,
  ) -> List[int]:
    if self.artifacts is None:
        return []
    kk = int(k if k is not None else self.params.rank_k_default)
    return rank_topk(self.artifacts, now_ts=now_ts, k=kk, candidate_labels=candidate_labels)

  def params_dict(self) -> Dict[str, Any]:
    return asdict(self.params)

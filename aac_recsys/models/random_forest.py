from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from aac_recsys.models.ranker_base import Ranker


@dataclass
class RandomForestParams:
  n_estimators: int = 300
  max_depth: int | None = None
  min_samples_leaf: int = 2
  min_samples_split: int = 2
  max_features: str | float | int = "sqrt"
  class_weight: str | dict | None = "balanced_subsample"
  random_state: int = 42
  n_jobs: int = -1

  rank_k_default: int = 60


class RandomForestRanker(Ranker):
  """
  Per-user RandomForest ranker.

  Trains a multi-class classifier to predict card_enc given context features.
  Ranking is produced by sorting classes by predict_proba.

  Important:
  - This ranker requires row-level features at inference time.
    Use rank_row(row, k), not rank(now_ts).
  """

  name = "random_forest"

  def __init__(self, params: RandomForestParams):
    self.params = params
    self.clf: RandomForestClassifier | None = None
    self.feature_cols: list[str] = []
    self.classes_: np.ndarray | None = None
    self.popularity_: list[int] = []

  def params_dict(self) -> Dict[str, Any]:
    return {
      "n_estimators": self.params.n_estimators,
      "max_depth": self.params.max_depth,
      "min_samples_leaf": self.params.min_samples_leaf,
      "min_samples_split": self.params.min_samples_split,
      "max_features": self.params.max_features,
      "class_weight": self.params.class_weight,
      "random_state": self.params.random_state,
      "rank_k_default": self.params.rank_k_default,
    }

  def _infer_feature_cols(self, df: pd.DataFrame, *, label_col: str, ts_col: str) -> list[str]:
    """
    Pick a conservative set of features from your processed.parquet.
    Assumes you already have:
      - cluster (int)
      - hour, week_num, year_num (int)
      - week_day_* one-hot
      - fuzzy memberships: dawn..nght
    """
    base = ["cluster", "hour", "week_num", "year_num", "dawn", "morn", "noon", "aftn", "even", "nght"]
    base = [c for c in base if c in df.columns]

    # one-hot week_day column names look like: week_day_Friday, week_day_Monday, ...
    week_ohe = [c for c in df.columns if c.startswith("week_day_")]

    # drop label/timestamp if they slip in
    cols = [c for c in (base + week_ohe) if c not in {label_col, ts_col}]

    # keep only numeric-ish columns
    numeric = []
    for c in cols:
      if pd.api.types.is_numeric_dtype(df[c]):
        numeric.append(c)
    return numeric

  def fit(
    self,
    train_df: pd.DataFrame,
    *,
    label_col: str,
    ts_col: str,
    train_end: Any = None,  # kept for interface compatibility
  ) -> None:
    df = train_df.copy()

    # Ensure numeric label
    y = pd.to_numeric(df[label_col], errors="coerce")
    df = df.dropna(subset=[label_col]).copy()
    y = y.loc[df.index].astype(int)

    # Store popularity fallback (most frequent labels in TRAIN)
    self.popularity_ = (
      df[label_col].astype(int).value_counts().index.astype(int).tolist()
    )

    self.feature_cols = self._infer_feature_cols(df, label_col=label_col, ts_col=ts_col)
    if not self.feature_cols:
      # No usable features => degenerate ranker
      self.clf = None
      self.classes_ = None
      return

    X = df[self.feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)

    self.clf = RandomForestClassifier(
      n_estimators=self.params.n_estimators,
      max_depth=self.params.max_depth,
      min_samples_leaf=self.params.min_samples_leaf,
      min_samples_split=self.params.min_samples_split,
      max_features=self.params.max_features,
      class_weight=self.params.class_weight,
      random_state=self.params.random_state,
      n_jobs=self.params.n_jobs,
    )

    self.clf.fit(X, y.to_numpy())
    self.classes_ = getattr(self.clf, "classes_", None)

  def rank(self, *, now_ts: Any, k: int | None = None) -> list[int]:
    """
    Baseline-compatible signature, but RF cannot rank using only time.
    We return popularity fallback.
    """
    kk = int(k or self.params.rank_k_default)
    return self.popularity_[:kk]

  def rank_row(self, row: pd.Series, *, k: int | None = None) -> list[int]:
    """
    Rank using row-level features (the correct way for RF).
    """
    kk = int(k or self.params.rank_k_default)

    if self.clf is None or not self.feature_cols:
      return self.popularity_[:kk]

    # Build a 1xD vector; missing columns default to 0
    x = []
    for c in self.feature_cols:
      v = row.get(c, 0.0)
      try:
        v = float(v)
      except Exception:
        v = 0.0
      x.append(v)

    X = np.asarray(x, dtype=np.float32).reshape(1, -1)

    try:
      proba = self.clf.predict_proba(X)[0]  # aligned with self.clf.classes_
      classes = self.clf.classes_.astype(int)

      order = np.argsort(-proba)  # descending
      ranked = classes[order].tolist()

      # in case you want to guarantee at least kk outputs, append popularity tail
      if len(ranked) < kk and self.popularity_:
        seen = set(ranked)
        for lab in self.popularity_:
          if lab not in seen:
            ranked.append(lab)
            if len(ranked) >= kk:
              break

      return ranked[:kk]
    except Exception:
      return self.popularity_[:kk]

"""
Ranker base protocol for AAC pictogram recommendation models.

Implements the Ranker protocol that all ranking models should follow.
"""

from __future__ import annotations
from typing import Protocol, Sequence, Any, Dict
from datetime import datetime
import pandas as pd

class Ranker(Protocol):
  name: str

  def fit(self, train_df: pd.DataFrame, *, label_col: str, ts_col: str, train_end: datetime) -> None: ...
  def rank(self, *, now_ts: datetime, k: int | None = None, candidate_labels: Sequence[int] | None = None) -> list[int]: ...
  def params_dict(self) -> Dict[str, Any]: ...

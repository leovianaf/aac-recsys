from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
from typing import Optional, Tuple
import numpy as np
import pandas as pd
import seaborn as sns

from aac_recsys.config import FIGURES_DIR, PROCESSED_DATA_DIR, REPORTS_DIR, logger

def plot_user_hexbin(df_u: pd.DataFrame, outpath: Path, user_idx: int, gridsize: int = 80) -> None:
  """Save a density heatmap (hexbin) for a user."""
  df_u = df_u.dropna(subset=["latitude", "longitude"]).copy()
  if df_u.empty:
      return

  plt.figure()
  plt.hexbin(df_u["longitude"].to_numpy(), df_u["latitude"].to_numpy(), gridsize=gridsize, bins="log")
  plt.colorbar(label="log10(density)")
  plt.xlabel("Longitude")
  plt.ylabel("Latitude")
  plt.title(f"Heatmap (hexbin) — user_{user_idx}")
  plt.tight_layout()
  plt.savefig(outpath, dpi=200)
  plt.close()


def plot_user_clusters_scatter(
    df_u: pd.DataFrame,
    outpath: Path,
    user_idx: int,
    max_points: int = 120_000,
) -> None:
  """Save a scatter plot colored by cluster id for a user."""
  df_u = df_u.dropna(subset=["latitude", "longitude", "cluster"]).copy()
  if df_u.empty:
    return

  if len(df_u) > max_points:
    df_u = df_u.sample(n=max_points, random_state=42)

  plt.figure()
  plt.scatter(
    df_u["longitude"].to_numpy(),
    df_u["latitude"].to_numpy(),
    c=df_u["cluster"].to_numpy(),
    s=4,
    alpha=0.4,
  )
  plt.colorbar(label="cluster id (-1 = noise)")
  plt.xlabel("Longitude")
  plt.ylabel("Latitude")
  plt.title(f"Clusters — user_{user_idx}")
  plt.tight_layout()
  plt.savefig(outpath, dpi=200)
  plt.close()


def plot_all_users_scatter(
    df: pd.DataFrame,
    outpath: Path,
    *,
    max_points: int = 300_000,
) -> None:
  """One map with all users. Each user receives a different color."""
  df = df.dropna(subset=["latitude", "longitude", "user_uuid"]).copy()
  if df.empty:
    return

  if len(df) > max_points:
    df = df.sample(n=max_points, random_state=42)

  user_codes, _ = pd.factorize(df["user_uuid"], sort=True)

  plt.figure()
  plt.scatter(
    df["longitude"].to_numpy(),
    df["latitude"].to_numpy(),
    c=user_codes,
    s=2,
    alpha=0.35,
    cmap="tab20",
  )
  plt.colorbar(label="user (code)")
  plt.xlabel("Longitude")
  plt.ylabel("Latitude")
  plt.title("All users — scatter by user (color)")
  plt.tight_layout()
  plt.savefig(outpath, dpi=220)
  plt.close()


def plot_all_users_hexbin(df: pd.DataFrame, outpath: Path, gridsize: int = 140) -> None:
  """One map with all users — density heatmap (hexbin)."""
  df = df.dropna(subset=["latitude", "longitude"]).copy()
  if df.empty:
    return

  plt.figure()
  plt.hexbin(df["longitude"].to_numpy(), df["latitude"].to_numpy(), gridsize=gridsize, bins="log")
  plt.colorbar(label="log10(density)")
  plt.xlabel("Longitude")
  plt.ylabel("Latitude")
  plt.title("All users — global density (hexbin)")
  plt.tight_layout()
  plt.savefig(outpath, dpi=220)
  plt.close()


def clear_dir(path: Path) -> None:
  """Remove directory if exists (safe) and recreate."""
  if path.exists():
    shutil.rmtree(path)
  path.mkdir(parents=True, exist_ok=True)


def select_top_users(df: pd.DataFrame, *, top_n: int = 15) -> pd.DataFrame:
  """Select top_n users by number of rows (interactions)."""
  if "user_uuid" not in df.columns:
      return df
  counts = df["user_uuid"].value_counts()
  top_users = counts.head(top_n).index
  return df[df["user_uuid"].isin(top_users)].copy()


def generate_per_user_plots(
    df: pd.DataFrame,
    output_dir: Path,
    *,
    clear_output_dir: bool,
    max_points_per_user: int,
    gridsize_user: int,
) -> None:
  """
  Generate per-user plots from an already loaded df (viz parquet).
  Expects columns: user_uuid, latitude, longitude, cluster.
  """
  required = {"user_uuid", "latitude", "longitude", "cluster"}
  missing = required - set(df.columns)
  if missing:
    raise KeyError(f"Missing columns in viz df: {missing}")

  if clear_output_dir:
    clear_dir(output_dir)
  else:
    output_dir.mkdir(parents=True, exist_ok=True)

  for idx, (_, df_u) in enumerate(df.groupby("user_uuid", sort=True)):
    df_u = df_u.dropna(subset=["latitude", "longitude", "cluster"]).copy()
    plot_user_hexbin(df_u, output_dir / f"user_{idx}_hexbin.png", user_idx=idx, gridsize=gridsize_user)
    plot_user_clusters_scatter(
      df_u,
      output_dir / f"user_{idx}_clusters.png",
      user_idx=idx,
      max_points=max_points_per_user,
    )


def run_plots(
  viz_path: Path,
  *,
  clear: bool = True,
  per_user: bool = False,
  clear_per_user: bool = True,
  top_n: int = 15,
) -> None:
  """
  Run plotting functions.

  - If clear=True, clears entire output_dir.
  - If per_user=True, generates per-user plots under output_dir/per_user.
  - Uses top_n users for global plots to improve readability.
  """
  output_dir = FIGURES_DIR / "cluster_heatmaps"

  if clear:
    clear_dir(output_dir)
  else:
    output_dir.mkdir(parents=True, exist_ok=True)

  df = pd.read_parquet(viz_path)

  if per_user:
    per_user_dir = output_dir / "per_user"
    generate_per_user_plots(
      df,
      per_user_dir,
      clear_output_dir=(clear or clear_per_user),
      max_points_per_user=120,
      gridsize_user=80,
    )

  df_top = select_top_users(df, top_n=top_n)
  plot_all_users_scatter(df_top, output_dir / f"all_users_scatter_top{top_n}.png", max_points=300_000)
  plot_all_users_hexbin(df_top, output_dir / f"all_users_hexbin_top{top_n}.png", gridsize=140)

  logger.success(f"Saved plots to: {output_dir}")


def parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser()

  p.add_argument(
    "--viz-path",
    type=Path,
    default=PROCESSED_DATA_DIR / "data_for_visualization.parquet",
    help="Path to data_for_visualization.parquet",
  )
  p.add_argument("--clear", action="store_true", help="Clear the entire output dir before generating plots")
  p.add_argument("--per-user", default=False, action="store_true", help="Generate per-user plots")
  p.add_argument("--clear-per-user", action="store_true", help="Clear only per_user subdir")
  p.add_argument("--top-n", type=int, default=15, help="Top N users (by interactions) to use in global plots")


  return p.parse_args()


BAND_ORDER = ("LOW", "MID", "HIGH")


def load_user_summary( *,
    filename: str = "predict_user_summary_baseline.csv",
) -> pd.DataFrame:
    """Load the per-user summary CSV produced by the evaluation pipeline."""
    path = REPORTS_DIR / filename
    return pd.read_csv(path)


def add_test_ratio_and_band(df: pd.DataFrame) -> pd.DataFrame:
    """Add test_ratio and band columns based on train/test proportions.

    Band definition:
      - LOW:  test_ratio < 0.15
      - MID:  0.15 <= test_ratio <= 0.35
      - HIGH: test_ratio > 0.35
    """
    out = df.copy()

    denom = out["total_n_test"] + out["total_n_train"]
    out["test_ratio"] = out["total_n_test"] / denom

    def classify_band(ratio: float) -> str:
        if ratio < 0.15:
            return "LOW"
        if ratio <= 0.35:
            return "MID"
        return "HIGH"

    out["band"] = out["test_ratio"].apply(classify_band)
    return out


def _save_dataframe_as_png(
    df_table: pd.DataFrame,
    *,
    out_path: Path,
    title: Optional[str] = None,
    font_size: int = 10,
    scale: Tuple[float, float] = (1.0, 1.2),
) -> None:
    """Render a DataFrame as a table and save it as a PNG image."""
    n_rows, n_cols = df_table.shape
    fig_w = max(8.0, n_cols * 1.3)
    fig_h = max(2.0, n_rows * 0.45)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    if title:
        ax.set_title(title, fontsize=12, pad=12)

    table = ax.table(
        cellText=df_table.values,
        colLabels=df_table.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(scale[0], scale[1])

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_train_test_ratio_table(
    df: pd.DataFrame,
    *,
    csv_name: str = "table_train_test_ratio_by_user.csv",
    png_name: str = "table_train_test_ratio_by_user.png",
    max_rows: Optional[int] = None,
) -> None:
    """Save a per-user train/test proportion table (CSV + PNG).

    If max_rows is provided, the table is truncated to the first max_rows rows
    after sorting by test_ratio (ascending).
    """
    tmp = df.copy()
    tmp["train_ratio"] = 1.0 - tmp["test_ratio"]

    table_df = tmp[
        ["user_id", "total_n_train", "train_ratio", "total_n_test", "test_ratio", "band"]
    ].copy()

    table_df = table_df.rename(
        columns={
            "user_id": "User ID",
            "total_n_train": "Train",
            "train_ratio": "Train %",
            "total_n_test": "Test",
            "test_ratio": "Test %",
            "band": "Band",
        }
    )

    table_df = table_df.sort_values("Test %", ascending=True)

    table_df["Train %"] = (table_df["Train %"] * 100).round(0).astype(int).astype(str) + "%"
    table_df["Test %"] = (table_df["Test %"] * 100).round(0).astype(int).astype(str) + "%"

    if max_rows is not None:
        table_df = table_df.head(int(max_rows)).copy()

    csv_path = FIGURES_DIR / csv_name
    png_path = FIGURES_DIR / png_name

    table_df.to_csv(csv_path, index=False)

    _save_dataframe_as_png(
        table_df,
        out_path=png_path,
        title="Train/Test Proportion per User",
        font_size=10,
        scale=(1.0, 1.2),
    )


def save_metrics_by_band_table(
    df: pd.DataFrame,
    *,
    csv_name: str = "table_metrics_by_band.csv",
    png_name: str = "table_metrics_by_band.png",
) -> None:
    """Save a metrics summary table (mean/std) grouped by band (CSV + PNG)."""
    metric_specs = [
        ("accuracy_top1_w", "Accuracy@1", None),
        ("recall@1_w", "Recall", 1),
        ("recall@3_w", "Recall", 3),
        ("recall@5_w", "Recall", 5),
        ("mrr@1_w", "MRR", 1),
        ("mrr@3_w", "MRR", 3),
        ("mrr@5_w", "MRR", 5),
    ]

    rows = []
    for band in BAND_ORDER:
        g = df[df["band"] == band]
        if g.empty:
            continue

        for col, metric_name, k_val in metric_specs:
            if col not in g.columns:
                continue

            values = pd.to_numeric(g[col], errors="coerce")
            mean = float(values.mean())
            std = float(values.std())

            rows.append(
                {
                    "Band": band,
                    "Metric": metric_name,
                    "K": "" if k_val is None else int(k_val),
                    "Mean": round(mean, 4),
                    "Std": round(std, 4),
                    "N users": int(g.shape[0]),
                }
            )

    out = pd.DataFrame(rows)

    metric_order = {"Accuracy@1": 0, "Recall": 1, "MRR": 2}
    out["_metric_order"] = out["Metric"].map(metric_order).fillna(9).astype(int)
    out["_k_order"] = pd.to_numeric(out["K"], errors="coerce").fillna(0).astype(int)

    out = out.sort_values(["Band", "_metric_order", "_k_order"]).drop(
        columns=["_metric_order", "_k_order"]
    )

    csv_path = FIGURES_DIR / csv_name
    png_path = FIGURES_DIR / png_name

    out.to_csv(csv_path, index=False)

    _save_dataframe_as_png(
        out,
        out_path=png_path,
        title="Evaluation Metrics (Mean ± Std) by Test Proportion Band",
        font_size=10,
        scale=(1.0, 1.2),
    )


def plot_recall_vs_train_size(df: pd.DataFrame) -> None:
    """
    Plot recall@5_w versus total training size (log scale).

    Each point represents a user.
    """
    plt.figure(figsize=(7, 5))

    x = np.log10(df["total_n_train"] + 1)
    y = df["recall@5_w"]

    plt.scatter(x, y, alpha=0.7)

    plt.xlabel("log10(Total training events)")
    plt.ylabel("Recall@5 (weighted)")
    plt.title("Recall@5 vs Training Data Volume")

    out_path = FIGURES_DIR / "scatter_recall5_vs_train_size.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_recall_vs_n_folds(df: pd.DataFrame) -> None:
    """
    Plot recall@5_w versus number of evaluation folds.
    """
    plt.figure(figsize=(7, 5))

    plt.scatter(
        df["n_folds"],
        df["recall@5_w"],
        alpha=0.7,
    )

    plt.xlabel("Number of folds")
    plt.ylabel("Recall@5 (weighted)")
    plt.title("Recall@5 vs Number of Rolling Folds")

    out_path = FIGURES_DIR / "scatter_recall5_vs_n_folds.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_recall_distribution(df: pd.DataFrame) -> None:
    """
    Plot the distribution of recall@5_w across users.
    """
    plt.figure(figsize=(7, 5))

    sns.histplot(
        df["recall@5_w"],
        bins=15,
        kde=True,
    )

    plt.xlabel("Recall@5 (weighted)")
    plt.ylabel("Number of users")
    plt.title("Distribution of Recall@5 Across Users")

    out_path = FIGURES_DIR / "hist_recall5_distribution.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def run_output_plots(
    *,
    user_summary_filename: str = "predict_user_summary_baseline.csv",
    max_users_in_ratio_table: Optional[int] = None,
) -> None:
    """
    Generate and save output tables and plots for evaluation results.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_user_summary(filename=user_summary_filename)
    df = add_test_ratio_and_band(df)

    save_train_test_ratio_table(df, max_rows=max_users_in_ratio_table)
    save_metrics_by_band_table(df)

    plot_recall_vs_train_size(df)
    plot_recall_vs_n_folds(df)
    plot_recall_distribution(df)
  

def main() -> None:
  args = parse_args()

  if not args.viz_path.exists():
    raise FileNotFoundError(f"viz parquet not found: {args.viz_path}")

  run_plots(
    args.viz_path,
    clear=args.clear,
    per_user=args.per_user,
    clear_per_user=args.clear_per_user,
    top_n=args.top_n,
  )


if __name__ == "__main__":
  main()
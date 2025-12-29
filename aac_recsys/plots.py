from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from aac_recsys.config import FIGURES_DIR, PROCESSED_DATA_DIR, logger

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
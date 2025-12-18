"""
Preprocessing pipeline for the AAC pictogram recommendation experiment.

This script:
- Loads raw clickstream data (parquet) from GS_DATASET_URL.
- Cleans rows with missing critical fields.
- Extracts temporal context features and fuzzy time memberships.
- Filters users by minimum activity and diversity constraints.
- Parses click_location into latitude/longitude.
- For each user:
  - Generates a per-user label vocabulary (label_vocab.json).
  - Trains a per-user LabelEncoder for card_written_text.
  - Runs DBSCAN clustering (optionally on a spatially stratified sample).
  - Saves per-user cluster centroids (location_vocab_user.json).
  - Assigns clusters to all user points by nearest centroid within EPS_METERS.
  - Trains a per-user OneHotEncoder for categorical context features.
  - Saves a per-user processed parquet dataset.
- Concatenates all processed users into a global parquet.
- Optionally generates cluster visualization plots from a visualization parquet.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.cluster import DBSCAN
from sklearn.neighbors import BallTree
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

import matplotlib.pyplot as plt


EARTH_RADIUS_METERS = 6_373_000
EPS_METERS = 400
MIN_SAMPLES = 3

REQUIRED_COLUMNS = ["user_uuid", "click_location", "card_written_text", "event_timestamp"]

FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports" / "figures" / "cluster_heatmaps"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--user-idx", type=int, default=None, help="Process only user_{idx}")
    parser.add_argument("--force", action="store_true", help="Reprocess even if outputs exist")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    return parser.parse_args()


def setup_logging(log_file: Optional[Path] = None) -> logging.Logger:
    """
    Configure logging to stdout and optionally to a log file.

    Parameters
    ----------
    log_file:
        If provided, writes logs to this file as well.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger("preprocess")
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def user_artifacts_exist(user_dir: Path) -> bool:
    """
    Return True if all expected per-user artifacts exist.

    Artifacts:
    - processed.parquet
    - label_vocab.json
    - location_vocab_user.json
    - label_encoder_card.pkl
    - onehot_encoder.pkl
    """
    needed = [
        user_dir / "processed.parquet",
        user_dir / "label_vocab.json",
        user_dir / "location_vocab_user.json",
        user_dir / "label_encoder_card.pkl",
        user_dir / "onehot_encoder.pkl",
    ]
    return all(p.exists() for p in needed)


def normalize_card_text(text: Any) -> str:
    """Normalize card text deterministically."""
    return " ".join(str(text).strip().lower().split())


def generate_label_vocab_json(
    df: pd.DataFrame,
    *,
    text_col: str,
    output_path: Path,
) -> Dict[str, Any]:
    """
    Generate a label vocabulary JSON mapping ids <-> normalized texts.

    IDs are deterministic (alphabetical order of unique normalized texts).

    Parameters
    ----------
    df:
        Input dataframe.
    text_col:
        Column containing card text.
    output_path:
        Where to save label_vocab.json.

    Returns
    -------
    dict
        Payload written to disk.
    """
    if text_col not in df.columns:
        raise KeyError(f"Column '{text_col}' not found in DataFrame.")

    texts = df[text_col].dropna().map(normalize_card_text)
    unique_texts = sorted(texts.unique().tolist())

    id_to_text = {int(i): t for i, t in enumerate(unique_texts)}
    text_to_id = {t: int(i) for i, t in enumerate(unique_texts)}

    payload: Dict[str, Any] = {
        "version": 1,
        "text_col": text_col,
        "num_labels": len(unique_texts),
        "id_to_text": id_to_text,
        "text_to_id": text_to_id,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return payload


def spatial_cell(lat: float, lon: float, precision: int = 3) -> str:
    """Create a coarse spatial cell id by rounding coordinates."""
    return f"{round(lat, precision)}_{round(lon, precision)}"


def load_location_vocab_df(vocab_path: Path) -> pd.DataFrame:
    """Load location_vocab_user.json as a DataFrame."""
    records = json.loads(vocab_path.read_text(encoding="utf-8"))
    return pd.DataFrame(records)


def gaussian_membership(x: float, center: float, sigma: float) -> float:
    """Compute Gaussian membership value."""
    return float(np.exp(-((x - center) ** 2) / (2 * (sigma**2))))


def compute_fuzzy_time_memberships(hour: int) -> Dict[str, float]:
    """Return fuzzy time memberships for a given hour [0..23]."""
    return {
        "dawn": gaussian_membership(hour, 2.5, 2.0),
        "morn": gaussian_membership(hour, 7.5, 2.0),
        "noon": gaussian_membership(hour, 11.5, 2.0),
        "aftn": gaussian_membership(hour, 13.5, 2.0),
        "even": gaussian_membership(hour, 17.5, 2.0),
        "nght": gaussian_membership(hour, 22.5, 2.0),
    }


def clusterize_locations(df: pd.DataFrame, eps_meters: float, min_samples: int) -> pd.DataFrame:
    """
    Run DBSCAN clustering (haversine) on latitude/longitude.

    Returns a copy with a 'cluster' column.
    """
    coords = df[["latitude", "longitude"]].to_numpy()
    coords_rad = np.radians(coords)
    eps_rad = eps_meters / EARTH_RADIUS_METERS

    model = DBSCAN(eps=eps_rad, min_samples=min_samples, metric="haversine")
    labels = model.fit_predict(coords_rad)

    out = df.copy()
    out["cluster"] = labels
    return out


def save_user_location_vocab(df_clustered: pd.DataFrame, *, user_tag: str, output_path: Path) -> None:
    """
    Save per-user cluster centroids (mean lat/lng) to JSON.

    Output schema:
        [{'lat':..., 'lng':..., 'clustering': int, 'user_uuid': str}, ...]
    """
    vocab_df = (
        df_clustered.loc[df_clustered["cluster"] != -1, ["latitude", "longitude", "cluster"]]
        .groupby("cluster", as_index=False)
        .agg(lat=("latitude", "mean"), lng=("longitude", "mean"))
    )

    vocab_df["clustering"] = vocab_df["cluster"].astype(int)
    vocab_df["user_uuid"] = user_tag
    vocab_df = vocab_df.drop(columns=["cluster"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(vocab_df.to_dict(orient="records"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def assign_clusters_by_centroids(
    df_points: pd.DataFrame,
    vocab_df: pd.DataFrame,
    eps_meters: float,
) -> pd.DataFrame:
    """
    Assign cluster id to each point by nearest centroid with a distance threshold.

    If distance(point, nearest_centroid) <= eps_meters:
        cluster = centroid.clustering
    else:
        cluster = -1
    """
    if vocab_df.empty:
        out = df_points.copy()
        out["cluster"] = -1
        return out

    centroids_rad = np.radians(vocab_df[["lat", "lng"]].to_numpy())
    points_rad = np.radians(df_points[["latitude", "longitude"]].to_numpy())

    tree = BallTree(centroids_rad, metric="haversine")
    dists_rad, idxs = tree.query(points_rad, k=1)

    eps_rad = eps_meters / EARTH_RADIUS_METERS
    idxs = idxs.reshape(-1)
    dists_rad = dists_rad.reshape(-1)

    out = df_points.copy()
    out["cluster"] = -1

    within = dists_rad <= eps_rad
    out.loc[within, "cluster"] = vocab_df.loc[idxs[within], "clustering"].astype(int).to_numpy()
    return out


def plot_user_hexbin(df_u: pd.DataFrame, outpath: Path, user_idx: int, gridsize: int = 80) -> None:
    """Save a density heatmap (hexbin) for a user."""
    lat = df_u["latitude"].to_numpy()
    lon = df_u["longitude"].to_numpy()

    plt.figure()
    plt.hexbin(lon, lat, gridsize=gridsize, bins="log")
    plt.colorbar(label="log10(contagem)")
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
    if len(df_u) > max_points:
        df_u = df_u.sample(n=max_points, random_state=42)

    lat = df_u["latitude"].to_numpy()
    lon = df_u["longitude"].to_numpy()
    cluster = df_u["cluster"].to_numpy()

    plt.figure()
    plt.scatter(lon, lat, c=cluster, s=4, alpha=0.4)
    plt.colorbar(label="cluster id (-1 = noise)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(f"Clusters — user_{user_idx}")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def generate_cluster_heatmaps(
    viz_parquet_path: Path,
    output_dir: Path,
    *,
    clear_output_dir: bool = True,
) -> None:
    """Generate per-user cluster plots from visualization parquet."""
    if clear_output_dir and output_dir.exists():
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(viz_parquet_path)

    required = {"user_uuid", "latitude", "longitude", "cluster"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns in viz parquet: {missing}")

    for idx, (_, df_u) in enumerate(df.groupby("user_uuid", sort=True)):
        df_u = df_u.dropna(subset=["latitude", "longitude", "cluster"]).copy()
        plot_user_hexbin(df_u, output_dir / f"user_{idx}_hexbin.png", user_idx=idx)
        plot_user_clusters_scatter(df_u, output_dir / f"user_{idx}_clusters.png", user_idx=idx)


def main() -> None:
    """Run preprocessing pipeline and optionally generate plots."""
    args = parse_args()
    load_dotenv()

    logger = setup_logging(PROJECT_ROOT / "preprocessing.log")
    logger.info("--- Starting Preprocessing Pipeline ---")

    gs_url = os.getenv("GS_DATASET_URL")
    if not gs_url:
        raise RuntimeError("Environment variable GS_DATASET_URL is not set.")

    logger.info("Loading dataset from: %s", gs_url)
    df = pd.read_parquet(gs_url)
    logger.info("Dataset loaded: %d rows", len(df))

    df_clean = df.dropna(subset=REQUIRED_COLUMNS).copy()
    logger.info("After cleaning: %d rows", len(df_clean))

    dt = (
        pd.to_datetime(df_clean["event_timestamp"], unit="us", utc=True, errors="coerce")
        .dt.tz_convert("America/Recife")
    )
    df_clean["week_day"] = dt.dt.day_name()
    df_clean["hour"] = dt.dt.hour
    df_clean["year_num"] = dt.dt.year
    df_clean["week_num"] = dt.dt.isocalendar().week

    df_clean["week_order"] = df_clean["year_num"].astype(str) + "-" + df_clean["week_num"].astype(str)
    df_clean["week_order"] = pd.Categorical(
        df_clean["week_order"],
        categories=df_clean["week_order"].unique(),
        ordered=True,
    )

    fuzzy_df = df_clean["hour"].apply(compute_fuzzy_time_memberships).apply(pd.Series)
    df_clean = pd.concat([df_clean.drop(columns=[]), fuzzy_df], axis=1)

    df_filtered = df_clean.copy()
    logger.info("Before filtering: rows=%d users=%d", len(df_filtered), df_filtered["user_uuid"].nunique())

    clicks = df_filtered.groupby("user_uuid")["event_timestamp"].count()
    df_filtered = df_filtered[df_filtered["user_uuid"].isin(clicks[clicks >= 50].index)]
    logger.info("After filter clicks>=50: rows=%d users=%d", len(df_filtered), df_filtered["user_uuid"].nunique())

    locs = df_filtered.groupby("user_uuid")["click_location"].nunique()
    df_filtered = df_filtered[df_filtered["user_uuid"].isin(locs[locs >= 10].index)]
    logger.info("After filter locs>=10: rows=%d users=%d", len(df_filtered), df_filtered["user_uuid"].nunique())

    hours = df_filtered.groupby("user_uuid")["hour"].nunique()
    df_filtered = df_filtered[df_filtered["user_uuid"].isin(hours[hours >= 20].index)]
    logger.info("After filter hours>=20: rows=%d users=%d", len(df_filtered), df_filtered["user_uuid"].nunique())

    weeks = df_filtered.groupby("user_uuid")["week_order"].nunique()
    df_filtered = df_filtered[df_filtered["user_uuid"].isin(weeks[weeks >= 3].index)]
    logger.info("After filter weeks>=3: rows=%d users=%d", len(df_filtered), df_filtered["user_uuid"].nunique())

    df_filtered["card_written_text"] = df_filtered["card_written_text"].map(normalize_card_text)

    coords = df_filtered["click_location"].astype(str).str.split(",", expand=True)
    df_filtered["latitude"] = pd.to_numeric(coords[0], errors="coerce")
    df_filtered["longitude"] = pd.to_numeric(coords[1], errors="coerce")
    df_filtered = df_filtered.dropna(subset=["latitude", "longitude"])

    selected_columns = [
        "user_uuid",
        "click_location",
        "card_written_text",
        "event_timestamp",
        "week_day",
        "hour",
        "year_num",
        "week_num",
        "week_order",
        "dawn",
        "morn",
        "noon",
        "aftn",
        "even",
        "nght",
        "latitude",
        "longitude",
    ]
    df_filtered = df_filtered.loc[:, selected_columns].sort_values(["user_uuid", "event_timestamp"])

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    filtered_parquet_path = PROCESSED_DIR / "df_filtered.parquet"
    df_filtered.to_parquet(filtered_parquet_path, index=False)
    logger.info("Saved filtered dataset: %s", filtered_parquet_path)

    user_list = df_filtered["user_uuid"].unique()
    le_user = LabelEncoder().fit(user_list)

    global_encoder_dir = PROJECT_ROOT / "models" / "global_encoders"
    global_encoder_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(le_user, global_encoder_dir / "label_encoder_user.pkl")
    logger.info("Saved global LabelEncoder (users): %s", global_encoder_dir / "label_encoder_user.pkl")

    processed_parts: List[pd.DataFrame] = []
    viz_parts: List[pd.DataFrame] = []

    large_user_threshold = 50_000
    max_points_per_cell = 300
    precision = 3

    for i, user_uuid in enumerate(user_list):
        if args.user_idx is not None and i != args.user_idx:
            continue

        logger.info("Processing user_%d", i)
        user_dir = PROJECT_ROOT / "models" / f"user_{i}"
        user_dir.mkdir(parents=True, exist_ok=True)

        if not args.force and user_artifacts_exist(user_dir):
            logger.info("Skipping user_%d (artifacts exist). Use --force to reprocess.", i)
            continue

        df_u = df_filtered[df_filtered["user_uuid"] == user_uuid].copy()
        df_u = df_u.sort_values("event_timestamp")

        generate_label_vocab_json(
            df_u,
            text_col="card_written_text",
            output_path=user_dir / "label_vocab.json",
        )

        le_card_user = LabelEncoder().fit(df_u["card_written_text"])
        joblib.dump(le_card_user, user_dir / "label_encoder_card.pkl")

        df_for_clustering = df_u
        if len(df_u) > large_user_threshold:
            tmp = df_u.copy()
            tmp["spatial_cell"] = tmp.apply(
                lambda r: spatial_cell(float(r["latitude"]), float(r["longitude"]), precision=precision),
                axis=1,
            )
            df_for_clustering = (
                tmp.groupby("spatial_cell", group_keys=False)
                .apply(
                    lambda g: g.sample(n=min(len(g), max_points_per_cell), random_state=42),
                    include_groups=False,
                )
                .drop(columns=["spatial_cell"], errors="ignore")
            )
            logger.info("User_%d sampled for clustering: %d -> %d", i, len(df_u), len(df_for_clustering))

        clustered = clusterize_locations(df_for_clustering, eps_meters=EPS_METERS, min_samples=MIN_SAMPLES)

        vocab_path = user_dir / "location_vocab_user.json"
        save_user_location_vocab(clustered, user_tag=f"user_{i}", output_path=vocab_path)

        vocab_df = load_location_vocab_df(vocab_path)
        df_u = assign_clusters_by_centroids(df_u, vocab_df, eps_meters=EPS_METERS)

        df_u["user_uuid_enc"] = le_user.transform(df_u["user_uuid"])
        df_u["card_enc"] = le_card_user.transform(df_u["card_written_text"])

        cat_cols = ["week_day", "cluster"]
        fuzzy_cols = ["dawn", "morn", "noon", "aftn", "even", "nght"]

        ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        ohe.fit(df_u[cat_cols])
        joblib.dump(ohe, user_dir / "onehot_encoder.pkl")

        cat_data = ohe.transform(df_u[cat_cols])
        cat_names = ohe.get_feature_names_out(cat_cols)
        df_cat = pd.DataFrame(cat_data, columns=cat_names, index=df_u.index)

        df_fuzzy = df_u[fuzzy_cols].astype(float)
        df_drop = df_u.drop(columns=["card_written_text", "user_uuid"] + cat_cols + fuzzy_cols)
        df_final = pd.concat([df_drop, df_cat, df_fuzzy], axis=1)

        out_path = user_dir / "processed.parquet"
        df_final.to_parquet(out_path, compression="snappy", index=False)

        processed_parts.append(df_final)
        viz_parts.append(df_u[["user_uuid", "latitude", "longitude", "cluster"]].copy())

        logger.info("Saved user_%d artifacts at: %s", i, user_dir)

    if processed_parts:
        df_all = pd.concat(processed_parts, ignore_index=True)
        out_all_path = PROCESSED_DIR / "all_users_processed.parquet"
        df_all.to_parquet(out_all_path, index=False, compression="snappy")
        logger.info("Saved global dataset: %s (rows=%d)", out_all_path, len(df_all))

    if viz_parts:
        df_viz = pd.concat(viz_parts, ignore_index=True)
        viz_path = PROCESSED_DIR / "data_for_visualization.parquet"
        df_viz.to_parquet(viz_path, index=False, compression="snappy")
        logger.info("Saved viz parquet: %s (rows=%d)", viz_path, len(df_viz))

        if not args.no_plots:
            generate_cluster_heatmaps(viz_path, REPORTS_DIR, clear_output_dir=True)
            logger.info("Saved plots to: %s", REPORTS_DIR)

    logger.info("--- Done ---")


if __name__ == "__main__":
    main()

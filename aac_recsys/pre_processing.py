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
import os
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.cluster import DBSCAN
from sklearn.neighbors import BallTree
from sklearn.preprocessing import LabelEncoder

from aac_recsys.config import PROJ_ROOT, PROCESSED_DATA_DIR, MODELS_DIR, logger
from aac_recsys.plots import run_plots


EARTH_RADIUS_METERS = 6_373_000
EPS_METERS = 400
MIN_SAMPLES = 3

REQUIRED_COLUMNS = ["user_uuid", "click_location", "card_written_text", "event_timestamp"]
PERIOD_KEYS = ["dawn", "morn", "noon", "aftn", "even", "nght"]

BASELINE_COLUMNS = ["timestamp", "event_timestamp", "card_enc"]
RF_COLUMNS = [
    "timestamp",
    "event_timestamp",
    "card_enc",
    "hour",
    "week_num",
    "week_day",
    "latitude",
    "longitude",
    "cluster",
    "dawn", "morn", "noon", "aftn", "even", "nght",
]

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="baseline", help="Model type to preprocess for")
    parser.add_argument("--user-idx", type=int, default=None, help="Process only user_{idx}")
    parser.add_argument("--force", action="store_true", help="Reprocess even if outputs exist")
    parser.add_argument("--plots", action="store_true", help="Generate plots")
    parser.add_argument("--plots-per-user", action="store_true", help="Generate plots per user")
    return parser.parse_args()


def user_artifacts_exist(user_data_dir: Path, user_model_dir: Path, model: str) -> bool:
    """
    Return True if all expected per-user artifacts exist.

    Data artifacts (data/processed):
    - {model}_processed.parquet
    - label_vocab.json
    - location_vocab_user.json

    Model artifacts (models):
    - label_encoder_card.pkl
    """
    base = [
        user_data_dir / "label_vocab.json",
        user_data_dir / "location_vocab_user.json",
        user_model_dir / "label_encoder_card.pkl",
    ]

    if model == "baseline":
        needed = base + [user_data_dir / "baseline_processed.parquet"]
    elif model == "random_forest":
        needed = base + [user_data_dir / "random_forest_processed.parquet"]
    elif model == "two_tower":
        needed = base + [user_data_dir / "two_tower_processed.parquet"]
    else:
        needed = base

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


def run_preprocess(*, model: str, user_idx: int | None = None, force: bool = False, plots: bool = False, plots_per_user: bool = False) -> None:
    """Run preprocessing pipeline and optionally generate plots."""
    load_dotenv()

    log_path = PROJ_ROOT / "preprocessing.log"
    sink_id = logger.add(str(log_path), level="INFO")
    logger.info("--- Starting Preprocessing Pipeline ---")
    logger.info("--- Model: {} ---", model)

    gs_url = os.getenv("GS_DATASET_URL")
    if not gs_url:
        raise RuntimeError("Environment variable GS_DATASET_URL is not set.")

    filtered_parquet_path = PROCESSED_DATA_DIR / "df_filtered.parquet"
    if filtered_parquet_path.exists():
        logger.info("--- Loading cached filtered dataset ---")
        df_filtered = pd.read_parquet(filtered_parquet_path)
    else:
        logger.info("--- Loading dataset from URL ---")
        df = pd.read_parquet(gs_url)
        logger.info("Dataset loaded: {} rows", len(df))

        df_clean = df.dropna(subset=REQUIRED_COLUMNS).copy()
        logger.info("After cleaning: {} rows", len(df_clean))

        dt = (
            pd.to_datetime(df_clean["event_timestamp"], unit="us", utc=True, errors="coerce")
            .dt.tz_convert("America/Recife")
        )
        df_clean["timestamp"] = dt
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
        logger.info("Before filtering: rows={} users={}", len(df_filtered), df_filtered["user_uuid"].nunique())

        clicks = df_filtered.groupby("user_uuid")["event_timestamp"].count()
        df_filtered = df_filtered[df_filtered["user_uuid"].isin(clicks[clicks >= 50].index)]
        logger.info("After filter clicks>=50: rows={} users={}", len(df_filtered), df_filtered["user_uuid"].nunique())

        locs = df_filtered.groupby("user_uuid")["click_location"].nunique()
        df_filtered = df_filtered[df_filtered["user_uuid"].isin(locs[locs >= 3].index)]
        logger.info("After filter locs>=3: rows={} users={}", len(df_filtered), df_filtered["user_uuid"].nunique())

        hours = df_filtered.groupby("user_uuid")["hour"].nunique()
        df_filtered = df_filtered[df_filtered["user_uuid"].isin(hours[hours >= 18].index)]
        logger.info("After filter hours>=18: rows={} users={}", len(df_filtered), df_filtered["user_uuid"].nunique())

        weeks = df_filtered.groupby("user_uuid")["week_order"].nunique()
        df_filtered = df_filtered[df_filtered["user_uuid"].isin(weeks[weeks >= 3].index)]
        logger.info("After filter weeks>=3: rows={} users={}", len(df_filtered), df_filtered["user_uuid"].nunique())

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
            "timestamp",
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
        df_filtered = df_filtered.loc[:, selected_columns].sort_values(["user_uuid", "timestamp"])

        df_filtered.to_parquet(filtered_parquet_path, index=False)
        logger.success("Saved filtered dataset: {}", filtered_parquet_path)

    user_list = df_filtered["user_uuid"].unique()
    le_user = LabelEncoder().fit(user_list)

    global_encoder_dir = MODELS_DIR / "global_encoders"
    global_encoder_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(le_user, global_encoder_dir / "label_encoder_user.pkl")
    logger.success("Saved global LabelEncoder (users): {}", global_encoder_dir / "label_encoder_user.pkl")

    viz_parts: List[pd.DataFrame] = []

    large_user_threshold = 50_000
    max_points_per_cell = 300
    precision = 3

    for i, user_uuid in enumerate(user_list):
        if user_idx is not None and i != user_idx:
            continue

        logger.info("Processing user_{}", i)
        user_data_dir = PROCESSED_DATA_DIR / f"user_{i}"
        user_model_dir = MODELS_DIR / f"user_{i}"

        user_data_dir.mkdir(parents=True, exist_ok=True)
        user_model_dir.mkdir(parents=True, exist_ok=True)

        if not force and user_artifacts_exist(user_data_dir, user_model_dir, model):
            logger.info("Skipping user_{} (artifacts exist). Use --force to reprocess.", i)
            continue

        df_u = df_filtered[df_filtered["user_uuid"] == user_uuid].copy()
        df_u = df_u.sort_values("timestamp")

        generate_label_vocab_json(
            df_u,
            text_col="card_written_text",
            output_path=user_data_dir / "label_vocab.json",
        )

        le_card_user = LabelEncoder().fit(df_u["card_written_text"])
        joblib.dump(le_card_user, user_model_dir / "label_encoder_card.pkl")

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
            logger.info("User_{} sampled for clustering: {} -> {}", i, len(df_u), len(df_for_clustering))

        clustered = clusterize_locations(df_for_clustering, eps_meters=EPS_METERS, min_samples=MIN_SAMPLES)

        vocab_path = user_data_dir / "location_vocab_user.json"
        save_user_location_vocab(clustered, user_tag=f"user_{i}", output_path=vocab_path)

        vocab_df = load_location_vocab_df(vocab_path)
        df_u = assign_clusters_by_centroids(df_u, vocab_df, eps_meters=EPS_METERS)

        df_u["card_enc"] = le_card_user.transform(df_u["card_written_text"])

        if model == "baseline":
            df_base = df_u.loc[:, BASELINE_COLUMNS].copy()

            base_out_path = user_data_dir / "baseline_processed.parquet"
            df_base.to_parquet(base_out_path, compression="snappy", index=False)
            logger.success("Saved Baseline dataset for user_{} at: {}", i, base_out_path)

        if model == "random_forest":
            rf_cols = [c for c in RF_COLUMNS if c in df_u.columns]
            missing = [c for c in RF_COLUMNS if c not in df_u.columns]
            if missing:
                logger.warning("User_{} RF missing cols (will be skipped): {}", i, missing)

            df_rf = df_u.loc[:, rf_cols].copy()

            df_rf["card_enc"] = df_rf["card_enc"].astype(int)
            df_rf["cluster"] = pd.to_numeric(df_rf["cluster"], errors="coerce").fillna(-1).astype(int)
            df_rf["hour"] = df_rf["hour"].astype(int)

            rf_out_path = user_data_dir / "random_forest_processed.parquet"
            df_rf.to_parquet(rf_out_path, compression="snappy", index=False)
            logger.success("Saved RF dataset for user_{} at: {}", i, rf_out_path)

        if model == "two_tower":
            # Placeholder for future Two-Tower preprocessing
            pass

        viz_parts.append(df_u[["user_uuid", "latitude", "longitude", "cluster"]].copy())

        logger.success("Saved user_{} data at: {} | models at: {}", i, user_data_dir, user_model_dir)

    if viz_parts and user_idx is None:
        df_viz = pd.concat(viz_parts, ignore_index=True)
        viz_path = PROCESSED_DATA_DIR / "data_for_visualization.parquet"
        df_viz.to_parquet(viz_path, index=False, compression="snappy")
        logger.success("Saved viz parquet: {} (rows={})", viz_path, len(df_viz))

        gen_plots = plots or plots_per_user
        if gen_plots:
            run_plots(viz_path, clear=True, per_user=plots_per_user, clear_per_user=plots_per_user, top_n=15)

    logger.success("--- Done ---")
    logger.remove(sink_id)

def main() -> None:
    args = parse_args()
    run_preprocess(model=args.model, user_idx=args.user_idx, force=args.force, plots=args.plots, plots_per_user=args.plots_per_user)

if __name__ == "__main__":
    main()

"""
Preprocessing pipeline for the AAC pictogram recommendation experiment.

This script:
    1. Loads the raw clickstream dataset.
    2. Cleans rows with missing critical fields.
    3. Extracts temporal context features (weekday, hour, week number, etc.).
    4. Filters users based on activity richness criteria.
    5. Normalizes card text casing.
    6. Parses click locations into latitude/longitude.
    7. Trains:
        - a global user LabelEncoder
        - per-user LabelEncoders for card text
        - per-user OneHotEncoders for contextual features
    8. Applies spatial DBSCAN clustering per user (train),
       and assigns cluster labels to test data based on nearest neighbors.
    9. Saves:
        - filtered base dataset
        - per-user processed train/test datasets
        - global concatenated train/test datasets
        - dataset for cluster visualization.
"""

import os
import json
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any
from math import radians, sin, cos, sqrt, atan2

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.neighbors import BallTree
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from dotenv import load_dotenv

# Constants
EARTH_RADIUS_METERS = 6_373_000
EPS_METERS = 400
EPS_KM = EPS_METERS / 1000
MIN_SAMPLES = 3
REQUIRED_COLUMNS = ['user_uuid', 'click_location', 'card_written_text', 'event_timestamp']

# Directory Paths
FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


# Helper Functions

def spatial_cell(lat, lon, precision=3):
    """
    Create a coarse spatial cell id by rounding coordinates.
    precision=3 ≈ 100m–150m
    """
    return f"{round(lat, precision)}_{round(lon, precision)}"


def haversine_distance_km(
    user_lat: float,
    user_lng: float,
    lat: float,
    lng: float,
) -> float:
    """
    Compute the Haversine (great-circle) distance between two points.

    Matches the original Kotlin implementation:
    - Earth radius = 6373.0 km
    - Returns distance in kilometers
    """
    earth_radius_km = 6373.0

    d_lat = radians(lat - user_lat)
    d_lng = radians(lng - user_lng)

    a = (
        sin(d_lat / 2) ** 2
        + cos(radians(user_lat))
        * cos(radians(lat))
        * sin(d_lng / 2) ** 2
    )

    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return earth_radius_km * c


def load_location_vocab(
    path: str | Path,
    *,
    user_uuid: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Load location vocabulary (cluster centroids) from JSON.

    Optionally filters by user_uuid.
    """
    path = Path(path)
    records = json.loads(path.read_text(encoding="utf-8"))

    if user_uuid is not None:
        records = [
            r for r in records
            if str(r.get("user_uuid")) == str(user_uuid)
        ]

    return records


def load_location_vocab_df(vocab_path: str | Path) -> pd.DataFrame:
    """
    Loads location_vocab_user.json into a DataFrame with columns:
    ['lat', 'lng', 'clustering', 'user_uuid'].
    """
    vocab_path = Path(vocab_path)
    records = json.loads(vocab_path.read_text(encoding="utf-8"))
    return pd.DataFrame(records)


def gaussian_membership(x: float, center: float, sigma: float) -> float:
    return float(np.exp(-((x - center) ** 2) / (2 * (sigma ** 2))))


def compute_fuzzy_time_memberships(hour: int) -> dict[str, float]:
    return {
        "dawn": gaussian_membership(hour, 2.5, 2.0),
        "morn": gaussian_membership(hour, 7.5, 2.0),
        "noon": gaussian_membership(hour, 11.5, 2.0),
        "aftn": gaussian_membership(hour, 13.5, 2.0),
        "even": gaussian_membership(hour, 17.5, 2.0),
        "nght": gaussian_membership(hour, 22.5, 2.0),
    }


def assign_clusters_by_centroids(
    df_points: pd.DataFrame,
    vocab_df: pd.DataFrame,
    eps_meters: float = 400,
) -> pd.DataFrame:
    """
    Assign cluster to each point using nearest centroid (hotspot) + eps threshold.
    Returns df with a 'cluster' column (cluster id or -1).
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

    out = df_points.copy()
    out["cluster"] = -1

    # idxs shape: (n,1) -> flatten
    idxs = idxs.reshape(-1)
    dists_rad = dists_rad.reshape(-1)

    within = dists_rad <= eps_rad
    # vocab_df['clustering'] é o id do cluster
    out.loc[within, "cluster"] = vocab_df.loc[idxs[within], "clustering"].astype(int).to_numpy()

    return out


def quantize_location(
    lat: float,
    lon: float,
    *,
    vocab_path: str | Path,
    user_uuid: Optional[str] = None,
) -> Tuple[Optional[int], float]:
    """
    Assign a location to the nearest spatial cluster centroid.

    Args:
        lat (float): Latitude of the query point
        lon (float): Longitude of the query point
        vocab_path (str | Path): Path to location_vocab_user.json
        user_uuid (Optional[str]): Filter clusters by user if needed

    Returns:
        (cluster_id, distance_km)
        - cluster_id: value from 'clustering' field
        - distance_km: Haversine distance in kilometers
    """
    hotspots = load_location_vocab(vocab_path, user_uuid=user_uuid)

    if not hotspots:
        return None, float("inf")

    best_cluster = None
    best_distance = float("inf")

    for h in hotspots:
        h_lat = float(h["lat"])
        h_lng = float(h["lng"])

        distance = haversine_distance_km(
            user_lat=lat,
            user_lng=lon,
            lat=h_lat,
            lng=h_lng,
        )

        if distance < best_distance:
            best_distance = distance
            best_cluster = int(h["clustering"])

    return best_cluster, best_distance

def quantize_location_with_threshold(
    lat: float,
    lon: float,
    *,
    vocab_path: str | Path,
    max_distance_km: float = 0.4,  # 400 meters
    user_uuid: Optional[str] = None,
) -> int:
    """
    Quantize location with a maximum distance threshold.
    Returns -1 if no cluster is close enough.
    """
    cluster_id, distance_km = quantize_location(
        lat,
        lon,
        vocab_path=vocab_path,
        user_uuid=user_uuid,
    )

    if distance_km > max_distance_km:
        return -1

    return cluster_id


def clusterize_locations(
    df: pd.DataFrame,
    eps_meters: float = 400,
    min_samples: int = 3
) -> pd.DataFrame:
    """
    Perform DBSCAN spatial clustering using latitude/longitude with haversine distance.

    Args:
        df (pd.DataFrame): DataFrame containing 'latitude' and 'longitude' columns.
        eps_meters (float): Maximum neighborhood radius in meters for cluster formation.
        min_samples (int): Minimum number of points required to form a cluster.

    Returns:
        pd.DataFrame: Copy of the input DataFrame with a new 'cluster' column.
                      Cluster labels follow DBSCAN's convention:
                      -1 denotes noise/outliers.
    """
    coords = df[['latitude', 'longitude']].to_numpy()
    coords_rad = np.radians(coords)
    eps_rad = eps_meters / EARTH_RADIUS_METERS

    db = DBSCAN(
        eps=eps_rad,
        min_samples=min_samples,
        metric="haversine"
    ).fit(coords_rad)

    df_out = df.copy()
    df_out['cluster'] = db.labels_
    return df_out


def assign_test_clusters(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    eps_meters: float = 400
) -> pd.DataFrame:
    """
    Assign cluster labels to test points based on proximity to clustered train points.

    For each test point, the nearest train point is found using a BallTree with
    haversine distance. If the distance is within eps_meters, the corresponding
    cluster label is copied; otherwise, the test point is labeled as noise (-1).

    Args:
        train_df (pd.DataFrame): Training DataFrame with 'latitude', 'longitude',
                                 and 'cluster' columns.
        test_df (pd.DataFrame): Test DataFrame with 'latitude' and 'longitude'.
        eps_meters (float): Maximum allowed distance in meters for a cluster match.

    Returns:
        pd.DataFrame: Test DataFrame copy with an added 'cluster' column.
    """
    train_coords = np.radians(train_df[['latitude', 'longitude']].to_numpy())
    test_coords = np.radians(test_df[['latitude', 'longitude']].to_numpy())
    eps_rad = eps_meters / EARTH_RADIUS_METERS

    tree = BallTree(train_coords, metric="haversine")
    dists, indices = tree.query(test_coords, k=1)

    df_out = test_df.copy()
    df_out['cluster'] = -1  # default for "no cluster found"

    cluster_col_idx = df_out.columns.get_loc('cluster')

    for i, (dist, idx) in enumerate(zip(dists, indices)):
        if dist < eps_rad:
            df_out.iloc[i, cluster_col_idx] = train_df.iloc[idx]['cluster'] # type: ignore

    return df_out

def save_user_location_vocab(
    df_train_clustered: pd.DataFrame,
    *,
    user_uuid: str,
    output_path: str,
) -> None:
    """
    Save per-user cluster centroids (mean lat/lng) to a JSON file.

    Output schema (list of dicts):
        - lat (float)
        - lng (float)
        - clustering (int)  # cluster id
        - user_uuid (str)

    Notes:
        - Noise points (cluster == -1) are excluded.
        - Uses mean latitude/longitude per cluster.
    """
    vocab_df = (
        df_train_clustered.loc[df_train_clustered["cluster"] != -1, ["latitude", "longitude", "cluster"]]
        .groupby("cluster", as_index=False)
        .agg(lat=("latitude", "mean"), lng=("longitude", "mean"))
    )

    vocab_df["clustering"] = vocab_df["cluster"].astype(int)
    vocab_df["user_uuid"] = user_uuid
    vocab_df = vocab_df.drop(columns=["cluster"])

    records = vocab_df.to_dict(orient="records")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def generate_user_clusters(df_train: pd.DataFrame, *, user_uuid: Optional[str]=None, vocab_output_path: Optional[str]=None) -> pd.DataFrame:
    """
    Run DBSCAN-based spatial clustering on train data and propagate cluster labels to test data.

    Optionally saves per-user cluster centroids to a JSON vocabulary file.
    """
    # Cluster train
    df_train_clustered = clusterize_locations(df_train, eps_meters=EPS_METERS, min_samples=MIN_SAMPLES)

    # Save vocab if requested
    if vocab_output_path is not None:
        if user_uuid is None:
            if "user_uuid" not in df_train_clustered.columns:
                raise ValueError(
                    "user_uuid not provided and 'user_uuid' column not found in df_train."
                )
            user_uuid = str(df_train_clustered["user_uuid"].iloc[0])

        save_user_location_vocab(
            df_train_clustered,
            user_uuid=user_uuid,
            output_path=vocab_output_path,
        )

    # Propagate to test
    return df_train_clustered


# Main Pipeline
def main() -> None:
    """
    Execute the full preprocessing pipeline:
    load → clean → feature engineering → filters → clustering → encoding → save.
    """
    load_dotenv()
    print("\n--- Starting Preprocessing Pipeline ---\n")

    # 1. Load dataset
    gs_url = os.getenv("GS_DATASET_URL")

    if not gs_url:
        raise RuntimeError(
            "Environment variable GS_DATASET_URL is not set. "
            "Please set GS_DATASET_URL to point to your dataset location."
        )

    print(f"Loading dataset from: {gs_url}")

    try:
        df = pd.read_parquet(gs_url)
        print(f"Dataset loaded successfully: {len(df):,} rows")
    except FileNotFoundError:
        raise FileNotFoundError(f"Parquet file not found at path: {gs_url}")
    except Exception as e:
        raise RuntimeError(f"Failed to load parquet file: {e}")

    # 2. Remove rows with missing critical columns
    df_clean = df.dropna(subset=REQUIRED_COLUMNS).copy()

    print(f"Total rows before cleaning: {len(df):,}")
    print(f"Total rows after cleaning:  {len(df_clean):,}")

    # 3. Temporal feature engineering
    # Parse timestamp as UTC and convert to production timezone
    df_clean["datetime"] = (
        pd.to_datetime(df_clean["event_timestamp"], unit="us", utc=True, errors="coerce")
        .dt.tz_convert("America/Recife")
    )

    # Day of week (e.g. Monday, Tuesday, ...)
    df_clean["week_day"] = df_clean["datetime"].dt.day_name() # type: ignore
    df_clean["weekday_prod"] = (df_clean["datetime"].dt.dayofweek + 1) % 7

    # Hour of the day [0–23]
    df_clean["hour"] = df_clean["datetime"].dt.hour # type: ignore

    # Year (used for week ordering)
    df_clean["year_num"] = df_clean["datetime"].dt.year # type: ignore

    # ISO week number [1–52]
    df_clean["week_num"] = df_clean["datetime"].dt.isocalendar().week # type: ignore

    # Week order identifier (year-week) as an ordered categorical
    df_clean["week_order"] = (
        df_clean["year_num"].astype(str) + "-" + df_clean["week_num"].astype(str)
    )
    df_clean["week_order"] = pd.Categorical(
        df_clean["week_order"],
        categories=df_clean["week_order"].unique(),
        ordered=True,
    )

    fuzzy_df = df_clean["hour"].apply(compute_fuzzy_time_memberships).apply(pd.Series)
    df_clean = pd.concat([df_clean, fuzzy_df], axis=1)

    # Drop intermediate datetime column
    df_clean = df_clean.drop(columns=["datetime"])

    # 4. User filtering: enforce activity and diversity constraints
    print("\n--- BEFORE FILTERING ---")
    print(f"Total rows:           {df_clean.shape[0]:,}")
    print(f"Unique users:         {df_clean['user_uuid'].nunique():,}")

    df_filtered = df_clean.copy()

    # Filter 1: Minimum 50 clicks per user
    print("\n>>> FILTER 1: Minimum 50 clicks per user")
    rows_before = df_filtered.shape[0]
    users_before = df_filtered["user_uuid"].nunique()

    user_clicks = df_filtered.groupby("user_uuid")["event_timestamp"].count()
    qualified_users = user_clicks[user_clicks >= 50].index
    df_filtered = df_filtered[df_filtered["user_uuid"].isin(qualified_users)]

    print(f"Rows removed:    {rows_before - df_filtered.shape[0]:,}")
    print(f"Users removed:   {users_before - df_filtered['user_uuid'].nunique():,}")
    print(
        f"Current status:  {df_filtered.shape[0]:,} rows, "
        f"{df_filtered['user_uuid'].nunique():,} users"
    )

    # Filter 2: Minimum 10 distinct click locations per user
    print("\n>>> FILTER 2: Minimum 10 distinct click locations per user")
    rows_before = df_filtered.shape[0]
    users_before = df_filtered["user_uuid"].nunique()

    user_locations = df_filtered.groupby("user_uuid")["click_location"].nunique()
    qualified_users = user_locations[user_locations >= 10].index
    df_filtered = df_filtered[df_filtered["user_uuid"].isin(qualified_users)]

    print(f"Rows removed:    {rows_before - df_filtered.shape[0]:,}")
    print(f"Users removed:   {users_before - df_filtered['user_uuid'].nunique():,}")
    print(
        f"Current status:  {df_filtered.shape[0]:,} rows, "
        f"{df_filtered['user_uuid'].nunique():,} users"
    )

    # Filter 3: Minimum 20 distinct click hours per user
    print("\n>>> FILTER 3: Minimum 20 distinct click hours per user")
    rows_before = df_filtered.shape[0]
    users_before = df_filtered["user_uuid"].nunique()

    user_hours = df_filtered.groupby("user_uuid")["hour"].nunique()
    qualified_users = user_hours[user_hours >= 20].index
    df_filtered = df_filtered[df_filtered["user_uuid"].isin(qualified_users)]

    print(f"Rows removed:    {rows_before - df_filtered.shape[0]:,}")
    print(f"Users removed:   {users_before - df_filtered['user_uuid'].nunique():,}")
    print(
        f"Current status:  {df_filtered.shape[0]:,} rows, "
        f"{df_filtered['user_uuid'].nunique():,} users"
    )

    # Filter 4: Minimum 1000 clicks per user
    print("\n>>> FILTER 4: Minimum 1000 clicks per user")
    rows_before = df_filtered.shape[0]
    users_before = df_filtered["user_uuid"].nunique()

    user_clicks_1000 = df_filtered.groupby("user_uuid")["event_timestamp"].count()
    qualified_users = user_clicks_1000[user_clicks_1000 >= 1000].index
    df_filtered = df_filtered[df_filtered["user_uuid"].isin(qualified_users)]

    print(f"Rows removed:    {rows_before - df_filtered.shape[0]:,}")
    print(f"Users removed:   {users_before - df_filtered['user_uuid'].nunique():,}")
    print(
        f"Current status:  {df_filtered.shape[0]:,} rows, "
        f"{df_filtered['user_uuid'].nunique():,} users"
    )

    # Filter 5: Minimum 3 distinct weeks of usage per user
    print("\n>>> FILTER 5: Minimum 3 weeks of usage per user")
    rows_before = df_filtered.shape[0]
    users_before = df_filtered["user_uuid"].nunique()

    user_weeks = df_filtered.groupby("user_uuid")["week_order"].nunique()
    qualified_users = user_weeks[user_weeks >= 3].index
    df_filtered = df_filtered[df_filtered["user_uuid"].isin(qualified_users)]

    print(f"Rows removed:    {rows_before - df_filtered.shape[0]:,}")
    print(f"Users removed:   {users_before - df_filtered['user_uuid'].nunique():,}")
    print(
        f"Current status:  {df_filtered.shape[0]:,} rows, "
        f"{df_filtered['user_uuid'].nunique():,} users"
    )

    # Final state after filtering
    print("\n--- FINAL FILTERED DATASET STATE ---")
    print(f"Total rows:      {df_filtered.shape[0]:,}")
    print(f"Unique users:    {df_filtered['user_uuid'].nunique():,}")

    # 5. Normalize card text (lowercase) and select relevant columns
    df_filtered["card_written_text"] = df_filtered["card_written_text"].apply(
        lambda s: s.lower() if isinstance(s, str) else s
    )

    location_coords = df_filtered["click_location"].astype(str).str.split(",", expand=True)
    df_filtered["latitude"] = pd.to_numeric(location_coords[0], errors="coerce")
    df_filtered["longitude"] = pd.to_numeric(location_coords[1], errors="coerce")
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
        "weekday_prod",
        "dawn", "morn", "noon", "aftn", "even", "nght",
        "latitude",
        "longitude",
    ]
    df_filtered = df_filtered.loc[:, selected_columns]

    # Save intermediate filtered dataset
    filtered_parquet_path = PROCESSED_DIR / "df_filtered.parquet"
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    df_filtered.to_parquet(filtered_parquet_path, index=False)
    print(f"\nFiltered DataFrame saved to: {filtered_parquet_path}")

    # 7. Prepare global encoders (users)
    print("\n--- Preparing Global User Encoder ---")
    global_encoder_path = "../models/global_encoders"
    os.makedirs(global_encoder_path, exist_ok=True)

    all_user_uuids = df_filtered["user_uuid"].unique()
    le_user = LabelEncoder().fit(all_user_uuids)
    joblib.dump(le_user, os.path.join(global_encoder_path, "label_encoder_user.pkl"))
    print(
        f"Global user LabelEncoder trained and saved. "
        f"Total users: {len(le_user.classes_)}"
    )

    # 8. Train/test split per user, clustering, and encoding
    train_parts: List[pd.DataFrame] = []
    test_parts: List[pd.DataFrame] = []
    train_parts_for_viz: List[pd.DataFrame] = []

    df_filtered_sorted = df_filtered.sort_values(["user_uuid", "event_timestamp"])
    user_list = df_filtered_sorted["user_uuid"].unique()

    print('\n--- Processing Each User Individually ---')
    import logging

    # Configure logger
    logging.basicConfig(
        filename="preprocessing.log",
        filemode="w",  # overwrite each run
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Also print to terminal
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console.setFormatter(formatter)

    logging.getLogger().addHandler(console)

    for i, user_id in enumerate(user_list):
        logging.info(f"--- Processing user_{i} (UUID={user_id}) ---")

        user_model_path = f"../models/user_{i}"
        os.makedirs(user_model_path, exist_ok=True)

        # User-specific subset
        df_u = df_filtered_sorted[df_filtered_sorted["user_uuid"] == user_id]
        logging.info(f"   User rows: {len(df_u):,}")

        # Card LabelEncoder (per user)
        try:
            le_card_user = LabelEncoder().fit(df_u["card_written_text"])
            joblib.dump(le_card_user, os.path.join(user_model_path, "label_encoder_card.pkl"))
            logging.info("   - Card LabelEncoder trained successfully.")
        except Exception as e:
            logging.error(f"   ERROR training LabelEncoder for user_{i}: {e}")
            continue

        # Week-based train/test split
        weeks = sorted(df_u["week_order"].unique())
        logging.info(f"   - Weeks found: {weeks}")

        if len(weeks) < 3:
            logging.warning(f"   - Skipping user_{i}: not enough weeks (<3)")
            continue

        df_u = df_u.sort_values("week_order")
        weeks_test = weeks[-2:]
        weeks_train = weeks[:-2]

        df_train = df_u[df_u["week_order"].isin(weeks_train)].copy()
        df_test = df_u[df_u["week_order"].isin(weeks_test)].copy()

        logging.info(f"   - Train size: {len(df_train):,}, Test size: {len(df_test):,}")

        if df_train.empty or df_test.empty:
            logging.warning(f"   - Skipping user_{i}: train or test split is empty")
            continue

        # ---------------------------
        # Spatial clustering
        # ---------------------------
        try:
            logging.info(f"   - Running DBSCAN clustering for user_{i}...")

            vocab_output_path = os.path.join(user_model_path, "location_vocab_user.json")

            # 1) Build a sampled train split ONLY for clustering (large users)
            LARGE_USER_THRESHOLD = 50_000
            MAX_POINTS_PER_CELL = 300
            PRECISION = 3  # ~100–150m cell size

            df_train_for_clustering = df_train

            if len(df_train) > LARGE_USER_THRESHOLD:
                logging.info(
                    f"   - Large train split detected ({len(df_train):,} rows). "
                    "Applying spatial stratified sampling before DBSCAN..."
                )

                tmp = df_train.copy()
                tmp["spatial_cell"] = tmp.apply(
                    lambda r: spatial_cell(float(r["latitude"]), float(r["longitude"]), precision=PRECISION),
                    axis=1,
                )

                df_train_for_clustering = (
                    tmp.groupby("spatial_cell", group_keys=False)
                    .apply(lambda g: g.sample(
                        n=min(len(g), MAX_POINTS_PER_CELL),
                        random_state=42,
                    ))
                    .drop(columns=["spatial_cell"])
                )

                logging.info(
                    f"   - Sampling done. Train for clustering reduced to "
                    f"{len(df_train_for_clustering):,} rows."
                )

            # 2) Cluster sampled train -> this produces clusters for the sampled points
            generate_user_clusters(
                df_train_for_clustering,
                user_uuid=str(user_id),
                vocab_output_path=vocab_output_path,
            )

            # 3) Load centroids vocab
            vocab_df = load_location_vocab_df(vocab_output_path)

            # 4) Assign clusters to FULL train and test using centroids (quantizeLocation logic)
            df_train = assign_clusters_by_centroids(df_train, vocab_df, eps_meters=EPS_METERS)
            df_test  = assign_clusters_by_centroids(df_test,  vocab_df, eps_meters=EPS_METERS)
            logging.info("   - Spatial clustering completed successfully.")

            df_train["cluster_prod"] = df_train["cluster"].astype(int)
            df_test["cluster_prod"]  = df_test["cluster"].astype(int)

            df_train["hour_prod"] = df_train["hour"].astype(int)
            df_test["hour_prod"]  = df_test["hour"].astype(int)

            df_train["weekday_prod"] = df_train["weekday_prod"].astype(int)
            df_test["weekday_prod"]  = df_test["weekday_prod"].astype(int)

            df_train["input_vector_prod"] = list(zip(
            df_train["cluster_prod"], df_train["weekday_prod"], df_train["hour_prod"]
            ))
            df_test["input_vector_prod"] = list(zip(
                df_test["cluster_prod"], df_test["weekday_prod"], df_test["hour_prod"]
            ))

            num_train_clusters = df_train["cluster"].nunique() - (1 if -1 in df_train["cluster"].values else 0)
            num_test_clusters = df_test["cluster"].nunique() - (1 if -1 in df_test["cluster"].values else 0)

            logging.info(f"       Train clusters (full train): {num_train_clusters}")
            logging.info(f"       Test clusters:              {num_test_clusters}")

        except Exception as e:
            logging.error(f"   ERROR during clustering for user_{i}: {e}")
            continue

        # Store for visualization
        df_train["user_uuid_enc"] = le_user.transform(df_train["user_uuid"])
        train_parts_for_viz.append(df_train.copy())

        # ---------------------------
        # Encoding
        # ---------------------------
        logging.info("   - Applying encoders...")

        try:
            df_train["card_enc"] = le_card_user.transform(df_train["card_written_text"])
            df_test["card_enc"] = le_card_user.transform(df_test["card_written_text"])

            df_train["user_uuid_enc"] = le_user.transform(df_train["user_uuid"])
            df_test["user_uuid_enc"] = le_user.transform(df_test["user_uuid"])

            cat_cols = ["weekday_prod", "cluster_prod"]
            fuzzy_cols = ["dawn", "morn", "noon", "aftn", "even", "nght"]

            ohe_ctx = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

            logging.info("   - Training OneHotEncoder...")
            ohe_ctx.fit(df_train[cat_cols])
            joblib.dump(ohe_ctx, os.path.join(user_model_path, "onehot_encoder.pkl"))
            logging.info("   - OneHotEncoder trained successfully.")

            for df_part, name in [(df_train, "train"), (df_test, "test")]:
                # 1) OHE só nas categóricas
                ctx_cat = ohe_ctx.transform(df_part[cat_cols])
                cat_feature_names = ohe_ctx.get_feature_names_out(cat_cols)
                df_cat = pd.DataFrame(ctx_cat, columns=cat_feature_names, index=df_part.index)

                # 2) fuzzy entra como numérico direto (sem OHE)
                df_fuzzy = df_part[fuzzy_cols].astype(float)

                # 3) drop do que não vai pro modelo
                df_part_drop = df_part.drop(columns=["card_written_text", "user_uuid"] + cat_cols + fuzzy_cols)

                # 4) concat final
                df_final_user = pd.concat([df_part_drop, df_cat, df_fuzzy], axis=1)

                output_path = os.path.join(user_model_path, f"{name}_processed.parquet")
                df_final_user.to_parquet(output_path, compression="snappy")
                logging.info(f"       Saved {name} dataset to: {output_path}")

                if name == "train":
                    train_parts.append(df_final_user)
                else:
                    test_parts.append(df_final_user)

        except Exception as e:
            logging.error(f"   ERROR during encoding for user_{i}: {e}")
            continue

    # 9. Concatenate all users and save global datasets
    print("\n--- Concatenating and saving final train/test datasets ---")
    df_train_final = pd.concat(train_parts, ignore_index=True)
    df_test_final = pd.concat(test_parts, ignore_index=True)
    df_for_viz = pd.concat(train_parts_for_viz, ignore_index=True)

    processed_train_path = "../data/processed/train_final_processed.parquet"
    processed_test_path = "../data/processed/test_final_processed.parquet"
    viz_path = "../data/processed/data_for_visualization.parquet"

    df_train_final.to_parquet(processed_train_path, index=False, compression="snappy")
    df_test_final.to_parquet(processed_test_path, index=False, compression="snappy")
    df_for_viz.to_parquet(viz_path, index=False, compression="snappy")

    print(f"   - Final train dataset saved to: {processed_train_path}")
    print(f"   - Final test dataset saved to:  {processed_test_path}")
    print(f"   - Visualization dataset saved to: {viz_path}")

    print("\n--- Preprocessing completed for all users ---")
    print(f"Final train size: {len(df_train_final):,} rows")
    print(f"Final test size:  {len(df_test_final):,} rows")


# Entry Point

if __name__ == "__main__":
    main()

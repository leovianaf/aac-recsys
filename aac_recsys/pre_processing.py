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
from pathlib import Path
from typing import Tuple, List

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.neighbors import BallTree
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from dotenv import load_dotenv


# Constants
EARTH_RADIUS_METERS = 6_371_000
REQUIRED_COLUMNS = ['user_uuid', 'click_location', 'card_written_text', 'event_timestamp']

# Directory Paths
FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


# Helper Functions
def get_period_of_day(hour: int) -> str:
    """
    Map an hour of day [0–23] to a coarse time period bucket.

    Args:
        hour (int): Hour of day in 24h format.

    Returns:
        str: One of {'midnight', 'dawn', 'morning', 'noon', 'afternoon', 'evening', 'night'}.
    """
    if 6 <= hour <= 8:
        return "dawn"
    if 9 <= hour <= 11:
        return "morning"
    if 12 <= hour <= 14:
        return "noon"
    if 15 <= hour <= 17:
        return "afternoon"
    if 18 <= hour <= 20:
        return "evening"
    if 21 <= hour <= 23:
        return "night"
    return "midnight"


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


def generate_user_clusters(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run DBSCAN-based spatial clustering on train data and propagate cluster labels to test data.

    Args:
        df_train (pd.DataFrame): User-specific training subset, including 'latitude'
                                 and 'longitude' columns.
        df_test (pd.DataFrame): User-specific test subset, including 'latitude'
                                and 'longitude' columns.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            - Training DataFrame with 'cluster' labels assigned by DBSCAN.
            - Test DataFrame with cluster labels inferred by nearest neighbor matching.
    """
    df_train_clustered = clusterize_locations(df_train)
    df_test_clustered = assign_test_clusters(df_train_clustered, df_test)
    return df_train_clustered, df_test_clustered


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
    # Extract datetime from event_timestamp (microseconds)
    df_clean["datetime"] = pd.to_datetime(
        df_clean["event_timestamp"],
        unit="us",
        errors="coerce"
    )

    # Day of week (e.g. Monday, Tuesday, ...)
    df_clean["week_day"] = df_clean["datetime"].dt.day_name() # type: ignore

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

    # Period of day (midnight, dawn, morning, etc.)
    df_clean["period_day"] = df_clean["hour"].apply(get_period_of_day)

    # Custom ordered categories for period of day
    period_order = [
        "midnight",
        "dawn",
        "morning",
        "noon",
        "afternoon",
        "evening",
        "night",
    ]
    df_clean["period_day"] = pd.Categorical(
        df_clean["period_day"],
        categories=period_order,
        ordered=True,
    )

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
        "period_day",
    ]
    df_filtered = df_filtered.loc[:, selected_columns]

    # Save intermediate filtered dataset
    filtered_parquet_path = PROCESSED_DIR / "df_filtered.parquet"
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    df_filtered.to_parquet(filtered_parquet_path, index=False)
    print(f"\nFiltered DataFrame saved to: {filtered_parquet_path}")

    # 6. Parse geolocation coordinates (latitude/longitude)
    location_coords = df_filtered["click_location"].astype(str).str.split(",", expand=True)
    df_filtered["latitude"] = pd.to_numeric(location_coords[0], errors="coerce")
    df_filtered["longitude"] = pd.to_numeric(location_coords[1], errors="coerce")

    # Drop rows with invalid coordinates
    df_filtered = df_filtered.dropna(subset=["latitude", "longitude"])

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

    for i, user_id in enumerate(user_list):
        print(f"\n--- Processing user_{i} ---")
        user_model_path = f"../models/user_{i}"
        os.makedirs(user_model_path, exist_ok=True)

        # User-specific subset
        df_u = df_filtered_sorted[df_filtered_sorted["user_uuid"] == user_id]

        # Card LabelEncoder (per user, fit on all user data)
        le_card_user = LabelEncoder().fit(df_u["card_written_text"])
        joblib.dump(le_card_user, os.path.join(user_model_path, "label_encoder_card.pkl"))
        print(f"   - Card LabelEncoder trained and saved at {user_model_path}")

        # Week-based train/test split
        weeks = sorted(df_u["week_order"].unique())
        if len(weeks) < 3:
            # Not enough distinct weeks to create train/test split
            continue

        df_u = df_u.sort_values("week_order")
        weeks_test = weeks[-2:]
        weeks_train = weeks[:-2]

        df_train = df_u[df_u["week_order"].isin(weeks_train)].copy()
        df_test = df_u[df_u["week_order"].isin(weeks_test)].copy()

        if df_train.empty or df_test.empty:
            continue

        # Spatial clustering per user
        df_train, df_test = generate_user_clusters(df_train, df_test)
        print("   - Spatial clustering completed.")
        print(f"   - Total clusters found in train: {df_train['cluster'].nunique() - (1 if -1 in df_train['cluster'].values else 0)}")
        print(f"   - Total clusters assigned in test: {df_test['cluster'].nunique() - (1 if -1 in df_test['cluster'].values else 0)}")

        # Store train subset for cluster visualization
        df_train["user_uuid_enc"] = le_user.transform(df_train["user_uuid"])
        train_parts_for_viz.append(df_train.copy())

        # Encoders: card + user + contextual one-hot
        print("   - Applying encoders...")

        # 1) Card encoder (per user)
        df_train["card_enc"] = le_card_user.transform(df_train["card_written_text"])
        df_test["card_enc"] = le_card_user.transform(df_test["card_written_text"])

        # 2) Global user encoder
        df_train["user_uuid_enc"] = le_user.transform(df_train["user_uuid"])
        df_test["user_uuid_enc"] = le_user.transform(df_test["user_uuid"])

        # 3) One-hot encoder for contextual features
        context_cols = ["period_day", "week_day", "cluster"]
        ohe_ctx = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

        ohe_ctx.fit(df_train[context_cols])
        joblib.dump(ohe_ctx, os.path.join(user_model_path, "onehot_encoder.pkl"))
        print(f"   - OneHotEncoder trained and saved at {user_model_path}")

        # Transform and save user-specific train and test datasets
        for df_part, name in [(df_train, "train"), (df_test, "test")]:
            ctx_encoded = ohe_ctx.transform(df_part[context_cols])
            feature_names = ohe_ctx.get_feature_names_out(context_cols)
            df_encoded = pd.DataFrame(ctx_encoded, columns=feature_names, index=df_part.index) # type: ignore

            # Drop original text and categorical context columns
            df_part = df_part.drop(columns=["card_written_text", "user_uuid"] + context_cols)
            df_final_user = pd.concat([df_part, df_encoded], axis=1)

            output_path = os.path.join(user_model_path, f"{name}_processed.parquet")
            df_final_user.to_parquet(output_path, compression="snappy")
            print(f"   - File saved: {output_path}")

            if name == "train":
                train_parts.append(df_final_user)
            else:
                test_parts.append(df_final_user)

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

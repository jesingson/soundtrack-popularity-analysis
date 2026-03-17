"""Functions for loading and preparing soundtrack analysis data.

This module contains reusable helper functions for reading the source
datasets, engineering album-level features, normalizing analysis fields,
and selecting the final columns used in the soundtrack analysis workflow.
These helpers keep data preparation logic separate from the command-line
entry point so the code is easier to test, reuse, and maintain.
"""

import pandas as pd

FILM_IDS = ["tmdb_id"]

FILM_FEATURES = [
    "film_vote_count", "film_popularity",
    "film_budget", "film_revenue",
    "film_rating",
    "days_since_film_release", "film_runtime_min",
    "film_is_action", "film_is_adventure", "film_is_animation",
    "film_is_comedy", "film_is_crime", "film_is_documentary",
    "film_is_drama", "film_is_family", "film_is_fantasy",
    "film_is_history", "film_is_horror", "film_is_music",
    "film_is_mystery", "film_is_romance", "film_is_science_fiction",
    "film_is_tv_movie", "film_is_thriller", "film_is_war",
    "film_is_western",
]

ALBUM_IDS = ["release_group_mbid"]

ALBUM_FEATURES = [
    "days_since_album_release",
    "n_tracks",
    "composer_album_count",
    "ambient_experimental",
    "classical_orchestral",
    "electronic",
    "hip_hop_rnb",
    "pop",
    "rock",
    "world_folk",
]

TARGET_COL = "log_lfm_album_listeners"
Y_FEATURE = [TARGET_COL]

US_SCORE_NOMINEE_COLS = [
    "oscar_score_nominee",
    "globes_score_nominee",
    "critics_score_nominee",
]

US_SONG_NOMINEE_COLS = [
    "oscar_song_nominee",
    "globes_song_nominee",
    "critics_song_nominee",
]

BAFTA_NOMINEE_COL = ["bafta_score_nominee"]

DERIVED_AWARD_COLS = [
    "us_score_nominee_count",
    "us_song_nominee_count",
    "bafta_nominee",
]

def load_input_data(
    album_file_path:str,
    wide_file_path:str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the album and wide-format CSV files.

    Args:
        album_file_path (str): Path to the album-level CSV file.
        wide_file_path (str): Path to the wide-format CSV file.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the loaded
            album dataframe and wide dataframe.
    """
    albums_df = pd.read_csv(album_file_path)
    # Use low_memory to force Pandas to resolve datatypes more holistically
    wide_df = pd.read_csv(wide_file_path, low_memory = False)

    return albums_df, wide_df

# Cleaning and transformation functions

def add_track_counts(
        albums_df: pd.DataFrame,
        wide_df: pd.DataFrame
) -> pd.DataFrame:
    """Add a per-album track count feature to the album dataframe.

    This function counts the number of distinct tracks associated with each
    album-film combination in the wide-format dataframe, then merges that
    count back into the album-level dataframe as ``n_tracks``. If an
    existing ``n_tracks`` column is already present, it is dropped first so
    repeated runs remain deterministic and do not create merge suffixes.

    Args:
        albums_df: Album-level dataframe.
        wide_df: Wide-format dataframe containing track-level rows.

    Returns:
        pd.DataFrame: Album dataframe with an ``n_tracks`` column added.
    """
    # Add n_tracks: number of tracks
    # Count unique tracks per (release_group_mbid, tmdb_id)
    # If you have already computed n_tracks before, this will overwrite it deterministically.
    n_tracks_df = (
        wide_df[["release_group_mbid", "tmdb_id", "track_id"]]
        .groupby(["release_group_mbid", "tmdb_id"], as_index=False)
        .agg(n_tracks=("track_id", "nunique"))
    )

    # Drop existing column to avoid _x/_y suffixes on repeated runs
    if "n_tracks" in albums_df.columns:
        albums_df = albums_df.drop(columns=["n_tracks"])

    albums_df = albums_df.merge(
        n_tracks_df,
        on=["release_group_mbid", "tmdb_id"],
        how="left",
        validate="1:1",
    )

    # print("n_tracks added. Nulls:", int(albums_df["n_tracks"].isna().sum()))
    # print(albums_df["n_tracks"].describe())

    return albums_df


def add_award_features(albums_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived award-related features for each album.

    Args:
        albums_df: Album-level dataframe.

    Returns:
        pd.DataFrame: Album dataframe with derived award-related columns
        added.
    """
    existing = [col for col in DERIVED_AWARD_COLS if col in albums_df.columns]
    if existing:
        albums_df = albums_df.drop(columns=existing)

    albums_df["us_score_nominee_count"] = (
        albums_df[US_SCORE_NOMINEE_COLS].sum(axis=1).astype(int)
    )
    albums_df["us_song_nominee_count"] = (
        albums_df[US_SONG_NOMINEE_COLS].sum(axis=1).astype(int)
    )
    albums_df["bafta_nominee"] = (
        albums_df[BAFTA_NOMINEE_COL].sum(axis=1) > 0
    ).astype(int)

    return albums_df

def add_composer_album_count(
        albums_df: pd.DataFrame
) -> pd.DataFrame:
    """Add a composer-level album count feature to the album dataframe.

    This function computes how many album rows are associated with each
    cleaned primary composer value and merges that count back into the
    album dataframe as ``composer_album_count``. If the column already
    exists, it is dropped first so repeated runs remain deterministic.

    Args:
        albums_df: Album-level dataframe containing
            ``composer_primary_clean``.

    Returns:
        pd.DataFrame: Album dataframe with ``composer_album_count`` added.
    """

    composer_counts = (
        albums_df[["composer_primary_clean", "release_group_mbid", "tmdb_id"]]
        .groupby("composer_primary_clean", as_index=False)
        .agg(composer_album_count=("release_group_mbid", "count"))
    )

    if "composer_album_count" in albums_df.columns:
        albums_df = albums_df.drop(columns=["composer_album_count"])

    albums_df = albums_df.merge(
        composer_counts,
        on="composer_primary_clean",
        how="left",
        validate="m:1",
    )

    return albums_df

def normalize_genre_flags(
        albums_df: pd.DataFrame
) -> pd.DataFrame:
    """Normalize genre indicator columns to consistent 0/1 integer flags.

    This function coerces the selected album genre columns into a stable
    binary representation suitable for downstream analysis. Missing values
    are treated as absent genre membership and converted to 0.

    Args:
        albums_df: Album-level dataframe containing genre indicator columns.

    Returns:
        pd.DataFrame: Album dataframe with normalized genre flag columns.
    """
    genre_cols = [
        "ambient_experimental", "classical_orchestral", "electronic",
        "hip_hop_rnb", "pop", "rock", "world_folk"
    ]

    # Idempotent: bool/float/object -> int(0/1); NaN -> 0
    albums_df[genre_cols] = (
        albums_df[genre_cols]
        .fillna(False)  # NaN -> False
        .astype(bool)  # anything truthy -> True, False stays False
        .astype(int)  # True/False -> 1/0
    )

    return albums_df


def select_analysis_columns(albums_df: pd.DataFrame) -> pd.DataFrame:
    """
    Select the final set of columns for album-level analytics.

    Args:
        albums_df: Album dataframe containing engineered features.

    Returns:
        pd.DataFrame: Analysis-ready dataframe restricted to the final
        identifier, feature, derived award, and target columns used in
        downstream analysis.
    """
    selected_cols = (
            FILM_IDS
            + FILM_FEATURES
            + ALBUM_IDS
            + ALBUM_FEATURES
            + DERIVED_AWARD_COLS
            + Y_FEATURE
    )

    return albums_df[selected_cols]

def build_album_analytics(
        albums_df: pd.DataFrame,
        wide_df: pd.DataFrame
) -> pd.DataFrame:
    """Build the final analysis-ready album dataframe.

    This function runs the core album-level feature engineering pipeline by
    adding track counts, deriving award features, computing composer album
    counts, normalizing genre flags, and selecting the final analysis
    columns used downstream in correlation and regression workflows.

    Args:
        albums_df: Album-level source dataframe.
        wide_df: Wide-format source dataframe containing track-level rows.

    Returns:
        pd.DataFrame: Final analysis-ready album dataframe.
    """

    albums_df = add_track_counts(albums_df, wide_df)
    albums_df = add_award_features(albums_df)
    albums_df = add_composer_album_count(albums_df)
    albums_df = normalize_genre_flags(albums_df)
    album_analytics_df = select_analysis_columns(albums_df)

    return album_analytics_df


def inspect_genre_columns(albums_df: pd.DataFrame) -> None:
    """
    Print basic diagnostics for genre flag columns.

    Args:
        albums_df: Album-level dataframe.
    """
    genre_cols = [
        "ambient_experimental",
        "classical_orchestral",
        "electronic",
        "hip_hop_rnb",
        "pop",
        "rock",
        "world_folk",
    ]

    print("\nGenre column dtypes:")
    print(albums_df[genre_cols].dtypes)

    for col in genre_cols:
        print(f"\nColumn: {col}")
        print("Unique non-null values:")
        print(albums_df[col].dropna().unique()[:10])

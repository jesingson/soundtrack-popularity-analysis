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

def derive_award_category(df: pd.DataFrame) -> pd.Series:
    """
    Create a mutually exclusive award category per album.

    Ensures each album is assigned to exactly one category using a
    priority hierarchy:
    - Oscars first (most prestigious)
    - Winners before nominees
    - Score before Song
    """

    def assign(row):
        # Oscars - Score
        if row["oscar_score_winner"] == 1:
            return "Oscar Score Winner"
        if row["oscar_score_nominee"] == 1:
            return "Oscar Score Nominee"

        # Oscars - Song
        if row["oscar_song_winner"] == 1:
            return "Oscar Song Winner"
        if row["oscar_song_nominee"] == 1:
            return "Oscar Song Nominee"

        # BAFTA - Score
        if row["bafta_score_winner"] == 1:
            return "BAFTA Score Winner"
        if row["bafta_score_nominee"] == 1:
            return "BAFTA Score Nominee"

        # Golden Globes - Score
        if row["globes_score_winner"] == 1:
            return "Golden Globe Score Winner"
        if row["globes_score_nominee"] == 1:
            return "Golden Globe Score Nominee"

        # Golden Globes - Song
        if row["globes_song_winner"] == 1:
            return "Golden Globe Song Winner"
        if row["globes_song_nominee"] == 1:
            return "Golden Globe Song Nominee"

        # Critics - Score
        if row["critics_score_winner"] == 1:
            return "Critics Score Winner"
        if row["critics_score_nominee"] == 1:
            return "Critics Score Nominee"

        # Critics - Song
        if row["critics_song_winner"] == 1:
            return "Critics Song Winner"
        if row["critics_song_nominee"] == 1:
            return "Critics Song Nominee"

        return "No Major Award"

    return df.apply(assign, axis=1)

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

def add_release_lag_days(
        albums_df: pd.DataFrame
) -> pd.DataFrame:
    """Add the day difference between album and film release dates.

    The derived column ``album_release_lag_days`` measures how many days
    passed between the film release date and the album U.S. release date.

    Interpretation:
    - negative: album released before the film
    - zero: same-day release
    - positive: album released after the film

    Args:
        albums_df: Album-level dataframe containing release date columns.

    Returns:
        pd.DataFrame: Album dataframe with ``album_release_lag_days`` added.
    """
    albums_df = albums_df.copy()

    film_dates = pd.to_datetime(
        albums_df["film_release_date"],
        errors="coerce",
    )
    album_dates = pd.to_datetime(
        albums_df["album_us_release_date"],
        errors="coerce",
    )

    albums_df["album_release_lag_days"] = (
        album_dates - film_dates
    ).dt.days

    return albums_df

def add_album_genres_display(
        albums_df: pd.DataFrame
) -> pd.DataFrame:
    """Build a human-readable album genre string from canonical genre flags.

    This creates a pipe-delimited display column from the seven canonical
    album genre indicator columns used in the analysis workflow.

    Args:
        albums_df: Album-level dataframe containing canonical genre flags.

    Returns:
        pd.DataFrame: Album dataframe with ``album_genres_display`` added.
    """
    albums_df = albums_df.copy()

    genre_label_map = {
        "ambient_experimental": "Ambient/Experimental",
        "classical_orchestral": "Classical/Orchestral",
        "electronic": "Electronic",
        "hip_hop_rnb": "Hip-Hop/R&B",
        "pop": "Pop",
        "rock": "Rock",
        "world_folk": "World/Folk",
    }

    genre_cols = list(genre_label_map.keys())

    for col in genre_cols:
        if col not in albums_df.columns:
            albums_df[col] = 0

    def build_genre_string(row: pd.Series) -> str:
        labels = [
            genre_label_map[col]
            for col in genre_cols
            if pd.notna(row[col]) and int(row[col]) == 1
        ]
        return "|".join(labels)

    albums_df["album_genres_display"] = albums_df.apply(
        build_genre_string,
        axis=1,
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

def build_album_explorer_dataset(
        albums_df: pd.DataFrame,
        wide_df: pd.DataFrame
) -> pd.DataFrame:
    """Build a rich album-level dataframe for exploratory app pages.

    This dataset is intended for user-facing exploration pages such as the
    Dataset Explorer, Distribution Explorer, Group Comparison Explorer,
    Relationship Explorer, and Concentration Explorer. It keeps the
    descriptive album metadata from the source CSV while adding a small
    number of intuitive engineered features that are useful for browsing,
    filtering, sorting, and group-level comparison.

    Unlike ``build_album_analytics()``, this function does not reduce the
    dataframe to a narrow modeling feature set.

    Args:
        albums_df: Album-level source dataframe.
        wide_df: Wide-format source dataframe containing track-level rows.

    Returns:
        pd.DataFrame: Enriched album-level exploration dataframe.
    """
    albums_df = add_track_counts(albums_df, wide_df)
    albums_df = add_award_features(albums_df)
    albums_df["award_category"] = derive_award_category(albums_df)
    albums_df = add_composer_album_count(albums_df)
    albums_df = add_release_lag_days(albums_df)
    albums_df = normalize_genre_flags(albums_df)
    albums_df = add_album_genres_display(albums_df)

    return albums_df

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

def _get_track_album_metadata(
        albums_df: pd.DataFrame,
        wide_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Build the album-level metadata slice needed for the track explorer.

    This helper reuses the existing album exploration enrichment pipeline so
    the track explorer inherits the same intuitive browsing and filtering
    fields already used elsewhere in the app.

    Args:
        albums_df: Album-level source dataframe.
        wide_df: Wide-format source dataframe containing track-level rows.

    Returns:
        pd.DataFrame: Album-level metadata restricted to the columns needed
        by the track explorer.
    """
    album_explorer_df = build_album_explorer_dataset(albums_df, wide_df).copy()

    metadata_cols = [
        "tmdb_id",
        "release_group_mbid",
        "album_title",
        "film_title",
        "composer_primary_clean",
        "label_names",
        "film_year",
        "film_genres",
        "album_genres_display",
        "award_category",
        "n_tracks",
        "album_release_lag_days",
        "composer_album_count",
        "ambient_experimental",
        "classical_orchestral",
        "electronic",
        "hip_hop_rnb",
        "pop",
        "rock",
        "world_folk",
    ]

    existing_cols = [
        col for col in metadata_cols
        if col in album_explorer_df.columns
    ]

    return album_explorer_df[existing_cols].copy()


def _prepare_track_base(
        wide_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Build a cleaned one-row-per-track base dataframe from the wide dataset.

    The goal is to keep the track explorer at true track grain while
    standardizing core track fields needed for later positional charts and
    album-level cohesion rollups.

    Args:
        wide_df: Wide-format source dataframe containing track-level rows.

    Returns:
        pd.DataFrame: Cleaned track-level dataframe with one row per track.
    """
    track_cols = [
        "tmdb_id",
        "release_group_mbid",
        "track_id",
        "track_number",
        "track_title",
        "lfm_track_listeners",
        "lfm_track_playcount",
        "spotify_popularity",
        "log_lfm_track_listeners",
        "log_lfm_track_playcount",
    ]

    existing_cols = [col for col in track_cols if col in wide_df.columns]
    track_df = wide_df[existing_cols].copy()

    track_df["track_number"] = pd.to_numeric(
        track_df["track_number"],
        errors="coerce",
    )

    track_df = track_df.dropna(
        subset=["tmdb_id", "release_group_mbid", "track_number"]
    ).copy()

    track_df = track_df[track_df["track_number"] >= 1].copy()
    track_df["track_number"] = track_df["track_number"].astype(int)

    if "track_title" in track_df.columns:
        track_df["track_title"] = (
            track_df["track_title"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    dedupe_key = [
        "tmdb_id",
        "release_group_mbid",
        "track_number",
    ]

    sort_cols = dedupe_key.copy()
    if "lfm_track_listeners" in track_df.columns:
        sort_cols.append("lfm_track_listeners")

    ascending = [True, True, True]
    if "lfm_track_listeners" in track_df.columns:
        ascending.append(False)

    track_df = (
        track_df.sort_values(sort_cols, ascending=ascending)
        .drop_duplicates(subset=dedupe_key, keep="first")
        .copy()
    )

    return track_df


def _add_track_structure_fields(
        track_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Add reusable track-structure helper columns.

    Args:
        track_df: Cleaned track-level dataframe with observed track counts.

    Returns:
        pd.DataFrame: Track dataframe with helper columns added.
    """
    track_df = track_df.copy()

    track_df["is_first_track"] = track_df["track_number"] == 1
    track_df["is_last_track"] = (
        track_df["track_number"] == track_df["max_track_number_observed"]
    )

    track_df["track_position_pct"] = 0.0

    valid_denominator = track_df["max_track_number_observed"] > 1
    track_df.loc[valid_denominator, "track_position_pct"] = (
        (track_df.loc[valid_denominator, "track_number"] - 1)
        / (track_df.loc[valid_denominator, "max_track_number_observed"] - 1)
    )

    return track_df


def build_track_explorer_dataset(
        albums_df: pd.DataFrame,
        wide_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Build a rich track-level dataframe for the Track Structure Explorer.

    This dataset keeps one row per track while attaching the album metadata
    needed for shared filtering, grouping, display, and tooltips. It is the
    base dataframe for Page 6 and is intentionally exploratory rather than
    modeling-focused.

    Args:
        albums_df: Album-level source dataframe.
        wide_df: Wide-format source dataframe containing track-level rows.

    Returns:
        pd.DataFrame: Enriched track-level exploration dataframe.
    """
    album_metadata_df = _get_track_album_metadata(albums_df, wide_df)
    track_df = _prepare_track_base(wide_df)

    observed_counts_df = (
        track_df.groupby(
            ["release_group_mbid", "tmdb_id"],
            as_index=False,
        )
        .agg(
            track_count_observed=("track_number", "nunique"),
            max_track_number_observed=("track_number", "max"),
        )
    )

    track_df = track_df.merge(
        observed_counts_df,
        on=["release_group_mbid", "tmdb_id"],
        how="left",
        validate="m:1",
    )

    track_df = track_df.merge(
        album_metadata_df,
        on=["release_group_mbid", "tmdb_id"],
        how="left",
        validate="m:1",
    )

    track_df = _add_track_structure_fields(track_df)

    sort_cols = [
        col for col in [
            "film_title",
            "album_title",
            "release_group_mbid",
            "tmdb_id",
            "track_number",
        ]
        if col in track_df.columns
    ]

    if sort_cols:
        track_df = track_df.sort_values(sort_cols).reset_index(drop=True)

    return track_df

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

"""Functions for loading and preparing soundtrack analysis data.

This module contains reusable helper functions for reading the source
datasets, engineering album-level features, normalizing analysis fields,
and selecting the final columns used in the soundtrack analysis workflow.
These helpers keep data preparation logic separate from the command-line
entry point so the code is easier to test, reuse, and maintain.
"""
import re
import unicodedata
import numpy as np
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
    "album_cohesion_score",
    "album_cohesion_has_audio_data",
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

LABEL_CANONICAL_MAP = {
    "editions milan music": "Milan",
    "milan entertainment inc": "Milan",
    "decca classics": "Decca Records",
    "filmtrax ltd": "Filmtrax",
    "netflix": "Netflix Music",
    "invada": "Invada Records",
    "virgin music": "Virgin Records",
    "emi": "EMI Records",
    "paramount music corporation": "Paramount Music",
    "universal music": "Universal",
    "universal music classics": "Universal",
    "universal records": "Universal",
}



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

def _normalize_label_key(label: str) -> str:
    """
    Convert a raw label string into a normalized lookup key.

    Args:
        label: Raw label name.

    Returns:
        str: Normalized key used for canonical mapping.
    """
    if not isinstance(label, str):
        return ""

    label = unicodedata.normalize("NFKD", label)
    label = label.encode("ascii", "ignore").decode("ascii")
    label = label.strip().lower()

    label = label.replace("‐", "-").replace("–", "-").replace("—", "-")
    label = label.replace("&", "and")

    # Remove punctuation but keep spaces.
    label = re.sub(r"[^a-z0-9\s-]", " ", label)
    label = re.sub(r"\s+", " ", label).strip()

    # Light legal-suffix cleanup.
    label = re.sub(r"\binc\b$", "", label).strip()
    label = re.sub(r"\bllc\b$", "", label).strip()
    label = re.sub(r"\bltd\b$", "", label).strip()
    label = re.sub(r"\s+", " ", label).strip()

    return label


def canonicalize_label_name(label: str) -> str:
    """
    Map a raw label name to a conservative canonical display label.

    Args:
        label: Raw label name.

    Returns:
        str: Canonicalized label name.
    """
    if not isinstance(label, str):
        return ""

    clean_label = label.strip()
    if not clean_label:
        return ""

    normalized_key = _normalize_label_key(clean_label)

    if normalized_key in LABEL_CANONICAL_MAP:
        return LABEL_CANONICAL_MAP[normalized_key]

    return clean_label


def normalize_label_names(label_value: object) -> str:
    """
    Normalize a raw multi-label field into a cleaned pipe-delimited string.

    Important:
    - split only on '|'
    - do not split on commas, because commas often appear inside company names

    Args:
        label_value: Raw label_names value.

    Returns:
        str: Canonicalized pipe-delimited label string.
    """
    if pd.isna(label_value):
        return ""

    raw_text = str(label_value).strip()
    if not raw_text:
        return ""

    raw_parts = [
        part.strip()
        for part in re.split(r"\s*\|\s*", raw_text)
        if part.strip()
    ]

    clean_parts = []
    seen = set()

    for part in raw_parts:
        canonical = canonicalize_label_name(part)
        if canonical and canonical not in seen:
            clean_parts.append(canonical)
            seen.add(canonical)

    return " | ".join(clean_parts)


def add_label_names_clean(
    albums_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Add a cleaned label field for exploratory filtering and relationship views.

    Args:
        albums_df: Album-level dataframe.

    Returns:
        pd.DataFrame: Album dataframe with cleaned label column added.
    """
    albums_df = albums_df.copy()

    if "label_names" not in albums_df.columns:
        albums_df["label_names_clean"] = ""
        return albums_df

    albums_df["label_names_clean"] = albums_df["label_names"].apply(
        normalize_label_names
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
    albums_df = _get_base_album_metadata(albums_df, wide_df)

    cohesion_band_df = build_album_cohesion_band_dataset(albums_df, wide_df)

    merge_cols = [
        col for col in [
            "release_group_mbid",
            "tmdb_id",
            "album_cohesion_score",
            "album_cohesion_band",
        ]
        if col in cohesion_band_df.columns
    ]

    if {"release_group_mbid", "tmdb_id", "album_cohesion_band"}.issubset(merge_cols):
        albums_df = albums_df.merge(
            cohesion_band_df[merge_cols].drop_duplicates(
                subset=["release_group_mbid", "tmdb_id"]
            ),
            on=["release_group_mbid", "tmdb_id"],
            how="left",
            validate="1:1",
        )

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
    albums_df = add_album_cohesion_analysis_features(albums_df, wide_df)

    album_analytics_df = select_analysis_columns(albums_df)
    return album_analytics_df

def _get_base_album_metadata(
    albums_df: pd.DataFrame,
    wide_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a minimal album-level metadata slice without invoking the full
    album explorer pipeline.

    This helper is safe to use inside downstream builders that must avoid
    recursive calls back into build_album_explorer_dataset(...).
    """
    base_df = albums_df.copy()

    base_df = add_track_counts(base_df, wide_df)
    base_df = add_award_features(base_df)
    base_df["award_category"] = derive_award_category(base_df)
    base_df = add_composer_album_count(base_df)
    base_df = add_release_lag_days(base_df)
    base_df = add_label_names_clean(base_df)
    base_df = normalize_genre_flags(base_df)
    base_df = add_album_genres_display(base_df)

    return base_df

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
    album_explorer_df = _get_base_album_metadata(albums_df, wide_df).copy()

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
    standardizing core track fields needed for later positional charts,
    album-level cohesion rollups, and audio-feature analysis.

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
        # RapidAPI audio fields
        "key",
        "mode",
        "camelot",
        "tempo",
        "duration",
        "popularity",
        "energy",
        "danceability",
        "happiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "speechiness",
        "loudness",
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

    # Existing normalized 0-1 position measure.
    track_df["track_position_pct"] = 0.0
    valid_denominator = track_df["max_track_number_observed"] > 1
    track_df.loc[valid_denominator, "track_position_pct"] = (
        (track_df.loc[valid_denominator, "track_number"] - 1)
        / (track_df.loc[valid_denominator, "max_track_number_observed"] - 1)
    )

    # Alias with clearer semantics for later pages.
    track_df["relative_track_position"] = track_df["track_position_pct"]

    # Position from the end of the visible album.
    track_df["reverse_track_position"] = (
        track_df["max_track_number_observed"] - track_df["track_number"] + 1
    )

    # Simple reusable buckets for early-album structure.
    track_df["is_first_three_tracks"] = track_df["track_number"] <= 3
    track_df["is_first_five_tracks"] = track_df["track_number"] <= 5

    # Relative bucket labels are handy for later exploratory/stat pages too.
    def bucket_track_position(p: float) -> str | None:
        if pd.isna(p):
            return None
        if p <= 0.20:
            return "Opening"
        if p <= 0.40:
            return "Early"
        if p <= 0.60:
            return "Middle"
        if p <= 0.80:
            return "Late"
        return "Closing"

    track_df["track_position_bucket"] = track_df["relative_track_position"].apply(
        bucket_track_position
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

def build_track_data_explorer_dataset(
    albums_df: pd.DataFrame,
    wide_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a rich track-level dataframe for the Track Data Explorer.

    This dataset is the broad track-grain foundation for Page 30. It keeps
    one row per visible track, attaches album metadata for shared filtering,
    adds track-structure helper fields, and applies reusable audio cleaning
    so cleaned audio features are available directly in the page.

    Args:
        albums_df: Album-level source dataframe.
        wide_df: Wide-format source dataframe containing track-level rows.

    Returns:
        pd.DataFrame: Enriched track-level exploration dataframe.
    """
    album_metadata_df = _get_track_album_metadata(albums_df, wide_df).copy()
    track_df = _prepare_track_base(wide_df).copy()
    track_df = clean_track_audio_features(track_df)

    # Bring in album-level popularity fields for contextual filtering.
    album_metric_cols = [
        "release_group_mbid",
        "tmdb_id",
        "lfm_album_listeners",
        "lfm_album_playcount",
    ]
    available_album_metric_cols = [
        col for col in album_metric_cols
        if col in albums_df.columns or col in ["release_group_mbid", "tmdb_id"]
    ]

    if {"lfm_album_listeners", "lfm_album_playcount"} - set(album_metadata_df.columns):
        album_metric_df = albums_df[available_album_metric_cols].drop_duplicates()
        album_metadata_df = album_metadata_df.merge(
            album_metric_df,
            on=["release_group_mbid", "tmdb_id"],
            how="left",
            validate="1:1",
        )

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

    # Audio completeness helpers for profile cards / filtering.
    audio_feature_cols = [
        col for col in [
            "energy",
            "danceability",
            "happiness",
            "acousticness",
            "instrumentalness",
            "liveness",
            "speechiness",
            "tempo",
            "loudness",
            "duration_seconds",
            "key",
            "mode",
            "camelot_number",
            "camelot_mode",
        ]
        if col in track_df.columns
    ]

    if audio_feature_cols:
        track_df["has_any_audio_features"] = track_df[audio_feature_cols].notna().any(axis=1)
        track_df["audio_feature_count"] = track_df[audio_feature_cols].notna().sum(axis=1)
    else:
        track_df["has_any_audio_features"] = False
        track_df["audio_feature_count"] = 0

    if {"lfm_track_listeners", "lfm_album_listeners"}.issubset(track_df.columns):
        track_df["track_share_of_album_listeners"] = safe_ratio(
            track_df["lfm_track_listeners"],
            track_df["lfm_album_listeners"],
        )

    if {"lfm_track_playcount", "lfm_album_playcount"}.issubset(track_df.columns):
        track_df["track_share_of_album_playcount"] = safe_ratio(
            track_df["lfm_track_playcount"],
            track_df["lfm_album_playcount"],
        )

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

TRACK_AUDIO_FEATURE_COLS = [
    "spotify_popularity",
    "key",
    "mode",
    "tempo",
    "energy",
    "danceability",
    "acousticness",
    "instrumentalness",
    "liveness",
    "speechiness",
    "loudness",
]

BOUNDED_0_1_AUDIO_COLS = [
    "energy",
    "danceability",
    "acousticness",
    "instrumentalness",
    "liveness",
    "speechiness",
]

OPTIONAL_TRACK_AUDIO_COLS = [
    "valence",
]

TRACK_ARCHETYPE_SCORE_COLS = [
    "track_intensity_score",
    "track_acoustic_orchestral_score",
    "track_speech_texture_score",
]


def _parse_duration_to_seconds(value: object) -> float:
    """
    Parse a mixed-format duration field into seconds.

    Supports:
    - mm:ss strings such as '3:00'
    - numeric minute-like values less than 20, which are interpreted as minutes
    - numeric second-like values >= 20, which are treated as already being seconds

    Args:
        value: Raw duration value.

    Returns:
        float: Duration in seconds, or NaN when parsing fails.
    """
    if pd.isna(value):
        return np.nan

    if isinstance(value, str):
        text = value.strip()
        if ":" in text:
            parts = text.split(":")
            if len(parts) == 2:
                try:
                    minutes = float(parts[0])
                    seconds = float(parts[1])
                    return minutes * 60 + seconds
                except ValueError:
                    return np.nan
        try:
            numeric_val = float(text)
        except ValueError:
            return np.nan
    else:
        try:
            numeric_val = float(value)
        except (TypeError, ValueError):
            return np.nan

    if numeric_val < 0:
        return np.nan

    # Heuristic: small decimal values in this field are likely minutes.
    if numeric_val < 20:
        return numeric_val * 60

    return numeric_val


def _parse_loudness_db(value: object) -> float:
    """
    Parse loudness strings like '-23 dB' into numeric dB values.

    Args:
        value: Raw loudness value.

    Returns:
        float: Loudness as a float, or NaN when parsing fails.
    """
    if pd.isna(value):
        return np.nan

    text = str(value).strip().replace(" dB", "").replace("db", "")
    try:
        return float(text)
    except ValueError:
        return np.nan


def _parse_camelot_number(value: object) -> float:
    """
    Extract the numeric component from a Camelot code like '10B' or '8A'.

    Args:
        value: Raw Camelot value.

    Returns:
        float: Camelot number, or NaN when parsing fails.
    """
    if pd.isna(value):
        return np.nan

    text = str(value).strip().upper()
    digits = "".join(ch for ch in text if ch.isdigit())
    if not digits:
        return np.nan

    try:
        return float(digits)
    except ValueError:
        return np.nan


def _parse_camelot_mode(value: object) -> float:
    """
    Extract the mode component from a Camelot code.

    A -> minor -> 0
    B -> major -> 1

    Args:
        value: Raw Camelot value.

    Returns:
        float: 0 for minor, 1 for major, or NaN when parsing fails.
    """
    if pd.isna(value):
        return np.nan

    text = str(value).strip().upper()
    if text.endswith("A"):
        return 0.0
    if text.endswith("B"):
        return 1.0
    return np.nan

def _zscore_series(series: pd.Series) -> pd.Series:
    """
    Return a simple z-score transform, preserving NaN values.

    If the series has zero variance or no valid values, return all-NaN so
    downstream composite scores do not create fake signal.
    """
    numeric = pd.to_numeric(series, errors="coerce")
    std = numeric.std(skipna=True)

    if pd.isna(std) or std == 0:
        return pd.Series(np.nan, index=series.index, dtype="float64")

    mean = numeric.mean(skipna=True)
    return (numeric - mean) / std


def _mean_if_any(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """
    Row-wise mean across the provided columns, returning NaN when all inputs
    are missing on a row.
    """
    if not cols:
        return pd.Series(np.nan, index=df.index, dtype="float64")

    out = df[cols].mean(axis=1, skipna=True)
    return out.where(df[cols].notna().any(axis=1), np.nan)


def add_track_archetype_scores(
    track_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add reusable track-level archetype scores derived only from cleaned audio
    features.

    Archetypes:
    - track_intensity_score: energy + loudness
    - track_acoustic_orchestral_score: acousticness + instrumentalness
    - track_speech_texture_score: speechiness only

    Notes:
    - These are descriptive character dimensions, not outcome variables.
    - Inputs are z-scored first so mixed scales can be combined sensibly.
    """
    out = track_df.copy()

    energy_z = _zscore_series(out["energy"]) if "energy" in out.columns else None
    loudness_z = _zscore_series(out["loudness"]) if "loudness" in out.columns else None
    acousticness_z = (
        _zscore_series(out["acousticness"]) if "acousticness" in out.columns else None
    )
    instrumentalness_z = (
        _zscore_series(out["instrumentalness"])
        if "instrumentalness" in out.columns else None
    )
    speechiness_z = (
        _zscore_series(out["speechiness"]) if "speechiness" in out.columns else None
    )

    if energy_z is not None or loudness_z is not None:
        intensity_parts = pd.DataFrame(index=out.index)
        if energy_z is not None:
            intensity_parts["energy_z"] = energy_z
        if loudness_z is not None:
            intensity_parts["loudness_z"] = loudness_z

        out["track_intensity_score"] = _mean_if_any(
            intensity_parts,
            list(intensity_parts.columns),
        )
    else:
        out["track_intensity_score"] = np.nan

    if acousticness_z is not None or instrumentalness_z is not None:
        acoustic_parts = pd.DataFrame(index=out.index)
        if acousticness_z is not None:
            acoustic_parts["acousticness_z"] = acousticness_z
        if instrumentalness_z is not None:
            acoustic_parts["instrumentalness_z"] = instrumentalness_z

        out["track_acoustic_orchestral_score"] = _mean_if_any(
            acoustic_parts,
            list(acoustic_parts.columns),
        )
    else:
        out["track_acoustic_orchestral_score"] = np.nan

    if speechiness_z is not None:
        out["track_speech_texture_score"] = speechiness_z
    else:
        out["track_speech_texture_score"] = np.nan

    return out

def _band_from_zscore(value: float) -> str | None:
    """
    Convert a z-scored archetype dimension into a simple categorical band.

    Thresholds are intentionally conservative:
    - <= -0.5 -> Low
    - >= 0.5 -> High
    - otherwise -> Medium
    """
    if pd.isna(value):
        return None
    if value <= -0.5:
        return "Low"
    if value >= 0.5:
        return "High"
    return "Medium"


def add_track_archetype_bands(
    track_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add user-facing categorical archetype bands from the internal archetype scores.

    Creates:
    - track_intensity_band
    - track_acoustic_orchestral_band
    - track_speech_texture_band
    """
    out = track_df.copy()

    if "track_intensity_score" in out.columns:
        out["track_intensity_band"] = out["track_intensity_score"].apply(_band_from_zscore)
    else:
        out["track_intensity_band"] = None

    if "track_acoustic_orchestral_score" in out.columns:
        out["track_acoustic_orchestral_band"] = out[
            "track_acoustic_orchestral_score"
        ].apply(_band_from_zscore)
    else:
        out["track_acoustic_orchestral_band"] = None

    if "track_speech_texture_score" in out.columns:
        out["track_speech_texture_band"] = out[
            "track_speech_texture_score"
        ].apply(_band_from_zscore)
    else:
        out["track_speech_texture_band"] = None

    return out

def clean_track_audio_features(
    track_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Clean and standardize track-level RapidAPI audio feature columns.

    This dataset uses mixed formats:
    - note names for key (e.g. 'D', 'Bb', 'C#')
    - text labels for mode ('major', 'minor')
    - Camelot codes (e.g. '10B', '8A')
    - duration strings like '3:00'
    - loudness strings like '-23 dB'
    - bounded perceptual features stored on a 0–100 scale

    Cleaning rules:
    - convert key note names to integers in [0, 11]
    - convert mode to binary (major=1, minor=0)
    - keep camelot as a cleaned string and derive helper numeric fields
    - convert duration to duration_seconds
    - parse loudness text into numeric dB
    - scale bounded perceptual features from 0–100 to 0–1
    - coerce tempo and both popularity fields to numeric
    - add convenience flags for later pages

    Args:
        track_df: Track-level dataframe.

    Returns:
        pd.DataFrame: Copy of the dataframe with cleaned audio columns and
        convenience helper fields added.
    """
    track_df = track_df.copy()

    key_map = {
        "C": 0,
        "C#": 1,
        "DB": 1,
        "D": 2,
        "D#": 3,
        "EB": 3,
        "E": 4,
        "F": 5,
        "F#": 6,
        "GB": 6,
        "G": 7,
        "G#": 8,
        "AB": 8,
        "A": 9,
        "A#": 10,
        "BB": 10,
        "B": 11,
    }

    if "key" in track_df.columns:
        track_df["key_label"] = track_df["key"].astype("string")
        track_df["key"] = (
            track_df["key"]
            .astype("string")
            .str.strip()
            .str.upper()
            .map(key_map)
        )

    if "mode" in track_df.columns:
        track_df["mode_label"] = track_df["mode"].astype("string")
        track_df["mode"] = (
            track_df["mode"]
            .astype("string")
            .str.strip()
            .str.lower()
            .map({"major": 1.0, "minor": 0.0})
        )

    if "camelot" in track_df.columns:
        track_df["camelot"] = (
            track_df["camelot"].astype("string").str.strip().str.upper()
        )
        track_df["camelot_number"] = track_df["camelot"].apply(_parse_camelot_number)
        track_df["camelot_mode"] = track_df["camelot"].apply(_parse_camelot_mode)

    if "duration" in track_df.columns:
        track_df["duration_seconds"] = track_df["duration"].apply(
            _parse_duration_to_seconds
        )

    if "loudness" in track_df.columns:
        track_df["loudness"] = track_df["loudness"].apply(_parse_loudness_db)

    for col in ["tempo", "popularity", "spotify_popularity"]:
        if col in track_df.columns:
            track_df[col] = pd.to_numeric(track_df[col], errors="coerce")

    if "spotify_popularity" in track_df.columns:
        track_df["spotify_popularity"] = track_df["spotify_popularity"].where(
            track_df["spotify_popularity"].between(0, 100),
            np.nan,
        )

    if "popularity" in track_df.columns:
        track_df["popularity"] = track_df["popularity"].where(
            track_df["popularity"].between(0, 100),
            np.nan,
        )

    if "tempo" in track_df.columns:
        track_df["tempo"] = track_df["tempo"].where(
            track_df["tempo"] > 0,
            np.nan,
        )

    bounded_100_cols = [
        col for col in [
            "energy",
            "danceability",
            "happiness",
            "acousticness",
            "instrumentalness",
            "liveness",
            "speechiness",
        ]
        if col in track_df.columns
    ]

    for col in bounded_100_cols:
        track_df[col] = pd.to_numeric(track_df[col], errors="coerce")
        track_df[col] = (track_df[col] / 100.0).clip(lower=0, upper=1)

    if "instrumentalness" in track_df.columns:
        track_df["is_instrumental"] = (
            track_df["instrumentalness"] >= 0.7
        ).astype("float")

    if "energy" in track_df.columns:
        track_df["is_high_energy"] = (
            track_df["energy"] >= 0.6
        ).astype("float")

    if "happiness" in track_df.columns:
        track_df["is_high_happiness"] = (
            track_df["happiness"] >= 0.6
        ).astype("float")

    if "mode" in track_df.columns:
        track_df["is_major_mode"] = track_df["mode"]

    track_df = add_track_archetype_scores(track_df)
    track_df = add_track_archetype_bands(track_df)

    return track_df

def _entropy_from_series(series: pd.Series) -> float:
    """
    Compute Shannon entropy for a discrete series.

    Args:
        series: Discrete-valued series with possible nulls.

    Returns:
        float: Entropy in nats, or NaN when no non-null values are present.
    """
    values = series.dropna()
    if values.empty:
        return np.nan

    probs = values.value_counts(normalize=True)
    return float(-(probs * np.log(probs)).sum())

def safe_ratio(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    """
    Compute a safe ratio that returns NaN when the denominator is missing or <= 0.

    Args:
        numerator: Numerator series.
        denominator: Denominator series.

    Returns:
        pd.Series: Ratio with invalid divisions replaced by NaN.
    """
    out = numerator / denominator
    out = out.where(denominator > 0)
    return out.replace([np.inf, -np.inf], np.nan)

def _zscore_series(series: pd.Series) -> pd.Series:
    """
    Return a z-score transform while preserving NaN values.

    If the input has zero variance or no valid values, return all-NaN so the
    downstream composite does not create fake separation.
    """
    numeric = pd.to_numeric(series, errors="coerce")
    std = numeric.std(skipna=True)

    if pd.isna(std) or std == 0:
        return pd.Series(np.nan, index=series.index, dtype="float64")

    mean = numeric.mean(skipna=True)
    return (numeric - mean) / std


def _row_mean_if_any(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """
    Row-wise mean across selected columns, returning NaN when all inputs are missing.
    """
    if not cols:
        return pd.Series(np.nan, index=df.index, dtype="float64")

    out = df[cols].mean(axis=1, skipna=True)
    return out.where(df[cols].notna().any(axis=1), np.nan)


def add_album_cohesion_features(
    cohesion_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add a composite album cohesion score and user-facing categorical cohesion band.

    The score is based on a small set of core variability metrics:
    - energy_variance_proxy_std
    - danceability_variance_proxy_std
    - tempo_variance_proxy_std

    Higher variance means lower cohesion, so the averaged z-score is inverted:
    higher album_cohesion_score = more cohesive album.
    """
    out = cohesion_df.copy()

    component_cols = [
        col for col in [
            "energy_variance_proxy_std",
            "danceability_variance_proxy_std",
            "tempo_variance_proxy_std",
        ]
        if col in out.columns
    ]

    if not component_cols:
        out["album_cohesion_score"] = np.nan
        out["album_cohesion_band"] = None
        return out

    z_df = pd.DataFrame(index=out.index)
    for col in component_cols:
        z_df[f"{col}_z"] = _zscore_series(out[col])

    # Invert so higher score = more cohesive / less variable.
    out["album_cohesion_score"] = -1.0 * _row_mean_if_any(
        z_df,
        list(z_df.columns),
    )

    def assign_band(value: float) -> str:
        if pd.isna(value):
            return "Insufficient Audio Data"
        if value >= 0.5:
            return "Highly Cohesive"
        if value <= -0.5:
            return "Diverse / Varied"
        return "Moderately Cohesive"

    out["album_cohesion_band"] = out["album_cohesion_score"].apply(assign_band)
    return out

def _first_non_null(series: pd.Series):
    """
    Return the first non-null value in a series, if any.

    Args:
        series: Input series.

    Returns:
        object: First non-null value, or NaN if none exist.
    """
    non_null = series.dropna()
    if non_null.empty:
        return np.nan
    return non_null.iloc[0]


def _compute_top3_sum(series: pd.Series) -> float:
    """
    Return the sum of the top 3 non-null numeric values in a series.

    Args:
        series: Input series.

    Returns:
        float: Sum of the top 3 values, or NaN when no valid values exist.
    """
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if len(values) == 0:
        return np.nan

    values = np.sort(values)[::-1]
    return float(values[:3].sum())


def _assign_album_ratio_bucket(value: float) -> str | None:
    """
    Bucket a top-track / album ratio.

    Args:
        value: Ratio value.

    Returns:
        str | None: Bucket label or None when missing.
    """
    if pd.isna(value):
        return None
    if value < 1:
        return "<1x"
    if value < 2:
        return "1–2x"
    if value < 5:
        return "2–5x"
    if value < 10:
        return "5–10x"
    return "10x+"


def _assign_total_share_bucket(value: float) -> str | None:
    """
    Bucket a top-track share of total track performance.

    Args:
        value: Share value.

    Returns:
        str | None: Bucket label or None when missing.
    """
    if pd.isna(value):
        return None
    if value < 0.10:
        return "<10%"
    if value < 0.20:
        return "10–20%"
    if value < 0.35:
        return "20–35%"
    if value < 0.50:
        return "35–50%"
    return "50%+"


def build_album_cohesion_band_dataset(
    albums_df: pd.DataFrame,
    wide_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a minimal one-row-per-album dataframe containing only the composite
    album cohesion score and user-facing cohesion band.

    This helper is intentionally lightweight so album exploration pages can
    use cohesion categories without depending on the full Page 22 dataset.
    """
    track_df = _prepare_track_base(wide_df).copy()
    track_df = clean_track_audio_features(track_df)

    group_keys = ["release_group_mbid", "tmdb_id"]

    agg_spec: dict[str, object] = {}

    for col in [
        "energy",
        "danceability",
        "tempo",
    ]:
        if col in track_df.columns:
            agg_spec[col] = ["std"]

    if not agg_spec:
        return pd.DataFrame(columns=[
            "release_group_mbid",
            "tmdb_id",
            "album_cohesion_score",
            "album_cohesion_band",
        ])

    cohesion_df = track_df.groupby(group_keys, dropna=False).agg(agg_spec)
    cohesion_df.columns = [
        "_".join([part for part in col if part]).strip("_")
        if isinstance(col, tuple) else col
        for col in cohesion_df.columns.to_flat_index()
    ]
    cohesion_df = cohesion_df.reset_index()

    rename_map = {
        "energy_std": "energy_variance_proxy_std",
        "danceability_std": "danceability_variance_proxy_std",
        "tempo_std": "tempo_variance_proxy_std",
    }
    existing_renames = {
        old: new for old, new in rename_map.items()
        if old in cohesion_df.columns
    }
    cohesion_df = cohesion_df.rename(columns=existing_renames)

    cohesion_df = add_album_cohesion_features(cohesion_df)

    return cohesion_df[
        [
            col for col in [
                "release_group_mbid",
                "tmdb_id",
                "album_cohesion_score",
                "album_cohesion_band",
            ]
            if col in cohesion_df.columns
        ]
    ].copy()

def add_album_cohesion_analysis_features(
    albums_df: pd.DataFrame,
    wide_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge analysis-ready cohesion features into the album analytics dataframe.

    Adds:
    - album_cohesion_score: continuous cohesion signal
    - album_cohesion_has_audio_data: 1 when cohesion could be computed, else 0

    The categorical explorer label `album_cohesion_band` remains useful for
    exploratory pages, but the statistical/modeling pages should start from
    a cleaner numeric representation.
    """
    out = albums_df.copy()

    cohesion_df = build_album_cohesion_band_dataset(albums_df, wide_df)

    merge_cols = [
        col for col in [
            "release_group_mbid",
            "tmdb_id",
            "album_cohesion_score",
        ]
        if col in cohesion_df.columns
    ]

    if {"release_group_mbid", "tmdb_id", "album_cohesion_score"}.issubset(merge_cols):
        out = out.merge(
            cohesion_df[merge_cols].drop_duplicates(
                subset=["release_group_mbid", "tmdb_id"]
            ),
            on=["release_group_mbid", "tmdb_id"],
            how="left",
            validate="1:1",
        )
    else:
        out["album_cohesion_score"] = np.nan

    out["album_cohesion_has_audio_data"] = (
        out["album_cohesion_score"].notna().astype(int)
    )

    return out

def build_track_album_relationship_dataset(
    albums_df: pd.DataFrame,
    wide_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a one-row-per-album dataset for the Track–Album Relationship Explorer.

    This dataset is designed to align with the shared explorer architecture:
    it preserves the common metadata fields needed by global filters
    (`film_year`, `film_genres`, `album_genres_display`,
    `composer_primary_clean`, `label_names`) while also adding the
    track-vs-album aggregation and dominance fields used by Page 21.

    Args:
        albums_df: Album-level source dataframe.
        wide_df: Wide-format source dataframe containing track-level rows.

    Returns:
        pd.DataFrame: One-row-per-album relationship dataset.
    """
    album_metadata_df = _get_track_album_metadata(albums_df, wide_df).copy()
    track_df = _prepare_track_base(wide_df).copy()

    group_keys = ["release_group_mbid", "tmdb_id"]

    base_track_cols = [
        "release_group_mbid",
        "tmdb_id",
        "track_number",
        "track_title",
        "lfm_track_listeners",
        "lfm_track_playcount",
    ]
    track_df = track_df[[col for col in base_track_cols if col in track_df.columns]].copy()

    for col in ["lfm_track_listeners", "lfm_track_playcount"]:
        if col in track_df.columns:
            track_df[col] = pd.to_numeric(track_df[col], errors="coerce")

    agg_spec: dict[str, object] = {
        "track_title": _first_non_null,
        "track_number": "nunique",
    }

    if "lfm_track_listeners" in track_df.columns:
        agg_spec["lfm_track_listeners"] = ["max", "mean", "median", "sum", _compute_top3_sum]

    if "lfm_track_playcount" in track_df.columns:
        agg_spec["lfm_track_playcount"] = ["max", "mean", "median", "sum", _compute_top3_sum]

    grouped = track_df.groupby(group_keys, dropna=False).agg(agg_spec)
    grouped.columns = [
        "_".join([part for part in col if part]).strip("_")
        if isinstance(col, tuple) else col
        for col in grouped.columns.to_flat_index()
    ]
    grouped = grouped.reset_index()

    rename_map = {
        "track_title__first_non_null": "sample_track_title",
        "track_title__first_non_null_": "sample_track_title",
        "track_title__first_non_null__": "sample_track_title",
        "track_title__first_non_null____": "sample_track_title",
        "track_title__first_non_null_____": "sample_track_title",
        "track_title__first_non_null______": "sample_track_title",
        "track_title__first_non_null_______": "sample_track_title",
        "track_title__first_non_null________": "sample_track_title",
        "track_title__first_non_null_________": "sample_track_title",
        "track_title__first_non_null__________": "sample_track_title",
        "track_title__first_non_null___________": "sample_track_title",
        "track_title__first_non_null____________": "sample_track_title",
        "track_title__first_non_null_____________": "sample_track_title",
        "track_title__first_non_null______________": "sample_track_title",
        "track_title__first_non_null_______________": "sample_track_title",
        "track_title__first_non_null________________": "sample_track_title",
        "track_title__first_non_null_________________": "sample_track_title",
        "track_title__first_non_null__________________": "sample_track_title",
        "track_title__first_non_null___________________": "sample_track_title",
        "track_title__first_non_null____________________": "sample_track_title",
        "track_title__first_non_null_____________________": "sample_track_title",
        "track_title__first_non_null______________________": "sample_track_title",
        "track_title__first_non_null_______________________": "sample_track_title",
        "track_title__first_non_null________________________": "sample_track_title",
        "track_title__first_non_null_________________________": "sample_track_title",
        "track_title__first_non_null__________________________": "sample_track_title",
        "track_title__first_non_null___________________________": "sample_track_title",
        "track_title__first_non_null____________________________": "sample_track_title",
        "track_title__first_non_null_____________________________": "sample_track_title",
        "track_title__first_non_null______________________________": "sample_track_title",
        "track_title__first_non_null_______________________________": "sample_track_title",
        "track_number_nunique": "track_count_observed",
        "lfm_track_listeners_max": "track_max_listeners",
        "lfm_track_listeners_mean": "track_mean_listeners",
        "lfm_track_listeners_median": "track_median_listeners",
        "lfm_track_listeners_sum": "track_total_listeners",
        "lfm_track_listeners__compute_top3_sum": "track_top3_listeners",
        "lfm_track_playcount_max": "track_max_playcount",
        "lfm_track_playcount_mean": "track_mean_playcount",
        "lfm_track_playcount_median": "track_median_playcount",
        "lfm_track_playcount_sum": "track_total_playcount",
        "lfm_track_playcount__compute_top3_sum": "track_top3_playcount",
    }

    # More robust rename handling for pandas flattening differences
    actual_rename_map = {}
    for col in grouped.columns:
        if col == "track_title__first_non_null" or col == "track_title__first_non_null_":
            actual_rename_map[col] = "sample_track_title"
        elif col == "track_title_first_non_null":
            actual_rename_map[col] = "sample_track_title"
        elif col in rename_map:
            actual_rename_map[col] = rename_map[col]

    grouped = grouped.rename(columns=actual_rename_map)

    metric_cols_to_merge = [
        "release_group_mbid",
        "tmdb_id",
        "lfm_album_listeners",
        "lfm_album_playcount",
    ]
    metric_slice = [
        col for col in metric_cols_to_merge
        if col in albums_df.columns or col in group_keys
    ]

    if {"lfm_album_listeners", "lfm_album_playcount"} - set(album_metadata_df.columns):
        album_metric_df = albums_df[metric_slice].drop_duplicates()
        album_metadata_df = album_metadata_df.merge(
            album_metric_df,
            on=group_keys,
            how="left",
            validate="1:1",
        )

    grouped = grouped.merge(
        album_metadata_df,
        on=group_keys,
        how="left",
        validate="1:1",
    )

    # Reconcile track-count columns so the final dataset always exposes
    # the shared plain `n_tracks` field expected by Page 21 and other helpers.
    if "n_tracks_x" in grouped.columns or "n_tracks_y" in grouped.columns:
        if "n_tracks_y" in grouped.columns:
            grouped["n_tracks"] = grouped["n_tracks_y"]
        else:
            grouped["n_tracks"] = grouped["n_tracks_x"]

        if "n_tracks_x" in grouped.columns:
            grouped["n_tracks"] = grouped["n_tracks"].fillna(grouped["n_tracks_x"])

        grouped = grouped.drop(
            columns=[col for col in ["n_tracks_x", "n_tracks_y"] if col in grouped.columns]
        )

    if "n_tracks" not in grouped.columns and "track_count_observed" in grouped.columns:
        grouped["n_tracks"] = grouped["track_count_observed"]

    if "n_tracks" in grouped.columns and "track_count_observed" in grouped.columns:
        grouped["n_tracks"] = grouped["n_tracks"].fillna(grouped["track_count_observed"])
    if {"track_max_listeners", "lfm_album_listeners"}.issubset(grouped.columns):
        grouped["top_to_album_listeners"] = safe_ratio(
            grouped["track_max_listeners"],
            grouped["lfm_album_listeners"],
        )

    if {"track_max_listeners", "track_total_listeners"}.issubset(grouped.columns):
        grouped["top_to_total_listeners"] = safe_ratio(
            grouped["track_max_listeners"],
            grouped["track_total_listeners"],
        )

    if {"track_max_playcount", "lfm_album_playcount"}.issubset(grouped.columns):
        grouped["top_to_album_playcount"] = safe_ratio(
            grouped["track_max_playcount"],
            grouped["lfm_album_playcount"],
        )

    if {"track_max_playcount", "track_total_playcount"}.issubset(grouped.columns):
        grouped["top_to_total_playcount"] = safe_ratio(
            grouped["track_max_playcount"],
            grouped["track_total_playcount"],
        )

    if {"track_max_listeners", "track_mean_listeners"}.issubset(grouped.columns):
        grouped["top_to_mean_track_listeners"] = safe_ratio(
            grouped["track_max_listeners"],
            grouped["track_mean_listeners"],
        )

    if {"track_max_listeners", "track_median_listeners"}.issubset(grouped.columns):
        grouped["top_to_median_track_listeners"] = safe_ratio(
            grouped["track_max_listeners"],
            grouped["track_median_listeners"],
        )

    if {"track_max_playcount", "track_mean_playcount"}.issubset(grouped.columns):
        grouped["top_to_mean_track_playcount"] = safe_ratio(
            grouped["track_max_playcount"],
            grouped["track_mean_playcount"],
        )

    if {"track_max_playcount", "track_median_playcount"}.issubset(grouped.columns):
        grouped["top_to_median_track_playcount"] = safe_ratio(
            grouped["track_max_playcount"],
            grouped["track_median_playcount"],
        )

    if "top_to_album_listeners" in grouped.columns:
        grouped["dominance_bucket_top_to_album_listeners"] = grouped[
            "top_to_album_listeners"
        ].apply(_assign_album_ratio_bucket)

    if "top_to_total_listeners" in grouped.columns:
        grouped["dominance_bucket_top_to_total_listeners"] = grouped[
            "top_to_total_listeners"
        ].apply(_assign_total_share_bucket)

    if "top_to_album_playcount" in grouped.columns:
        grouped["dominance_bucket_top_to_album_playcount"] = grouped[
            "top_to_album_playcount"
        ].apply(_assign_album_ratio_bucket)

    if "top_to_total_playcount" in grouped.columns:
        grouped["dominance_bucket_top_to_total_playcount"] = grouped[
            "top_to_total_playcount"
        ].apply(_assign_total_share_bucket)

    # Attach top-track identity separately for listeners and playcount.
    identity_cols = [
        "release_group_mbid",
        "tmdb_id",
        "track_title",
        "track_number",
        "lfm_track_listeners",
        "lfm_track_playcount",
    ]
    identity_df = track_df[[col for col in identity_cols if col in track_df.columns]].copy()

    if {"lfm_track_listeners", "track_title", "track_number"}.issubset(identity_df.columns):
        top_listener_df = (
            identity_df.sort_values(
                ["release_group_mbid", "tmdb_id", "lfm_track_listeners", "track_number"],
                ascending=[True, True, False, True],
            )
            .drop_duplicates(subset=group_keys, keep="first")
            [[
                "release_group_mbid",
                "tmdb_id",
                "track_title",
                "track_number",
                "lfm_track_listeners",
            ]]
            .rename(columns={
                "track_title": "top_track_title_listeners",
                "track_number": "top_track_number_listeners",
                "lfm_track_listeners": "top_track_listeners_raw",
            })
        )

        grouped = grouped.merge(
            top_listener_df,
            on=group_keys,
            how="left",
            validate="1:1",
        )

    if {"lfm_track_playcount", "track_title", "track_number"}.issubset(identity_df.columns):
        top_playcount_df = (
            identity_df.sort_values(
                ["release_group_mbid", "tmdb_id", "lfm_track_playcount", "track_number"],
                ascending=[True, True, False, True],
            )
            .drop_duplicates(subset=group_keys, keep="first")
            [[
                "release_group_mbid",
                "tmdb_id",
                "track_title",
                "track_number",
                "lfm_track_playcount",
            ]]
            .rename(columns={
                "track_title": "top_track_title_playcount",
                "track_number": "top_track_number_playcount",
                "lfm_track_playcount": "top_track_playcount_raw",
            })
        )

        grouped = grouped.merge(
            top_playcount_df,
            on=group_keys,
            how="left",
            validate="1:1",
        )

    sort_cols = [
        col for col in ["film_title", "album_title", "release_group_mbid", "tmdb_id"]
        if col in grouped.columns
    ]
    if sort_cols:
        grouped = grouped.sort_values(sort_cols).reset_index(drop=True)

    return grouped

def build_track_audio_cohesion_dataset(
    albums_df: pd.DataFrame,
    wide_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a one-row-per-album dataset for track-audio cohesion analysis.

    This is intended for Page 22 and future track-audio pages. It joins
    track-level audio dispersion metrics back to a compact album-level frame
    that also contains album popularity and top-track dominance measures.

    Args:
        albums_df: Album-level source dataframe.
        wide_df: Wide-format source dataframe containing track-level rows.

    Returns:
        pd.DataFrame: Album-level dataframe with audio cohesion metrics.
    """
    album_metadata_df = _get_track_album_metadata(albums_df, wide_df).copy()
    track_df = _prepare_track_base(wide_df)
    track_df = clean_track_audio_features(track_df)

    audio_cols = [
        col for col in [
            "spotify_popularity",
            "popularity",
            "tempo",
            "duration_seconds",
            "energy",
            "danceability",
            "happiness",
            "acousticness",
            "instrumentalness",
            "liveness",
            "speechiness",
            "loudness",
            "key",
            "mode",
            "camelot_number",
            "camelot_mode",
            "is_instrumental",
            "is_high_energy",
            "is_high_happiness",
        ]
        if col in track_df.columns
    ]

    base_cols = [
        "release_group_mbid",
        "tmdb_id",
        "track_number",
        "track_title",
        "lfm_track_listeners",
        "lfm_track_playcount",
        *audio_cols,
    ]
    track_df = track_df[[col for col in base_cols if col in track_df.columns]].copy()

    group_keys = ["release_group_mbid", "tmdb_id"]

    agg_spec: dict[str, object] = {
        "lfm_track_listeners": ["max", "mean", "median", "sum"],
        "lfm_track_playcount": ["max", "mean", "median", "sum"],
    }

    for col in [
        "spotify_popularity",
        "popularity",
        "tempo",
        "duration_seconds",
        "energy",
        "danceability",
        "happiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "speechiness",
        "loudness",
    ]:
        if col in track_df.columns:
            agg_spec[col] = ["mean", "std", "min", "max"]

    if "is_instrumental" in track_df.columns:
        agg_spec["is_instrumental"] = ["mean"]

    if "is_high_energy" in track_df.columns:
        agg_spec["is_high_energy"] = ["mean"]

    if "is_high_happiness" in track_df.columns:
        agg_spec["is_high_happiness"] = ["mean"]

    cohesion_df = track_df.groupby(group_keys, dropna=False).agg(agg_spec)
    cohesion_df.columns = [
        "_".join([part for part in col if part]).strip("_")
        if isinstance(col, tuple) else col
        for col in cohesion_df.columns.to_flat_index()
    ]
    cohesion_df = cohesion_df.reset_index()

    if "key" in track_df.columns:
        key_entropy_df = (
            track_df.groupby(group_keys, dropna=False)["key"]
            .apply(_entropy_from_series)
            .reset_index(name="key_entropy")
        )
        cohesion_df = cohesion_df.merge(
            key_entropy_df,
            on=group_keys,
            how="left",
            validate="1:1",
        )

    if "mode" in track_df.columns:
        mode_entropy_df = (
            track_df.groupby(group_keys, dropna=False)["mode"]
            .apply(_entropy_from_series)
            .reset_index(name="mode_entropy")
        )
        cohesion_df = cohesion_df.merge(
            mode_entropy_df,
            on=group_keys,
            how="left",
            validate="1:1",
        )

    rename_map = {
        "lfm_track_listeners_max": "track_max_listeners",
        "lfm_track_listeners_mean": "track_mean_listeners",
        "lfm_track_listeners_median": "track_median_listeners",
        "lfm_track_listeners_sum": "track_total_listeners",
        "lfm_track_playcount_max": "track_max_playcount",
        "lfm_track_playcount_mean": "track_mean_playcount",
        "lfm_track_playcount_median": "track_median_playcount",
        "lfm_track_playcount_sum": "track_total_playcount",
        "tempo_std": "tempo_variance_proxy_std",
        "tempo_min": "tempo_min",
        "tempo_max": "tempo_max",
        "energy_std": "energy_variance_proxy_std",
        "danceability_std": "danceability_variance_proxy_std",
        "happiness_std": "happiness_variance_proxy_std",
        "acousticness_std": "acousticness_variance_proxy_std",
        "instrumentalness_std": "instrumentalness_variance_proxy_std",
        "liveness_std": "liveness_variance_proxy_std",
        "speechiness_std": "speechiness_variance_proxy_std",
        "loudness_std": "loudness_variance_proxy_std",
        "duration_seconds_std": "duration_variance_proxy_std",
        "is_instrumental_mean": "pct_instrumental_tracks",
        "is_high_energy_mean": "pct_high_energy_tracks",
        "is_high_happiness_mean": "pct_high_happiness_tracks",
    }
    existing_renames = {
        old: new for old, new in rename_map.items()
        if old in cohesion_df.columns
    }
    cohesion_df = cohesion_df.rename(columns=existing_renames)

    if "tempo_max" in cohesion_df.columns and "tempo_min" in cohesion_df.columns:
        cohesion_df["tempo_range"] = (
            cohesion_df["tempo_max"] - cohesion_df["tempo_min"]
        )

    if (
        ("lfm_album_listeners" not in album_metadata_df.columns)
        and ("lfm_album_listeners" in albums_df.columns)
    ):
        album_metric_slice = albums_df[
            ["release_group_mbid", "tmdb_id", "lfm_album_listeners", "lfm_album_playcount"]
        ].drop_duplicates()
        album_metadata_df = album_metadata_df.merge(
            album_metric_slice,
            on=group_keys,
            how="left",
            validate="1:1",
        )

    cohesion_df = cohesion_df.merge(
        album_metadata_df,
        on=group_keys,
        how="left",
        validate="1:1",
    )

    if "track_max_listeners" in cohesion_df.columns and "lfm_album_listeners" in cohesion_df.columns:
        cohesion_df["top_to_album_listeners"] = safe_ratio(
            cohesion_df["track_max_listeners"],
            cohesion_df["lfm_album_listeners"],
        )

    if "track_max_listeners" in cohesion_df.columns and "track_total_listeners" in cohesion_df.columns:
        cohesion_df["top_to_total_listeners"] = safe_ratio(
            cohesion_df["track_max_listeners"],
            cohesion_df["track_total_listeners"],
        )

    if "track_max_playcount" in cohesion_df.columns and "lfm_album_playcount" in cohesion_df.columns:
        cohesion_df["top_to_album_playcount"] = safe_ratio(
            cohesion_df["track_max_playcount"],
            cohesion_df["lfm_album_playcount"],
        )

    if "track_max_playcount" in cohesion_df.columns and "track_total_playcount" in cohesion_df.columns:
        cohesion_df["top_to_total_playcount"] = safe_ratio(
            cohesion_df["track_max_playcount"],
            cohesion_df["track_total_playcount"],
        )

    sort_cols = [
        col for col in ["film_title", "album_title", "release_group_mbid", "tmdb_id"]
        if col in cohesion_df.columns
    ]
    if sort_cols:
        cohesion_df = cohesion_df.sort_values(sort_cols).reset_index(drop=True)

    return cohesion_df

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

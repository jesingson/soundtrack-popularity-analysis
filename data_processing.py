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
    albums_df = add_track_counts(albums_df, wide_df)
    albums_df = add_award_features(albums_df)
    albums_df["award_category"] = derive_award_category(albums_df)
    albums_df = add_composer_album_count(albums_df)
    albums_df = add_release_lag_days(albums_df)
    albums_df = add_label_names_clean(albums_df)
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
    track_df["track_position_bucket"] = pd.cut(
        track_df["relative_track_position"],
        bins=[-0.001, 0.25, 0.50, 0.75, 1.001],
        labels=["Early", "Early-mid", "Late-mid", "Late"],
    ).astype(str)

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

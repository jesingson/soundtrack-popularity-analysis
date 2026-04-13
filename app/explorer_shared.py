from __future__ import annotations

import numpy as np
import pandas as pd

from app.data_filters import split_multivalue_genres
from app.ui import rename_columns_for_display, get_display_label
from data_processing import (
    TRACK_CONTEXT_CONTINUOUS_COLS,
    TRACK_CONTEXT_BINARY_COLS,
    TRACK_CONTEXT_CATEGORICAL_COLS,
    TRACK_CONTEXT_DERIVED_AWARD_COLS,
)

TRACK_NUMERIC_PRIORITY = [
    "lfm_track_listeners",
    "lfm_track_playcount",
    "track_share_of_album_listeners",
    "track_share_of_album_playcount",
    "spotify_popularity",
    "track_number",
    "relative_track_position",
    "energy",
    "danceability",
    "happiness",
    "instrumentalness",
    "acousticness",
    "liveness",
    "speechiness",
    "tempo",
    "loudness",
    "duration_seconds",
]


TRACK_GROUP_PRIORITY = [
    "track_position_bucket",
    "track_type",
    "track_intensity_band",
    "track_acoustic_orchestral_band",
    "track_speech_texture_band",
    "key_mode_label",
    "mode_label",
    "key_label",
    "album_genre_group",
    "film_genre_group",
    "award_category",
    "film_year_bucket",
    "composer_primary_clean",
]

TRACK_CONTEXT_NUMERIC_PRIORITY = [
    "film_year",
    "film_vote_count",
    "film_popularity",
    "film_budget",
    "film_revenue",
    "film_rating",
    "film_runtime_min",
    "days_since_film_release",
    "n_tracks",
    "album_release_lag_days",
    "composer_album_count",
    "album_cohesion_score",
]

TRACK_CONTEXT_GROUP_PRIORITY = [
    "film_genre_group",
    "album_genre_group",
    "film_year_bucket",
    "award_category",
    "film_is_action",
    "film_is_adventure",
    "film_is_animation",
    "film_is_comedy",
    "film_is_crime",
    "film_is_documentary",
    "film_is_drama",
    "film_is_family",
    "film_is_fantasy",
    "film_is_history",
    "film_is_horror",
    "film_is_music",
    "film_is_mystery",
    "film_is_romance",
    "film_is_science_fiction",
    "film_is_tv_movie",
    "film_is_thriller",
    "film_is_war",
    "film_is_western",
    "ambient_experimental",
    "classical_orchestral",
    "electronic",
    "hip_hop_rnb",
    "pop",
    "rock",
    "world_folk",
    "bafta_nominee",
]

TRACK_LEVEL_NUMERIC_COLS = {
    "lfm_track_listeners",
    "lfm_track_playcount",
    "spotify_popularity",
    "log_lfm_track_listeners",
    "log_lfm_track_playcount",
    "track_number",
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
    "track_intensity_score",
    "track_acoustic_orchestral_score",
    "track_speech_texture_score",
    "track_count_observed",
    "max_track_number_observed",
    "track_position_pct",
    "relative_track_position",
    "reverse_track_position",
    "audio_feature_count",
    "track_share_of_album_listeners",
    "track_share_of_album_playcount",
}

TRACK_LEVEL_GROUP_COLS = {
    "track_position_bucket",
    "track_intensity_band",
    "track_acoustic_orchestral_band",
    "track_speech_texture_band",
    "is_first_track",
    "is_last_track",
    "is_first_three_tracks",
    "is_first_five_tracks",
    "is_instrumental",
    "is_high_energy",
    "is_high_happiness",
    "is_major_mode",
    "has_any_audio_features",
    "key_label",
    "mode_label",
    "camelot",
}


def is_track_context_column(col: str) -> bool:
    """
    Return True when a column is album-/film-level context attached to the
    track-grain dataset rather than a native track-level field.
    """
    return col in (
        set(TRACK_CONTEXT_CONTINUOUS_COLS)
        | set(TRACK_CONTEXT_BINARY_COLS)
        | set(TRACK_CONTEXT_CATEGORICAL_COLS)
        | set(TRACK_CONTEXT_DERIVED_AWARD_COLS)
        | {
            "album_title",
            "film_title",
            "composer_primary_clean",
            "label_names",
        }
    )


def get_track_page_display_label(col: str) -> str:
    """
    Return a track-page-specific display label that visually separates
    track-level features from attached album/film context.
    """
    base_label = get_display_label(col)

    if col in TRACK_LEVEL_NUMERIC_COLS or col in TRACK_LEVEL_GROUP_COLS:
        return f"Track · {base_label}"

    if is_track_context_column(col):
        if col.startswith("film_") or col in {
            "film_year",
            "film_genres",
            "film_title",
            "days_since_film_release",
            "film_vote_count",
            "film_popularity",
            "film_budget",
            "film_revenue",
            "film_rating",
            "film_runtime_min",
        }:
            return f"Film · {base_label}"

        return f"Album · {base_label}"

    return base_label


def rename_track_page_columns_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename dataframe columns using track-page-specific labels so track-level
    and album/film-level fields are visually segregated in tables.
    """
    rename_map = {
        col: get_track_page_display_label(col)
        for col in df.columns
    }
    return df.rename(columns=rename_map)

def get_track_numeric_options(
    df: pd.DataFrame,
    include_context_features: bool = False,
) -> list[str]:
    """
    Return track-level numeric options in preferred order, with optional
    film/album context metrics appended after the native track metrics.
    """
    base_cols = [col for col in TRACK_NUMERIC_PRIORITY if col in df.columns]

    if not include_context_features:
        return base_cols

    context_cols = [
        col for col in TRACK_CONTEXT_NUMERIC_PRIORITY
        if col in df.columns and col not in base_cols
    ]
    return base_cols + context_cols


def get_track_group_options(
    df: pd.DataFrame,
    include_context_features: bool = False,
) -> list[str]:
    """
    Return track-level grouping options in preferred order, with optional
    film/album grouping fields appended after native track groupings.
    """
    base_cols = [col for col in TRACK_GROUP_PRIORITY if col in df.columns]

    if not include_context_features:
        return base_cols

    context_cols = [
        col for col in TRACK_CONTEXT_GROUP_PRIORITY
        if col in df.columns and col not in base_cols
    ]
    return base_cols + context_cols

def derive_multi_label_group_from_flags(
    df: pd.DataFrame,
    flag_cols: list[str],
    label_map: dict[str, str],
    output_col: str,
) -> pd.DataFrame:
    """
    Collapse multiple binary genre flags into a single grouped label.

    Rules:
    - one active flag -> that label
    - multiple active flags -> "Multi-genre"
    - no active flags -> "Unknown"

    Args:
        df: Input dataframe.
        flag_cols: Binary flag columns to inspect.
        label_map: Mapping from raw flag column to display label.
        output_col: Output grouped column name.

    Returns:
        pd.DataFrame: Copy with grouped output column added.
    """
    out_df = df.copy()
    available_cols = [col for col in flag_cols if col in out_df.columns]

    if not available_cols:
        out_df[output_col] = "Unknown"
        return out_df

    def assign_group(row: pd.Series) -> str:
        active_cols = [col for col in available_cols if row[col] == 1]
        if len(active_cols) == 1:
            return label_map[active_cols[0]]
        if len(active_cols) > 1:
            return "Multi-genre"
        return "Unknown"

    out_df[output_col] = out_df[available_cols].apply(assign_group, axis=1)
    return out_df


def derive_single_value_group_from_multivalue(
    value: object,
) -> str:
    """
    Collapse a multivalue text field into one grouped display label.

    Rules:
    - one value -> that value
    - multiple values -> "Multi-genre"
    - no values -> "Unknown"

    Args:
        value: Raw multivalue field.

    Returns:
        str: Grouped label.
    """
    values = split_multivalue_genres(pd.Series([value]))
    if len(values) == 1:
        return values[0]
    if len(values) > 1:
        return "Multi-genre"
    return "Unknown"


def add_standard_multivalue_groups(
    df: pd.DataFrame,
    album_col: str = "album_genres_display",
    film_col: str = "film_genres",
    album_output_col: str = "album_genre_group",
    film_output_col: str = "film_genre_group",
) -> pd.DataFrame:
    """
    Add single-value grouped genre fields from multivalue text columns.

    Args:
        df: Input dataframe.
        album_col: Raw album multivalue genre column.
        film_col: Raw film multivalue genre column.
        album_output_col: Output album grouped genre column.
        film_output_col: Output film grouped genre column.

    Returns:
        pd.DataFrame: Copy with grouped genre fields added.
    """
    out_df = df.copy()

    if album_col in out_df.columns:
        out_df[album_output_col] = out_df[album_col].apply(
            derive_single_value_group_from_multivalue
        )
    else:
        out_df[album_output_col] = "Unknown"

    if film_col in out_df.columns:
        out_df[film_output_col] = out_df[film_col].apply(
            derive_single_value_group_from_multivalue
        )
    else:
        out_df[film_output_col] = "Unknown"

    return out_df


def add_film_year_bucket(
    df: pd.DataFrame,
    film_year_col: str = "film_year",
    output_col: str = "film_year_bucket",
) -> pd.DataFrame:
    """
    Add a categorical year bucket used for cleaner legends and grouping.

    Args:
        df: Input dataframe.
        film_year_col: Raw film year column.
        output_col: Output year-bucket column.

    Returns:
        pd.DataFrame: Copy with film year bucket added.
    """
    out_df = df.copy()

    if film_year_col not in out_df.columns:
        out_df[output_col] = "Unknown"
        return out_df

    out_df[film_year_col] = pd.to_numeric(out_df[film_year_col], errors="coerce")

    def bucket_year(val: float) -> str:
        if pd.isna(val):
            return "Unknown"
        year = int(val)
        if year <= 2017:
            return "2015–2017"
        if year <= 2020:
            return "2018–2020"
        if year <= 2023:
            return "2021–2023"
        return "2024–2025"

    out_df[output_col] = out_df[film_year_col].apply(bucket_year)
    return out_df

def add_key_mode_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a human-readable harmonic label combining key and mode.

    Creates:
        - key_mode_label: e.g. 'C major', 'F minor'
    """
    out = df.copy()

    if "key_label" not in out.columns or "mode" not in out.columns:
        out["key_mode_label"] = "Unknown"
        return out

    mode_map = {
        1.0: "major",
        0.0: "minor",
        1: "major",
        0: "minor",
    }

    mode_series = (
        pd.to_numeric(out["mode"], errors="coerce")
        .map(mode_map)
        .fillna("unknown")
    )

    key_series = (
        out["key_label"]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
        .replace("", "Unknown")
    )

    out["key_mode_label"] = np.where(
        (key_series != "Unknown") & (mode_series != "unknown"),
        key_series + " " + mode_series,
        "Unknown",
    )

    return out

def get_global_filter_inputs(
    df: pd.DataFrame,
    film_year_col: str = "film_year",
    film_genres_col: str = "film_genres",
    album_genres_col: str = "album_genres_display",
) -> dict:
    """
    Build the common global-filter inputs used by multiple explorer pages.

    Args:
        df: Input dataframe.
        film_year_col: Film year column.
        film_genres_col: Multivalue film-genre column.
        album_genres_col: Multivalue album-genre column.

    Returns:
        dict: Filter input metadata.
    """
    year_series = pd.to_numeric(df[film_year_col], errors="coerce").dropna().astype(int)

    return {
        "min_year": int(year_series.min()),
        "max_year": int(year_series.max()),
        "film_genre_options": split_multivalue_genres(df[film_genres_col]),
        "album_genre_options": split_multivalue_genres(df[album_genres_col]),
    }


def get_clean_composer_options(
    df: pd.DataFrame,
    composer_col: str = "composer_primary_clean",
) -> list[str]:
    """
    Build a cleaned sorted composer option list for sidebar controls.

    Args:
        df: Input dataframe.
        composer_col: Composer column.

    Returns:
        list[str]: Clean composer names.
    """
    if composer_col not in df.columns:
        return []

    return sorted(
        df[composer_col]
        .dropna()
        .astype(str)
        .str.strip()
        .replace("", pd.NA)
        .dropna()
        .unique()
        .tolist()
    )


def select_unique_existing_columns(
    df: pd.DataFrame,
    preferred_cols: list[str],
) -> list[str]:
    """
    Keep only existing columns, preserving order and removing duplicates.

    Args:
        df: Input dataframe.
        preferred_cols: Candidate columns in desired order.

    Returns:
        list[str]: Existing unique columns.
    """
    selected = []
    seen = set()

    for col in preferred_cols:
        if col in df.columns and col not in seen:
            selected.append(col)
            seen.add(col)

    return selected


def rename_and_dedupe_for_display(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply display labels and dedupe resulting column names.

    Args:
        df: Input dataframe.

    Returns:
        pd.DataFrame: Display-ready dataframe with unique column names.
    """
    out_df = rename_columns_for_display(df.copy())

    deduped_cols = []
    seen = {}

    for col in out_df.columns:
        if col not in seen:
            seen[col] = 1
            deduped_cols.append(col)
        else:
            seen[col] += 1
            deduped_cols.append(f"{col} ({seen[col]})")

    out_df.columns = deduped_cols
    return out_df
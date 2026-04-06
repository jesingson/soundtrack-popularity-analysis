from __future__ import annotations

import pandas as pd

from app.data_filters import split_multivalue_genres
from app.ui import rename_columns_for_display


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
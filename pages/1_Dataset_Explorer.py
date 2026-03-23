from __future__ import annotations

import re

import pandas as pd
import streamlit as st

from app.app_controls import get_dataset_controls
from app.app_data import load_explorer_data
from app.data_filters import filter_dataset, split_multivalue_genres
from app.ui import (
    apply_app_styles,
    get_display_label,
    rename_columns_for_display,
)

DISPLAY_COLUMNS = [
    "album_title",
    "film_title",
    "film_release_date",
    "album_us_release_date",
    "album_release_lag_days",
    "composer_primary_clean",
    "label_names",
    "film_genres",
    "album_genres_display",
    "lfm_album_listeners",
    "lfm_album_playcount",
    "n_tracks",
    "oscar_score_nominee",
    "oscar_song_nominee",
    "globes_score_nominee",
    "globes_song_nominee",
    "critics_score_nominee",
    "critics_song_nominee",
    "bafta_score_nominee",
    "us_score_nominee_count",
    "us_song_nominee_count",
    "bafta_nominee",
    "composer_album_count",
    "film_year",
    "days_since_film_release",
    "days_since_album_release",
]

def split_multivalue_genres(series: pd.Series) -> list[str]:
    """Extract unique genre values from pipe- or comma-delimited strings."""
    values: set[str] = set()

    for value in series.dropna():
        parts = re.split(r"\s*\|\s*|\s*,\s*", str(value))
        for part in parts:
            part = part.strip()
            if part:
                values.add(part)

    return sorted(values)

def contains_any_genre(value: str, selected_genres: list[str]) -> bool:
    """Return True if a multi-value genre string contains any selected genre."""
    if pd.isna(value) or not selected_genres:
        return False

    parts = {
        part.strip().lower()
        for part in re.split(r"\s*\|\s*|\s*,\s*", str(value))
        if part.strip()
    }
    selected = {genre.strip().lower() for genre in selected_genres}

    return bool(parts & selected)

def apply_text_search(df: pd.DataFrame, search_text: str) -> pd.DataFrame:
    """Filter rows by case-insensitive text search across key fields."""
    if not search_text or not search_text.strip():
        return df

    search_text = search_text.strip().lower()

    search_cols = [
        "album_title",
        "film_title",
        "composer_primary_clean",
        "label_names",
    ]

    combined = pd.Series("", index=df.index)

    for col in search_cols:
        if col in df.columns:
            combined = combined + " " + df[col].fillna("").astype(str)

    mask = combined.str.lower().str.contains(
        re.escape(search_text),
        regex=True,
    )

    return df[mask]

def filter_dataset(df: pd.DataFrame, controls: dict) -> pd.DataFrame:
    """Apply Dataset Explorer filters to the album dataframe."""
    filtered = df.copy()

    year_min, year_max = controls["year_range"]
    filtered = filtered[
        filtered["film_year"].between(year_min, year_max, inclusive="both")
    ]

    filtered = filtered[filtered["n_tracks"].fillna(0) >= controls["min_tracks"]]

    if controls["listeners_only"]:
        filtered = filtered[filtered["lfm_album_listeners"].notna()]

    if controls["selected_composers"]:
        filtered = filtered[
            filtered["composer_primary_clean"].isin(
                controls["selected_composers"]
            )
        ]

    if controls["selected_labels"]:
        filtered = filtered[
            filtered["label_names"].fillna("").isin(controls["selected_labels"])
        ]

    if controls["selected_film_genres"]:
        filtered = filtered[
            filtered["film_genres"].apply(
                lambda value: contains_any_genre(
                    value,
                    controls["selected_film_genres"],
                )
            )
        ]

    if controls["selected_album_genres"]:
        filtered = filtered[
            filtered["album_genres_display"].apply(
                lambda value: contains_any_genre(
                    value,
                    controls["selected_album_genres"],
                )
            )
        ]

    filtered = apply_text_search(
        filtered,
        controls["search_text"],
    )

    return filtered


def get_safe_display_columns(df: pd.DataFrame) -> list[str]:
    """Return the subset of preferred display columns present in the dataframe."""
    return [col for col in DISPLAY_COLUMNS if col in df.columns]


def format_display_df(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Prepare the filtered dataframe for display."""
    display_df = df[columns].copy()
    display_df = rename_columns_for_display(display_df)
    display_df = format_date_columns(display_df)
    return display_df

def format_date_columns(display_df: pd.DataFrame) -> pd.DataFrame:
    """Format known date columns for cleaner table display."""
    formatted_df = display_df.copy()

    date_cols = [
        "Film Release Date",
        "Album Release Date",
    ]

    for col in date_cols:
        if col in formatted_df.columns:
            formatted_df[col] = pd.to_datetime(
                formatted_df[col],
                errors="coerce",
            ).dt.strftime("%Y-%m-%d")

    return formatted_df

def main() -> None:
    """Render the Dataset Explorer page."""
    apply_app_styles()
    st.title("Dataset Explorer")
    st.write(
        """
        Explore the album-level soundtrack dataset used throughout this app.
        Each row represents one soundtrack album.
        """
    )

    explorer_df = load_explorer_data()

    default_display_columns = get_safe_display_columns(explorer_df)

    year_series = explorer_df["film_year"].dropna().astype(int)
    min_year = int(year_series.min())
    max_year = int(year_series.max())

    film_genre_options = split_multivalue_genres(explorer_df["film_genres"])
    album_genre_options = split_multivalue_genres(explorer_df["album_genres_display"])
    composer_options = sorted(
        explorer_df["composer_primary_clean"].dropna().astype(str).unique().tolist()
    )
    label_options = sorted(
        explorer_df["label_names"].dropna().astype(str).unique().tolist()
    )

    controls = get_dataset_controls(
        min_year=min_year,
        max_year=max_year,
        film_genre_options=film_genre_options,
        album_genre_options=album_genre_options,
        composer_options=composer_options,
        label_options=label_options,
    )

    filtered_df = filter_dataset(explorer_df, controls)

    sort_col = "lfm_album_listeners"
    if sort_col in filtered_df.columns:
        filtered_df = filtered_df.sort_values(
            by=sort_col,
            ascending=False,
            na_position="last",
        )

    album_count = len(filtered_df)
    film_count = filtered_df["film_title"].nunique() if "film_title" in filtered_df.columns else 0
    composer_count = (
        filtered_df["composer_primary_clean"].nunique()
        if "composer_primary_clean" in filtered_df.columns else 0
    )
    label_count = (
        filtered_df["label_names"].nunique()
        if "label_names" in filtered_df.columns else 0
    )

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    metric_col1.metric("Albums", f"{album_count:,}")
    metric_col2.metric("Films", f"{film_count:,}")
    metric_col3.metric("Composers", f"{composer_count:,}")
    metric_col4.metric("Labels", f"{label_count:,}")

    st.markdown("### Table Columns")

    default_selected_columns = [
        col for col in [
            "album_title",
            "film_title",
            "film_release_date",
            "album_us_release_date",
            "album_release_lag_days",
            "composer_primary_clean",
            "label_names",
            "film_genres",
            "album_genres_display",
            "lfm_album_listeners",
            "lfm_album_playcount",
            "n_tracks",
            "oscar_score_nominee",
            "oscar_song_nominee",
            "globes_score_nominee",
            "globes_song_nominee",
            "critics_score_nominee",
            "critics_song_nominee",
            "bafta_score_nominee",
        ]
        if col in default_display_columns
    ]

    selected_columns = st.multiselect(
        "Choose columns to display",
        options=default_display_columns,
        default=default_selected_columns,
        format_func=get_display_label,
    )

    if not selected_columns:
        st.warning("Select at least one column to display.")
        return

    display_df = format_display_df(filtered_df, selected_columns)

    st.markdown(f"### Filtered Albums ({len(display_df):,} rows)")

    if controls["show_data_table"]:
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
        )

    csv_bytes = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download filtered dataset as CSV",
        data=csv_bytes,
        file_name="filtered_soundtrack_albums.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
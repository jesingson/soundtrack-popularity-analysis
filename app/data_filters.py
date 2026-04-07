from __future__ import annotations

import re

import pandas as pd


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
        "track_title",
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
    """
    Apply shared album-level filters.

    Supports both the full Dataset Explorer controls and the lightweight
    global-filter subset used by other pages.
    """
    filtered = df.copy()

    year_min, year_max = controls["year_range"]
    if "film_year" in filtered.columns:
        filtered = filtered[
            filtered["film_year"].between(year_min, year_max, inclusive="both")
        ]

    if "min_tracks" in controls and "n_tracks" in filtered.columns:
        filtered = filtered[
            filtered["n_tracks"].fillna(0) >= controls["min_tracks"]
        ]

    if controls.get("listeners_only", False) and "lfm_album_listeners" in filtered.columns:
        filtered = filtered[filtered["lfm_album_listeners"].notna()]

    if controls.get("selected_composers"):
        filtered = filtered[
            filtered["composer_primary_clean"].isin(controls["selected_composers"])
        ]

    if controls.get("selected_labels"):
        filtered = filtered[
            filtered["label_names"].fillna("").isin(controls["selected_labels"])
        ]

    if controls.get("selected_film_genres"):
        filtered = filtered[
            filtered["film_genres"].apply(
                lambda value: contains_any_genre(
                    value,
                    controls["selected_film_genres"],
                )
            )
        ]

    if controls.get("selected_album_genres"):
        filtered = filtered[
            filtered["album_genres_display"].apply(
                lambda value: contains_any_genre(
                    value,
                    controls["selected_album_genres"],
                )
            )
        ]

    if "search_text" in controls:
        filtered = apply_text_search(filtered, controls["search_text"])

    return filtered
from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from app.app_data import load_track_data_explorer_data
from app.app_controls import (
    get_global_filter_controls,
    get_track_dataset_explorer_controls,
)
from app.data_filters import filter_dataset
from app.explorer_shared import (
    add_film_year_bucket,
    add_standard_multivalue_groups,
    get_clean_composer_options,
    get_global_filter_inputs,
    rename_track_page_columns_for_display,
    select_unique_existing_columns,
    get_track_page_display_label,
)
from app.ui import apply_app_styles


DISPLAY_COLUMNS = [
    "film_title",
    "album_title",
    "track_title",
    "track_number",
    "track_position_bucket",
    "relative_track_position",
    "composer_primary_clean",
    "label_names",
    "film_year",
    "film_genres",
    "album_genres_display",
    "album_genre_group",
    "film_genre_group",
    "award_category",
    "film_vote_count",
    "film_popularity",
    "film_rating",
    "film_runtime_min",
    "days_since_film_release",
    "n_tracks",
    "album_release_lag_days",
    "composer_album_count",
    "album_cohesion_score",
    "bafta_nominee",
    "lfm_album_listeners",
    "lfm_album_playcount",
    "lfm_track_listeners",
    "lfm_track_playcount",
    "track_share_of_album_listeners",
    "track_share_of_album_playcount",
    "spotify_popularity",
    "track_count_observed",
    "max_track_number_observed",
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
    "is_instrumental",
    "is_high_energy",
    "is_high_happiness",
    "is_major_mode",
]

AUDIO_COVERAGE_COLS = [
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
    "spotify_popularity",
]


def add_track_data_display_fields(track_df: pd.DataFrame) -> pd.DataFrame:
    """Add grouped display fields used across the Track Dataset Explorer."""
    df = add_standard_multivalue_groups(track_df)
    df = add_film_year_bucket(df)

    if "composer_primary_clean" in df.columns:
        df["composer_primary_clean"] = (
            df["composer_primary_clean"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    if "label_names" in df.columns:
        df["label_names"] = (
            df["label_names"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    return df


def filter_track_dataset(df: pd.DataFrame, controls: dict) -> pd.DataFrame:
    """Apply shared filters plus Track Dataset Explorer-specific filters."""
    filtered = filter_dataset(df, controls).copy()

    if "track_number" in filtered.columns:
        filtered = filtered[
            filtered["track_number"] <= controls["max_track_position"]
        ].copy()

    if "lfm_album_listeners" in filtered.columns:
        filtered = filtered[
            filtered["lfm_album_listeners"].fillna(0) >= controls["min_album_listeners"]
        ].copy()

    if controls.get("audio_only", False):
        required_audio_cols = [
            col
            for col in ["energy", "danceability", "happiness", "instrumentalness"]
            if col in filtered.columns
        ]
        if required_audio_cols:
            filtered = filtered.dropna(subset=required_audio_cols).copy()

    return filtered


def build_filter_context_caption(controls: dict) -> str:
    """Build a short caption describing the current filter scope."""
    parts = []

    year_min, year_max = controls["year_range"]
    parts.append(f"Film years {year_min}–{year_max}")

    if controls.get("selected_film_genres"):
        shown = ", ".join(controls["selected_film_genres"][:5])
        if len(controls["selected_film_genres"]) > 5:
            shown += ", ..."
        parts.append(f"Film genres: {shown}")

    if controls.get("selected_album_genres"):
        shown = ", ".join(controls["selected_album_genres"][:5])
        if len(controls["selected_album_genres"]) > 5:
            shown += ", ..."
        parts.append(f"Album genres: {shown}")

    if controls.get("selected_composers"):
        shown = ", ".join(controls["selected_composers"][:3])
        if len(controls["selected_composers"]) > 3:
            shown += ", ..."
        parts.append(f"Composers: {shown}")

    if controls["min_album_listeners"] > 0:
        parts.append(f"Min album listeners: {controls['min_album_listeners']:,}")

    parts.append(f"Max track position: {controls['max_track_position']}")

    if controls["audio_only"]:
        parts.append("Only tracks with core audio features")

    return " | ".join(parts)


def build_audio_coverage_df(filtered_df: pd.DataFrame) -> pd.DataFrame:
    """Build coverage summary for track audio fields."""
    available_cols = [col for col in AUDIO_COVERAGE_COLS if col in filtered_df.columns]

    rows = []
    total_rows = len(filtered_df)

    for col in available_cols:
        non_null = int(filtered_df[col].notna().sum())
        rows.append(
            {
                "field": col,
                "field_label": get_track_page_display_label(col),
                "non_null_count": non_null,
                "coverage_share": (non_null / total_rows) if total_rows > 0 else 0.0,
            }
        )

    coverage_df = pd.DataFrame(rows)
    if not coverage_df.empty:
        coverage_df = coverage_df.sort_values(
            ["coverage_share", "field_label"],
            ascending=[False, True],
        ).reset_index(drop=True)

    return coverage_df


def create_audio_coverage_chart(coverage_df: pd.DataFrame) -> alt.Chart:
    """Create a bar chart of audio field coverage."""
    return (
        alt.Chart(coverage_df)
        .mark_bar()
        .encode(
            y=alt.Y("field_label:N", sort="-x", title="Field"),
            x=alt.X("coverage_share:Q", title="Coverage Share", axis=alt.Axis(format="%")),
            tooltip=[
                alt.Tooltip("field_label:N", title="Field"),
                alt.Tooltip("non_null_count:Q", title="Tracks", format=",.0f"),
                alt.Tooltip("coverage_share:Q", title="Coverage", format=".1%"),
            ],
        )
        .properties(
            height=max(320, min(28 * len(coverage_df), 520)),
            title="Audio Field Coverage",
        )
    )

def build_dataset_supporting_insight(
    filtered_df: pd.DataFrame,
    coverage_df: pd.DataFrame,
    audio_complete_share: float,
) -> str:
    """Build a short educational insight for the current dataset view."""
    if filtered_df.empty:
        return "No tracks remain under the current filters."

    track_count = len(filtered_df)

    if coverage_df.empty:
        return (
            f"💡 The current view contains {track_count:,} tracks, but no audio-field "
            "coverage summary is available."
        )

    top_row = coverage_df.iloc[0]
    bottom_row = coverage_df.iloc[-1]

    return (
        f"💡 The current view contains {track_count:,} tracks, and "
        f"{audio_complete_share:.1%} have at least one audio feature available. "
        f"{top_row['field_label']} is the most complete audio field "
        f"({top_row['coverage_share']:.1%} coverage), while {bottom_row['field_label']} "
        f"is the sparsest ({bottom_row['coverage_share']:.1%})."
    )

def get_safe_display_columns(df: pd.DataFrame) -> list[str]:
    """Return the subset of preferred display columns present in the dataframe."""
    return [col for col in DISPLAY_COLUMNS if col in df.columns]


def main() -> None:
    """Render the Track Dataset Explorer page."""
    st.set_page_config(
        page_title="Track Dataset Explorer",
        layout="wide",
    )
    apply_app_styles()

    st.title("Track Dataset Explorer")
    st.write(
        """
        Understand what is in the visible track-level dataset before moving into
        distribution, grouped comparison, or relationship analysis.
        """
    )

    track_df = load_track_data_explorer_data()
    track_df = add_track_data_display_fields(track_df)

    filter_inputs = get_global_filter_inputs(track_df)
    composer_options = get_clean_composer_options(track_df)

    global_controls = get_global_filter_controls(
        min_year=filter_inputs["min_year"],
        max_year=filter_inputs["max_year"],
        film_genre_options=filter_inputs["film_genre_options"],
        album_genre_options=filter_inputs["album_genre_options"],
    )

    local_controls = get_track_dataset_explorer_controls(
        composer_options=composer_options,
    )

    controls = {
        **global_controls,
        **local_controls,
    }

    filtered_df = filter_track_dataset(track_df, controls)

    if filtered_df.empty:
        st.warning("No tracks remain under the current filters.")
        return

    st.markdown("**Filter Context**")
    st.caption(build_filter_context_caption(controls))

    track_count = len(filtered_df)
    album_count = (
        filtered_df[["release_group_mbid", "tmdb_id"]]
        .drop_duplicates()
        .shape[0]
        if {"release_group_mbid", "tmdb_id"}.issubset(filtered_df.columns)
        else 0
    )
    film_count = (
        filtered_df["film_title"].nunique()
        if "film_title" in filtered_df.columns
        else 0
    )
    composer_count = (
        filtered_df["composer_primary_clean"]
        .replace("", pd.NA)
        .dropna()
        .nunique()
        if "composer_primary_clean" in filtered_df.columns
        else 0
    )
    audio_complete_share = (
        filtered_df["has_any_audio_features"].mean()
        if "has_any_audio_features" in filtered_df.columns
        else 0.0
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Tracks", f"{track_count:,}")
    c2.metric("Albums", f"{album_count:,}")
    c3.metric("Films", f"{film_count:,}")
    c4.metric("Composers", f"{composer_count:,}")
    c5.metric("Tracks with Any Audio", f"{audio_complete_share:.1%}")

    coverage_df = build_audio_coverage_df(filtered_df)

    st.caption(
        build_dataset_supporting_insight(
            filtered_df=filtered_df,
            coverage_df=coverage_df,
            audio_complete_share=audio_complete_share,
        )
    )

    st.markdown("### Filtered Tracks")
    st.caption(
        "These rows represent the currently visible source tracks after the active dataset filters."
    )

    if "lfm_track_listeners" in filtered_df.columns:
        filtered_df = filtered_df.sort_values(
            by="lfm_track_listeners",
            ascending=False,
            na_position="last",
        ).copy()

    default_display_columns = get_safe_display_columns(filtered_df)
    default_selected_columns = [
        col
        for col in [
            "film_title",
            "album_title",
            "track_title",
            "track_number",
            "track_position_bucket",
            "composer_primary_clean",
            "album_genres_display",
            "film_genres",
            "lfm_track_listeners",
            "lfm_track_playcount",
            "track_share_of_album_listeners",
            "energy",
            "danceability",
            "happiness",
            "instrumentalness",
            "tempo",
        ]
        if col in default_display_columns
    ]

    selected_columns = st.multiselect(
        "Choose columns to display",
        options=default_display_columns,
        default=default_selected_columns,
        format_func=get_track_page_display_label,
    )

    if not selected_columns:
        st.warning("Select at least one column to display.")
        return

    table_cols = select_unique_existing_columns(filtered_df, selected_columns)
    display_df = rename_track_page_columns_for_display(filtered_df[table_cols].copy())

    if controls["show_data_table"]:
        st.dataframe(
            display_df,
            width="stretch",
            hide_index=True,
        )

    csv_bytes = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download filtered track dataset as CSV",
        data=csv_bytes,
        file_name="filtered_soundtrack_tracks.csv",
        mime="text/csv",
    )

    st.markdown("### Field Coverage")
    st.caption(
        "These bars show how fully populated the main track-audio fields are in the current view."
    )

    if coverage_df.empty:
        st.info("No audio coverage summary is available in the current view.")
    else:
        st.altair_chart(create_audio_coverage_chart(coverage_df), width="stretch")

        coverage_display_df = coverage_df.rename(
            columns={
                "field_label": "Field",
                "non_null_count": "Non-null Track Count",
                "coverage_share": "Coverage Share",
            }
        )

        st.dataframe(
            coverage_display_df,
            width="stretch",
            hide_index=True,
        )


if __name__ == "__main__":
    main()
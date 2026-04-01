from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from app.app_controls import get_track_album_relationship_controls
from app.app_data import load_source_data
from app.ui import apply_app_styles, rename_columns_for_display


ALBUM_FILTER_FLAG_MAP = {
    "Ambient / Experimental": "album_ambient_experimental",
    "Classical / Orchestral": "album_classical_orchestral",
    "Electronic": "album_electronic",
    "Hip-Hop / R&B": "album_hip_hop_rnb",
    "Pop": "album_pop",
    "Rock": "album_rock",
    "World / Folk": "album_world_folk",
}

FILM_FILTER_FLAG_MAP = {
    "Action": "film_is_action",
    "Adventure": "film_is_adventure",
    "Animation": "film_is_animation",
    "Comedy": "film_is_comedy",
    "Crime": "film_is_crime",
    "Documentary": "film_is_documentary",
    "Drama": "film_is_drama",
    "Family": "film_is_family",
    "Fantasy": "film_is_fantasy",
    "History": "film_is_history",
    "Horror": "film_is_horror",
    "Music": "film_is_music",
    "Mystery": "film_is_mystery",
    "Romance": "film_is_romance",
    "Science Fiction": "film_is_science_fiction",
    "TV Movie": "film_is_tv_movie",
    "Thriller": "film_is_thriller",
    "War": "film_is_war",
    "Western": "film_is_western",
}

ALBUM_GENRE_FLAG_COLS = list(ALBUM_FILTER_FLAG_MAP.values())
FILM_GENRE_FLAG_COLS = list(FILM_FILTER_FLAG_MAP.values())

GENRE_LABEL_MAP = {
    **{col: label for label, col in ALBUM_FILTER_FLAG_MAP.items()},
    **{col: label for label, col in FILM_FILTER_FLAG_MAP.items()},
}

DOMINANCE_BUCKET_ORDER = ["<1x", "1–2x", "2–5x", "5–10x", "10x+"]

TRACK_AGG_LABELS = {
    "max": "Top Track",
    "mean": "Mean Track",
    "median": "Median Track",
    "top3": "Top 3 Sum",
}

METRIC_FIELD_MAP = {
    "listeners": {
        "album": "lfm_album_listeners",
        "track": "lfm_track_listeners",
        "max": "track_max_listeners",
        "mean": "track_mean_listeners",
        "median": "track_median_listeners",
        "top3": "track_top3_listeners",
        "total": "track_total_listeners",
        "top_to_album": "top_to_album_listeners",
        "top_to_total": "top_to_total_listeners",
        "dominance_bucket_top_to_album": "dominance_bucket_top_to_album_listeners",
        "dominance_bucket_top_to_total": "dominance_bucket_top_to_total_listeners",
        "metric_label": "Listeners",
    },
    "playcount": {
        "album": "lfm_album_playcount",
        "track": "lfm_track_playcount",
        "max": "track_max_playcount",
        "mean": "track_mean_playcount",
        "median": "track_median_playcount",
        "top3": "track_top3_playcount",
        "total": "track_total_playcount",
        "top_to_album": "top_to_album_playcount",
        "top_to_total": "top_to_total_playcount",
        "dominance_bucket_top_to_album": "dominance_bucket_top_to_album_playcount",
        "dominance_bucket_top_to_total": "dominance_bucket_top_to_total_playcount",
        "metric_label": "Playcount",
    },
}

DOMINANCE_BUCKET_ORDER_ALBUM = ["<1x", "1–2x", "2–5x", "5–10x", "10x+"]
DOMINANCE_BUCKET_ORDER_TOTAL = ["<10%", "10–20%", "20–35%", "35–50%", "50%+"]

def first_non_null(series: pd.Series):
    """Return the first non-null value in a series, if any."""
    non_null = series.dropna()
    if non_null.empty:
        return np.nan
    return non_null.iloc[0]


def compute_top3_sum(series: pd.Series) -> float:
    """Return the sum of the top 3 non-null values in a series."""
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if len(values) == 0:
        return np.nan
    values = np.sort(values)[::-1]
    return float(values[:3].sum())


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Compute a safe ratio that returns NaN when the denominator is missing or <= 0."""
    out = numerator / denominator
    out = out.where(denominator > 0)
    return out.replace([np.inf, -np.inf], np.nan)


def assign_album_ratio_bucket(value: float) -> str | None:
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

def assign_total_share_bucket(value: float) -> str | None:
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

def get_dominance_label(dominance_metric_key: str, metric_label: str) -> str:
    """Return a user-facing label for the selected dominance metric."""
    if dominance_metric_key == "top_to_album":
        return f"Top Track / Album {metric_label}"
    return f"Top Track / Total Track {metric_label}"

def get_dominance_bucket_order(dominance_metric_key: str) -> list[str]:
    """Return the correct bucket ordering for the selected dominance metric."""
    if dominance_metric_key == "top_to_album":
        return DOMINANCE_BUCKET_ORDER_ALBUM
    return DOMINANCE_BUCKET_ORDER_TOTAL

def derive_multi_label_group(
    df: pd.DataFrame,
    flag_cols: list[str],
    output_col: str,
) -> pd.DataFrame:
    """
    Collapse binary genre flags into a single grouped label.

    One active flag -> genre label
    Multiple active flags -> Multi-genre
    No active flags -> Unknown
    """
    available_cols = [col for col in flag_cols if col in df.columns]

    if not available_cols:
        df[output_col] = "Unknown"
        return df

    def assign_group(row: pd.Series) -> str:
        active_cols = [col for col in available_cols if row[col] == 1]
        if len(active_cols) == 1:
            return GENRE_LABEL_MAP[active_cols[0]]
        if len(active_cols) > 1:
            return "Multi-genre"
        return "Unknown"

    df[output_col] = df[available_cols].apply(assign_group, axis=1)
    return df


@st.cache_data(show_spinner=False)
def build_track_album_relationship_df() -> pd.DataFrame:
    """
    Build the one-row-per-album dataframe used by the Track–Album Relationship Explorer.
    """
    _, wide_df = load_source_data()

    keep_cols = [
        "tmdb_id",
        "release_group_mbid",
        "album_title",
        "film_title",
        "composer_primary_clean",
        "label_names",
        "film_year",
        "track_id",
        "lfm_track_listeners",
        "lfm_track_playcount",
        "lfm_album_listeners",
        "lfm_album_playcount",
        *ALBUM_GENRE_FLAG_COLS,
        *FILM_GENRE_FLAG_COLS,
    ]

    available_cols = [col for col in keep_cols if col in wide_df.columns]
    df = wide_df[available_cols].copy()

    for col in [
        "lfm_track_listeners",
        "lfm_track_playcount",
        "lfm_album_listeners",
        "lfm_album_playcount",
        "film_year",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    group_cols = ["tmdb_id", "release_group_mbid"]

    agg_dict = {
        "album_title": first_non_null,
        "film_title": first_non_null,
        "composer_primary_clean": first_non_null,
        "label_names": first_non_null,
        "film_year": first_non_null,
        "lfm_album_listeners": first_non_null,
        "lfm_album_playcount": first_non_null,
        "track_id": pd.Series.nunique,
        "lfm_track_listeners": ["max", "mean", "median", "sum", compute_top3_sum],
        "lfm_track_playcount": ["max", "mean", "median", "sum", compute_top3_sum],
    }

    for col in ALBUM_GENRE_FLAG_COLS + FILM_GENRE_FLAG_COLS:
        if col in df.columns:
            agg_dict[col] = "max"

    grouped = df.groupby(group_cols, dropna=False).agg(agg_dict)
    grouped.columns = [
        "_".join([part for part in col if part]).strip("_")
        if isinstance(col, tuple) else col
        for col in grouped.columns.to_flat_index()
    ]
    grouped = grouped.reset_index()

    grouped = grouped.rename(
        columns={
            "album_title_first_non_null": "album_title",
            "film_title_first_non_null": "film_title",
            "composer_primary_clean_first_non_null": "composer_primary_clean",
            "label_names_first_non_null": "label_names",
            "film_year_first_non_null": "film_year",
            "lfm_album_listeners_first_non_null": "lfm_album_listeners",
            "lfm_album_playcount_first_non_null": "lfm_album_playcount",
            "track_id_nunique": "n_tracks",
            "lfm_track_listeners_max": "track_max_listeners",
            "lfm_track_listeners_mean": "track_mean_listeners",
            "lfm_track_listeners_median": "track_median_listeners",
            "lfm_track_listeners_sum": "track_total_listeners",
            "lfm_track_listeners_compute_top3_sum": "track_top3_listeners",
            "lfm_track_playcount_max": "track_max_playcount",
            "lfm_track_playcount_mean": "track_mean_playcount",
            "lfm_track_playcount_median": "track_median_playcount",
            "lfm_track_playcount_sum": "track_total_playcount",
            "lfm_track_playcount_compute_top3_sum": "track_top3_playcount",
            "album_ambient_experimental_max": "album_ambient_experimental",
            "album_classical_orchestral_max": "album_classical_orchestral",
            "album_electronic_max": "album_electronic",
            "album_hip_hop_rnb_max": "album_hip_hop_rnb",
            "album_pop_max": "album_pop",
            "album_rock_max": "album_rock",
            "album_world_folk_max": "album_world_folk",
            "film_is_action_max": "film_is_action",
            "film_is_adventure_max": "film_is_adventure",
            "film_is_animation_max": "film_is_animation",
            "film_is_comedy_max": "film_is_comedy",
            "film_is_crime_max": "film_is_crime",
            "film_is_documentary_max": "film_is_documentary",
            "film_is_drama_max": "film_is_drama",
            "film_is_family_max": "film_is_family",
            "film_is_fantasy_max": "film_is_fantasy",
            "film_is_history_max": "film_is_history",
            "film_is_horror_max": "film_is_horror",
            "film_is_music_max": "film_is_music",
            "film_is_mystery_max": "film_is_mystery",
            "film_is_romance_max": "film_is_romance",
            "film_is_science_fiction_max": "film_is_science_fiction",
            "film_is_tv_movie_max": "film_is_tv_movie",
            "film_is_thriller_max": "film_is_thriller",
            "film_is_war_max": "film_is_war",
            "film_is_western_max": "film_is_western",
        }
    )

    grouped["top_to_album_listeners"] = safe_ratio(
        grouped["track_max_listeners"],
        grouped["lfm_album_listeners"],
    )
    grouped["top_to_total_listeners"] = safe_ratio(
        grouped["track_max_listeners"],
        grouped["track_total_listeners"],
    )
    grouped["top_to_album_playcount"] = safe_ratio(
        grouped["track_max_playcount"],
        grouped["lfm_album_playcount"],
    )
    grouped["top_to_total_playcount"] = safe_ratio(
        grouped["track_max_playcount"],
        grouped["track_total_playcount"],
    )

    grouped["dominance_bucket_top_to_album_listeners"] = grouped[
        "top_to_album_listeners"
    ].apply(assign_album_ratio_bucket)
    grouped["dominance_bucket_top_to_total_listeners"] = grouped[
        "top_to_total_listeners"
    ].apply(assign_total_share_bucket)

    grouped["dominance_bucket_top_to_album_playcount"] = grouped[
        "top_to_album_playcount"
    ].apply(assign_album_ratio_bucket)
    grouped["dominance_bucket_top_to_total_playcount"] = grouped[
        "top_to_total_playcount"
    ].apply(assign_total_share_bucket)

    grouped = derive_multi_label_group(grouped, ALBUM_GENRE_FLAG_COLS, "album_genre_group")
    grouped = derive_multi_label_group(grouped, FILM_GENRE_FLAG_COLS, "film_genre_group")

    return grouped


def build_parity_line_df(plot_df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    """Build a parity reference line covering the plotted x/y range."""
    min_val = float(min(plot_df[x_col].min(), plot_df[y_col].min()))
    max_val = float(max(plot_df[x_col].max(), plot_df[y_col].max()))
    return pd.DataFrame(
        {
            "x_value": [min_val, max_val],
            "y_value": [min_val, max_val],
        }
    )

def get_color_tooltip_fields(color_mode: str) -> list[alt.Tooltip]:
    """
    Return tooltip fields for the active color encoding.
    """
    if color_mode == "Dominance Bucket":
        return [
            alt.Tooltip(
                "dominance_bucket_for_display:N",
                title="Dominance Bucket",
            )
        ]
    if color_mode == "Album Genre":
        return [
            alt.Tooltip(
                "album_genre_group:N",
                title="Album Genre Group",
            )
        ]
    if color_mode == "Film Genre":
        return [
            alt.Tooltip(
                "film_genre_group:N",
                title="Film Genre Group",
            )
        ]
    if color_mode == "Film Year":
        return [
            alt.Tooltip(
                "film_year:Q",
                title="Film Year",
                format=",.0f",
            )
        ]
    return []

def create_hero_scatter(
    plot_df: pd.DataFrame,
    parity_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_mode: str,
    use_log_scale: bool,
    metric_label: str,
    y_agg_label: str,
    dominance_metric_key: str,
    dominance_label: str,
) -> alt.Chart:
    """Create the anchor scatterplot for album vs. track aggregation."""
    x_scale = alt.Scale(type="log") if use_log_scale else alt.Scale()
    y_scale = alt.Scale(type="log") if use_log_scale else alt.Scale()

    base_tooltip = [
        alt.Tooltip("film_title:N", title="Film"),
        alt.Tooltip("album_title:N", title="Album"),
        alt.Tooltip("composer_primary_clean:N", title="Composer"),
        alt.Tooltip("film_year:Q", title="Film Year", format=",.0f"),
        alt.Tooltip("n_tracks:Q", title="Track Count", format=",.0f"),
        alt.Tooltip(f"{x_col}:Q", title=f"Album {metric_label}", format=",.0f"),
        alt.Tooltip(f"{y_col}:Q", title=f"{y_agg_label} {metric_label}", format=",.0f"),
        alt.Tooltip(
            "top_to_album_for_display:Q",
            title="Top Track / Album",
            format=".2f",
        ),
        alt.Tooltip(
            "top_to_total_for_display:Q",
            title="Top Track / Total Track",
            format=".2f",
        ),
    ]

    tooltip = base_tooltip + get_color_tooltip_fields(color_mode)

    base = alt.Chart(plot_df).mark_circle(size=55, opacity=0.45).encode(
        x=alt.X(x_col, title=f"Album {metric_label}", scale=x_scale),
        y=alt.Y(y_col, title=f"{y_agg_label} {metric_label}", scale=y_scale),
        tooltip=tooltip,
    )

    if color_mode == "Dominance Bucket":
        points = base.encode(
            color=alt.Color(
                "dominance_bucket_for_display:N",
                title="Dominance Bucket",
                scale=alt.Scale(domain=get_dominance_bucket_order(dominance_metric_key)),
            )
        )
    elif color_mode == "Album Genre":
        points = base.encode(
            color=alt.Color("album_genre_group:N", title="Album Genre Group")
        )
    elif color_mode == "Film Genre":
        points = base.encode(
            color=alt.Color("film_genre_group:N", title="Film Genre Group")
        )
    else:
        points = base.encode(
            color=alt.Color("film_year:Q", title="Film Year")
        )

    parity_line = (
        alt.Chart(parity_df)
        .mark_line(strokeDash=[6, 4], strokeWidth=2, opacity=0.8)
        .encode(
            x=alt.X("x_value:Q", scale=x_scale),
            y=alt.Y("y_value:Q", scale=y_scale),
        )
    )

    return (points + parity_line).properties(
        width=760,
        height=500,
        title={
            "text": f"Album {metric_label} vs. {y_agg_label} {metric_label}",
            "subtitle": [
                (
                    f"Each point is an album. The dashed line marks parity between the album metric "
                    f"and the selected track aggregation. Color shows {dominance_label.lower()}."
                )
            ],
        },
    )


def create_dominance_histogram(
    plot_df: pd.DataFrame,
    dominance_col: str,
    dominance_label: str,
) -> alt.Chart:
    """Create a histogram of album dominance ratios."""
    return (
        alt.Chart(plot_df)
        .mark_bar(opacity=0.85)
        .encode(
            x=alt.X(
                f"{dominance_col}:Q",
                bin=alt.Bin(maxbins=30),
                title=dominance_label,
            ),
            y=alt.Y("count():Q", title="Album Count"),
            tooltip=[
                alt.Tooltip("count():Q", title="Album Count", format=",.0f"),
            ],
        )
        .properties(
            width=760,
            height=280,
            title={
                "text": "Dominance Distribution",
                "subtitle": [
                    f"Distribution of {dominance_label.lower()} across albums in the current view."
                ],
            },
        )
    )


def create_scaling_scatter(
    plot_df: pd.DataFrame,
    album_col: str,
    dominance_col: str,
    metric_label: str,
    use_log_scale: bool,
    color_mode: str,
    dominance_label: str,
    dominance_metric_key: str,
) -> alt.Chart:
    """Create a scaling scatter of album size vs. dominance."""
    x_scale = alt.Scale(type="log") if use_log_scale else alt.Scale()

    base_tooltip = [
        alt.Tooltip("film_title:N", title="Film"),
        alt.Tooltip("album_title:N", title="Album"),
        alt.Tooltip("composer_primary_clean:N", title="Composer"),
        alt.Tooltip("film_year:Q", title="Film Year", format=",.0f"),
        alt.Tooltip("n_tracks:Q", title="Track Count", format=",.0f"),
        alt.Tooltip(
            f"{album_col}:Q",
            title=f"Album {metric_label}",
            format=",.0f",
        ),
        alt.Tooltip(
            f"{dominance_col}:Q",
            title=dominance_label,
            format=".3f",
        ),
        alt.Tooltip(
            "top_to_album_for_display:Q",
            title="Top Track / Album",
            format=".2f",
        ),
        alt.Tooltip(
            "top_to_total_for_display:Q",
            title="Top Track / Total Track",
            format=".2f",
        ),
    ]

    tooltip = base_tooltip + get_color_tooltip_fields(color_mode)

    base = (
        alt.Chart(plot_df)
        .mark_circle(size=55, opacity=0.45)
        .encode(
            x=alt.X(album_col, title=f"Album {metric_label}", scale=x_scale),
            y=alt.Y(
                dominance_col,
                title=dominance_label,
            ),
            tooltip=tooltip,
        )
    )

    reference_lines = None
    if dominance_metric_key == "top_to_total":
        ref_df = pd.DataFrame(
            {"y_ref": [0.10, 0.25, 0.50]}
        )
        reference_lines = (
            alt.Chart(ref_df)
            .mark_rule(strokeDash=[4, 4], opacity=0.35)
            .encode(y="y_ref:Q")
        )

    if color_mode == "Dominance Bucket":
        points = base.encode(
            color=alt.Color(
                "dominance_bucket_for_display:N",
                title="Dominance Bucket",
                scale=alt.Scale(domain=get_dominance_bucket_order(dominance_metric_key)),
            )
        )
    elif color_mode == "Album Genre":
        points = base.encode(
            color=alt.Color("album_genre_group:N", title="Album Genre Group")
        )
    elif color_mode == "Film Genre":
        points = base.encode(
            color=alt.Color("film_genre_group:N", title="Film Genre Group")
        )
    else:
        points = base.encode(
            color=alt.Color("film_year:Q", title="Film Year")
        )

    chart = points
    if reference_lines is not None:
        chart = reference_lines + points

    return chart.properties(
        width=760,
        height=360,
        title={
            "text": f"Album Size vs. {dominance_label}",
            "subtitle": [
                (
                    f"Do larger albums rely more or less on a single dominant track "
                    f"based on {dominance_label.lower()}?"
                )
            ],
        },
    )

def build_track_album_scope_caption(
    controls: dict,
    plot_df: pd.DataFrame,
    metric_label: str,
    y_agg_label: str,
    dominance_label: str,
) -> str:
    """
    Build a short caption describing the current Track–Album analysis scope.

    Args:
        controls: Sidebar control selections.
        plot_df: Already filtered one-row-per-album dataframe.
        metric_label: Display label for the selected comparison metric.
        y_agg_label: Display label for the selected track aggregation.
        dominance_label: Display label for the selected dominance framing.

    Returns:
        str: Human-readable scope caption.
    """
    parts = [
        f"{len(plot_df):,} albums in view",
        f"album {metric_label.lower()} vs {y_agg_label.lower()} {metric_label.lower()}",
        f"dominance: {dominance_label.lower()}",
        f"color: {controls['color_mode'].lower()}",
    ]

    if controls.get("use_log_scale"):
        parts.append("log scale on")

    year_range = controls.get("year_range")
    if year_range:
        parts.append(f"film years {year_range[0]}–{year_range[1]}")

    if controls.get("selected_album_genres"):
        parts.append(
            f"album genres: {', '.join(controls['selected_album_genres'])}"
        )

    if controls.get("selected_film_genres"):
        parts.append(
            f"film genres: {', '.join(controls['selected_film_genres'])}"
        )

    if controls.get("selected_composers"):
        parts.append(
            f"{len(controls['selected_composers'])} selected composers"
        )

    if controls.get("selected_labels"):
        parts.append(
            f"{len(controls['selected_labels'])} selected labels"
        )

    return "Current scope: " + "; ".join(parts) + "."


def build_track_album_insight_summary(
    plot_df: pd.DataFrame,
    album_col: str,
    y_col: str,
    dominance_col: str,
    dominance_label: str,
) -> dict[str, str]:
    """
    Build reactive top-row insight cards for the Track–Album Relationship Explorer.

    Args:
        plot_df: Already filtered one-row-per-album dataframe.
        album_col: Selected album metric column.
        y_col: Selected track aggregation column.
        dominance_col: Selected dominance metric column.
        dominance_label: Display label for the selected dominance framing.

    Returns:
        dict[str, str]: Titles, values, and captions for three insight cards.
    """
    if plot_df.empty:
        return {
            "card1_title": "Parity Pattern",
            "card1_value": "None",
            "card1_caption": "No visible albums remain under the current settings.",
            "card2_title": "Typical Dominance",
            "card2_value": "None",
            "card2_caption": "No dominance summary is available.",
            "card3_title": "Most Common Bucket",
            "card3_value": "None",
            "card3_caption": "No dominance bucket pattern is available.",
        }

    above_parity_mask = plot_df[y_col] >= plot_df[album_col]
    above_count = int(above_parity_mask.sum())
    total_count = int(len(plot_df))
    above_share = above_count / total_count if total_count > 0 else 0.0

    if above_share >= 0.60:
        parity_value = "Mostly above parity"
    elif above_share <= 0.40:
        parity_value = "Mostly below parity"
    else:
        parity_value = "Mixed parity"

    parity_caption = (
        f"{above_share:.1%} of visible albums have the selected track aggregation "
        "at or above the album metric."
    )

    dominance_median = float(plot_df[dominance_col].median())
    dominance_value = (
        f"{dominance_median:.2f}"
        if "total track" in dominance_label.lower()
        else f"{dominance_median:.2f}x"
    )

    dominance_caption = (
        f"Median visible {dominance_label.lower()} across the current filtered albums."
    )

    bucket_mode_series = (
        plot_df["dominance_bucket_for_display"]
        .dropna()
        .astype(str)
        .value_counts()
    )

    if bucket_mode_series.empty:
        bucket_value = "None"
        bucket_caption = "No visible dominance buckets are available."
    else:
        bucket_value = str(bucket_mode_series.index[0])
        bucket_count = int(bucket_mode_series.iloc[0])
        bucket_share = bucket_count / total_count if total_count > 0 else 0.0
        bucket_caption = (
            f"The most common visible dominance bucket contains {bucket_count:,} albums "
            f"({bucket_share:.1%} of the current view)."
        )

    return {
        "card1_title": "Parity Pattern",
        "card1_value": parity_value,
        "card1_caption": parity_caption,
        "card2_title": "Typical Dominance",
        "card2_value": dominance_value,
        "card2_caption": dominance_caption,
        "card3_title": "Most Common Bucket",
        "card3_value": bucket_value,
        "card3_caption": bucket_caption,
    }


def render_track_album_insight_cards(
    plot_df: pd.DataFrame,
    album_col: str,
    y_col: str,
    dominance_col: str,
    dominance_label: str,
) -> None:
    """
    Render reactive insight cards for the Track–Album Relationship Explorer.

    Args:
        plot_df: Already filtered one-row-per-album dataframe.
        album_col: Selected album metric column.
        y_col: Selected track aggregation column.
        dominance_col: Selected dominance metric column.
        dominance_label: Display label for the selected dominance framing.
    """
    insights = build_track_album_insight_summary(
        plot_df=plot_df,
        album_col=album_col,
        y_col=y_col,
        dominance_col=dominance_col,
        dominance_label=dominance_label,
    )

    st.markdown("### 🧠 Key Insights")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(insights["card1_title"], insights["card1_value"])
        st.caption(insights["card1_caption"])

    with col2:
        st.metric(insights["card2_title"], insights["card2_value"])
        st.caption(insights["card2_caption"])

    with col3:
        st.metric(insights["card3_title"], insights["card3_value"])
        st.caption(insights["card3_caption"])


def build_hero_scatter_supporting_insight(
    plot_df: pd.DataFrame,
    album_col: str,
    y_col: str,
    y_agg_label: str,
    metric_label: str,
) -> str:
    """
    Build a short supporting insight for the hero scatter.

    Args:
        plot_df: Already filtered one-row-per-album dataframe.
        album_col: Selected album metric column.
        y_col: Selected track aggregation column.
        y_agg_label: Display label for the selected track aggregation.
        metric_label: Display label for the selected comparison metric.

    Returns:
        str: Supporting sentence.
    """
    if plot_df.empty:
        return "No hero-scatter insight is available."

    diff = plot_df[y_col] - plot_df[album_col]
    median_diff = float(diff.median())
    above_share = float((plot_df[y_col] >= plot_df[album_col]).mean())

    if above_share >= 0.60:
        pattern = "sit mostly above parity"
    elif above_share <= 0.40:
        pattern = "sit mostly below parity"
    else:
        pattern = "straddle parity fairly evenly"

    if median_diff > 0:
        diff_phrase = (
            f"The typical visible album has {y_agg_label.lower()} {metric_label.lower()} "
            "above the album-level metric."
        )
    elif median_diff < 0:
        diff_phrase = (
            f"The typical visible album has {y_agg_label.lower()} {metric_label.lower()} "
            "below the album-level metric."
        )
    else:
        diff_phrase = (
            "The typical visible album sits almost exactly at parity."
        )

    return (
        f"💡 Under the current settings, albums {pattern}: "
        f"{above_share:.1%} are at or above the parity line. {diff_phrase}"
    )


def build_histogram_supporting_insight(
    plot_df: pd.DataFrame,
    dominance_col: str,
    dominance_label: str,
    dominance_metric_key: str,
) -> str:
    """
    Build a short supporting insight for the dominance histogram.

    Args:
        plot_df: Already filtered one-row-per-album dataframe.
        dominance_col: Selected dominance metric column.
        dominance_label: Display label for the selected dominance framing.
        dominance_metric_key: Dominance metric key from the controls.

    Returns:
        str: Supporting sentence.
    """
    if plot_df.empty:
        return "No dominance-distribution insight is available."

    median_value = float(plot_df[dominance_col].median())
    bucket_counts = (
        plot_df["dominance_bucket_for_display"]
        .dropna()
        .astype(str)
        .value_counts()
    )

    if bucket_counts.empty:
        bucket_phrase = "No visible dominance bucket stands out."
    else:
        top_bucket = str(bucket_counts.index[0])
        top_bucket_share = float(bucket_counts.iloc[0] / len(plot_df))
        bucket_phrase = (
            f"The most common visible bucket is {top_bucket} "
            f"({top_bucket_share:.1%} of albums)."
        )

    if dominance_metric_key == "top_to_total":
        value_phrase = (
            f"The median visible {dominance_label.lower()} is {median_value:.2f}, "
            "so the top track typically accounts for a moderate share of total track performance."
        )
    else:
        value_phrase = (
            f"The median visible {dominance_label.lower()} is {median_value:.2f}x, "
            "which is best read as a diagnostic comparison rather than a literal share."
        )

    return f"💡 {value_phrase} {bucket_phrase}"


def build_scaling_supporting_insight(
    plot_df: pd.DataFrame,
    album_col: str,
    dominance_col: str,
    dominance_label: str,
) -> str:
    """
    Build a short supporting insight for the album-size vs dominance scatter.

    Args:
        plot_df: Already filtered one-row-per-album dataframe.
        album_col: Selected album metric column.
        dominance_col: Selected dominance metric column.
        dominance_label: Display label for the selected dominance framing.

    Returns:
        str: Supporting sentence.
    """
    if plot_df.empty:
        return "No scaling insight is available."

    corr = plot_df[album_col].corr(plot_df[dominance_col], method="pearson")

    if pd.isna(corr):
        return (
            "The current filtered sample does not support a stable linear read on "
            "album size versus dominance."
        )

    if corr >= 0.20:
        pattern = "larger albums tend to look more top-heavy"
    elif corr <= -0.20:
        pattern = "larger albums tend to look less top-heavy"
    else:
        pattern = "album size and dominance look broadly weakly related"

    return (
        f"💡 In the current view, {pattern} based on {dominance_label.lower()} "
        f"(Pearson r = {corr:.3f})."
    )

def build_parity_explainer(
    y_agg_label: str,
    metric_label: str,
) -> str:
    """
    Build a short plain-English explainer for the hero scatter parity line.

    Args:
        y_agg_label: Display label for the selected track aggregation.
        metric_label: Display label for the selected comparison metric.

    Returns:
        str: Short explainer sentence.
    """
    return (
        f"Points above the dashed parity line have {y_agg_label.lower()} "
        f"{metric_label.lower()} at or above the album-level {metric_label.lower()}; "
        "points below the line fall short of album-level performance."
    )


def build_extreme_album_caption(
    plot_df: pd.DataFrame,
    dominance_col: str,
    dominance_label: str,
) -> str:
    """
    Build a short caption highlighting the most dominant visible album.

    Args:
        plot_df: Already filtered one-row-per-album dataframe.
        dominance_col: Selected dominance metric column.
        dominance_label: Display label for the selected dominance framing.

    Returns:
        str: Short caption naming the most extreme visible album.
    """
    if plot_df.empty:
        return "No extreme album callout is available."

    top_row = plot_df.sort_values(
        [dominance_col, "film_title", "album_title"],
        ascending=[False, True, True],
    ).iloc[0]

    value = float(top_row[dominance_col])
    value_text = (
        f"{value:.2f}"
        if "total track" in dominance_label.lower()
        else f"{value:.2f}x"
    )

    return (
        f"Most top-heavy visible album: {top_row['film_title']} — "
        f"{top_row['album_title']} ({value_text} on {dominance_label.lower()})."
    )


def build_top3_mode_caption(
    y_agg_key: str,
    y_agg_label: str,
) -> str:
    """
    Build an optional caption for Top 3 mode.

    Args:
        y_agg_key: Raw track aggregation key.
        y_agg_label: Display label for the selected track aggregation.

    Returns:
        str: Optional explanatory caption.
    """
    if y_agg_key != "top3":
        return ""

    return (
        f"You are using {y_agg_label.lower()}, so this view emphasizes clustered "
        "multi-track strength rather than dependence on a single breakout track."
    )

def main() -> None:
    """Render the Track–Album Relationship Explorer page."""
    st.set_page_config(
        page_title="Track–Album Relationship Explorer",
        layout="wide",
    )
    apply_app_styles()

    st.title("Track–Album Relationship Explorer")
    st.write(
        """
        Explore how track-level performance aggregates into album-level success.
        This page compares album outcomes with selected track summaries so you can
        see whether stronger soundtracks are driven mainly by one breakout track,
        a stronger top-three cluster, or broader depth across the album.
        """
    )

    explorer_df = build_track_album_relationship_df()

    year_min = int(explorer_df["film_year"].dropna().min())
    year_max = int(explorer_df["film_year"].dropna().max())

    album_genre_options = sorted(
        [
            label
            for label, col in ALBUM_FILTER_FLAG_MAP.items()
            if col in explorer_df.columns and explorer_df[col].fillna(0).sum() > 0
        ]
    )
    film_genre_options = sorted(
        [
            label
            for label, col in FILM_FILTER_FLAG_MAP.items()
            if col in explorer_df.columns and explorer_df[col].fillna(0).sum() > 0
        ]
    )
    composer_options = sorted(
        explorer_df["composer_primary_clean"].dropna().astype(str).unique().tolist()
    )
    label_options = sorted(
        explorer_df["label_names"].dropna().astype(str).unique().tolist()
    )

    controls = get_track_album_relationship_controls(
        min_year=year_min,
        max_year=year_max,
        album_genre_options=album_genre_options,
        film_genre_options=film_genre_options,
        composer_options=composer_options,
        label_options=label_options,
    )

    metric_key = controls["comparison_metric"]
    y_agg_key = controls["track_aggregation"]
    dominance_metric_key = controls["dominance_metric"]

    fields = METRIC_FIELD_MAP[metric_key]
    album_col = fields["album"]
    y_col = fields[y_agg_key]
    top_to_album_col = fields["top_to_album"]
    top_to_total_col = fields["top_to_total"]
    dominance_col = fields[dominance_metric_key]
    dominance_bucket_col = fields[f"dominance_bucket_{dominance_metric_key}"]
    metric_label = fields["metric_label"]

    color_mode = controls["color_mode"]
    use_log_scale = controls["use_log_scale"]
    selected_years = controls["year_range"]
    selected_album_genres = controls["selected_album_genres"]
    selected_film_genres = controls["selected_film_genres"]
    selected_composers = controls["selected_composers"]
    selected_labels = controls["selected_labels"]
    show_table = controls["show_data_table"]

    y_agg_label = TRACK_AGG_LABELS[y_agg_key]
    dominance_label = get_dominance_label(dominance_metric_key, metric_label)

    if dominance_metric_key == "top_to_total":
        st.info(
            f"You are viewing dominance as **{dominance_label}**. "
            "This measures how much of the soundtrack's total track performance is captured by its top track. "
            "Values range from 0 to 1, where higher values indicate a more top-heavy soundtrack."
        )
    else:
        st.info(
            f"You are viewing dominance as **{dominance_label}**. "
            "This compares the top track directly to the album-level Last.fm metric. "
            "Because album and track metrics may be tabulated differently, treat this as a diagnostic comparison rather than a literal share."
        )

    top3_caption = build_top3_mode_caption(
        y_agg_key=y_agg_key,
        y_agg_label=y_agg_label,
    )
    if top3_caption:
        st.caption(top3_caption)

    plot_df = explorer_df.copy()

    plot_df = plot_df[
        (plot_df["film_year"].fillna(year_min) >= selected_years[0])
        & (plot_df["film_year"].fillna(year_max) <= selected_years[1])
    ].copy()

    if selected_album_genres:
        selected_album_flags = [
            ALBUM_FILTER_FLAG_MAP[label]
            for label in selected_album_genres
            if label in ALBUM_FILTER_FLAG_MAP
            and ALBUM_FILTER_FLAG_MAP[label] in plot_df.columns
        ]
        if selected_album_flags:
            plot_df = plot_df[
                plot_df[selected_album_flags].fillna(0).sum(axis=1) > 0
            ].copy()

    if selected_film_genres:
        selected_film_flags = [
            FILM_FILTER_FLAG_MAP[label]
            for label in selected_film_genres
            if label in FILM_FILTER_FLAG_MAP
            and FILM_FILTER_FLAG_MAP[label] in plot_df.columns
        ]
        if selected_film_flags:
            plot_df = plot_df[
                plot_df[selected_film_flags].fillna(0).sum(axis=1) > 0
            ].copy()

    if selected_composers:
        plot_df = plot_df[
            plot_df["composer_primary_clean"].isin(selected_composers)
        ].copy()

    if selected_labels:
        plot_df = plot_df[
            plot_df["label_names"].fillna("").isin(selected_labels)
        ].copy()

    plot_df = plot_df.dropna(
        subset=[album_col, y_col, dominance_col, dominance_bucket_col]
    ).copy()

    if use_log_scale:
        plot_df = plot_df[
            (plot_df[album_col] > 0) & (plot_df[y_col] > 0)
        ].copy()

    if plot_df.empty:
        st.warning("No rows remain after applying the current filters.")
        return

    plot_df["top_to_album_for_display"] = plot_df[top_to_album_col]
    plot_df["top_to_total_for_display"] = plot_df[top_to_total_col]
    plot_df["dominance_bucket_for_display"] = plot_df[dominance_bucket_col]

    parity_df = build_parity_line_df(plot_df, album_col, y_col)

    st.caption(
        build_track_album_scope_caption(
            controls=controls,
            plot_df=plot_df,
            metric_label=metric_label,
            y_agg_label=y_agg_label,
            dominance_label=dominance_label,
        )
    )

    render_track_album_insight_cards(
        plot_df=plot_df,
        album_col=album_col,
        y_col=y_col,
        dominance_col=dominance_col,
        dominance_label=dominance_label,
    )

    with st.expander("Quick read of this page", expanded=False):
        st.write(
            build_parity_explainer(
                y_agg_label=y_agg_label,
                metric_label=metric_label,
            )
        )

    util_col1, util_col2, util_col3 = st.columns(3)

    with util_col1:
        st.metric("Albums in View", f"{len(plot_df):,}")

    with util_col2:
        st.metric("Median Track Count", f"{plot_df['n_tracks'].median():.0f}")

    with util_col3:
        if dominance_metric_key == "top_to_total":
            st.metric(
                f"Median {dominance_label}",
                f"{plot_df[dominance_col].median():.2f}",
            )
        else:
            st.metric(
                f"Median {dominance_label}",
                f"{plot_df[dominance_col].median():.2f}x",
            )

    st.subheader("Album vs. Track Aggregation")
    hero_chart = create_hero_scatter(
        plot_df=plot_df,
        parity_df=parity_df,
        x_col=album_col,
        y_col=y_col,
        color_mode=color_mode,
        use_log_scale=use_log_scale,
        metric_label=metric_label,
        y_agg_label=y_agg_label,
        dominance_metric_key=dominance_metric_key,
        dominance_label=dominance_label,
    )
    st.altair_chart(hero_chart, width="stretch")
    st.caption(
        build_hero_scatter_supporting_insight(
            plot_df=plot_df,
            album_col=album_col,
            y_col=y_col,
            y_agg_label=y_agg_label,
            metric_label=metric_label,
        )
    )

    st.subheader("Dominance Distribution")
    hist_chart = create_dominance_histogram(
        plot_df=plot_df,
        dominance_col=dominance_col,
        dominance_label=dominance_label,
    )
    st.altair_chart(hist_chart, width="stretch")
    st.caption(
        build_histogram_supporting_insight(
            plot_df=plot_df,
            dominance_col=dominance_col,
            dominance_label=dominance_label,
            dominance_metric_key=dominance_metric_key,
        )
    )
    st.caption(
        build_extreme_album_caption(
            plot_df=plot_df,
            dominance_col=dominance_col,
            dominance_label=dominance_label,
        )
    )

    if dominance_metric_key == "top_to_total":
        st.caption(
            "In this framing, 0.50 means the top track contributes half of the soundtrack's visible total track performance."
        )
    else:
        st.caption(
            "In this framing, values above 1x mean the top track exceeds the album-level count, which can happen because album and track metrics are tabulated differently."
        )

    st.subheader("Album Size vs. Dominance")
    scaling_chart = create_scaling_scatter(
        plot_df=plot_df,
        album_col=album_col,
        dominance_col=dominance_col,
        metric_label=metric_label,
        use_log_scale=use_log_scale,
        color_mode=color_mode,
        dominance_label=dominance_label,
        dominance_metric_key=dominance_metric_key,
    )
    st.altair_chart(scaling_chart, width="stretch")
    st.caption(
        build_scaling_supporting_insight(
            plot_df=plot_df,
            album_col=album_col,
            dominance_col=dominance_col,
            dominance_label=dominance_label,
        )
    )

    if show_table:
        preferred_cols = [
            "film_title",
            "album_title",
            "composer_primary_clean",
            "film_year",
            "album_genre_group",
            "film_genre_group",
            "n_tracks",
            album_col,
            y_col,
            top_to_album_col,
            dominance_col,
            dominance_bucket_col,
        ]

        table_cols = []
        seen = set()
        for col in preferred_cols:
            if col in plot_df.columns and col not in seen:
                table_cols.append(col)
                seen.add(col)

        table_df = rename_columns_for_display(plot_df[table_cols]).copy()

        deduped_cols = []
        seen_display = {}
        for col in table_df.columns:
            if col not in seen_display:
                seen_display[col] = 1
                deduped_cols.append(col)
            else:
                seen_display[col] += 1
                deduped_cols.append(f"{col} ({seen_display[col]})")
        table_df.columns = deduped_cols

        st.subheader("Source Data")
        st.dataframe(
            table_df,
            width="stretch",
            hide_index=True,
        )


if __name__ == "__main__":
    main()
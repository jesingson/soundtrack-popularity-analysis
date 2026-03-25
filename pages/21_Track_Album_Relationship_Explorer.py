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

        This page asks whether strong albums are driven by one breakout track,
        a small cluster of standout tracks, or broader consistency across the
        soundtrack.
        """
    )

    st.caption(
        "Use this page to compare album-level success to track-level strength. "
        "The hero scatter compares the album metric to a selected track aggregation, "
        "the histogram shows how concentrated performance is in the top track, "
        "and the scaling chart asks whether larger albums are more balanced or more top-heavy."
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
        plot_df = plot_df[(plot_df[album_col] > 0) & (plot_df[y_col] > 0)].copy()

    if plot_df.empty:
        st.warning("No rows remain after applying the current filters.")
        return

    plot_df["top_to_album_for_display"] = plot_df[top_to_album_col]
    plot_df["top_to_total_for_display"] = plot_df[top_to_total_col]
    plot_df["dominance_bucket_for_display"] = plot_df[dominance_bucket_col]

    parity_df = build_parity_line_df(plot_df, album_col, y_col)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Albums in View", f"{len(plot_df):,}")

    with col2:
        st.metric("Median Track Count", f"{plot_df['n_tracks'].median():.0f}")

    with col3:
        st.metric(
            "Median Top / Album",
            f"{plot_df[top_to_album_col].median():.2f}x",
        )

    with col4:
        st.metric(
            f"Median {dominance_label}",
            f"{plot_df[dominance_col].median():.2f}x",
        )

    hero_chart = create_hero_scatter(
        plot_df=plot_df,
        parity_df=parity_df,
        x_col=album_col,
        y_col=y_col,
        color_mode=color_mode,
        use_log_scale=use_log_scale,
        metric_label=metric_label,
        y_agg_label=TRACK_AGG_LABELS[y_agg_key],
        dominance_metric_key=dominance_metric_key,
        dominance_label=dominance_label,
    )
    st.altair_chart(hero_chart, width="stretch")

    hist_chart = create_dominance_histogram(
        plot_df=plot_df,
        dominance_col=dominance_col,
        dominance_label=dominance_label,
    )
    st.altair_chart(hist_chart, width="stretch")

    if dominance_metric_key == "top_to_total":
        st.caption(
            "This histogram shows how concentrated total track performance is in the top track. "
            "For example, 0.50 means the top track accounts for half of all track-level performance."
        )
    else:
        st.caption(
            "This histogram compares the top track directly to the album-level metric. "
            "Values above 1x mean the top track exceeds the album-level count, which can happen because album and track metrics are tabulated differently."
        )

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

    if dominance_metric_key == "top_to_total":
        st.caption(
            "In this mode, the y-axis ranges from 0 to 1. "
            "Higher values mean the top track contributes a larger share of total track performance."
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
        table_cols = [col for col in preferred_cols if col in plot_df.columns]
        st.subheader("Source Data")
        st.dataframe(
            rename_columns_for_display(plot_df[table_cols]),
            width="stretch",
            hide_index=True,
        )


if __name__ == "__main__":
    main()
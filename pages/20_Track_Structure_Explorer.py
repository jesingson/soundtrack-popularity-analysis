from __future__ import annotations

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

from app.app_controls import (
    get_global_filter_controls,
    get_track_structure_controls,
)
from app.app_data import load_track_explorer_data
from app.data_filters import filter_dataset, split_multivalue_genres
from app.ui import get_display_label
from app.explorer_shared import (
    get_clean_composer_options,
    get_global_filter_inputs,
)

TRACK_METRIC_OPTIONS = [
    "lfm_track_listeners",
    "lfm_track_playcount",
    "spotify_popularity",
]

SCATTER_COLOR_OPTIONS = [
    "film_year",
    "award_category",
]


def apply_track_structure_filters(
    track_df: pd.DataFrame,
    global_controls: dict,
    track_controls: dict,
) -> pd.DataFrame:
    """
    Apply global album-style filters plus Page 6 track-specific filters.

    Args:
        track_df: Track-level explorer dataframe.
        global_controls: Shared global filter control values.
        track_controls: Page 6-specific control values.

    Returns:
        pd.DataFrame: Filtered track dataframe for Stage 2 charts.
    """
    filtered = filter_dataset(track_df, global_controls).copy()

    if track_controls.get("selected_composers"):
        filtered = filtered[
            filtered["composer_primary_clean"].isin(
                track_controls["selected_composers"]
            )
        ].copy()

    filtered = filtered[
        filtered["track_count_observed"] >= track_controls["min_tracks_per_album"]
    ].copy()

    filtered = filtered[
        filtered["track_number"] <= track_controls["max_track_position"]
    ].copy()

    metric = track_controls["metric"]
    filtered = filtered[filtered[metric].notna()].copy()

    return filtered


def add_display_metric(
    track_df: pd.DataFrame,
    metric_col: str,
    transform_y: str,
) -> pd.DataFrame:
    """
    Add a chart-ready display metric column.

    Args:
        track_df: Filtered track dataframe.
        metric_col: Selected raw track metric.
        transform_y: Display transform option.

    Returns:
        pd.DataFrame: Track dataframe with display metric columns added.
    """
    track_df = track_df.copy()
    track_df["metric_raw"] = track_df[metric_col]

    if transform_y == "Log1p":
        positive_mask = track_df["metric_raw"] >= 0
        track_df = track_df[positive_mask].copy()
        track_df["metric_display"] = np.log1p(track_df["metric_raw"])
    else:
        track_df["metric_display"] = track_df["metric_raw"]

    return track_df

def add_metric_structure_fields(
    track_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add metric-specific within-album structure fields based on metric_raw.

    Args:
        track_df: Filtered track dataframe with metric_raw already added.

    Returns:
        pd.DataFrame: Track dataframe with within-album helper fields added.
    """
    track_df = track_df.copy()

    album_keys = ["release_group_mbid", "tmdb_id"]
    grouped = track_df.groupby(album_keys)["metric_raw"]

    track_df["album_metric_total"] = grouped.transform("sum")
    track_df["album_metric_mean"] = grouped.transform("mean")
    track_df["album_metric_median"] = grouped.transform("median")

    track_df["track_share"] = np.where(
        track_df["album_metric_total"] > 0,
        track_df["metric_raw"] / track_df["album_metric_total"],
        np.nan,
    )

    track_df["track_rank_within_album"] = grouped.rank(
        method="dense",
        ascending=False,
    )

    track_df["is_top_track"] = track_df["track_rank_within_album"] == 1

    return track_df

def add_analysis_basis_fields(
    track_df: pd.DataFrame,
    analysis_basis: str,
    transform_y: str,
) -> pd.DataFrame:
    """
    Add active plotted metric columns based on the selected analysis basis.

    Args:
        track_df: Track dataframe with metric_raw, metric_display, and
            track_share already available.
        analysis_basis: Selected analysis basis.
        transform_y: Display transform option.

    Returns:
        pd.DataFrame: Track dataframe with active display columns added.
    """
    track_df = track_df.copy()

    if analysis_basis == "Within-album share":
        track_df["analysis_value_raw"] = track_df["track_share"]
        track_df["analysis_value_display"] = track_df["track_share"]
    else:
        track_df["analysis_value_raw"] = track_df["metric_raw"]
        track_df["analysis_value_display"] = track_df["metric_display"]

    # Keep share untransformed for interpretability.
    if analysis_basis == "Raw metric" and transform_y == "Log1p":
        analysis_label_suffix = " (log1p)"
    else:
        analysis_label_suffix = ""

    track_df["analysis_label_suffix"] = analysis_label_suffix

    return track_df

def add_scatter_x_position(
    track_df: pd.DataFrame,
    apply_jitter: bool,
    jitter_strength: float,
) -> pd.DataFrame:
    """
    Add scatterplot X positions with optional jitter.

    Args:
        track_df: Track dataframe for plotting.
        apply_jitter: Whether to apply horizontal jitter.
        jitter_strength: Maximum jitter offset.

    Returns:
        pd.DataFrame: Track dataframe with scatter X positions added.
    """
    track_df = track_df.copy()
    track_df["track_number_plot"] = track_df["track_number"].astype(float)

    if apply_jitter:
        rng = np.random.default_rng(42)
        track_df["track_number_plot"] = (
            track_df["track_number_plot"]
            + rng.uniform(
                low=-jitter_strength,
                high=jitter_strength,
                size=len(track_df),
            )
        )

    return track_df


def build_position_summary_df(
    track_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate the filtered track dataframe by track position.

    Args:
        track_df: Filtered track dataframe with metric_display.

    Returns:
        pd.DataFrame: Summary dataframe by track position.
    """
    summary_df = (
        track_df.groupby("track_number", as_index=False)
        .agg(
            n_tracks=("analysis_value_display", "size"),
            mean_value=("analysis_value_display", "mean"),
            median_value=("analysis_value_display", "median"),
            q25_value=("analysis_value_display", lambda x: x.quantile(0.25)),
            q75_value=("analysis_value_display", lambda x: x.quantile(0.75)),
        )
        .sort_values("track_number")
        .reset_index(drop=True)
    )

    return summary_df

def build_top_track_position_df(
    track_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a summary of where the strongest visible track occurs.

    Probability is computed as:
    top-track count at a position / number of visible albums that have that
    position available.

    Args:
        track_df: Filtered track dataframe with metric_raw and is_top_track.

    Returns:
        pd.DataFrame: Summary dataframe by track position.
    """
    album_keys = ["release_group_mbid", "tmdb_id"]

    top_track_df = (
        track_df.sort_values(
            album_keys + ["metric_raw", "track_number"],
            ascending=[True, True, False, True],
        )
        .drop_duplicates(subset=album_keys, keep="first")
        .copy()
    )

    top_counts_df = (
        top_track_df.groupby("track_number", as_index=False)
        .agg(top_track_count=("track_number", "size"))
    )

    eligible_rows = []
    max_position = int(track_df["track_number"].max())

    album_max_df = (
        track_df[album_keys + ["max_track_number_observed"]]
        .drop_duplicates()
        .copy()
    )

    for pos in range(1, max_position + 1):
        eligible_count = int(
            (album_max_df["max_track_number_observed"] >= pos).sum()
        )
        eligible_rows.append(
            {
                "track_number": pos,
                "eligible_album_count": eligible_count,
            }
        )

    eligible_df = pd.DataFrame(eligible_rows)

    position_df = eligible_df.merge(
        top_counts_df,
        on="track_number",
        how="left",
    )

    position_df["top_track_count"] = (
        position_df["top_track_count"].fillna(0).astype(int)
    )

    position_df["top_track_probability"] = np.where(
        position_df["eligible_album_count"] > 0,
        position_df["top_track_count"] / position_df["eligible_album_count"],
        np.nan,
    )

    return position_df

def plot_track_position_summary(
    summary_df: pd.DataFrame,
    summary_stat: str,
    show_ribbon: bool,
    metric_label: str,
) -> alt.Chart:
    """
    Plot the primary summary-by-position chart.

    Args:
        summary_df: Aggregated summary dataframe by track position.
        summary_stat: Chosen summary statistic mode.
        show_ribbon: Whether to show the interquartile ribbon.
        metric_label: Display label for the selected metric.

    Returns:
        alt.Chart: Altair summary chart.
    """
    base = alt.Chart(summary_df)

    layers = []

    if show_ribbon:
        ribbon = base.mark_area(opacity=0.18).encode(
            x=alt.X("track_number:Q", title="Track position"),
            y=alt.Y("q25_value:Q", title=metric_label),
            y2="q75_value:Q",
            tooltip=[
                alt.Tooltip("track_number:Q", title="Track position"),
                alt.Tooltip("q25_value:Q", title="25th percentile", format=".2f"),
                alt.Tooltip("q75_value:Q", title="75th percentile", format=".2f"),
                alt.Tooltip("n_tracks:Q", title="Tracks"),
            ],
        )
        layers.append(ribbon)

    if summary_stat in ["Median", "Both"]:
        median_line = base.mark_line(point=True).encode(
            x=alt.X("track_number:Q", title="Track position"),
            y=alt.Y("median_value:Q", title=metric_label),
            tooltip=[
                alt.Tooltip("track_number:Q", title="Track position"),
                alt.Tooltip("median_value:Q", title="Median", format=".2f"),
                alt.Tooltip("n_tracks:Q", title="Tracks"),
            ],
        )
        layers.append(median_line)

    if summary_stat in ["Mean", "Both"]:
        mean_line = base.mark_line(strokeDash=[5, 4], point=True).encode(
            x=alt.X("track_number:Q", title="Track position"),
            y=alt.Y("mean_value:Q", title=metric_label),
            tooltip=[
                alt.Tooltip("track_number:Q", title="Track position"),
                alt.Tooltip("mean_value:Q", title="Mean", format=".2f"),
                alt.Tooltip("n_tracks:Q", title="Tracks"),
            ],
        )
        layers.append(mean_line)

    return alt.layer(*layers).properties(height=380)

def plot_top_track_position_chart(
    position_df: pd.DataFrame,
) -> alt.Chart:
    """
    Plot the probability that the strongest visible track occurs at each
    track position.

    Args:
        position_df: Top-track position summary dataframe.

    Returns:
        alt.Chart: Altair bar + line chart.
    """
    base = alt.Chart(position_df)

    bars = base.mark_bar(opacity=0.75).encode(
        x=alt.X("track_number:Q", title="Track position"),
        y=alt.Y(
            "top_track_probability:Q",
            title="Probability strongest track occurs here",
            axis=alt.Axis(format="%"),
        ),
        tooltip=[
            alt.Tooltip("track_number:Q", title="Track position"),
            alt.Tooltip("top_track_count:Q", title="Top-track albums"),
            alt.Tooltip("eligible_album_count:Q", title="Eligible albums"),
            alt.Tooltip(
                "top_track_probability:Q",
                title="Probability",
                format=".1%",
            ),
        ],
    )

    line = base.mark_line(point=True).encode(
        x="track_number:Q",
        y=alt.Y("top_track_probability:Q", axis=alt.Axis(format="%")),
    )

    return (bars + line).properties(height=320)

def plot_track_position_scatter(
    track_df: pd.DataFrame,
    color_col: str,
    show_trendline: bool,
    metric_label: str,
) -> alt.Chart:
    """
    Plot track position versus track performance scatter.

    Args:
        track_df: Filtered track dataframe with display metric and scatter X.
        color_col: Optional color grouping field.
        show_trendline: Whether to overlay a fitted line.
        metric_label: Display label for the selected metric.

    Returns:
        alt.Chart: Altair scatter chart.
    """
    tooltip_cols = [
        alt.Tooltip("film_title:N", title="Film"),
        alt.Tooltip("album_title:N", title="Album"),
        alt.Tooltip("track_title:N", title="Track"),
        alt.Tooltip("track_number:Q", title="Track position"),
        alt.Tooltip("metric_raw:Q", title="Selected metric (raw)", format=".2f"),
        alt.Tooltip("track_share:Q", title="Track share", format=".3f"),
        alt.Tooltip("analysis_value_display:Q", title=metric_label, format=".3f"),
        alt.Tooltip("track_count_observed:Q", title="Observed tracks"),
    ]

    if color_col != "None":
        tooltip_cols.append(
            alt.Tooltip(
                f"{color_col}:N",
                title=get_display_label(color_col),
            )
        )

    base = alt.Chart(track_df)

    if color_col != "None":
        points = base.mark_circle(opacity=0.45, size=45).encode(
            x=alt.X("track_number_plot:Q", title="Track position"),
            y=alt.Y("analysis_value_display:Q", title=metric_label),
            color=alt.Color(f"{color_col}:N", title=get_display_label(color_col)),
            tooltip=tooltip_cols,
        )
    else:
        points = base.mark_circle(opacity=0.45, size=45).encode(
            x=alt.X("track_number_plot:Q", title="Track position"),
            y=alt.Y("analysis_value_display:Q", title=metric_label),
            tooltip=tooltip_cols,
        )

    if show_trendline:
        trend = base.transform_regression(
            "track_number",
            "analysis_value_display",
        ).mark_line().encode(
            x="track_number:Q",
            y="analysis_value_display:Q",
        )
        return (points + trend).properties(height=380)

    return points.properties(height=380)

def build_album_cohesion_df(
    track_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build an album-level cohesion summary from the filtered track dataframe.

    Cohesion metrics are computed from the selected raw track metric so they
    remain interpretable even when the display layer uses a transformed value.

    Args:
        track_df: Filtered track dataframe with metric_raw available.

    Returns:
        pd.DataFrame: One row per album with within-album cohesion metrics.
    """
    album_keys = [
        "release_group_mbid",
        "tmdb_id",
        "album_title",
        "film_title",
        "composer_primary_clean",
        "track_count_observed",
    ]

    cohesion_df = (
        track_df.groupby(album_keys, as_index=False)
        .agg(
            total_track_metric=("metric_raw", "sum"),
            top_track_value=("metric_raw", "max"),
            median_track_value=("metric_raw", "median"),
            track_metric_std_dev=("metric_raw", "std"),
        )
        .copy()
    )

    cohesion_df["track_metric_std_dev"] = (
        cohesion_df["track_metric_std_dev"].fillna(0)
    )

    cohesion_df["top_track_share"] = np.where(
        cohesion_df["total_track_metric"] > 0,
        cohesion_df["top_track_value"] / cohesion_df["total_track_metric"],
        np.nan,
    )

    cohesion_df["top_track_to_median_ratio"] = np.where(
        cohesion_df["median_track_value"] > 0,
        cohesion_df["top_track_value"] / cohesion_df["median_track_value"],
        np.nan,
    )

    first_three_df = (
        track_df[track_df["track_number"] <= 3]
        .groupby(["release_group_mbid", "tmdb_id"], as_index=False)
        .agg(first_three_track_value=("metric_raw", "sum"))
    )

    cohesion_df = cohesion_df.merge(
        first_three_df,
        on=["release_group_mbid", "tmdb_id"],
        how="left",
        validate="1:1",
    )

    cohesion_df["first_three_track_value"] = (
        cohesion_df["first_three_track_value"].fillna(0)
    )

    cohesion_df["first_three_share"] = np.where(
        cohesion_df["total_track_metric"] > 0,
        cohesion_df["first_three_track_value"] / cohesion_df["total_track_metric"],
        np.nan,
    )

    top_track_rows = (
        track_df.sort_values(
            [
                "release_group_mbid",
                "tmdb_id",
                "metric_raw",
                "track_number",
            ],
            ascending=[True, True, False, True],
        )
        .drop_duplicates(
            subset=["release_group_mbid", "tmdb_id"],
            keep="first",
        )[[
            "release_group_mbid",
            "tmdb_id",
            "track_title",
            "track_number",
            "metric_raw",
            "track_share",
        ]]
        .rename(
            columns={
                "track_title": "top_track_title",
                "track_number": "top_track_position",
                "metric_raw": "top_track_value_detail",
                "track_share": "top_track_share_detail",
            }
        )
        .copy()
    )

    cohesion_df = cohesion_df.merge(
        top_track_rows,
        on=["release_group_mbid", "tmdb_id"],
        how="left",
        validate="1:1",
    )

    return cohesion_df

def plot_album_cohesion_ranking(
    cohesion_df: pd.DataFrame,
    cohesion_metric: str,
) -> alt.Chart:
    """
    Plot a horizontal ranking of albums by cohesion metric.

    Args:
        cohesion_df: Album-level cohesion dataframe.
        cohesion_metric: Selected album cohesion ranking metric.

    Returns:
        alt.Chart: Altair horizontal bar chart.
    """
    plot_df = cohesion_df.copy().sort_values(
        cohesion_metric,
        ascending=False,
    )

    plot_df["album_label"] = (
        plot_df["film_title"].fillna("")
        + " — "
        + plot_df["album_title"].fillna("")
    )

    chart = alt.Chart(plot_df).mark_bar().encode(
        y=alt.Y(
            "album_label:N",
            sort="-x",
            title=None,
        ),
        x=alt.X(
            f"{cohesion_metric}:Q",
            title=get_display_label(cohesion_metric),
        ),
        tooltip=[
            alt.Tooltip("film_title:N", title="Film"),
            alt.Tooltip("album_title:N", title="Album"),
            alt.Tooltip("composer_primary_clean:N", title="Composer"),
            alt.Tooltip("top_track_title:N", title="Top track"),
            alt.Tooltip("top_track_position:Q", title="Top track position"),
            alt.Tooltip("track_count_observed:Q", title="Observed tracks"),
            alt.Tooltip("top_track_share:Q", title="Top track share", format=".3f"),
            alt.Tooltip(
                "first_three_share:Q",
                title="First 3 track share",
                format=".3f",
            ),
            alt.Tooltip(
                "top_track_to_median_ratio:Q",
                title="Top/median ratio",
                format=".2f",
            ),
            alt.Tooltip(
                "track_metric_std_dev:Q",
                title="Std dev",
                format=".2f",
            ),
            alt.Tooltip(
                "top_track_value:Q",
                title="Top track value",
                format=".2f",
            ),
            alt.Tooltip(
                "median_track_value:Q",
                title="Median track value",
                format=".2f",
            ),
            alt.Tooltip(
                "total_track_metric:Q",
                title="Total track metric",
                format=".2f",
            ),
        ],
    ).properties(
        height=max(320, min(36 * len(plot_df), 700)),
    )

    return chart

def build_album_drilldown_options(
    track_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a unique album list for the drilldown selector.

    Args:
        track_df: Filtered track dataframe.

    Returns:
        pd.DataFrame: One row per album with a readable display label.
    """
    option_cols = [
        "release_group_mbid",
        "tmdb_id",
        "film_title",
        "album_title",
        "composer_primary_clean",
    ]

    options_df = (
        track_df[option_cols]
        .drop_duplicates()
        .copy()
    )

    options_df["album_label"] = (
        options_df["film_title"].fillna("Unknown film")
        + " — "
        + options_df["album_title"].fillna("Unknown album")
    )

    return options_df.sort_values("album_label").reset_index(drop=True)

def build_album_drilldown_df(
    track_df: pd.DataFrame,
    selected_album_label: str,
) -> pd.DataFrame:
    """
    Build the selected album's track-by-track drilldown dataframe.

    Args:
        track_df: Filtered track dataframe with metric columns already added.
        selected_album_label: Chosen album display label.

    Returns:
        pd.DataFrame: Track rows for the selected album only.
    """
    drilldown_df = track_df.copy()
    drilldown_df["album_label"] = (
        drilldown_df["film_title"].fillna("Unknown film")
        + " — "
        + drilldown_df["album_title"].fillna("Unknown album")
    )

    drilldown_df = drilldown_df[
        drilldown_df["album_label"] == selected_album_label
    ].copy()

    drilldown_df = drilldown_df.sort_values("track_number").reset_index(drop=True)

    return drilldown_df

def plot_album_drilldown(
    drilldown_df: pd.DataFrame,
    metric_label: str,
) -> alt.Chart:
    """
    Plot track-by-track performance for one selected soundtrack.

    Args:
        drilldown_df: Track rows for one selected album.
        metric_label: Display label for the selected metric.

    Returns:
        alt.Chart: Altair bar chart for the album drilldown.
    """
    chart = alt.Chart(drilldown_df).mark_bar().encode(
        x=alt.X("track_number:Q", title="Track position"),
        y=alt.Y("analysis_value_display:Q", title=metric_label),
        tooltip=[
            alt.Tooltip("film_title:N", title="Film"),
            alt.Tooltip("album_title:N", title="Album"),
            alt.Tooltip("track_title:N", title="Track"),
            alt.Tooltip("track_number:Q", title="Track position"),
            alt.Tooltip("metric_raw:Q", title="Selected metric (raw)", format=".2f"),
            alt.Tooltip("track_share:Q", title="Track share", format=".3f"),
            alt.Tooltip("analysis_value_display:Q", title=metric_label, format=".3f"),
            alt.Tooltip("track_count_observed:Q", title="Observed tracks"),
        ],
    ).properties(
        height=360,
    )

    return chart

def build_track_structure_scope_caption(
    global_controls: dict,
    track_controls: dict,
    filtered_tracks: pd.DataFrame,
    album_options_df: pd.DataFrame,
) -> str:
    """
    Build a short caption describing the current analysis scope.

    Args:
        global_controls: Shared global filter values.
        track_controls: Track Structure control values.
        filtered_tracks: Filtered track dataframe.
        album_options_df: Unique visible albums for drilldown.

    Returns:
        str: Human-readable scope caption.
    """
    metric_phrase = get_display_label(track_controls["metric"])
    if track_controls["transform_y"] == "Log1p":
        metric_phrase = f"{metric_phrase} (log1p)"

    parts = [
        f"showing {len(filtered_tracks):,} visible tracks across {len(album_options_df):,} albums",
        f"metric: {metric_phrase}",
        f"positions 1–{track_controls['max_track_position']}",
        f"minimum {track_controls['min_tracks_per_album']} observed tracks per album",
    ]

    if track_controls.get("selected_composers"):
        parts.append(
            f"{len(track_controls['selected_composers'])} selected composers"
        )

    year_range = global_controls.get("year_range")
    if year_range:
        parts.append(f"film years {year_range[0]}–{year_range[1]}")

    return "Current scope: " + "; ".join(parts) + "."

def build_track_structure_insight_summary(
    summary_df: pd.DataFrame,
    cohesion_ranked_df: pd.DataFrame,
    track_controls: dict,
    filtered_tracks: pd.DataFrame,
) -> dict[str, str]:
    """
    Build reactive top-row insight cards for the Track Structure Explorer.

    This helper keeps the top-row metrics aligned with the current visible
    scope of the page. The cards summarize three complementary ideas:

    1. Strongest visible track position
       Uses the currently selected summary statistic (median by default,
       or mean if chosen) to identify which track number performs best
       across the visible filtered set.

    2. Front-loading
       Summarizes how much of an album's visible performance is captured
       by its first three tracks. This is computed album by album using
       track_share, then summarized across the visible albums using the
       median to reduce sensitivity to outliers.

    3. Cohesion leader
       Identifies the top visible album under the currently selected
       cohesion ranking metric.

    Args:
        summary_df: Track-position summary dataframe used by the main
            position chart.
        cohesion_ranked_df: Visible ranked cohesion dataframe after sorting
            and Top-N truncation.
        track_controls: Page control values.
        filtered_tracks: Filtered track dataframe with track_share already
            added.

    Returns:
        dict[str, str]: Titles, values, and captions for the three insight
        cards rendered at the top of the page.
    """
    if summary_df.empty:
        return {
            "card1_title": "Strongest Visible Position",
            "card1_value": "None",
            "card1_caption": "No track-position summary is available.",
            "card2_title": "Front-loading",
            "card2_value": "None",
            "card2_caption": "No front-loading summary is available.",
            "card3_title": "Cohesion Leader",
            "card3_value": "None",
            "card3_caption": "No cohesion ranking is available.",
        }

    if track_controls["summary_stat"] == "Mean":
        stat_col = "mean_value"
        stat_label = "mean"
    else:
        stat_col = "median_value"
        stat_label = "median"

    strongest_row = summary_df.sort_values(
        [stat_col, "track_number"],
        ascending=[False, True],
    ).iloc[0]

    frontload_base_df = filtered_tracks.copy()
    frontload_base_df["first_three_track_share_component"] = np.where(
        frontload_base_df["track_number"] <= 3,
        frontload_base_df["track_share"],
        0.0,
    )

    album_frontload_df = (
        frontload_base_df.groupby(
            ["release_group_mbid", "tmdb_id"],
            as_index=False,
        )
        .agg(
            total_share=("track_share", "sum"),
            first_three_share=("first_three_track_share_component", "sum"),
        )
    )

    if album_frontload_df.empty:
        frontload_value = "None"
        frontload_caption = "No front-loading summary is available."
    else:
        median_frontload = float(album_frontload_df["first_three_share"].median())
        frontload_value = f"{median_frontload:.1%}"
        frontload_caption = (
            "Across visible albums, the median share captured by the first 3 tracks "
            f"is {median_frontload:.1%}."
        )

    if cohesion_ranked_df.empty:
        cohesion_value = "None"
        cohesion_caption = "No visible album cohesion result remains."
    else:
        cohesion_metric = track_controls["cohesion_metric"]
        top_row = cohesion_ranked_df.iloc[0]
        cohesion_value = (
            f"{top_row['film_title']} — {top_row['album_title']}"
        )
        cohesion_caption = (
            f"Highest visible {get_display_label(cohesion_metric).lower()} "
            f"among the top {len(cohesion_ranked_df)} ranked albums at "
            f"{float(top_row[cohesion_metric]):.3f}."
        )

    strongest_caption = (
        f"Track {int(strongest_row['track_number'])} has the highest visible "
        f"{stat_label} value at {float(strongest_row[stat_col]):.2f}."
    )
    if track_controls["summary_stat"] == "Both":
        strongest_caption += (
            " With Both selected, this card keys off the median line."
        )

    return {
        "card1_title": "Strongest Visible Position",
        "card1_value": f"Track {int(strongest_row['track_number'])}",
        "card1_caption": strongest_caption,
        "card2_title": "Front-loading",
        "card2_value": frontload_value,
        "card2_caption": frontload_caption,
        "card3_title": "Cohesion Leader",
        "card3_value": cohesion_value,
        "card3_caption": cohesion_caption,
    }

def render_track_structure_insight_cards(
    summary_df: pd.DataFrame,
    cohesion_ranked_df: pd.DataFrame,
    track_controls: dict,
    filtered_tracks: pd.DataFrame,
) -> None:
    """
    Render reactive insight cards for the Track Structure Explorer.

    Args:
        summary_df: Track-position summary dataframe.
        cohesion_ranked_df: Visible ranked cohesion dataframe.
        track_controls: Page control values.
        filtered_tracks: Filtered track dataframe with track_share already
            added.
    """
    insights = build_track_structure_insight_summary(
        summary_df=summary_df,
        cohesion_ranked_df=cohesion_ranked_df,
        track_controls=track_controls,
        filtered_tracks=filtered_tracks,
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

def build_position_supporting_insight(
    summary_df: pd.DataFrame,
    track_controls: dict,
) -> str:
    """
    Build a short supporting insight for the position summary chart.

    Args:
        summary_df: Track-position summary dataframe.
        track_controls: Page control values.

    Returns:
        str: Supporting sentence.
    """
    if summary_df.empty:
        return "No position insight is available."

    if track_controls["summary_stat"] == "Mean":
        stat_col = "mean_value"
        stat_label = "mean"
    else:
        stat_col = "median_value"
        stat_label = "median"

    ordered_df = summary_df.sort_values("track_number").reset_index(drop=True)
    strongest_row = ordered_df.sort_values(
        [stat_col, "track_number"],
        ascending=[False, True],
    ).iloc[0]
    weakest_row = ordered_df.sort_values(
        [stat_col, "track_number"],
        ascending=[True, True],
    ).iloc[0]

    sample_note = ""
    if "n_tracks" in ordered_df.columns:
        first_n = int(ordered_df.iloc[0]["n_tracks"])
        last_n = int(ordered_df.iloc[-1]["n_tracks"])
        sample_note = (
            f" Sample sizes range from {first_n:,} tracks at position "
            f"{int(ordered_df.iloc[0]['track_number'])} to {last_n:,} at "
            f"position {int(ordered_df.iloc[-1]['track_number'])}."
        )

    return (
        f"💡 Using the visible {stat_label} line, track "
        f"{int(strongest_row['track_number'])} is strongest "
        f"({float(strongest_row[stat_col]):.2f}) while track "
        f"{int(weakest_row['track_number'])} is weakest "
        f"({float(weakest_row[stat_col]):.2f}).{sample_note}"
    )

def build_top_track_position_insight(
    position_df: pd.DataFrame,
) -> str:
    """
    Build a short supporting insight for the top-track position chart.

    Args:
        position_df: Top-track position summary dataframe.

    Returns:
        str: Supporting sentence.
    """
    if position_df.empty:
        return "No top-track position insight is available."

    strongest_row = position_df.sort_values(
        ["top_track_probability", "track_number"],
        ascending=[False, True],
    ).iloc[0]

    return (
        f"💡 Among albums that visibly include each position, track "
        f"{int(strongest_row['track_number'])} is the most likely place for the "
        f"strongest track to appear "
        f"({float(strongest_row['top_track_probability']):.1%}; "
        f"{int(strongest_row['top_track_count'])} of "
        f"{int(strongest_row['eligible_album_count'])} eligible albums)."
    )

def build_scatter_supporting_insight(
    scatter_df: pd.DataFrame,
    track_controls: dict,
) -> str:
    """
    Build a short supporting insight for the scatter section.

    Args:
        scatter_df: Scatterplot dataframe.
        track_controls: Page control values.

    Returns:
        str: Supporting sentence.
    """
    if scatter_df.empty:
        return "No scatter insight is available."

    corr = scatter_df["track_number"].corr(
        scatter_df["analysis_value_display"]
    )
    if pd.isna(corr):
        return (
            "The scatterplot shows the visible track-level spread directly, "
            "but a simple linear trend is not stable under the current filters."
        )

    if corr <= -0.10:
        direction = "downward"
    elif corr >= 0.10:
        direction = "upward"
    else:
        direction = "mostly flat"

    color_note = ""
    if track_controls.get("selected_composers"):
        color_note = " Colors are locked to composer because a composer subset is selected."
    elif track_controls["color_col"] != "None":
        color_note = (
            f" Colors are grouped by {get_display_label(track_controls['color_col']).lower()}."
        )

    trend_note = ""
    if track_controls["show_trendline"]:
        trend_note = " The fitted line should broadly match this direction."

    return (
        f"💡 The visible point cloud suggests a {direction} relationship between "
        f"track position and displayed performance (Pearson r = {corr:.3f})."
        f"{trend_note}{color_note}"
    )


def get_cohesion_metric_explainer(cohesion_metric: str) -> str:
    """
    Return a short explanation of the selected cohesion metric.

    Args:
        cohesion_metric: Selected cohesion metric column.

    Returns:
        str: Human-readable explanation.
    """
    explainers = {
        "top_track_share": (
            "Top track share measures how much of an album's total visible track performance "
            "comes from its single strongest track."
        ),
        "top_track_to_median_ratio": (
            "Top track to median ratio compares the strongest track with the album's typical "
            "track, highlighting breakout-versus-balanced soundtracks."
        ),
        "track_metric_std_dev": (
            "Track metric standard deviation measures how uneven visible track performance is "
            "within the album overall."
        ),
    }
    return explainers.get(cohesion_metric, get_display_label(cohesion_metric))


def build_cohesion_supporting_insight(
    cohesion_ranked_df: pd.DataFrame,
    cohesion_metric: str,
) -> str:
    """
    Build a short supporting insight for the cohesion ranking chart.

    Args:
        cohesion_ranked_df: Visible ranked cohesion dataframe.
        cohesion_metric: Selected cohesion metric.

    Returns:
        str: Supporting sentence.
    """
    if cohesion_ranked_df.empty:
        return "No cohesion insight is available."

    top_row = cohesion_ranked_df.iloc[0]
    median_value = float(cohesion_ranked_df[cohesion_metric].median())

    return (
        f"💡 Within the visible ranked set, the strongest album on "
        f"{get_display_label(cohesion_metric).lower()} is "
        f"{top_row['film_title']} — {top_row['album_title']} "
        f"({float(top_row[cohesion_metric]):.3f}). The median across the "
        f"currently ranked albums is {median_value:.3f}."
    )


def build_album_drilldown_caption(
    drilldown_df: pd.DataFrame,
    cohesion_metric: str,
) -> str:
    """
    Build a short reactive caption for the selected album drilldown.

    Args:
        drilldown_df: Track rows for the selected album.
        cohesion_metric: Selected cohesion metric.

    Returns:
        str: Supporting sentence.
    """
    if drilldown_df.empty:
        return "No drilldown insight is available."

    top_row = drilldown_df.sort_values(
        ["metric_raw", "track_number"],
        ascending=[False, True],
    ).iloc[0]

    total_metric = float(drilldown_df["metric_raw"].sum())
    top_share = (
        float(top_row["metric_raw"]) / total_metric
        if total_metric > 0 else np.nan
    )

    first_three_total = float(
        drilldown_df.loc[
            drilldown_df["track_number"] <= 3,
            "metric_raw",
        ].sum()
    )
    first_three_share = (
        first_three_total / total_metric
        if total_metric > 0 else np.nan
    )

    cohesion_note = ""
    if cohesion_metric == "top_track_share" and not pd.isna(top_share):
        cohesion_note = (
            f" Its top track accounts for {top_share:.1%} of the album's visible total."
        )

    frontload_note = ""
    if not pd.isna(first_three_share):
        frontload_note = (
            f" The first 3 tracks account for {first_three_share:.1%} of the visible total."
        )

    return (
        f"Top visible track for this soundtrack is '{top_row['track_title']}' "
        f"at position {int(top_row['track_number'])}, with a raw value of "
        f"{float(top_row['metric_raw']):.2f}."
        f"{cohesion_note}{frontload_note}"
    )

def main() -> None:
    """Render the Track Structure Explorer."""
    st.title("Track Structure Explorer")
    st.markdown(
        """
        Analyze how performance unfolds within a soundtrack. This page combines
        track-position summaries, track-level scatter, album cohesion ranking,
        and soundtrack drilldown to show whether albums are front-loaded,
        evenly distributed, or dominated by a standout track.
        """
    )

    track_df = load_track_explorer_data()

    filter_inputs = get_global_filter_inputs(track_df)
    composer_options = get_clean_composer_options(track_df)

    global_controls = get_global_filter_controls(
        min_year=filter_inputs["min_year"],
        max_year=filter_inputs["max_year"],
        film_genre_options=filter_inputs["film_genre_options"],
        album_genre_options=filter_inputs["album_genre_options"],
    )

    track_controls = get_track_structure_controls(
        metric_options=TRACK_METRIC_OPTIONS,
        color_options=SCATTER_COLOR_OPTIONS,
        composer_options=composer_options,
    )

    filtered_tracks = apply_track_structure_filters(
        track_df=track_df,
        global_controls=global_controls,
        track_controls=track_controls,
    )

    if filtered_tracks.empty:
        st.warning(
            "No tracks remain under the current filters. Try broadening the film year "
            "range, reducing the minimum tracks per album, or clearing composer restrictions."
        )
        return

    filtered_tracks = add_display_metric(
        track_df=filtered_tracks,
        metric_col=track_controls["metric"],
        transform_y=track_controls["transform_y"],
    )

    if filtered_tracks.empty:
        st.warning(
            "No tracks remain after applying the selected display transform. "
            "Try switching Transform Y to None or broadening the current filters."
        )
        return

    # Add within-album metric helpers tied to the selected raw metric.
    filtered_tracks = add_metric_structure_fields(filtered_tracks)

    # Add the active plotted metric based on the selected analysis basis.
    filtered_tracks = add_analysis_basis_fields(
        track_df=filtered_tracks,
        analysis_basis=track_controls["analysis_basis"],
        transform_y=track_controls["transform_y"],
    )

    scatter_df = add_scatter_x_position(
        track_df=filtered_tracks,
        apply_jitter=track_controls["apply_jitter"],
        jitter_strength=track_controls["jitter_strength"],
    )

    summary_df = build_position_summary_df(filtered_tracks)

    top_track_position_df = build_top_track_position_df(filtered_tracks)

    cohesion_all_df = build_album_cohesion_df(filtered_tracks)
    cohesion_metric = track_controls["cohesion_metric"]
    cohesion_all_df = cohesion_all_df[
        cohesion_all_df[cohesion_metric].notna()
    ].copy()

    cohesion_ranked_df = cohesion_all_df.sort_values(
        cohesion_metric,
        ascending=False,
    ).head(track_controls["top_n_albums"]).copy()

    album_options_df = build_album_drilldown_options(filtered_tracks)

    metric_label = get_display_label(track_controls["metric"])
    if track_controls["analysis_basis"] == "Within-album share":
        metric_label = "Track share of album total"
    elif track_controls["transform_y"] == "Log1p":
        metric_label = f"{metric_label} (log1p)"

    st.caption(
        build_track_structure_scope_caption(
            global_controls=global_controls,
            track_controls=track_controls,
            filtered_tracks=filtered_tracks,
            album_options_df=album_options_df,
        )
    )

    render_track_structure_insight_cards(
        summary_df=summary_df,
        cohesion_ranked_df=cohesion_ranked_df,
        track_controls=track_controls,
        filtered_tracks=filtered_tracks,
    )

    st.subheader("Popularity by Track Position")
    st.altair_chart(
        plot_track_position_summary(
            summary_df=summary_df,
            summary_stat=track_controls["summary_stat"],
            show_ribbon=track_controls["show_ribbon"],
            metric_label=metric_label,
        ),
        use_container_width=True,
    )
    st.caption(
        build_position_supporting_insight(
            summary_df=summary_df,
            track_controls=track_controls,
        )
    )

    st.subheader("Where the Strongest Track Appears")
    st.markdown(
        """
        This view shows how often the strongest visible track in a soundtrack
        occurs at each track position, adjusting for how many albums are long
        enough to include that position.
        """
    )
    st.altair_chart(
        plot_top_track_position_chart(top_track_position_df),
        use_container_width=True,
    )
    st.caption(
        build_top_track_position_insight(top_track_position_df)
    )

    if track_controls["show_summary_table"]:
        st.dataframe(summary_df, use_container_width=True)

    if track_controls["selected_composers"]:
        scatter_color_col = "composer_primary_clean"
    else:
        scatter_color_col = track_controls["color_col"]

    if track_controls["show_scatter"]:
        if track_controls["analysis_basis"] == "Within-album share":
            scatter_title = "Track Position vs Within-Album Share"
        else:
            scatter_title = "Track Position vs Performance"

        st.subheader(scatter_title)
        st.altair_chart(
            plot_track_position_scatter(
                track_df=scatter_df,
                color_col=scatter_color_col,
                show_trendline=track_controls["show_trendline"],
                metric_label=metric_label,
            ),
            use_container_width=True,
        )
        st.caption(
            build_scatter_supporting_insight(
                scatter_df=scatter_df,
                track_controls=track_controls,
            )
        )

    st.subheader("Album Cohesion Ranking")
    st.markdown(
        """
        Rank soundtracks by how concentrated or uneven track performance is
        within each album.
        """
    )
    st.caption(
        get_cohesion_metric_explainer(cohesion_metric)
    )

    st.altair_chart(
        plot_album_cohesion_ranking(
            cohesion_df=cohesion_ranked_df,
            cohesion_metric=cohesion_metric,
        ),
        use_container_width=True,
    )
    st.caption(
        build_cohesion_supporting_insight(
            cohesion_ranked_df=cohesion_ranked_df,
            cohesion_metric=cohesion_metric,
        )
    )

    if track_controls["show_cohesion_table"]:
        st.dataframe(cohesion_ranked_df, use_container_width=True)

    st.subheader("Inspect an Album")
    st.markdown(
        """
        Select one soundtrack to inspect its track-by-track performance profile.
        """
    )

    if not album_options_df.empty:
        selected_album_label = st.selectbox(
            "Album",
            options=album_options_df["album_label"].tolist(),
            index=0,
        )

        drilldown_df = build_album_drilldown_df(
            track_df=filtered_tracks,
            selected_album_label=selected_album_label,
        )

        st.altair_chart(
            plot_album_drilldown(
                drilldown_df=drilldown_df,
                metric_label=metric_label,
            ),
            use_container_width=True,
        )
        st.caption(
            build_album_drilldown_caption(
                drilldown_df=drilldown_df,
                cohesion_metric=cohesion_metric,
            )
        )

        if track_controls["show_album_track_table"]:
            display_cols = [
                col for col in [
                    "film_title",
                    "album_title",
                    "track_number",
                    "track_title",
                    "metric_raw",
                    "metric_display",
                    "analysis_value_display",
                    "track_share",
                    "track_rank_within_album",
                    "is_top_track",
                    "track_count_observed",
                ]
                if col in drilldown_df.columns
            ]
            st.dataframe(
                drilldown_df[display_cols],
                use_container_width=True,
            )
    else:
        st.info("No albums available for drilldown under the current filters.")

    if track_controls["show_track_table"]:
        st.subheader("Filtered Track Data")
        display_cols = [
            col for col in [
                "film_title",
                "album_title",
                "track_number",
                "track_title",
                "track_count_observed",
                "lfm_track_listeners",
                "lfm_track_playcount",
                "spotify_popularity",
                "track_share",
                "track_rank_within_album",
                "is_top_track",
                "relative_track_position",
            ]
            if col in filtered_tracks.columns
        ]
        st.dataframe(
            filtered_tracks[display_cols],
            use_container_width=True,
        )

if __name__ == "__main__":
    main()
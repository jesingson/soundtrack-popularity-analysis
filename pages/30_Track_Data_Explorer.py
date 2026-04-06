from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from app.app_controls import (
    get_global_filter_controls,
    get_track_data_explorer_controls,
)
from app.app_data import load_track_data_explorer_data
from app.data_filters import filter_dataset
from app.explorer_shared import (
    add_film_year_bucket,
    add_standard_multivalue_groups,
    get_clean_composer_options,
    get_global_filter_inputs,
    rename_and_dedupe_for_display,
    select_unique_existing_columns,
)
from app.ui import apply_app_styles, get_display_label


TRACK_METRIC_OPTIONS = [
    # Popularity
    "lfm_track_listeners",
    "lfm_track_playcount",
    "spotify_popularity",

    # Structure
    "track_number",
    "relative_track_position",

    # Audio
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

HEAVY_TAILED_TRACK_METRICS = {
    "lfm_track_listeners",
    "lfm_track_playcount",
}

GROUP_OPTIONS = [
    "None",
    "album_genre_group",
    "film_genre_group",
    "track_position_bucket",
    "composer_primary_clean",
    "award_category",
    "film_year_bucket",
]

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
    "lfm_album_listeners",
    "lfm_album_playcount",
    "lfm_track_listeners",
    "lfm_track_playcount",
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


def add_track_data_display_fields(track_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add grouped display fields used across the Track Data Explorer.
    """
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


def filter_track_data_explorer_df(
    track_df: pd.DataFrame,
    global_controls: dict,
    controls: dict,
) -> pd.DataFrame:
    """
    Apply shared global filters plus Page 30-specific filters.
    """
    merged_controls = {
        **global_controls,
        "selected_composers": controls.get("selected_composers", []),
        "search_text": controls.get("search_text", ""),
    }

    filtered = filter_dataset(track_df, merged_controls).copy()

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
            col for col in [
                "energy",
                "danceability",
                "happiness",
                "instrumentalness",
                "tempo",
                "loudness",
            ]
            if col in filtered.columns
        ]
        if required_audio_cols:
            filtered = filtered.dropna(subset=required_audio_cols).copy()

    return filtered


def prepare_metric_plot_df(
    track_df: pd.DataFrame,
    metric: str,
    use_log: bool,
    group_var: str,
    selected_groups: list[str],
    top_n: int | None,
) -> pd.DataFrame:
    """
    Prepare the main selected-metric dataframe for plotting.
    """
    required_cols = [metric]
    if group_var != "None":
        required_cols.append(group_var)

    plot_df = track_df[[col for col in required_cols if col in track_df.columns]].copy()
    plot_df = plot_df.dropna(subset=[metric]).copy()

    if group_var != "None":
        plot_df = plot_df.dropna(subset=[group_var]).copy()
        plot_df["group"] = plot_df[group_var].astype(str)

        if selected_groups:
            plot_df = plot_df[plot_df["group"].isin(selected_groups)].copy()
        elif top_n is not None:
            top_groups = (
                plot_df["group"]
                .value_counts()
                .head(top_n)
                .index
                .tolist()
            )
            plot_df = plot_df[plot_df["group"].isin(top_groups)].copy()

    if use_log:
        plot_df = plot_df[plot_df[metric] > 0].copy()
        plot_df["plot_value"] = np.log10(plot_df[metric])
    else:
        plot_df["plot_value"] = plot_df[metric]

    plot_df = plot_df.dropna(subset=["plot_value"]).copy()
    return plot_df


def build_track_filter_context_caption(
    global_controls: dict,
    controls: dict,
) -> str:
    """
    Build a short caption describing the active Track Data Explorer filters.
    """
    parts = []

    year_min, year_max = global_controls["year_range"]
    parts.append(f"Film years {year_min}–{year_max}")

    if global_controls.get("selected_film_genres"):
        shown = ", ".join(global_controls["selected_film_genres"][:5])
        if len(global_controls["selected_film_genres"]) > 5:
            shown += ", ..."
        parts.append(f"Film genres: {shown}")

    if global_controls.get("selected_album_genres"):
        shown = ", ".join(global_controls["selected_album_genres"][:5])
        if len(global_controls["selected_album_genres"]) > 5:
            shown += ", ..."
        parts.append(f"Album genres: {shown}")

    if controls["selected_composers"]:
        shown = ", ".join(controls["selected_composers"][:3])
        if len(controls["selected_composers"]) > 3:
            shown += ", ..."
        parts.append(f"Composers: {shown}")

    if controls["min_album_listeners"] > 0:
        parts.append(f"Min album listeners: {controls['min_album_listeners']:,}")

    parts.append(f"Max track position: {controls['max_track_position']}")

    if controls["audio_only"]:
        parts.append("Only tracks with core audio features")

    return " | ".join(parts) if parts else "Showing the full visible track dataset."


def build_metric_view_context_caption(
    metric: str,
    use_log: bool,
    group_var: str,
    selected_groups: list[str],
    top_n: int | None,
) -> str:
    """
    Build a short caption for the main metric panel.
    """
    metric_label = get_display_label(metric)

    if group_var == "None":
        if use_log:
            return f"Showing the distribution of {metric_label} on the log10 scale across visible tracks."
        return f"Showing the distribution of {metric_label} across visible tracks."

    group_label = get_display_label(group_var).lower()

    if selected_groups:
        if len(selected_groups) <= 5:
            group_scope = f"the selected {group_label} values ({', '.join(selected_groups)})"
        else:
            shown = ", ".join(selected_groups[:5])
            group_scope = f"the selected {group_label} values ({shown}, ...)"
    else:
        group_scope = f"the top {top_n} visible {group_label} groups by track count"

    if use_log:
        return f"Comparing {metric_label} on the log10 scale across {group_scope}."
    return f"Comparing {metric_label} across {group_scope}."


def build_track_data_insight_summary(
    filtered_df: pd.DataFrame,
    plot_df: pd.DataFrame,
    metric: str,
    use_log: bool,
) -> dict[str, str]:
    """
    Build top-row insight cards for the Track Data Explorer.
    """
    if filtered_df.empty or plot_df.empty:
        return {
            "card1_title": "Tracks in View",
            "card1_value": "0",
            "card1_caption": "No tracks remain under the current filters.",
            "card2_title": "Most Common Position Bucket",
            "card2_value": "NA",
            "card2_caption": "No position-bucket insight is available.",
            "card3_title": "Top Track Concentration",
            "card3_value": "NA",
            "card3_caption": "Not available under current filters.",
        }

    metric_label = get_display_label(metric)
    median_value = float(plot_df["plot_value"].median())
    suffix = " (log10)" if use_log else ""

    position_counts = (
        filtered_df["track_position_bucket"]
        .dropna()
        .astype(str)
        .value_counts()
        if "track_position_bucket" in filtered_df.columns
        else pd.Series(dtype=int)
    )

    if not position_counts.empty:
        bucket_value = str(position_counts.index[0])
        bucket_caption = (
            f"{int(position_counts.iloc[0]):,} visible tracks fall in this sequencing bucket."
        )
    else:
        bucket_value = "NA"
        bucket_caption = "No position-bucket insight is available."

    if {"lfm_track_listeners", "track_count_observed"}.issubset(filtered_df.columns):
        album_track_summary = (
            filtered_df.groupby(["release_group_mbid", "tmdb_id"], as_index=False)
            .agg(
                top_track_listeners=("lfm_track_listeners", "max"),
                mean_track_listeners=("lfm_track_listeners", "mean"),
            )
        )
        album_track_summary["top_to_mean_track_listeners"] = (
                album_track_summary["top_track_listeners"]
                / album_track_summary["mean_track_listeners"]
        )
        concentration_value = (
            float(album_track_summary["top_to_mean_track_listeners"].median())
            if not album_track_summary.empty
            else np.nan
        )
        concentration_display = (
            f"{concentration_value:.2f}x"
            if pd.notna(concentration_value)
            else "NA"
        )
        concentration_caption = (
            "Median top-track / mean-track listeners across visible albums."
        )
    else:
        concentration_display = "NA"
        concentration_caption = "Not available under current filters."

    return {
        "card1_title": "Tracks in View",
        "card1_value": f"{len(filtered_df):,}",
        "card1_caption": "Visible tracks after current global and track-specific filters.",
        "card2_title": "Most Common Position Bucket",
        "card2_value": bucket_value,
        "card2_caption": bucket_caption,
        "card3_title": "Top Track Concentration",
        "card3_value": concentration_display,
        "card3_caption": concentration_caption,
    }


def build_position_bucket_supporting_insight(filtered_df: pd.DataFrame) -> str:
    """
    Build a short supporting insight for the track-position composition chart.
    """
    if "track_position_bucket" not in filtered_df.columns or filtered_df.empty:
        return "No track-position composition insight is available."

    counts = (
        filtered_df["track_position_bucket"]
        .dropna()
        .astype(str)
        .value_counts()
    )

    if counts.empty:
        return "No track-position composition insight is available."

    top_bucket = counts.index[0]
    top_count = int(counts.iloc[0])
    top_share = top_count / len(filtered_df)

    return (
        f"💡 The visible track set is led by the {top_bucket.lower()} segment, "
        f"which contains {top_count:,} tracks ({top_share:.1%} of tracks in view)."
    )


def build_group_composition_supporting_insight(
        filtered_df: pd.DataFrame,
        group_col: str,
) -> str:
    """
    Build a short supporting insight for the secondary composition chart.
    """
    if group_col not in filtered_df.columns or filtered_df.empty:
        return "No composition insight is available."

    counts = (
        filtered_df[group_col]
        .fillna("Unknown")
        .astype(str)
        .value_counts()
    )

    if counts.empty:
        return "No composition insight is available."

    top_group = counts.index[0]
    top_count = int(counts.iloc[0])
    top_share = top_count / len(filtered_df)

    return (
        f"💡 {top_group} is the largest visible {get_display_label(group_col).lower()} "
        f"group, accounting for {top_count:,} tracks ({top_share:.1%} of tracks in view)."
    )

def render_track_data_insight_cards(
    filtered_df: pd.DataFrame,
    plot_df: pd.DataFrame,
    metric: str,
    use_log: bool,
) -> None:
    insights = build_track_data_insight_summary(
        filtered_df=filtered_df,
        plot_df=plot_df,
        metric=metric,
        use_log=use_log,
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


def create_track_metric_histogram(
    plot_df: pd.DataFrame,
    metric: str,
    use_log: bool,
    bins: int,
    group_var: str,
) -> alt.Chart:
    """
    Create the main histogram for the selected track metric.
    """
    x_title = (
        f"log10({get_display_label(metric)})"
        if use_log
        else get_display_label(metric)
    )

    if group_var != "None":
        chart = (
            alt.Chart(plot_df)
            .mark_bar(opacity=0.45)
            .encode(
                x=alt.X("plot_value:Q", bin=alt.Bin(maxbins=bins), title=x_title),
                y=alt.Y("count():Q", title="Track Count"),
                color=alt.Color("group:N", title=get_display_label(group_var)),
                tooltip=[
                    alt.Tooltip("group:N", title=get_display_label(group_var)),
                    alt.Tooltip("count():Q", title="Tracks"),
                ],
            )
        )
    else:
        chart = (
            alt.Chart(plot_df)
            .mark_bar(opacity=0.85)
            .encode(
                x=alt.X("plot_value:Q", bin=alt.Bin(maxbins=bins), title=x_title),
                y=alt.Y("count():Q", title="Track Count"),
                tooltip=[alt.Tooltip("count():Q", title="Tracks")],
            )
        )

    subtitle = "Histogram of visible track-level values"
    if use_log:
        subtitle += " on log10 scale"
    if group_var != "None":
        subtitle += f" | Grouped by {get_display_label(group_var)}"

    return chart.properties(
        height=420,
        title={
            "text": f"Distribution of {get_display_label(metric)}",
            "subtitle": [subtitle],
        },
    )


def build_position_bucket_chart(filtered_df: pd.DataFrame) -> alt.Chart:
    """
    Build a simple track-count chart by track position bucket.
    """
    if "track_position_bucket" not in filtered_df.columns:
        return alt.Chart(pd.DataFrame({"track_position_bucket": [], "count": []})).mark_bar()

    order = ["Opening", "Early", "Middle", "Late", "Closing"]
    plot_df = (
        filtered_df.groupby("track_position_bucket", as_index=False)
        .size()
        .rename(columns={"size": "track_count"})
    )

    return (
        alt.Chart(plot_df)
        .mark_bar()
        .encode(
            x=alt.X("track_position_bucket:N", sort=order, title="Track Position Bucket"),
            y=alt.Y("track_count:Q", title="Track Count"),
            tooltip=[
                alt.Tooltip("track_position_bucket:N", title="Bucket"),
                alt.Tooltip("track_count:Q", title="Tracks"),
            ],
        )
        .properties(
            height=300,
            title="Track Counts by Position Bucket",
        )
    )


def build_group_composition_chart(
    filtered_df: pd.DataFrame,
    group_col: str = "album_genre_group",
    top_n: int = 10,
) -> alt.Chart:
    """
    Build a simple composition chart for a selected grouping field.
    """
    if group_col not in filtered_df.columns:
        return alt.Chart(pd.DataFrame({group_col: [], "track_count": []})).mark_bar()

    plot_df = (
        filtered_df[group_col]
        .fillna("Unknown")
        .astype(str)
        .value_counts()
        .head(top_n)
        .reset_index()
    )
    plot_df.columns = [group_col, "track_count"]

    return (
        alt.Chart(plot_df)
        .mark_bar()
        .encode(
            y=alt.Y(f"{group_col}:N", sort="-x", title=get_display_label(group_col)),
            x=alt.X("track_count:Q", title="Track Count"),
            tooltip=[
                alt.Tooltip(f"{group_col}:N", title=get_display_label(group_col)),
                alt.Tooltip("track_count:Q", title="Tracks"),
            ],
        )
        .properties(
            height=max(280, min(36 * len(plot_df), 420)),
            title=f"Track Counts by {get_display_label(group_col)}",
        )
    )


def build_track_data_supporting_insight(
    plot_df: pd.DataFrame,
    metric: str,
    use_log: bool,
    group_var: str,
) -> str:
    """
    Build a short supporting narrative for the selected metric.
    """
    if plot_df.empty:
        return "No visible pattern remains to summarize."

    metric_label = get_display_label(metric)
    values = plot_df["plot_value"].dropna().astype(float)

    median = float(values.median())
    mean = float(values.mean())
    p10 = float(values.quantile(0.10))
    p90 = float(values.quantile(0.90))

    if use_log:
        shape = "meaningfully right-skewed" if (mean - median) >= 0.20 else "fairly compact"
        scale_note = (
            " Because log scaling compresses large values, upper-tail differences "
            "look less extreme here than they do on the raw scale."
        )
    else:
        ratio = (p90 / median) if median not in [0, np.nan] and median != 0 else np.nan
        shape = "strongly right-skewed" if pd.notna(ratio) and ratio >= 3 else "moderately spread"
        scale_note = ""

    if group_var == "None":
        return (
            f"💡 {metric_label} appears {shape} across visible tracks. "
            f"The median plotted value is {median:,.2f}, while the middle-80% range runs "
            f"from {p10:,.2f} to {p90:,.2f}."
            f"{scale_note}"
        )

    group_summary = (
        plot_df.groupby("group")["plot_value"]
        .agg(["count", "median"])
        .reset_index()
        .sort_values(["median", "count", "group"], ascending=[False, False, True])
        .reset_index(drop=True)
    )

    top_group = group_summary.iloc[0]
    return (
        f"💡 Across visible groups, {top_group['group']} has the highest median plotted "
        f"{metric_label.lower()} at {float(top_group['median']):,.2f}. "
        f"The current view remains {shape} overall."
        f"{scale_note}"
    )


def get_safe_display_columns(df: pd.DataFrame) -> list[str]:
    """
    Return the subset of preferred display columns present in the dataframe.
    """
    return [col for col in DISPLAY_COLUMNS if col in df.columns]


def main() -> None:
    """Render the Track Data Explorer page."""
    st.set_page_config(
        page_title="Track Data Explorer",
        layout="wide",
    )
    apply_app_styles()

    st.title("Track Data Explorer")
    st.write(
        """
        Explore the track-level soundtrack dataset directly. Each row represents
        one track, enriched with album context, track-position helpers, and
        cleaned audio features where available.
        """
    )
    st.caption(
        "This page helps you understand how track-level performance and audio "
        "characteristics are distributed before analyzing album-level outcomes."
    )

    track_df = load_track_data_explorer_data()
    track_df = add_track_data_display_fields(track_df)

    filter_inputs = get_global_filter_inputs(track_df)
    composer_options = get_clean_composer_options(track_df)

    metric_options = [
        col for col in TRACK_METRIC_OPTIONS
        if col in track_df.columns
    ]
    group_options = [
        col for col in GROUP_OPTIONS
        if col == "None" or col in track_df.columns
    ]

    global_controls = get_global_filter_controls(
        min_year=filter_inputs["min_year"],
        max_year=filter_inputs["max_year"],
        film_genre_options=filter_inputs["film_genre_options"],
        album_genre_options=filter_inputs["album_genre_options"],
    )

    group_value_options_map: dict[str, list[str]] = {}

    for group_col in group_options:
        if group_col == "None" or group_col not in track_df.columns:
            continue

        values = (
            track_df[group_col]
            .dropna()
            .astype(str)
            .str.strip()
            .replace("", pd.NA)
            .dropna()
            .unique()
            .tolist()
        )
        group_value_options_map[group_col] = sorted(values)

    controls = get_track_data_explorer_controls(
        metric_options=metric_options,
        group_options=group_options,
        group_value_options_map=group_value_options_map,
        composer_options=composer_options,
    )

    filtered_df = filter_track_data_explorer_df(
        track_df=track_df,
        global_controls=global_controls,
        controls=controls,
    )

    if filtered_df.empty:
        st.warning("No tracks remain under the current filters.")
        return

    st.markdown("**Filter Context**")
    st.caption(
        build_track_filter_context_caption(
            global_controls=global_controls,
            controls=controls,
        )
    )

    # Dataset profile cards
    track_count = len(filtered_df)
    album_count = (
        filtered_df[["release_group_mbid", "tmdb_id"]]
        .drop_duplicates()
        .shape[0]
        if {"release_group_mbid", "tmdb_id"}.issubset(filtered_df.columns)
        else 0
    )
    film_count = filtered_df["film_title"].nunique() if "film_title" in filtered_df.columns else 0
    audio_share = (
        filtered_df["energy"].notna().mean()
        if "energy" in filtered_df.columns
        else 0.0
    )

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    metric_col1.metric("Tracks", f"{track_count:,}")
    metric_col2.metric("Albums", f"{album_count:,}")
    metric_col3.metric("Films", f"{film_count:,}")
    metric_col4.metric("Tracks with Audio Data", f"{audio_share:.1%}")

    # Main metric panel
    metric = controls["metric"]
    use_log = controls["use_log"]
    NON_LOG_SAFE_METRICS = {
        "loudness",
        "relative_track_position",
        "track_number",
        "key",
        "mode",
        "camelot_number",
        "camelot_mode",
    }

    if metric in NON_LOG_SAFE_METRICS:
        use_log = False

    if controls["use_log"] and metric in NON_LOG_SAFE_METRICS:
        st.caption(
            f"Log scale is not applied to {get_display_label(metric)} because this metric includes non-positive or bounded values."
        )

    if metric in HEAVY_TAILED_TRACK_METRICS and not controls.get("log_user_explicitly_set", False):
        use_log = True

    group_var = controls["group_var"]
    selected_groups = controls["selected_groups"]
    top_n = controls["top_n"]

    plot_df = prepare_metric_plot_df(
        track_df=filtered_df,
        metric=metric,
        use_log=use_log,
        group_var=group_var,
        selected_groups=selected_groups,
        top_n=top_n,
    )

    if plot_df.empty:
        st.warning("No valid rows remain for the selected metric under the current settings.")
        return

    render_track_data_insight_cards(
        filtered_df=filtered_df,
        plot_df=plot_df,
        metric=metric,
        use_log=use_log,
    )

    st.markdown("### Distribution")
    st.caption(
        build_metric_view_context_caption(
            metric=metric,
            use_log=use_log,
            group_var=group_var,
            selected_groups=selected_groups,
            top_n=top_n,
        )
    )
    dynamic_bins = int(np.sqrt(len(plot_df)))
    dynamic_bins = max(10, min(dynamic_bins, 60))

    hist_chart = create_track_metric_histogram(
        plot_df=plot_df,
        metric=metric,
        use_log=use_log,
        bins=dynamic_bins,
        group_var=group_var,
    )
    st.altair_chart(hist_chart, width="stretch")
    st.caption(
        build_track_data_supporting_insight(
            plot_df=plot_df,
            metric=metric,
            use_log=use_log,
            group_var=group_var,
        )
    )

    # Composition snapshot
    st.subheader("Track Composition")
    st.caption(
        "These charts summarize how the currently visible tracks are distributed "
        "by sequencing and album context."
    )
    comp_col1, comp_col2 = st.columns(2)

    with comp_col1:
        position_chart = build_position_bucket_chart(filtered_df)
        st.altair_chart(position_chart, width="stretch")
        st.caption(
            build_position_bucket_supporting_insight(filtered_df)
        )

    with comp_col2:
        composition_group_col = "composer_primary_clean"
        group_chart = build_group_composition_chart(
            filtered_df=filtered_df,
            group_col=composition_group_col,
            top_n=10,
        )
        st.altair_chart(group_chart, width="stretch")
        st.caption(
            build_group_composition_supporting_insight(
                filtered_df=filtered_df,
                group_col=composition_group_col,
            )
        )

    # Table controls
    st.markdown("### Table Columns")

    if "lfm_track_listeners" in filtered_df.columns:
        filtered_df = filtered_df.sort_values(
            by="lfm_track_listeners",
            ascending=False,
            na_position="last",
        ).copy()

    default_display_columns = get_safe_display_columns(filtered_df)
    default_selected_columns = [
        col for col in [
            "film_title",
            "album_title",
            "track_title",
            "track_number",
            "composer_primary_clean",
            "label_names",
            "film_year",
            "album_genres_display",
            "film_genres",
            "lfm_track_listeners",
            "lfm_track_playcount",
            "spotify_popularity",
            "energy",
            "danceability",
            "happiness",
            "instrumentalness",
            "tempo",
            "loudness",
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

    table_cols = select_unique_existing_columns(filtered_df, selected_columns)
    display_df = rename_and_dedupe_for_display(filtered_df[table_cols])

    st.markdown(f"### Filtered Tracks ({len(display_df):,} rows)")
    st.caption(
        "Each row represents one track after the current filters. "
        "Use the column selector to tailor the table, or download the filtered track dataset as CSV."
    )

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


if __name__ == "__main__":
    main()
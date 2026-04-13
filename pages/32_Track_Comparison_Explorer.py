from __future__ import annotations

import altair as alt
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import streamlit as st

from app.app_controls import (
    get_global_filter_controls,
    get_track_comparison_controls,
)
from app.app_data import load_track_data_explorer_data
from app.data_filters import filter_dataset
from app.explorer_shared import (
    add_standard_multivalue_groups,
    add_film_year_bucket,
    add_key_mode_label,
    get_global_filter_inputs,
    get_track_numeric_options,
    get_track_group_options,
    get_clean_composer_options,
    rename_and_dedupe_for_display,
    get_track_page_display_label,
)
from app.ui import apply_app_styles


DETAIL_COLUMNS = [
    "film_title",
    "album_title",
    "track_title",
    "track_number",
    "track_position_bucket",
    "track_type",
    "composer_primary_clean",
    "film_year",
    "film_genres",
    "album_genres_display",
    "award_category",
    "mode_label",
    "key_label",
    "key_mode_label",
    "lfm_track_listeners",
    "lfm_track_playcount",
    "track_share_of_album_listeners",
    "track_share_of_album_playcount",
    "spotify_popularity",
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


def add_track_comparison_display_fields(track_df: pd.DataFrame) -> pd.DataFrame:
    """Add grouped display fields used by the Track Comparison Explorer."""
    df = add_standard_multivalue_groups(track_df)
    df = add_film_year_bucket(df)
    df = add_key_mode_label(df)

    if "composer_primary_clean" in df.columns:
        df["composer_primary_clean"] = (
            df["composer_primary_clean"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    if "mode" in df.columns:
        df["mode_label"] = df["mode"].map({1.0: "Major", 0.0: "Minor"}).fillna("Unknown")

    if "key_label" in df.columns:
        df["key_label"] = (
            df["key_label"]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
            .replace("", "Unknown")
        )

    if "is_instrumental" in df.columns:
        df["track_type"] = np.where(
            pd.to_numeric(df["is_instrumental"], errors="coerce").fillna(0).astype(int) == 1,
            "Instrumental",
            "Vocal / Mixed",
        )
    elif "instrumentalness" in df.columns:
        df["track_type"] = np.where(
            df["instrumentalness"] >= 0.70,
            "Instrumental",
            "Vocal / Mixed",
        )
    else:
        df["track_type"] = "Unknown"

    return df

def filter_track_comparison_df(
    track_df: pd.DataFrame,
    global_controls: dict,
    controls: dict,
) -> pd.DataFrame:
    """Apply shared global filters plus Page 32-specific filters."""
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


def prepare_track_comparison_data(
    df: pd.DataFrame,
    metric: str,
    group_var: str,
    selected_groups: list[str],
    top_n: int | None,
    min_group_size: int,
    use_log: bool,
) -> pd.DataFrame:
    """Prepare row-level data for boxplot / violin / ranking."""
    required_cols = [metric, group_var]
    plot_df = df[[col for col in required_cols if col in df.columns]].copy()
    plot_df = plot_df.dropna(subset=[metric, group_var]).copy()

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

    group_counts = plot_df["group"].value_counts()
    valid_groups = group_counts[group_counts >= min_group_size].index.tolist()
    plot_df = plot_df[plot_df["group"].isin(valid_groups)].copy()

    if use_log:
        plot_df = plot_df[plot_df[metric] > 0].copy()
        plot_df["value"] = np.log10(plot_df[metric])
    else:
        plot_df["value"] = plot_df[metric]

    plot_df = plot_df.dropna(subset=["value"]).copy()
    return plot_df


def build_group_summary_df(plot_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate row-level group data into one summary row per group."""
    if plot_df.empty:
        return pd.DataFrame(
            columns=["group", "count", "median", "mean", "total", "q1", "q3", "iqr"]
        )

    summary_df = (
        plot_df.groupby("group", as_index=False)
        .agg(
            count=("value", "size"),
            median=("value", "median"),
            mean=("value", "mean"),
            total=("value", "sum"),
            q1=("value", lambda s: float(s.quantile(0.25))),
            q3=("value", lambda s: float(s.quantile(0.75))),
        )
    )
    summary_df["iqr"] = summary_df["q3"] - summary_df["q1"]
    return summary_df


def get_view_metric_column(ranking_stat: str) -> str:
    stat_map = {
        "Median": "median",
        "Mean": "mean",
        "Total": "total",
        "Count": "count",
    }
    return stat_map.get(ranking_stat, "median")


def build_context_caption(
    metric: str,
    group_var: str,
    view_mode: str,
    use_log: bool,
    selected_groups: list[str],
    top_n: int | None,
    min_group_size: int,
    ranking_stat: str | None,
) -> str:
    metric_label = get_track_page_display_label(metric)
    group_label = get_track_page_display_label(group_var).lower()

    if selected_groups:
        if len(selected_groups) <= 5:
            group_scope = f"the selected {group_label} values ({', '.join(selected_groups)})"
        else:
            shown = ", ".join(selected_groups[:5])
            group_scope = f"the selected {group_label} values ({shown}, ...)"
    else:
        group_scope = f"the top {top_n} visible {group_label} groups by track count"

    base = f"Comparing {metric_label}"
    if use_log:
        base += " on the log10 scale"
    base += f" across {group_scope}, excluding groups with fewer than {min_group_size} tracks."

    if view_mode == "Bar Ranking" and ranking_stat:
        base += f" Groups are ordered by {ranking_stat.lower()} {metric_label}." if ranking_stat != "Count" else " Groups are ordered by track count."

    return base


def build_insight_summary(
    summary_df: pd.DataFrame,
    view_mode: str,
    ranking_stat: str | None,
) -> dict[str, str]:
    if summary_df.empty:
        return {
            "card1_title": "Top Group",
            "card1_value": "None",
            "card1_caption": "No groups remain in view.",
            "card2_title": "Most Consistent",
            "card2_value": "None",
            "card2_caption": "No consistency insight is available.",
            "card3_title": "Groups in View",
            "card3_value": "0",
            "card3_caption": "No visible groups remain.",
        }

    if view_mode in {"Boxplot", "Violin"}:
        top_median_row = summary_df.sort_values(
            ["median", "count", "group"],
            ascending=[False, False, True],
        ).iloc[0]

        most_consistent_row = summary_df.sort_values(
            ["iqr", "count", "group"],
            ascending=[True, False, True],
        ).iloc[0]

        return {
            "card1_title": "Top Median Group",
            "card1_value": str(top_median_row["group"]),
            "card1_caption": f"Median = {top_median_row['median']:,.2f}",
            "card2_title": "Most Consistent",
            "card2_value": str(most_consistent_row["group"]),
            "card2_caption": f"Lowest IQR = {most_consistent_row['iqr']:,.2f}",
            "card3_title": "Groups in View",
            "card3_value": f"{len(summary_df):,}",
            "card3_caption": "Groups currently visible in the chart.",
        }

    stat_col = get_view_metric_column(ranking_stat or "Median")
    ranked_df = summary_df.sort_values(
        [stat_col, "count", "group"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    top_row = ranked_df.iloc[0]
    return {
        "card1_title": "Top Group",
        "card1_value": str(top_row["group"]),
        "card1_caption": f"{ranking_stat} = {top_row[stat_col]:,.2f}" if ranking_stat != "Count" else f"Count = {int(top_row[stat_col]):,}",
        "card2_title": "Largest Group",
        "card2_value": str(summary_df.sort_values(['count', 'median', 'group'], ascending=[False, False, True]).iloc[0]['group']),
        "card2_caption": f"{int(summary_df['count'].max()):,} visible tracks",
        "card3_title": "Groups in View",
        "card3_value": f"{len(summary_df):,}",
        "card3_caption": "Groups currently visible in the chart.",
    }


def render_insight_cards(
    summary_df: pd.DataFrame,
    view_mode: str,
    ranking_stat: str | None,
) -> None:
    insights = build_insight_summary(summary_df, view_mode, ranking_stat)

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


def create_boxplot_chart(
    plot_df: pd.DataFrame,
    metric: str,
    group_var: str,
    use_log: bool,
    show_points: bool,
) -> alt.Chart:
    """
    Create a manual boxplot chart with:
    - true whiskers based on the 1.5 * IQR rule
    - separate summary tooltip for the box
    - separate row-level tooltip for outlier points
    """
    y_title = (
        f"log10({get_track_page_display_label(metric)})"
        if use_log
        else get_track_page_display_label(metric)
    )

    summary_rows = []
    outlier_parts = []

    for group_name, group_df in plot_df.groupby("group"):
        values = group_df["value"].dropna().astype(float)

        if values.empty:
            continue

        q1 = float(values.quantile(0.25))
        median = float(values.quantile(0.50))
        q3 = float(values.quantile(0.75))
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        non_outliers = group_df[
            (group_df["value"] >= lower_bound) & (group_df["value"] <= upper_bound)
        ].copy()
        outliers = group_df[
            (group_df["value"] < lower_bound) | (group_df["value"] > upper_bound)
        ].copy()

        if non_outliers.empty:
            lower_whisker = q1
            upper_whisker = q3
        else:
            lower_whisker = float(non_outliers["value"].min())
            upper_whisker = float(non_outliers["value"].max())

        summary_rows.append(
            {
                "group": group_name,
                "count": int(len(group_df)),
                "q1_value": q1,
                "median_value": median,
                "q3_value": q3,
                "lower_whisker": lower_whisker,
                "upper_whisker": upper_whisker,
                "raw_min": float(values.min()),
                "raw_max": float(values.max()),
            }
        )

        if not outliers.empty:
            outlier_parts.append(outliers.copy())

    summary_df = pd.DataFrame(summary_rows).sort_values(
        "median_value",
        ascending=False,
    )
    group_order = summary_df["group"].tolist()

    if outlier_parts:
        outlier_df = pd.concat(outlier_parts, ignore_index=True)
    else:
        outlier_df = pd.DataFrame(columns=plot_df.columns.tolist())

    lower_stems = (
        alt.Chart(summary_df)
        .mark_rule(color="white", strokeWidth=2, tooltip=False)
        .encode(
            x=alt.X(
                "group:N",
                title=get_track_page_display_label(group_var),
                sort=group_order,
                axis=alt.Axis(labelAngle=-35),
            ),
            y=alt.Y("lower_whisker:Q", title=y_title),
            y2="q1_value:Q",
        )
    )

    upper_stems = (
        alt.Chart(summary_df)
        .mark_rule(color="white", strokeWidth=2, tooltip=False)
        .encode(
            x=alt.X("group:N", sort=group_order),
            y=alt.Y("q3_value:Q", title=y_title),
            y2="upper_whisker:Q",
        )
    )

    lower_caps = (
        alt.Chart(summary_df)
        .mark_tick(color="white", size=28, thickness=2, tooltip=False)
        .encode(
            x=alt.X("group:N", sort=group_order),
            y=alt.Y("lower_whisker:Q"),
        )
    )

    upper_caps = (
        alt.Chart(summary_df)
        .mark_tick(color="white", size=28, thickness=2, tooltip=False)
        .encode(
            x=alt.X("group:N", sort=group_order),
            y=alt.Y("upper_whisker:Q"),
        )
    )

    boxes = (
        alt.Chart(summary_df)
        .mark_bar(size=35, opacity=0.8, tooltip=False)
        .encode(
            x=alt.X(
                "group:N",
                title=get_track_page_display_label(group_var),
                sort=group_order,
                axis=alt.Axis(labelAngle=-35),
            ),
            y=alt.Y("q1_value:Q", title=y_title),
            y2="q3_value:Q",
        )
    )

    medians = (
        alt.Chart(summary_df)
        .mark_tick(color="white", size=35, thickness=2, tooltip=False)
        .encode(
            x=alt.X("group:N", sort=group_order),
            y=alt.Y("median_value:Q"),
        )
    )

    summary_tooltips = (
        alt.Chart(summary_df)
        .mark_circle(opacity=0, size=1200)
        .encode(
            x=alt.X("group:N", sort=group_order),
            y=alt.Y("median_value:Q"),
            tooltip=[
                alt.Tooltip("group:N", title=get_track_page_display_label(group_var)),
                alt.Tooltip("count:Q", title="Tracks", format=",.0f"),
                alt.Tooltip("lower_whisker:Q", title="Lower Whisker", format=",.3f"),
                alt.Tooltip("q1_value:Q", title="Q1", format=",.3f"),
                alt.Tooltip("median_value:Q", title="Median", format=",.3f"),
                alt.Tooltip("q3_value:Q", title="Q3", format=",.3f"),
                alt.Tooltip("upper_whisker:Q", title="Upper Whisker", format=",.3f"),
            ],
        )
    )

    if not outlier_df.empty:
        tooltip_fields = [
            alt.Tooltip("group:N", title=get_track_page_display_label(group_var)),
        ]

        if metric in outlier_df.columns:
            metric_format = (
                ",.0f"
                if metric in {
                    "lfm_track_listeners",
                    "lfm_track_playcount",
                }
                else ",.3f"
            )
            tooltip_fields.append(
                alt.Tooltip(
                    f"{metric}:Q",
                    title=get_track_page_display_label(metric),
                    format=metric_format,
                )
            )

        tooltip_fields.append(
            alt.Tooltip("value:Q", title="Displayed Value", format=",.3f")
        )

        outliers = (
            alt.Chart(outlier_df)
            .mark_circle(size=45, opacity=0.9)
            .encode(
                x=alt.X("group:N", sort=group_order),
                y=alt.Y("value:Q"),
                tooltip=tooltip_fields,
            )
        )
    else:
        outliers = alt.Chart(
            pd.DataFrame({"group": [], "value": []})
        ).mark_circle()

    chart = (
        boxes
        + lower_stems
        + upper_stems
        + lower_caps
        + upper_caps
        + medians
        + outliers
        + summary_tooltips
    )

    if show_points:
        chart = chart + create_strip_overlay(
            plot_df=plot_df,
            group_order=group_order,
            group_var=group_var,
        )

    return chart.properties(
        width=750,
        height=450,
        title={
            "text": f"{get_track_page_display_label(metric)} by {get_track_page_display_label(group_var)}",
            "subtitle": [
                "Boxplot comparison across groups"
                + (" on log10 scale" if use_log else "")
            ],
        },
    )

def create_strip_overlay(
    plot_df: pd.DataFrame,
    group_order: list[str],
    group_var: str,
) -> alt.Chart:
    """Create a jittered strip overlay of individual track points."""
    tooltip_fields = [
        alt.Tooltip("group:N", title=get_track_page_display_label(group_var)),
        alt.Tooltip("value:Q", title="Displayed Value", format=",.3f"),
    ]

    return (
        alt.Chart(plot_df)
        .transform_calculate(
            jitter="(random() - 0.5) * 0.5"
        )
        .mark_circle(size=26, opacity=0.25)
        .encode(
            x=alt.X("group:N", sort=group_order),
            xOffset="jitter:Q",
            y=alt.Y("value:Q"),
            tooltip=tooltip_fields,
        )
    )


def create_violin_chart(
    plot_df: pd.DataFrame,
    metric: str,
    group_var: str,
    use_log: bool,
    show_points: bool,
) -> go.Figure:
    """
    Create a violin plot using Plotly.
    """
    summary_df = (
        plot_df.groupby("group", as_index=False)["value"]
        .median()
        .rename(columns={"value": "group_median"})
        .sort_values("group_median", ascending=False)
    )
    group_order = summary_df["group"].tolist()

    y_title = (
        f"log10({get_track_page_display_label(metric)})"
        if use_log
        else get_track_page_display_label(metric)
    )

    fig = go.Figure()

    for group_name in group_order:
        group_df = plot_df[plot_df["group"] == group_name].copy()

        fig.add_trace(
            go.Violin(
                x=[group_name] * len(group_df),
                y=group_df["value"],
                name=str(group_name),
                box_visible=True,
                meanline_visible=False,
                points="all" if show_points else False,
                jitter=0.18 if show_points else 0,
                pointpos=0,
                marker=dict(size=5, opacity=0.35),
                line=dict(width=1),
                spanmode="soft",
                hovertemplate=(
                    f"{get_track_page_display_label(group_var)}: {group_name}<br>"
                    "Displayed Value: %{y:,.3f}<extra></extra>"
                ),
                showlegend=False,
            )
        )

    fig.update_layout(
        title=(
                f"{get_track_page_display_label(metric)} by {get_track_page_display_label(group_var)}"
                + (" (log10 scale)" if use_log else "")
        ),
        xaxis_title=get_track_page_display_label(group_var),
        yaxis_title=y_title,
        violingap=0.15,
        violinmode="overlay",
        height=500,
        margin=dict(l=40, r=20, t=60, b=100),
    )

    fig.update_xaxes(categoryorder="array", categoryarray=group_order)

    return fig


def create_bar_ranking_chart(
    summary_df: pd.DataFrame,
    metric: str,
    group_var: str,
    ranking_stat: str,
) -> alt.Chart:
    stat_col = get_view_metric_column(ranking_stat)

    x_title = "Track Count" if ranking_stat == "Count" else f"{ranking_stat} {get_track_page_display_label(metric)}"

    return (
        alt.Chart(summary_df.sort_values(stat_col, ascending=False))
        .mark_bar()
        .encode(
            x=alt.X(f"{stat_col}:Q", title=x_title),
            y=alt.Y("group:N", sort="-x", title=get_track_page_display_label(group_var)),
            tooltip=[
                alt.Tooltip("group:N", title=get_track_page_display_label(group_var)),
                alt.Tooltip("count:Q", title="Tracks", format=",.0f"),
                alt.Tooltip("median:Q", title="Median", format=",.3f"),
                alt.Tooltip("mean:Q", title="Mean", format=",.3f"),
                alt.Tooltip("total:Q", title="Total", format=",.3f"),
            ],
        )
        .properties(
            height=max(320, min(700, 40 * len(summary_df))),
            title=f"{ranking_stat} {get_track_page_display_label(metric)} by {get_track_page_display_label(group_var)}" if ranking_stat != "Count" else f"Track Count by {get_track_page_display_label(group_var)}",
        )
    )


def build_supporting_table(
    plot_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    metric: str,
    group_var: str,
    view_mode: str,
) -> pd.DataFrame:
    if view_mode == "Bar Ranking":
        table_df = summary_df.copy()
        return rename_and_dedupe_for_display(table_df)

    detail_cols = [col for col in DETAIL_COLUMNS if col in plot_df.columns]
    source_cols = ["group", "value"] + detail_cols
    source_cols = [col for col in source_cols if col in plot_df.columns]
    table_df = plot_df[source_cols].copy()
    table_df = table_df.rename(columns={"group": get_track_page_display_label(group_var), "value": "Displayed Value"})
    return rename_and_dedupe_for_display(table_df.head(200))

def build_comparison_supporting_insight(
    summary_df: pd.DataFrame,
    metric: str,
    group_var: str,
    view_mode: str,
    ranking_stat: str | None,
) -> str:
    """Build a short educational interpretation for the current comparison view."""
    if summary_df.empty:
        return "No visible comparison remains."

    metric_label = get_track_page_display_label(metric).lower()
    group_label = get_track_page_display_label(group_var).lower()

    if view_mode in {"Boxplot", "Violin"}:
        top_median_row = summary_df.sort_values(
            ["median", "count", "group"],
            ascending=[False, False, True],
        ).iloc[0]

        most_consistent_row = summary_df.sort_values(
            ["iqr", "count", "group"],
            ascending=[True, False, True],
        ).iloc[0]

        return (
            f"💡 {top_median_row['group']} has the highest median {metric_label}, "
            f"while {most_consistent_row['group']} shows the tightest spread. "
            f"This means the page is helping you compare both typical values and within-group variability across {group_label}."
        )

    stat_col = get_view_metric_column(ranking_stat or "Median")
    ranked_df = summary_df.sort_values(
        [stat_col, "count", "group"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    top_row = ranked_df.iloc[0]

    if len(ranked_df) >= 2:
        second_row = ranked_df.iloc[1]
        gap_value = float(top_row[stat_col] - second_row[stat_col])
        gap_text = f"The lead over {second_row['group']} is {gap_value:,.2f}."
    else:
        gap_text = "Only one group remains in view."

    return (
        f"💡 {top_row['group']} leads on {ranking_stat.lower()} {metric_label}. "
        f"{gap_text} This ranking view is best for identifying which {group_label} groups lead on the selected statistic."
    )

def main() -> None:
    st.set_page_config(
        page_title="Track Comparison Explorer",
        layout="wide",
    )
    apply_app_styles()

    st.title("Track Comparison Explorer")
    st.write(
        """
        Compare how a selected track-level metric varies across categorical track,
        album, film, and musical-structure groupings.
        """
    )

    track_df = load_track_data_explorer_data()
    track_df = add_track_comparison_display_fields(track_df)

    filter_inputs = get_global_filter_inputs(track_df)
    composer_options = get_clean_composer_options(track_df)

    st.sidebar.header("Track Comparison Controls")
    include_context_features = st.sidebar.checkbox(
        "Include film & album context",
        value=False,
        help=(
            "Adds film- and album-level grouping and metric options while keeping "
            "track-native options first."
        ),
    )

    metric_options = get_track_numeric_options(
        track_df,
        include_context_features=include_context_features,
    )
    group_options = get_track_group_options(
        track_df,
        include_context_features=include_context_features,
    )

    group_value_options_map: dict[str, list[str]] = {}
    for group_col in group_options:
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

    global_controls = get_global_filter_controls(
        min_year=filter_inputs["min_year"],
        max_year=filter_inputs["max_year"],
        film_genre_options=filter_inputs["film_genre_options"],
        album_genre_options=filter_inputs["album_genre_options"],
    )

    controls = get_track_comparison_controls(
        metric_options=metric_options,
        group_options=group_options,
        group_value_options_map=group_value_options_map,
        composer_options=composer_options,
    )

    controls["include_context_features"] = include_context_features

    filtered_df = filter_track_comparison_df(
        track_df=track_df,
        global_controls=global_controls,
        controls=controls,
    )

    if filtered_df.empty:
        st.warning("No tracks remain under the current filters.")
        return

    metric = controls["metric"]
    group_var = controls["group_var"]
    view_mode = controls["view_mode"]

    st.markdown("**Filter Context**")
    st.caption(
        build_context_caption(
            metric=metric,
            group_var=group_var,
            view_mode=view_mode,
            use_log=controls["use_log"],
            selected_groups=controls["selected_groups"],
            top_n=controls["top_n"],
            min_group_size=controls["min_group_size"],
            ranking_stat=controls["ranking_stat"],
        )
    )

    plot_df = filtered_df.copy()

    # keep detail columns for supporting table
    for col in DETAIL_COLUMNS:
        if col not in plot_df.columns and col in filtered_df.columns:
            plot_df[col] = filtered_df[col]

    prepared_df = prepare_track_comparison_data(
        df=plot_df,
        metric=metric,
        group_var=group_var,
        selected_groups=controls["selected_groups"],
        top_n=controls["top_n"],
        min_group_size=controls["min_group_size"],
        use_log=controls["use_log"],
    )

    if prepared_df.empty:
        st.warning("No valid rows remain for the selected comparison.")
        return

    summary_df = build_group_summary_df(prepared_df)

    render_insight_cards(
        summary_df=summary_df,
        view_mode=view_mode,
        ranking_stat=controls["ranking_stat"],
    )

    st.markdown("### Comparison View")

    if view_mode == "Boxplot":
        chart = create_boxplot_chart(
            plot_df=prepared_df,
            metric=metric,
            group_var=group_var,
            use_log=controls["use_log"],
            show_points=controls["show_points"],
        )
        st.altair_chart(chart, width="stretch")

    elif view_mode == "Violin":
        violin_fig = create_violin_chart(
            plot_df=prepared_df,
            metric=metric,
            group_var=group_var,
            use_log=controls["use_log"],
            show_points=controls["show_points"],
        )
        st.plotly_chart(violin_fig, width="stretch")

    else:
        chart = create_bar_ranking_chart(
            summary_df=summary_df,
            metric=metric,
            group_var=group_var,
            ranking_stat=controls["ranking_stat"] or "Median",
        )
        st.altair_chart(chart, width="stretch")

    st.caption(
        build_comparison_supporting_insight(
            summary_df=summary_df,
            metric=metric,
            group_var=group_var,
            view_mode=view_mode,
            ranking_stat=controls["ranking_stat"],
        )
    )

    if controls["show_source_table"]:
        st.markdown("### Supporting Table")
        st.caption(
            "This table is tied to the current chart view rather than acting as a separate dataset explorer."
        )
        table_df = build_supporting_table(
            plot_df=prepared_df,
            summary_df=summary_df,
            metric=metric,
            group_var=group_var,
            view_mode=view_mode,
        )
        st.dataframe(
            table_df,
            width="stretch",
            hide_index=True,
        )


if __name__ == "__main__":
    main()
from __future__ import annotations

import re
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from app.app_controls import (
    get_global_filter_controls,
    get_group_comparison_controls,
)
from app.app_data import load_explorer_data
from app.data_filters import filter_dataset, split_multivalue_genres
from app.ui import (
    apply_app_styles,
    get_display_label,
    rename_columns_for_display,
)


PREFERRED_NUMERIC_COLS = [
    "lfm_album_listeners",
    "lfm_album_playcount",
    "n_tracks",
    "album_release_lag_days",
    "days_since_album_release",
    "days_since_film_release",
    "film_vote_count",
    "film_popularity",
    "film_rating",
    "film_runtime_min",
    "composer_album_count",
    "us_score_nominee_count",
    "us_song_nominee_count",
]

ALBUM_GENRE_FLAG_COLS = [
    "ambient_experimental",
    "classical_orchestral",
    "electronic",
    "hip_hop_rnb",
    "pop",
    "rock",
    "world_folk",
]

FILM_GENRE_FLAG_COLS = [
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
]

GENRE_LABEL_MAP = {
    "ambient_experimental": "Ambient / Experimental",
    "classical_orchestral": "Classical / Orchestral",
    "electronic": "Electronic",
    "hip_hop_rnb": "Hip-Hop / R&B",
    "pop": "Pop",
    "rock": "Rock",
    "world_folk": "World / Folk",
    "film_is_action": "Action",
    "film_is_adventure": "Adventure",
    "film_is_animation": "Animation",
    "film_is_comedy": "Comedy",
    "film_is_crime": "Crime",
    "film_is_documentary": "Documentary",
    "film_is_drama": "Drama",
    "film_is_family": "Family",
    "film_is_fantasy": "Fantasy",
    "film_is_history": "History",
    "film_is_horror": "Horror",
    "film_is_music": "Music",
    "film_is_mystery": "Mystery",
    "film_is_romance": "Romance",
    "film_is_science_fiction": "Science Fiction",
    "film_is_tv_movie": "TV Movie",
    "film_is_thriller": "Thriller",
    "film_is_war": "War",
    "film_is_western": "Western",
}

PREFERRED_GROUP_COLS = [
    "composer_primary_clean",
    "label_names",
    "album_genre_group",
    "film_genre_group",
    "album_us_release_year",
    "bafta_nominee",
    "oscar_score_nominee",
    "oscar_song_nominee",
    "globes_score_nominee",
    "globes_song_nominee",
    "critics_score_nominee",
    "critics_song_nominee",
]

def derive_multi_label_group(
    df: pd.DataFrame,
    flag_cols: list[str],
    label_map: dict[str, str],
    output_col: str,
) -> pd.DataFrame:
    """
    Collapse multi-label genre flags into a single grouping column.

    Rules:
        - one positive flag -> that genre label
        - multiple positive flags -> "Multi-genre"
        - no positive flags -> "Unknown"
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


def build_group_comparison_df(explorer_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived grouping fields used by the Group Comparison Explorer.
    """
    out_df = derive_multi_label_group(
        df=explorer_df,
        flag_cols=ALBUM_GENRE_FLAG_COLS,
        label_map=GENRE_LABEL_MAP,
        output_col="album_genre_group",
    )

    out_df = derive_multi_label_group(
        df=out_df,
        flag_cols=FILM_GENRE_FLAG_COLS,
        label_map=GENRE_LABEL_MAP,
        output_col="film_genre_group",
    )

    return out_df


def get_group_options(df: pd.DataFrame) -> list[str]:
    """
    Return curated grouping fields for the Group Comparison Explorer.
    """
    group_options = []

    for col in PREFERRED_GROUP_COLS:
        if col not in df.columns:
            continue

        non_null = df[col].dropna()
        if non_null.empty:
            continue

        nunique = non_null.astype(str).nunique()

        if col in {"composer_primary_clean", "label_names"}:
            if nunique >= 2:
                group_options.append(col)
        else:
            if 2 <= nunique <= 20:
                group_options.append(col)

    return group_options


def build_group_value_options_map(
    df: pd.DataFrame,
    group_options: list[str],
) -> dict[str, list[str]]:
    """
    Build selectable value lists for each grouping field.
    """
    value_map: dict[str, list[str]] = {}

    for col in group_options:
        if col == "album_us_release_year":
            year_values = (
                pd.to_numeric(df[col], errors="coerce")
                .dropna()
                .astype(int)
                .astype(str)
                .unique()
                .tolist()
            )
            values = sorted(year_values, key=lambda x: int(x))
        else:
            values = (
                df[col]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )
            values = sorted(values)

        value_map[col] = values

    return value_map

def explode_multivalue_group_rows(
    df: pd.DataFrame,
    raw_col: str,
) -> pd.DataFrame:
    """
    Explode a pipe- or comma-delimited genre column into one row per genre value.

    Args:
        df: Source dataframe.
        raw_col: Raw multivalue genre column such as 'film_genres' or
            'album_genres_display'.

    Returns:
        pd.DataFrame: Expanded dataframe with one row per genre membership and
        a 'group' column containing the individual genre value.
    """
    exploded_rows = []

    for _, row in df.iterrows():
        raw_value = row.get(raw_col)

        if pd.isna(raw_value):
            continue

        parts = [
            part.strip()
            for part in re.split(r"\s*\|\s*|\s*,\s*", str(raw_value))
            if part.strip()
        ]

        for part in parts:
            new_row = row.copy()
            new_row["group"] = part
            exploded_rows.append(new_row)

    if not exploded_rows:
        return pd.DataFrame(columns=list(df.columns) + ["group"])

    return pd.DataFrame(exploded_rows)

def prepare_group_comparison_data(
    df: pd.DataFrame,
    metric: str,
    group_var: str,
    selected_groups: list[str],
    top_n: int | None,
    min_group_size: int,
    use_log: bool,
    view_mode: str,
    genre_mode: str,
) -> pd.DataFrame:
    """
    Prepare row-level data for group comparison charts.
    """
    if (
        view_mode in ["Boxplot", "Violin"]
        and genre_mode == "Include albums in all matching genres"
        and group_var in ["album_genre_group", "film_genre_group"]
    ):
        raw_group_col = (
            "album_genres_display"
            if group_var == "album_genre_group"
            else "film_genres"
        )

        plot_df = df.copy().dropna(subset=[metric, raw_group_col])
        plot_df = explode_multivalue_group_rows(
            df=plot_df,
            raw_col=raw_group_col,
        )

    else:
        plot_df = df.copy().dropna(subset=[metric, group_var])

        if group_var == "album_us_release_year":
            plot_df["group"] = (
                pd.to_numeric(plot_df[group_var], errors="coerce")
                .astype("Int64")
                .astype(str)
            )
        else:
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


def build_group_summary_df(
    plot_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate row-level group data into one summary row per group.
    """
    summary_df = (
        plot_df.groupby("group", as_index=False)
        .agg(
            count=("value", "size"),
            median=("value", "median"),
            mean=("value", "mean"),
            total=("value", "sum"),
        )
    )

    return summary_df

def prepare_stratified_ranking_data(
    plot_df: pd.DataFrame,
    metric: str,
    ranking_stat: str,
    stratify_by: str,
    max_strata: int,
) -> pd.DataFrame:
    """
    Aggregate row-level data for stacked bar ranking.

    Args:
        plot_df: Row-level group comparison dataframe.
        metric: Selected metric.
        ranking_stat: Aggregation statistic for ranking.
        stratify_by: Secondary field used for stacked segments.
        max_strata: Maximum number of strata to display before rolling the rest
            into "Others".

    Returns:
        pd.DataFrame: Aggregated dataframe with columns:
            - group
            - stratum
            - value
    """
    rank_df = plot_df.copy()

    if stratify_by == "None":
        rank_df["stratum"] = "All"
    elif stratify_by == "album_us_release_year":
        rank_df["stratum"] = (
            pd.to_numeric(rank_df[stratify_by], errors="coerce")
            .astype("Int64")
            .astype(str)
        )
    else:
        rank_df["stratum"] = rank_df[stratify_by].astype(str)

    # Keep only the largest strata, roll the rest into Others
    if stratify_by != "None":
        top_strata = (
            rank_df["stratum"]
            .value_counts()
            .head(max_strata)
            .index
            .tolist()
        )
        rank_df["stratum"] = rank_df["stratum"].where(
            rank_df["stratum"].isin(top_strata),
            "Others",
        )

    grouped = rank_df.groupby(["group", "stratum"], as_index=False)

    if ranking_stat == "Count":
        agg_df = grouped.size().rename(columns={"size": "value"})
    elif ranking_stat == "Total":
        agg_df = grouped[metric].sum().rename(columns={metric: "value"})
    elif ranking_stat == "Mean":
        agg_df = grouped[metric].mean().rename(columns={metric: "value"})
    else:
        agg_df = grouped[metric].median().rename(columns={metric: "value"})

    return agg_df

def create_stacked_bar_chart(
    agg_df: pd.DataFrame,
    metric: str,
    group_var: str,
    ranking_stat: str,
    stratify_by: str,
) -> alt.Chart:
    """
    Create a stacked horizontal bar chart for stratified ranking.
    """
    group_totals = (
        agg_df.groupby("group", as_index=False)["value"]
        .sum()
        .sort_values("value", ascending=False)
    )
    group_order = group_totals["group"].tolist()

    if ranking_stat == "Count":
        x_title = "Album Count"
    else:
        x_title = f"{ranking_stat} {get_display_label(metric)}"

    subtitle = "Ranked comparison across groups"
    if stratify_by != "None":
        subtitle += f" | Stacked by {get_display_label(stratify_by)}"

    return (
        alt.Chart(agg_df)
        .mark_bar()
        .encode(
            y=alt.Y(
                "group:N",
                sort=group_order,
                title=get_display_label(group_var),
            ),
            x=alt.X(
                "value:Q",
                title=x_title,
            ),
            color=alt.Color(
                "stratum:N",
                title="Stratum" if stratify_by == "None" else get_display_label(stratify_by),
            ),
            tooltip=[
                alt.Tooltip("group:N", title=get_display_label(group_var)),
                alt.Tooltip(
                    "stratum:N",
                    title="Stratum" if stratify_by == "None" else get_display_label(stratify_by),
                ),
                alt.Tooltip(
                    "value:Q",
                    title="Value",
                    format=",.0f" if ranking_stat == "Count" else ",.3f",
                ),
            ],
        )
        .properties(
            width=750,
            height=450,
            title={
                "text": (
                    f"{ranking_stat} {get_display_label(metric)} by {get_display_label(group_var)}"
                    if ranking_stat != "Count"
                    else f"Album Count by {get_display_label(group_var)}"
                ),
                "subtitle": [subtitle],
            },
        )
    )


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
        f"log10({get_display_label(metric)})"
        if use_log
        else get_display_label(metric)
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

    # Whisker stems: Q1 down to lower whisker, and Q3 up to upper whisker
    lower_stems = (
        alt.Chart(summary_df)
        .mark_rule(tooltip=False)
        .encode(
            x=alt.X(
                "group:N",
                title=get_display_label(group_var),
                sort=group_order,
                axis=alt.Axis(labelAngle=-35),
            ),
            y=alt.Y("lower_whisker:Q", title=y_title),
            y2="q1_value:Q",
        )
    )

    upper_stems = (
        alt.Chart(summary_df)
        .mark_rule(tooltip=False)
        .encode(
            x=alt.X("group:N", sort=group_order),
            y=alt.Y("q3_value:Q", title=y_title),
            y2="upper_whisker:Q",
        )
    )

    # Whisker caps
    lower_caps = (
        alt.Chart(summary_df)
        .mark_tick(size=28, thickness=2, tooltip=False)
        .encode(
            x=alt.X("group:N", sort=group_order),
            y=alt.Y("lower_whisker:Q"),
        )
    )

    upper_caps = (
        alt.Chart(summary_df)
        .mark_tick(size=28, thickness=2, tooltip=False)
        .encode(
            x=alt.X("group:N", sort=group_order),
            y=alt.Y("upper_whisker:Q"),
        )
    )

    # IQR box
    boxes = (
        alt.Chart(summary_df)
        .mark_bar(size=35, opacity=0.8, tooltip=False)
        .encode(
            x=alt.X(
                "group:N",
                title=get_display_label(group_var),
                sort=group_order,
                axis=alt.Axis(labelAngle=-35),
            ),
            y=alt.Y("q1_value:Q", title=y_title),
            y2="q3_value:Q",
        )
    )

    # Median tick
    medians = (
        alt.Chart(summary_df)
        .mark_tick(color="white", size=35, thickness=2, tooltip=False)
        .encode(
            x=alt.X("group:N", sort=group_order),
            y=alt.Y("median_value:Q"),
        )
    )

    # Invisible summary tooltip layer for the box as a whole
    summary_tooltips = (
        alt.Chart(summary_df)
        .mark_circle(opacity=0, size=1200)
        .encode(
            x=alt.X("group:N", sort=group_order),
            y=alt.Y("median_value:Q"),
            tooltip=[
                alt.Tooltip("group:N", title=get_display_label(group_var)),
                alt.Tooltip("count:Q", title="Albums", format=",.0f"),
                alt.Tooltip("lower_whisker:Q", title="Lower Whisker", format=",.3f"),
                alt.Tooltip("q1_value:Q", title="Q1", format=",.3f"),
                alt.Tooltip("median_value:Q", title="Median", format=",.3f"),
                alt.Tooltip("q3_value:Q", title="Q3", format=",.3f"),
                alt.Tooltip("upper_whisker:Q", title="Upper Whisker", format=",.3f"),
            ],
        )
    )

    # Outlier tooltip: row-level context
    if not outlier_df.empty:
        tooltip_fields = []

        if "album_title" in outlier_df.columns:
            tooltip_fields.append(
                alt.Tooltip("album_title:N", title="Album")
            )

        if "film_title" in outlier_df.columns:
            tooltip_fields.append(
                alt.Tooltip("film_title:N", title="Film")
            )

        tooltip_fields.append(
            alt.Tooltip("group:N", title=get_display_label(group_var))
        )

        if metric in outlier_df.columns:
            metric_format = (
                ",.0f"
                if metric in {
                    "lfm_album_listeners",
                    "lfm_album_playcount",
                    "n_tracks",
                }
                else ",.3f"
            )
            tooltip_fields.append(
                alt.Tooltip(
                    f"{metric}:Q",
                    title=get_display_label(metric),
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
            lower_stems
            + upper_stems
            + lower_caps
            + upper_caps
            + boxes
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
            "text": f"{get_display_label(metric)} by {get_display_label(group_var)}",
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
    """
    Create a jittered strip overlay of individual album points.
    """
    tooltip_fields = []

    if "album_title" in plot_df.columns:
        tooltip_fields.append(
            alt.Tooltip("album_title:N", title="Album")
        )

    if "film_title" in plot_df.columns:
        tooltip_fields.append(
            alt.Tooltip("film_title:N", title="Film")
        )

    tooltip_fields.append(
        alt.Tooltip("group:N", title=get_display_label(group_var))
    )
    tooltip_fields.append(
        alt.Tooltip("value:Q", title="Displayed Value", format=",.3f")
    )

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

def create_bar_ranking_chart(
    summary_df: pd.DataFrame,
    metric: str,
    group_var: str,
    ranking_stat: str,
) -> alt.Chart:
    """
    Create a horizontal bar chart ranking groups by the selected statistic.
    """
    stat_map = {
        "Median": "median",
        "Mean": "mean",
        "Total": "total",
        "Count": "count",
    }
    stat_col = stat_map[ranking_stat]

    sort_df = summary_df.sort_values(stat_col, ascending=False)
    group_order = sort_df["group"].tolist()

    chart = (
        alt.Chart(summary_df)
        .mark_bar()
        .encode(
            y=alt.Y(
                "group:N",
                title=get_display_label(group_var),
                sort=group_order,
            ),
            x=alt.X(
                f"{stat_col}:Q",
                title=f"{ranking_stat} {get_display_label(metric)}"
                if ranking_stat != "Count"
                else "Album Count",
            ),
            tooltip=[
                alt.Tooltip("group:N", title=get_display_label(group_var)),
                alt.Tooltip("count:Q", title="Albums", format=",.0f"),
                alt.Tooltip("median:Q", title="Median", format=",.3f"),
                alt.Tooltip("mean:Q", title="Mean", format=",.3f"),
                alt.Tooltip("total:Q", title="Total", format=",.3f"),
            ],
        )
        .properties(
            width=750,
            height=450,
            title={
                "text": (
                    f"{ranking_stat} {get_display_label(metric)} "
                    f"by {get_display_label(group_var)}"
                    if ranking_stat != "Count"
                    else f"Album Count by {get_display_label(group_var)}"
                ),
                "subtitle": ["Ranked comparison across groups"],
            },
        )
    )

    return chart


def build_boxplot_source_table(
    plot_df: pd.DataFrame,
    metric: str,
) -> pd.DataFrame:
    """
    Build row-level source table for boxplot mode.
    """
    table_df = plot_df[[metric, "group", "value"]].copy()
    table_df = table_df.rename(columns={"value": "plot_value"})
    return rename_columns_for_display(table_df)


def build_ranking_source_table(
    summary_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build summary table for ranking mode.
    """
    return rename_columns_for_display(summary_df)


def build_group_drilldown_table(
    df: pd.DataFrame,
    metric: str,
    group_var: str,
    inspect_group: str,
    use_log: bool,
) -> pd.DataFrame:
    """
    Build a row-level drill-down table for one selected group.

    Uses the richer comparison dataframe rather than the narrow plot dataframe,
    so users can see actual album metadata.
    """
    detail_df = df.copy()

    if group_var == "album_us_release_year":
        detail_df["group"] = (
            pd.to_numeric(detail_df[group_var], errors="coerce")
            .astype("Int64")
            .astype(str)
        )
    else:
        detail_df["group"] = detail_df[group_var].astype(str)

    detail_df = detail_df[detail_df["group"] == inspect_group].copy()

    detail_df = detail_df.dropna(subset=[metric])

    if use_log:
        detail_df = detail_df[detail_df[metric] > 0].copy()
        detail_df["plot_value"] = np.log10(detail_df[metric])
    else:
        detail_df["plot_value"] = detail_df[metric]

    preferred_cols = [
        "album_title",
        "film_title",
        "composer_primary_clean",
        "label_names",
        "album_genre_group",
        "film_genre_group",
        "album_us_release_year",
        "film_year",
        "n_tracks",
        "lfm_album_listeners",
        "lfm_album_playcount",
        metric,
        "plot_value",
    ]

    cols = []
    for col in preferred_cols:
        if col in detail_df.columns and col not in cols:
            cols.append(col)

    detail_df = detail_df[cols].copy()

    sort_col = "plot_value" if "plot_value" in detail_df.columns else metric
    if sort_col in detail_df.columns:
        detail_df = detail_df.sort_values(
            sort_col,
            ascending=False,
            na_position="last",
        )

    return rename_columns_for_display(detail_df)

def create_violin_chart(
    plot_df: pd.DataFrame,
    metric: str,
    group_var: str,
    use_log: bool,
    show_points: bool,
) -> go.Figure:
    """
    Create a violin plot using Plotly.

    Args:
        plot_df: Row-level plot dataframe containing at least 'group' and 'value'.
        metric: Selected raw metric.
        group_var: Grouping field.
        use_log: Whether the displayed value is log10-transformed.
        show_points: Whether to overlay individual album points.

    Returns:
        go.Figure: Plotly violin chart.
    """
    summary_df = (
        plot_df.groupby("group", as_index=False)["value"]
        .median()
        .rename(columns={"value": "group_median"})
        .sort_values("group_median", ascending=False)
    )
    group_order = summary_df["group"].tolist()

    y_title = (
        f"log10({get_display_label(metric)})"
        if use_log
        else get_display_label(metric)
    )

    fig = go.Figure()

    for group_name in group_order:
        group_df = plot_df[plot_df["group"] == group_name].copy()

        customdata_cols = []
        if "album_title" in group_df.columns:
            customdata_cols.append("album_title")
        if "film_title" in group_df.columns:
            customdata_cols.append("film_title")
        if metric in group_df.columns:
            customdata_cols.append(metric)

        customdata = group_df[customdata_cols].to_numpy() if customdata_cols else None

        hover_parts = [f"{get_display_label(group_var)}: {group_name}"]

        cd_index = 0
        if "album_title" in group_df.columns:
            hover_parts.append("Album: %{customdata[" + str(cd_index) + "]}")
            cd_index += 1
        if "film_title" in group_df.columns:
            hover_parts.append("Film: %{customdata[" + str(cd_index) + "]}")
            cd_index += 1
        if metric in group_df.columns:
            metric_format = (
                ":,.0f"
                if metric in {"lfm_album_listeners", "lfm_album_playcount", "n_tracks"}
                else ":,.3f"
            )
            hover_parts.append(
                f"{get_display_label(metric)}: %{{customdata[{cd_index}]{metric_format}}}"
            )

        hover_parts.append("Displayed Value: %{y:,.3f}")
        hovertemplate = "<br>".join(hover_parts) + "<extra></extra>"

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
                customdata=customdata,
                hovertemplate=hovertemplate,
                showlegend=False,
            )
        )

    fig.update_layout(
        title=(
            f"{get_display_label(metric)} by {get_display_label(group_var)}"
            + (" (log10 scale)" if use_log else "")
        ),
        xaxis_title=get_display_label(group_var),
        yaxis_title=y_title,
        violingap=0.15,
        violinmode="overlay",
        height=500,
        margin=dict(l=40, r=20, t=60, b=100),
    )

    fig.update_xaxes(categoryorder="array", categoryarray=group_order)

    return fig

def main() -> None:
    """Render the Group Comparison Explorer page."""
    st.set_page_config(
        page_title="Group Comparison Explorer",
        layout="wide",
    )
    apply_app_styles()

    st.title("Group Comparison Explorer")
    st.write(
        """
        Compare how an album-level metric differs across groups.

        Each row represents one album.
        """
    )

    explorer_df = load_explorer_data()
    # Build options
    year_series = explorer_df["film_year"].dropna().astype(int)
    min_year = int(year_series.min())
    max_year = int(year_series.max())

    film_genre_options = split_multivalue_genres(explorer_df["film_genres"])
    album_genre_options = split_multivalue_genres(explorer_df["album_genres_display"])

    # Controls
    global_filters = get_global_filter_controls(
        min_year=min_year,
        max_year=max_year,
        film_genre_options=film_genre_options,
        album_genre_options=album_genre_options,
    )

    # Apply filters
    filtered_df = filter_dataset(explorer_df, global_filters)

    # Then proceed
    comparison_df = build_group_comparison_df(filtered_df)

    numeric_options = [
        col for col in PREFERRED_NUMERIC_COLS
        if col in comparison_df.columns
    ]
    group_options = get_group_options(comparison_df)
    group_value_options_map = build_group_value_options_map(
        comparison_df,
        group_options,
    )

    controls = get_group_comparison_controls(
        numeric_options=numeric_options,
        group_options=group_options,
        group_value_options_map=group_value_options_map,
    )

    metric = controls["metric"]
    view_mode = controls["view_mode"]
    group_var = controls["group_var"]
    selected_groups = controls["selected_groups"]
    top_n = controls["top_n"]
    min_group_size = controls["min_group_size"]
    use_log = controls["use_log"]
    ranking_stat = controls["ranking_stat"]

    plot_df = prepare_group_comparison_data(
        df=comparison_df,
        metric=metric,
        group_var=group_var,
        selected_groups=selected_groups,
        top_n=top_n,
        min_group_size=min_group_size,
        use_log=use_log,
        view_mode=view_mode,
        genre_mode=controls["genre_mode"],
    )

    if plot_df.empty:
        st.warning(
            "No valid rows remain for this metric after applying the current settings."
        )
        return

    summary_df = build_group_summary_df(plot_df)

    if summary_df.empty:
        st.warning("No groups remain after applying the minimum group size filter.")
        return

    displayed_groups = sorted(summary_df["group"].astype(str).tolist())

    groups_shown = summary_df["group"].nunique()
    albums_after_global_filters = len(filtered_df)
    if (
            view_mode in ["Boxplot", "Violin"]
            and controls["genre_mode"] == "Include albums in all matching genres"
            and group_var in ["album_genre_group", "film_genre_group"]
    ):
        if "album_title" in plot_df.columns:
            albums_in_view = plot_df["album_title"].nunique()
        else:
            albums_in_view = len(plot_df)
        membership_rows = len(plot_df)
    else:
        albums_in_view = len(plot_df)
        membership_rows = None
    median_group_size = summary_df["count"].median()

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Groups Shown", f"{groups_shown:,}")

    with col2:
        st.metric(
            "Albums After Global Filters",
            f"{albums_after_global_filters:,}",
            help=(
                "Album rows remaining after the page-level global filters "
                "such as year and genre."
            ),
        )

    with col3:
        st.metric(
            "Albums in View",
            f"{albums_in_view:,}",
            help=(
                "Album rows currently included in the chart after applying "
                "group selection, Top N groups, minimum group size, and "
                "metric-specific filtering such as log-scale validity."
            ),
        )

    with col4:
        st.metric(
            "Median Group Size",
            f"{median_group_size:.0f}",
            help=(
                "Median number of albums per group among the groups currently "
                "shown in the chart."
            ),
        )

    with col5:
        if view_mode == "Boxplot":
            st.metric("Metric", get_display_label(metric))
        else:
            st.metric("Ranking", ranking_stat)

    if selected_groups:
        st.caption(
            f"Showing selected {get_display_label(group_var)} values: "
            f"{', '.join(selected_groups[:5])}"
            + (" ..." if len(selected_groups) > 5 else "")
            + ". Albums in View reflects only albums that remain after "
              "global filters, group selection, minimum group size, and "
              "metric-specific filtering."
        )
    elif top_n is not None:
        st.caption(
            f"Showing top {top_n} {get_display_label(group_var)} groups by album count "
            f"with minimum group size = {min_group_size}. "
            f"Albums in View reflects only the albums included in those displayed groups."
        )

    if membership_rows is not None:
        st.caption(
            f"Albums in View counts unique albums. "
            f"The current chart contains {membership_rows:,} genre-membership rows "
            f"because albums can appear in multiple genres."
        )

    if view_mode == "Boxplot":
        chart = create_boxplot_chart(
            plot_df=plot_df,
            metric=metric,
            group_var=group_var,
            use_log=use_log,
            show_points=controls["show_points"],
        )
        st.altair_chart(chart, width="stretch")

    elif view_mode == "Violin":
        violin_fig = create_violin_chart(
            plot_df=plot_df,
            metric=metric,
            group_var=group_var,
            use_log=use_log,
            show_points=controls["show_points"],
        )
        st.plotly_chart(violin_fig, width="stretch")

    else:
        stratify_by = controls["stratify_by"]
        max_strata = controls["max_strata"]

        if stratify_by == "None":
            chart = create_bar_ranking_chart(
                summary_df=summary_df,
                metric=metric,
                group_var=group_var,
                ranking_stat=ranking_stat,
            )
        else:
            agg_df = prepare_stratified_ranking_data(
                plot_df=plot_df,
                metric=metric,
                ranking_stat=ranking_stat,
                stratify_by=stratify_by,
                max_strata=max_strata,
            )

            chart = create_stacked_bar_chart(
                agg_df=agg_df,
                metric=metric,
                group_var=group_var,
                ranking_stat=ranking_stat,
                stratify_by=stratify_by,
            )

        st.altair_chart(chart, width="stretch")

    if controls["show_table"]:
        st.subheader("Source Data")
        if view_mode == "Boxplot":
            table_df = build_boxplot_source_table(
                plot_df=plot_df,
                metric=metric,
            )
        else:
            table_df = build_ranking_source_table(summary_df)

        st.dataframe(
            table_df,
            width="stretch",
            hide_index=True,
        )

    st.subheader("Inspect a Group")

    inspect_group = st.selectbox(
        "Choose a displayed group to inspect",
        options=["None"] + displayed_groups,
        index=0,
        help="Show the album rows that belong to one currently displayed group.",
    )

    if inspect_group != "None":
        detail_table = build_group_drilldown_table(
            df=plot_df,
            metric=metric,
            group_var=group_var,
            inspect_group=inspect_group,
            use_log=use_log,
        )

        st.markdown(f"**Albums in {inspect_group}**")
        st.dataframe(
            detail_table,
            width="stretch",
            hide_index=True,
        )


if __name__ == "__main__":
    main()
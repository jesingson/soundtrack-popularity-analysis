from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from app.app_controls import (
    get_concentration_controls,
    get_global_filter_controls,
)
from app.app_data import load_explorer_data
from app.data_filters import filter_dataset, split_multivalue_genres
from app.ui import (
    apply_app_styles,
    get_display_label,
    rename_columns_for_display,
)


PREFERRED_CONCENTRATION_METRICS = [
    "lfm_album_listeners",
    "lfm_album_playcount",
]

PREFERRED_GROUP_COLS = [
    "composer_primary_clean",
    "label_names",
    "film_genre_group",
    "album_genre_group",
    "film_year",
    "award_category",
]

TOOLTIP_METADATA_CANDIDATES = [
    "album_title",
    "film_title",
    "composer_primary_clean",
    "label_names",
    "film_year",
    "n_tracks",
]

def derive_single_label_group(value: str) -> str:
    """
    Collapse a pipe- or comma-delimited multi-label string to a single group.

    Args:
        value: Raw multi-value string.

    Returns:
        str: Single grouped label, "Multi-genre", or "Unknown".
    """
    if pd.isna(value) or not str(value).strip():
        return "Unknown"

    parts = [
        part.strip()
        for part in str(value).replace("|", ",").split(",")
        if part.strip()
    ]

    unique_parts = list(dict.fromkeys(parts))

    if not unique_parts:
        return "Unknown"
    if len(unique_parts) == 1:
        return unique_parts[0]
    return "Multi-genre"


def build_concentration_explorer_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a page-specific dataframe with single-assignment grouping fields.

    Args:
        df: Explorer dataframe from load_explorer_data().

    Returns:
        pd.DataFrame: Copy with Page 5 grouping fields added where possible.
    """
    out_df = df.copy()

    if "film_genres" in out_df.columns and "film_genre_group" not in out_df.columns:
        out_df["film_genre_group"] = out_df["film_genres"].apply(derive_single_label_group)

    if (
        "album_genres_display" in out_df.columns
        and "album_genre_group" not in out_df.columns
    ):
        out_df["album_genre_group"] = out_df["album_genres_display"].apply(
            derive_single_label_group
        )

    if "film_year" in out_df.columns:
        out_df["film_year"] = pd.to_numeric(out_df["film_year"], errors="coerce")

    return out_df

def get_concentration_group_options(df: pd.DataFrame) -> list[str]:
    """
    Return grouping fields eligible for concentration analysis.

    Args:
        df: Album-level exploration dataframe.

    Returns:
        list[str]: Supported grouping fields with at least two distinct values.
    """
    group_options = []

    for col in PREFERRED_GROUP_COLS:
        if col not in df.columns:
            continue

        non_null = df[col].dropna()
        if non_null.empty:
            continue

        if non_null.astype(str).nunique() >= 2:
            group_options.append(col)

    return group_options


def build_group_value_options_map(
    df: pd.DataFrame,
    group_options: list[str],
) -> dict[str, list[str]]:
    """
    Build selectable value lists for each grouping field.

    Args:
        df: Album-level exploration dataframe.
        group_options: Supported grouping fields.

    Returns:
        dict[str, list[str]]: Sorted unique values by grouping field.
    """
    value_map: dict[str, list[str]] = {}

    for col in group_options:
        values = (
            df[col]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
        value_map[col] = sorted(values)

    return value_map


def prepare_concentration_data(
    df: pd.DataFrame,
    metric: str,
    group_var: str,
    selected_groups: list[str],
    top_n: int | None,
    min_group_size: int,
) -> pd.DataFrame:
    """
    Prepare row-level album data for concentration analysis.

    Args:
        df: Globally filtered album dataframe.
        metric: Selected metric.
        group_var: Selected grouping field.
        selected_groups: Optional explicit groups to keep.
        top_n: Top-N groups by album count when explicit selection is blank.
        min_group_size: Minimum albums required for a group to be retained.

    Returns:
        pd.DataFrame: Row-level dataframe with normalized 'group' column.
    """
    required_cols = [metric, group_var]
    metadata_cols = [
        col for col in TOOLTIP_METADATA_CANDIDATES
        if col in df.columns
    ]
    required_cols.extend(metadata_cols)
    required_cols = list(dict.fromkeys(required_cols))

    plot_df = df[required_cols].copy()
    plot_df = plot_df.dropna(subset=[metric, group_var]).copy()

    plot_df[metric] = pd.to_numeric(plot_df[metric], errors="coerce")
    plot_df = plot_df.dropna(subset=[metric]).copy()
    plot_df = plot_df[plot_df[metric] > 0].copy()

    plot_df["group"] = plot_df[group_var].astype(str)

    if selected_groups:
        plot_df = plot_df[plot_df["group"].isin(selected_groups)].copy()

    group_sizes = (
        plot_df.groupby("group", dropna=False)
        .size()
        .rename("album_count")
        .reset_index()
    )
    eligible_groups = group_sizes[
        group_sizes["album_count"] >= min_group_size
    ]["group"].tolist()

    plot_df = plot_df[plot_df["group"].isin(eligible_groups)].copy()

    if not selected_groups and top_n is not None:
        top_groups = (
            plot_df["group"]
            .value_counts()
            .head(top_n)
            .index
            .tolist()
        )
        plot_df = plot_df[plot_df["group"].isin(top_groups)].copy()

    return plot_df


def compute_top_k_share(values: pd.Series, k: int) -> float:
    """
    Compute the share of total metric contributed by the top k items.

    Args:
        values: Positive numeric values for one group.
        k: Number of top items to include.

    Returns:
        float: Top-k share in [0, 1].
    """
    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)

    if len(arr) == 0:
        return np.nan

    total = arr.sum()
    if total <= 0:
        return np.nan

    arr_sorted = np.sort(arr)[::-1]
    return float(arr_sorted[:k].sum() / total)


def compute_gini(values: pd.Series) -> float:
    """
    Compute the Gini coefficient for nonnegative values.

    Args:
        values: Positive numeric values for one group.

    Returns:
        float: Gini coefficient in [0, 1], where higher values indicate
        greater concentration.
    """
    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)

    if len(arr) == 0:
        return np.nan

    arr = arr[arr >= 0]
    if len(arr) == 0:
        return np.nan

    total = arr.sum()
    if total == 0:
        return 0.0

    arr = np.sort(arr)
    n = len(arr)
    index = np.arange(1, n + 1)

    gini = np.sum((2 * index - n - 1) * arr) / (n * total)
    return float(gini)

def compute_albums_to_threshold(
    values: pd.Series,
    threshold: float,
) -> float:
    """
    Compute how many top-ranked albums are needed to reach a cumulative
    share threshold within a group.

    Args:
        values: Positive numeric values for one group.
        threshold: Target cumulative share in [0, 1].

    Returns:
        float: Number of albums needed to reach the threshold.
    """
    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)

    if len(arr) == 0:
        return np.nan

    arr = arr[arr > 0]
    if len(arr) == 0:
        return np.nan

    total = arr.sum()
    if total <= 0:
        return np.nan

    arr_sorted = np.sort(arr)[::-1]
    cumulative_share = np.cumsum(arr_sorted) / total

    first_idx = np.argmax(cumulative_share >= threshold)
    return float(first_idx + 1)

def build_concentration_summary_df(
    plot_df: pd.DataFrame,
    metric: str,
) -> pd.DataFrame:
    """
    Build one-row-per-group concentration summary.

    Args:
        plot_df: Row-level concentration dataframe.
        metric: Selected metric.

    Returns:
        pd.DataFrame: Group summary with top-k shares, Gini, and threshold
        album counts.
    """
    parts = []

    for group_name, group_df in plot_df.groupby("group", dropna=False):
        values = group_df[metric]

        parts.append({
            "group": group_name,
            "album_count": int(len(group_df)),
            "total_metric": float(values.sum()),
            "mean_metric": float(values.mean()),
            "median_metric": float(values.median()),
            "top_1_share": compute_top_k_share(values, 1),
            "top_3_share": compute_top_k_share(values, 3),
            "top_5_share": compute_top_k_share(values, 5),
            "gini": compute_gini(values),
            "albums_to_50pct": compute_albums_to_threshold(values, 0.50),
            "albums_to_80pct": compute_albums_to_threshold(values, 0.80),
        })

    summary_df = pd.DataFrame(parts)

    if summary_df.empty:
        return summary_df

    return summary_df.reset_index(drop=True)


def sort_summary_df(
    summary_df: pd.DataFrame,
    ranking_metric: str,
) -> pd.DataFrame:
    """
    Sort the summary dataframe by the selected ranking metric.

    Args:
        summary_df: One-row-per-group summary dataframe.
        ranking_metric: Selected ranking metric.

    Returns:
        pd.DataFrame: Sorted summary dataframe.
    """
    return summary_df.sort_values(
        [ranking_metric, "total_metric", "album_count"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def build_lorenz_curve_df(
    plot_df: pd.DataFrame,
    metric: str,
    selected_groups: list[str],
) -> pd.DataFrame:
    """
    Build Lorenz curve points for selected groups.

    Args:
        plot_df: Row-level concentration dataframe.
        metric: Selected metric.
        selected_groups: Groups to include in the Lorenz curve.

    Returns:
        pd.DataFrame: Long dataframe with cumulative album and metric shares.
    """
    parts = []

    for group_name in selected_groups:
        group_df = plot_df[plot_df["group"] == group_name].copy()
        values = (
            pd.to_numeric(group_df[metric], errors="coerce")
            .dropna()
            .to_numpy(dtype=float)
        )

        values = values[values > 0]
        if len(values) == 0:
            continue

        values = np.sort(values)
        total = values.sum()
        n = len(values)

        cum_metric_share = np.cumsum(values) / total
        cum_album_share = np.arange(1, n + 1) / n

        group_part = pd.DataFrame({
            "group": [group_name] * (n + 1),
            "cum_album_share": np.concatenate([[0.0], cum_album_share]),
            "cum_metric_share": np.concatenate([[0.0], cum_metric_share]),
        })
        parts.append(group_part)

    if not parts:
        return pd.DataFrame(
            columns=["group", "cum_album_share", "cum_metric_share"]
        )

    return pd.concat(parts, ignore_index=True)


def build_equality_line_df() -> pd.DataFrame:
    """
    Build the 45-degree equality reference line for the Lorenz chart.

    Returns:
        pd.DataFrame: Equality line endpoints.
    """
    return pd.DataFrame({
        "cum_album_share": [0.0, 1.0],
        "cum_metric_share": [0.0, 1.0],
    })


def get_axis_format(ranking_metric: str) -> str:
    """
    Return axis format string for a ranking metric.

    Args:
        ranking_metric: Selected ranking metric.

    Returns:
        str: Altair axis format string.
    """
    percent_metrics = {"top_1_share", "top_3_share", "top_5_share"}
    if ranking_metric in percent_metrics:
        return ".0%"
    if ranking_metric == "gini":
        return ".3f"
    return ",.0f"


def get_composition_top_k(ranking_metric: str) -> int:
    """
    Return the number of named album segments to show in the composition chart.

    Args:
        ranking_metric: Selected ranking metric.

    Returns:
        int: Number of individually named top albums to show.
    """
    if ranking_metric == "top_1_share":
        return 1
    if ranking_metric == "top_3_share":
        return 3
    return 5


def get_composition_basis(ranking_metric: str) -> str:
    """
    Return the composition basis for the stacked composition chart.

    Args:
        ranking_metric: Selected ranking metric.

    Returns:
        str: Composition basis for the stacked chart.
    """
    if ranking_metric in {"total_metric", "gini", "top_1_share", "top_3_share", "top_5_share"}:
        return "metric"
    return "metric"


def build_top_k_composition_df(
    plot_df: pd.DataFrame,
    metric: str,
    ranking_metric: str,
    displayed_groups: list[str],
    include_others: bool,
) -> pd.DataFrame:
    """
    Build stacked composition data across all displayed groups.

    For Top-K share metrics, the chart shows the exact Top-K window plus an
    optional Others tail. For album_count, total_metric, and gini, the chart
    shows top 5 albums plus Others. Album_count uses equal album-count share
    per segment; all other ranking views use metric share.

    Args:
        plot_df: Row-level concentration dataframe.
        metric: Selected metric.
        ranking_metric: Selected ranking metric.
        displayed_groups: Groups currently shown in the ranking view.
        include_others: Whether to add an Others segment.

    Returns:
        pd.DataFrame: One-row-per-segment composition dataframe.
    """
    top_k = get_composition_top_k(ranking_metric)
    basis = get_composition_basis(ranking_metric)

    filtered_df = plot_df[plot_df["group"].isin(displayed_groups)].copy()
    parts = []

    group_order_map = {group_name: idx for idx, group_name in enumerate(displayed_groups)}

    for group_name, group_df in filtered_df.groupby("group", dropna=False):
        group_df = group_df.sort_values(metric, ascending=False).reset_index(drop=True)
        group_df["rank_within_group"] = np.arange(1, len(group_df) + 1)

        total_metric = float(group_df[metric].sum())
        total_album_count = int(len(group_df))

        if total_album_count == 0:
            continue

        top_df = group_df.head(top_k).copy()

        if basis == "album_count":
            top_df["composition_value"] = 1.0
            top_df["composition_share"] = 1.0 / total_album_count
            x_title = "Share of Group Album Count"
            tooltip_share_title = "Share of Group Album Count"
        else:
            if total_metric <= 0:
                continue
            top_df["composition_value"] = top_df[metric].astype(float)
            top_df["composition_share"] = top_df[metric] / total_metric
            x_title = "Share of Group Total"
            tooltip_share_title = "Share of Group Total"

        top_df["segment_label"] = "#" + top_df["rank_within_group"].astype(str)
        top_df["segment_order"] = top_df["rank_within_group"]
        top_df["group_order"] = group_order_map[group_name]
        top_df["x_title"] = x_title
        top_df["tooltip_share_title"] = tooltip_share_title
        parts.append(top_df)

        if include_others and len(group_df) > top_k:
            others_df = group_df.iloc[top_k:].copy()

            if basis == "album_count":
                others_value = float(len(others_df))
                others_share = float(len(others_df) / total_album_count)
                others_metric_value = float(others_df[metric].sum())
            else:
                others_metric_value = float(others_df[metric].sum())
                others_value = others_metric_value
                others_share = float(others_metric_value / total_metric)

            others_row = pd.DataFrame({
                "group": [group_name],
                "album_title": ["Others"],
                "film_title": [f"{len(others_df)} additional albums"],
                "rank_within_group": [top_k + 1],
                "segment_order": [top_k + 1],
                "segment_label": ["Others"],
                metric: [others_metric_value],
                "composition_value": [others_value],
                "composition_share": [others_share],
                "group_order": [group_order_map[group_name]],
                "x_title": [x_title],
                "tooltip_share_title": [tooltip_share_title],
            })
            parts.append(others_row)

    if not parts:
        return pd.DataFrame()

    composition_df = pd.concat(parts, ignore_index=True)
    return composition_df


def create_concentration_ranking_chart(
    summary_df: pd.DataFrame,
    group_var: str,
    ranking_metric: str,
    metric: str,
    height: int,
    label_font_size: int,
) -> alt.Chart:
    """
    Create horizontal ranking chart for concentration metrics.

    Args:
        summary_df: One-row-per-group concentration summary.
        group_var: Grouping field.
        ranking_metric: Selected group ranking statistic.
        metric: Selected album performance metric.
        height: Chart height in pixels.
        label_font_size: Y-axis label font size.

    Returns:
        alt.Chart: Ranking bar chart.
    """
    axis_format = get_axis_format(ranking_metric)
    metric_label = get_display_label(metric)

    if ranking_metric == "total_metric":
        title_text = f"Total {metric_label} by {get_display_label(group_var)}"
        subtitle_text = f"Ranked by summed {metric_label.lower()} across albums in each group"
    elif ranking_metric == "gini":
        title_text = f"Gini by {get_display_label(group_var)}"
        subtitle_text = f"Ranked by inequality in the distribution of {metric_label.lower()}"
    else:
        title_text = (
            f"{get_display_label(ranking_metric)} by "
            f"{get_display_label(group_var)}"
        )
        subtitle_text = (
            f"Ranked by {get_display_label(ranking_metric).lower()} "
            f"of total {metric_label.lower()}"
        )

    chart = (
        alt.Chart(summary_df)
        .mark_bar()
        .encode(
            x=alt.X(
                f"{ranking_metric}:Q",
                title=get_display_label(ranking_metric),
                axis=alt.Axis(format=axis_format),
            ),
            y=alt.Y(
                "group:N",
                sort=alt.SortField(field=ranking_metric, order="descending"),
                title=get_display_label(group_var),
                axis=alt.Axis(labelFontSize=label_font_size),
            ),
            tooltip=[
                alt.Tooltip("group:N", title=get_display_label(group_var)),
                alt.Tooltip("album_count:Q", title="Albums", format=",.0f"),
                alt.Tooltip(
                    "total_metric:Q",
                    title=f"Total {metric_label}",
                    format=",.0f",
                ),
                alt.Tooltip("top_1_share:Q", title="Top 1 Share", format=".1%"),
                alt.Tooltip("top_3_share:Q", title="Top 3 Share", format=".1%"),
                alt.Tooltip("top_5_share:Q", title="Top 5 Share", format=".1%"),
                alt.Tooltip("gini:Q", title="Gini", format=".3f"),
                alt.Tooltip("albums_to_50pct:Q", title="Albums to 50%", format=",.0f"),
                alt.Tooltip("albums_to_80pct:Q", title="Albums to 80%", format=",.0f"),
            ],
        )
        .properties(
            width=750,
            height=height,
            title={
                "text": title_text,
                "subtitle": [subtitle_text],
            },
        )
    )

    return chart


def create_top_k_composition_chart(
    composition_df: pd.DataFrame,
    group_var: str,
    metric: str,
    ranking_metric: str,
    height: int,
    label_font_size: int,
) -> alt.Chart:
    """
    Create a multi-group stacked composition chart.

    Args:
        composition_df: One-row-per-segment composition dataframe.
        group_var: Grouping field.
        metric: Selected album performance metric.
        ranking_metric: Selected group ranking statistic.
        height: Chart height in pixels.
        label_font_size: Y-axis label font size.

    Returns:
        alt.Chart: Horizontal stacked composition chart.
    """
    if composition_df.empty:
        return alt.Chart(pd.DataFrame({"x": [], "y": []})).mark_bar()

    metric_label = get_display_label(metric)

    if ranking_metric == "top_1_share":
        subtitle = f"Top contributing album in each group, based on {metric_label.lower()}"
    elif ranking_metric == "top_3_share":
        subtitle = f"Top 3 contributing albums in each group, based on {metric_label.lower()}"
    elif ranking_metric == "top_5_share":
        subtitle = f"Top 5 contributing albums in each group, based on {metric_label.lower()}"
    elif ranking_metric == "total_metric":
        subtitle = f"Top 5 albums plus Others, based on total {metric_label.lower()}"
    else:
        subtitle = f"Top 5 albums plus Others, shown alongside Gini and based on {metric_label.lower()}"

    chart = (
        alt.Chart(composition_df)
        .mark_bar()
        .encode(
            x=alt.X(
                "composition_share:Q",
                stack="zero",
                title=f"Share of Group Total {metric_label}",
                axis=alt.Axis(format=".0%"),
            ),
            y=alt.Y(
                "group:N",
                sort=alt.SortField(field="group_order", order="ascending"),
                title=get_display_label(group_var),
                axis=alt.Axis(labelFontSize=label_font_size),
            ),
            color=alt.Color(
                "segment_label:N",
                title="Album Segment",
                legend=alt.Legend(orient="bottom"),
            ),
            order=alt.Order("segment_order:Q", sort="ascending"),
            tooltip=[
                alt.Tooltip("group:N", title=get_display_label(group_var)),
                alt.Tooltip("segment_label:N", title="Segment"),
                alt.Tooltip("album_title:N", title="Album"),
                alt.Tooltip("film_title:N", title="Film"),
                alt.Tooltip("rank_within_group:Q", title="Album Rank", format=",.0f"),
                alt.Tooltip(metric, title=metric_label, format=",.0f"),
                alt.Tooltip(
                    "composition_share:Q",
                    title=f"Share of Group Total {metric_label}",
                    format=".1%",
                ),
            ],
        )
        .properties(
            width=750,
            height=height,
            title={
                "text": f"Composition by {get_display_label(group_var)}",
                "subtitle": [subtitle],
            },
        )
    )

    return chart


def create_lorenz_curve_chart(
    lorenz_df: pd.DataFrame,
    equality_df: pd.DataFrame,
    group_var: str,
    metric: str,
) -> alt.Chart:
    """
    Create Lorenz curve chart with equality reference line.

    Args:
        lorenz_df: Long dataframe of Lorenz points.
        equality_df: Equality line dataframe.
        group_var: Grouping field.
        metric: Selected album performance metric.

    Returns:
        alt.Chart: Layered Lorenz curve chart.
    """
    metric_label = get_display_label(metric)

    equality_line = (
        alt.Chart(equality_df)
        .mark_line(strokeDash=[6, 4], opacity=0.8)
        .encode(
            x=alt.X(
                "cum_album_share:Q",
                title="Cumulative Share of Albums",
                axis=alt.Axis(format="%"),
            ),
            y=alt.Y(
                "cum_metric_share:Q",
                title=f"Cumulative Share of {metric_label}",
                axis=alt.Axis(format="%"),
            ),
        )
    )

    lorenz_lines = (
        alt.Chart(lorenz_df)
        .mark_line(strokeWidth=2.5)
        .encode(
            x=alt.X(
                "cum_album_share:Q",
                title="Cumulative Share of Albums",
                axis=alt.Axis(format="%"),
            ),
            y=alt.Y(
                "cum_metric_share:Q",
                title=f"Cumulative Share of {metric_label}",
                axis=alt.Axis(format="%"),
            ),
            color=alt.Color("group:N", title=get_display_label(group_var)),
            tooltip=[
                alt.Tooltip("group:N", title=get_display_label(group_var)),
                alt.Tooltip(
                    "cum_album_share:Q",
                    title="Cumulative Album Share",
                    format=".1%",
                ),
                alt.Tooltip(
                    "cum_metric_share:Q",
                    title=f"Cumulative {metric_label} Share",
                    format=".1%",
                ),
            ],
        )
    )

    lorenz_points = (
        alt.Chart(lorenz_df)
        .mark_point(size=45, filled=True)
        .encode(
            x=alt.X(
                "cum_album_share:Q",
                title="Cumulative Share of Albums",
                axis=alt.Axis(format="%"),
            ),
            y=alt.Y(
                "cum_metric_share:Q",
                title=f"Cumulative Share of {metric_label}",
                axis=alt.Axis(format="%"),
            ),
            color=alt.Color("group:N", title=get_display_label(group_var)),
            tooltip=[
                alt.Tooltip("group:N", title=get_display_label(group_var)),
                alt.Tooltip(
                    "cum_album_share:Q",
                    title="Cumulative Album Share",
                    format=".1%",
                ),
                alt.Tooltip(
                    "cum_metric_share:Q",
                    title=f"Cumulative {metric_label} Share",
                    format=".1%",
                ),
            ],
        )
    )

    return (
        (equality_line + lorenz_lines + lorenz_points)
        .properties(
            width=750,
            height=450,
            title={
                "text": "Lorenz Curve",
                "subtitle": [
                    (
                        f"Compare cumulative catalog share to cumulative "
                        f"{metric_label.lower()} share"
                    )
                ],
            },
        )
    )


def create_concentration_histogram(
    summary_df: pd.DataFrame,
    ranking_metric: str,
    bins: int,
    metric: str,
) -> alt.Chart:
    """
    Create histogram of group-level concentration values.

    Args:
        summary_df: One-row-per-group concentration summary.
        ranking_metric: Selected group ranking statistic.
        bins: Histogram bin count.
        metric: Selected album performance metric.

    Returns:
        alt.Chart: Histogram of group concentration values.
    """
    axis_format = get_axis_format(ranking_metric)
    metric_label = get_display_label(metric)

    if ranking_metric == "total_metric":
        subtitle = f"Histogram of summed {metric_label.lower()} across displayed groups"
    elif ranking_metric == "gini":
        subtitle = f"Histogram of inequality in {metric_label.lower()} distribution"
    else:
        subtitle = f"Histogram of {get_display_label(ranking_metric).lower()} of total {metric_label.lower()}"

    return (
        alt.Chart(summary_df)
        .mark_bar(opacity=0.85)
        .encode(
            x=alt.X(
                f"{ranking_metric}:Q",
                bin=alt.Bin(maxbins=bins),
                title=get_display_label(ranking_metric),
                axis=alt.Axis(format=axis_format),
            ),
            y=alt.Y("count():Q", title="Group Count"),
        )
        .properties(
            width=750,
            height=320,
            title={
                "text": f"Distribution of {get_display_label(ranking_metric)}",
                "subtitle": [subtitle],
            },
        )
    )


def build_concentration_source_table(
    summary_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build display table for group-level concentration metrics.

    Args:
        summary_df: One-row-per-group concentration summary.

    Returns:
        pd.DataFrame: Renamed summary table.
    """
    return rename_columns_for_display(summary_df)


def build_group_concentration_drilldown_table(
    plot_df: pd.DataFrame,
    metric: str,
    inspect_group: str,
) -> pd.DataFrame:
    """
    Build album-level drilldown table for one selected group.

    Args:
        plot_df: Row-level concentration dataframe.
        metric: Selected metric.
        inspect_group: Group name to inspect.

    Returns:
        pd.DataFrame: Album-level detail table with rank and shares.
    """
    detail_df = plot_df[plot_df["group"] == inspect_group].copy()

    if detail_df.empty:
        return pd.DataFrame()

    detail_df = detail_df.sort_values(metric, ascending=False).reset_index(drop=True)
    detail_df["rank_within_group"] = np.arange(1, len(detail_df) + 1)

    total_metric = detail_df[metric].sum()
    detail_df["share_of_group_total"] = detail_df[metric] / total_metric
    detail_df["cumulative_share"] = detail_df["share_of_group_total"].cumsum()

    preferred_cols = [
        "rank_within_group",
        "album_title",
        "film_title",
        "composer_primary_clean",
        "label_names",
        "film_year",
        "n_tracks",
        metric,
        "share_of_group_total",
        "cumulative_share",
    ]

    cols = [col for col in preferred_cols if col in detail_df.columns]
    detail_df = detail_df[cols].copy()

    return rename_columns_for_display(detail_df)

def build_pareto_df(
    plot_df: pd.DataFrame,
    metric: str,
    inspect_group: str,
) -> pd.DataFrame:
    """
    Build Pareto-chart data for one selected group.

    Args:
        plot_df: Row-level concentration dataframe.
        metric: Selected album performance metric.
        inspect_group: Group name to inspect.

    Returns:
        pd.DataFrame: Album-level dataframe sorted descending by metric with
        rank, share of group total, and cumulative share.
    """
    pareto_df = plot_df[plot_df["group"] == inspect_group].copy()

    if pareto_df.empty:
        return pd.DataFrame()

    pareto_df = pareto_df.sort_values(metric, ascending=False).reset_index(drop=True)
    pareto_df["rank_within_group"] = np.arange(1, len(pareto_df) + 1)

    total_metric = float(pareto_df[metric].sum())
    if total_metric <= 0:
        return pd.DataFrame()

    pareto_df["share_of_group_total"] = pareto_df[metric] / total_metric
    pareto_df["cumulative_share"] = pareto_df["share_of_group_total"].cumsum()

    return pareto_df

def build_pareto_df(
    plot_df: pd.DataFrame,
    metric: str,
    inspect_group: str,
) -> pd.DataFrame:
    """
    Build Pareto-chart data for one selected group.

    Args:
        plot_df: Row-level concentration dataframe.
        metric: Selected album performance metric.
        inspect_group: Group name to inspect.

    Returns:
        pd.DataFrame: Album-level dataframe sorted descending by metric with
        rank, share of group total, and cumulative share.
    """
    pareto_df = plot_df[plot_df["group"] == inspect_group].copy()

    if pareto_df.empty:
        return pd.DataFrame()

    pareto_df = pareto_df.sort_values(metric, ascending=False).reset_index(drop=True)
    pareto_df["rank_within_group"] = np.arange(1, len(pareto_df) + 1)

    total_metric = float(pareto_df[metric].sum())
    if total_metric <= 0:
        return pd.DataFrame()

    pareto_df["share_of_group_total"] = pareto_df[metric] / total_metric
    pareto_df["cumulative_share"] = pareto_df["share_of_group_total"].cumsum()

    return pareto_df

def create_pareto_chart(
    pareto_df: pd.DataFrame,
    metric: str,
    inspect_group: str,
    group_var: str,
) -> alt.Chart:
    """
    Create a Pareto chart for one selected group.

    Args:
        pareto_df: Album-level Pareto dataframe for one group.
        metric: Selected album performance metric.
        inspect_group: Group name being inspected.
        group_var: Grouping field used on the page.

    Returns:
        alt.Chart: Layered Pareto chart with bars and cumulative-share line.
    """
    if pareto_df.empty:
        return alt.Chart(pd.DataFrame({"x": [], "y": []})).mark_bar()

    metric_label = get_display_label(metric)

    bars = (
        alt.Chart(pareto_df)
        .mark_bar()
        .encode(
            x=alt.X(
                "rank_within_group:O",
                title="Album Rank Within Group",
            ),
            y=alt.Y(
                f"{metric}:Q",
                title=metric_label,
            ),
            tooltip=[
                alt.Tooltip("group:N", title=get_display_label(group_var)),
                alt.Tooltip("rank_within_group:Q", title="Album Rank", format=",.0f"),
                alt.Tooltip("album_title:N", title="Album"),
                alt.Tooltip("film_title:N", title="Film"),
                alt.Tooltip(metric, title=metric_label, format=",.0f"),
                alt.Tooltip(
                    "share_of_group_total:Q",
                    title=f"Share of Group Total {metric_label}",
                    format=".1%",
                ),
                alt.Tooltip(
                    "cumulative_share:Q",
                    title="Cumulative Share",
                    format=".1%",
                ),
            ],
        )
    )

    line = (
        alt.Chart(pareto_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("rank_within_group:O"),
            y=alt.Y(
                "cumulative_share:Q",
                title="Cumulative Share",
                axis=alt.Axis(format=".0%"),
            ),
            tooltip=[
                alt.Tooltip("rank_within_group:Q", title="Album Rank", format=",.0f"),
                alt.Tooltip("album_title:N", title="Album"),
                alt.Tooltip(
                    "cumulative_share:Q",
                    title="Cumulative Share",
                    format=".1%",
                ),
            ],
        )
    )

    return (
        alt.layer(bars, line)
        .resolve_scale(y="independent")
        .properties(
            width=750,
            height=360,
            title={
                "text": f"Pareto View: {inspect_group}",
                "subtitle": [
                    f"Albums ranked by {metric_label.lower()} with cumulative share overlay"
                ],
            },
        )
    )

def main() -> None:
    """Render the Concentration Explorer page."""
    st.set_page_config(
        page_title="Concentration Explorer",
        layout="wide",
    )
    apply_app_styles()

    st.title("Concentration Explorer")
    st.write(
        """
        Evaluate whether album success within a composer or label catalog is
        broadly distributed or concentrated in a few breakout albums.

        Top-K Share, Gini, and Total Metric are computed from the selected
        album performance metric.
        """
    )

    explorer_df = load_explorer_data()
    concentration_df = build_concentration_explorer_df(explorer_df)

    min_year = int(explorer_df["film_year"].dropna().min())
    max_year = int(explorer_df["film_year"].dropna().max())

    film_genre_options = split_multivalue_genres(explorer_df["film_genres"])
    album_genre_options = split_multivalue_genres(
        explorer_df["album_genres_display"]
    )

    global_filters = get_global_filter_controls(
        min_year=min_year,
        max_year=max_year,
        film_genre_options=film_genre_options,
        album_genre_options=album_genre_options,
    )

    filtered_df = filter_dataset(concentration_df, global_filters)

    metric_options = [
        col for col in PREFERRED_CONCENTRATION_METRICS
        if col in filtered_df.columns
    ]
    group_options = get_concentration_group_options(filtered_df)
    group_value_options_map = build_group_value_options_map(
        filtered_df,
        group_options,
    )

    if not metric_options:
        st.warning("No supported concentration metrics are available.")
        return

    if not group_options:
        st.warning("No supported grouping fields are available.")
        return

    controls = get_concentration_controls(
        metric_options=metric_options,
        group_options=group_options,
        group_value_options_map=group_value_options_map,
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Composition Chart")

    composition_include_others = st.sidebar.checkbox(
        "Include Others segment",
        value=True,
        help="Complete the stacked composition chart with a remaining-catalog segment.",
    )

    metric = controls["metric"]
    group_var = controls["group_var"]
    selected_groups = controls["selected_groups"]
    top_n = controls["top_n"]
    min_group_size = controls["min_group_size"]
    ranking_metric = controls["ranking_metric"]

    metric_label = get_display_label(metric)

    plot_df = prepare_concentration_data(
        df=filtered_df,
        metric=metric,
        group_var=group_var,
        selected_groups=selected_groups,
        top_n=top_n,
        min_group_size=min_group_size,
    )

    if plot_df.empty:
        st.warning(
            "No valid rows remain after applying the current filters and concentration settings."
        )
        return

    summary_df = build_concentration_summary_df(
        plot_df=plot_df,
        metric=metric,
    )

    if summary_df.empty:
        st.warning("No valid groups remain for concentration analysis.")
        return

    summary_df = sort_summary_df(
        summary_df=summary_df,
        ranking_metric=ranking_metric,
    )
    displayed_groups = summary_df["group"].tolist()

    n_groups = len(displayed_groups)
    label_font_size = 12 if n_groups <= 8 else 10

    ranking_chart_height = max(420, min(1000, n_groups * 40))
    composition_chart_height = max(440, min(1000, n_groups * 50))

    lorenz_groups = controls["lorenz_groups"]
    if not lorenz_groups:
        lorenz_groups = summary_df.head(6)["group"].tolist()

    lorenz_groups = [group for group in lorenz_groups if group in displayed_groups]

    lorenz_df = build_lorenz_curve_df(
        plot_df=plot_df,
        metric=metric,
        selected_groups=lorenz_groups,
    )
    equality_df = build_equality_line_df()

    composition_df = build_top_k_composition_df(
        plot_df=plot_df,
        metric=metric,
        ranking_metric=ranking_metric,
        displayed_groups=displayed_groups,
        include_others=composition_include_others,
    )

    median_group_size = summary_df["album_count"].median()
    median_gini = summary_df["gini"].median()
    median_top_1 = summary_df["top_1_share"].median()

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Groups Shown", f"{len(summary_df):,}")

    with col2:
        st.metric("Albums in View", f"{len(plot_df):,}")

    with col3:
        st.metric("Median Group Size", f"{median_group_size:.0f}")

    with col4:
        st.metric("Median Gini", f"{median_gini:.3f}")

    with col5:
        st.metric("Median Top 1 Share", f"{median_top_1:.1%}")

    st.caption(
        f"Using **{metric_label}** as the album performance metric. "
        "Total Metric, Top-K Share, and Gini are all computed from this selection."
    )

    if selected_groups:
        st.caption(
            f"Showing selected {get_display_label(group_var)} values: "
            f"{', '.join(selected_groups[:5])}"
            + (" ..." if len(selected_groups) > 5 else "")
            + ". Albums in View reflects only albums that remain after global "
              "filters, metric filtering, and minimum group size filtering."
        )
    elif top_n is not None:
        st.caption(
            f"Showing top {top_n} {get_display_label(group_var)} groups by album count "
            f"with minimum group size = {min_group_size}."
        )

    ranking_chart = create_concentration_ranking_chart(
        summary_df=summary_df,
        group_var=group_var,
        ranking_metric=ranking_metric,
        metric=metric,
        height=ranking_chart_height,
        label_font_size=label_font_size,
    )
    st.altair_chart(ranking_chart, width="stretch")

    if not composition_df.empty:
        composition_chart = create_top_k_composition_chart(
            composition_df=composition_df,
            group_var=group_var,
            metric=metric,
            ranking_metric=ranking_metric,
            height=composition_chart_height,
            label_font_size=label_font_size,
        )
        st.altair_chart(composition_chart, width="stretch")

    if not lorenz_df.empty:
        lorenz_chart = create_lorenz_curve_chart(
            lorenz_df=lorenz_df,
            equality_df=equality_df,
            group_var=group_var,
            metric=metric,
        )
        st.altair_chart(lorenz_chart, width="stretch")
    else:
        st.info("No Lorenz curve groups are currently available to display.")

    histogram_chart = create_concentration_histogram(
        summary_df=summary_df,
        ranking_metric=ranking_metric,
        bins=controls["histogram_bins"],
        metric=metric,
    )
    st.altair_chart(histogram_chart, width="stretch")

    if controls["show_summary_table"]:
        st.subheader("Concentration Summary")
        table_df = build_concentration_source_table(summary_df)
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
        help="Show album-level contribution details for one displayed group.",
    )

    if inspect_group != "None":
        pareto_df = build_pareto_df(
            plot_df=plot_df,
            metric=metric,
            inspect_group=inspect_group,
        )

        if not pareto_df.empty:
            pareto_chart = create_pareto_chart(
                pareto_df=pareto_df,
                metric=metric,
                inspect_group=inspect_group,
                group_var=group_var,
            )
            st.altair_chart(pareto_chart, width="stretch")

            st.caption(
                "Bars show album-level performance ranked from highest to lowest. "
                "The line shows cumulative share of the group's total."
            )

        if controls["show_detail_table"]:
            detail_table = build_group_concentration_drilldown_table(
                plot_df=plot_df,
                metric=metric,
                inspect_group=inspect_group,
            )

            st.markdown(f"**Albums in {inspect_group}**")
            st.dataframe(
                detail_table,
                width="stretch",
                hide_index=True,
            )


if __name__ == "__main__":
    main()
from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from app.app_controls import (
    get_distribution_controls,
    get_global_filter_controls,
)
from app.app_data import load_explorer_data
from app.data_filters import filter_dataset
from app.explorer_shared import get_global_filter_inputs
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
    "album_cohesion_band",
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


def build_distribution_explorer_df(explorer_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived grouping fields used by the Distribution Explorer.
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


def get_distribution_group_options(df: pd.DataFrame) -> list[str]:
    """
    Return curated grouping fields for the Distribution Explorer.
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
        if col in {"album_us_release_year"}:
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


def prepare_distribution_data(
    df: pd.DataFrame,
    metric: str,
    use_log: bool,
    group_var: str,
    selected_groups: list[str],
    top_n: int | None,
) -> pd.DataFrame:
    """
    Prepare plot-ready data for distribution charts.
    """
    required_cols = [metric]
    grouped = group_var != "None"

    if grouped:
        required_cols.append(group_var)

    plot_df = df[required_cols].copy().dropna(subset=[metric])

    if grouped:
        plot_df = plot_df.dropna(subset=[group_var]).copy()

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

    if use_log:
        plot_df = plot_df[plot_df[metric] > 0].copy()
        plot_df["value"] = np.log10(plot_df[metric])
    else:
        plot_df["value"] = plot_df[metric]

    plot_df = plot_df.dropna(subset=["value"]).copy()

    return plot_df


def build_density_curve_df(
    plot_df: pd.DataFrame,
    grouped: bool,
) -> pd.DataFrame:
    """
    Build a simple smoothed density-style curve from histogram densities.
    """
    if plot_df.empty:
        return pd.DataFrame(columns=["x", "density", "group"])

    def single_density(values: pd.Series, group_name: str) -> pd.DataFrame:
        values = values.dropna().astype(float)
        if len(values) < 2 or values.nunique() < 2:
            return pd.DataFrame(columns=["x", "density", "group"])

        counts, edges = np.histogram(values, bins=40, density=True)
        centers = (edges[:-1] + edges[1:]) / 2
        density = pd.Series(counts).rolling(window=3, center=True, min_periods=1).mean()

        out = pd.DataFrame({
            "x": centers,
            "density": density,
            "group": group_name,
        })
        return out

    if grouped:
        parts = []
        for group_name, group_df in plot_df.groupby("group"):
            part = single_density(group_df["value"], group_name)
            if not part.empty:
                parts.append(part)

        if not parts:
            return pd.DataFrame(columns=["x", "density", "group"])

        return pd.concat(parts, ignore_index=True)

    return single_density(plot_df["value"], "All Albums")


def build_cdf_df(
    plot_df: pd.DataFrame,
    grouped: bool,
) -> pd.DataFrame:
    """
    Build empirical CDF data.
    """
    if plot_df.empty:
        return pd.DataFrame(columns=["x", "cdf", "group"])

    def single_cdf(values: pd.Series, group_name: str) -> pd.DataFrame:
        values = np.sort(values.dropna().astype(float).to_numpy())
        if len(values) == 0:
            return pd.DataFrame(columns=["x", "cdf", "group"])

        cdf = np.arange(1, len(values) + 1) / len(values)
        return pd.DataFrame({
            "x": values,
            "cdf": cdf,
            "group": group_name,
        })

    if grouped:
        parts = []
        for group_name, group_df in plot_df.groupby("group"):
            part = single_cdf(group_df["value"], group_name)
            if not part.empty:
                parts.append(part)

        if not parts:
            return pd.DataFrame(columns=["x", "cdf", "group"])

        return pd.concat(parts, ignore_index=True)

    return single_cdf(plot_df["value"], "All Albums")


def create_histogram_chart(
    plot_df: pd.DataFrame,
    metric: str,
    use_log: bool,
    bins: int,
    group_var: str,
) -> alt.Chart:
    """
    Create histogram chart.
    """
    x_title = (
        f"log10({get_display_label(metric)})"
        if use_log
        else get_display_label(metric)
    )

    grouped = group_var != "None"

    if grouped:
        chart = (
            alt.Chart(plot_df)
            .mark_bar(opacity=0.45)
            .encode(
                x=alt.X("value:Q", bin=alt.Bin(maxbins=bins), title=x_title),
                y=alt.Y("count():Q", title="Album Count"),
                color=alt.Color("group:N", title=get_display_label(group_var)),
                tooltip=[
                    alt.Tooltip("group:N", title=get_display_label(group_var)),
                    alt.Tooltip("count():Q", title="Albums"),
                ],
            )
        )
    else:
        chart = (
            alt.Chart(plot_df)
            .mark_bar(opacity=0.8)
            .encode(
                x=alt.X("value:Q", bin=alt.Bin(maxbins=bins), title=x_title),
                y=alt.Y("count():Q", title="Album Count"),
                tooltip=[alt.Tooltip("count():Q", title="Albums")],
            )
        )

    subtitle = "Histogram of album-level values"
    if use_log:
        subtitle += " on log10 scale"
    if grouped:
        subtitle += f" | Grouped by {get_display_label(group_var)}"

    return chart.properties(
        width=750,
        height=450,
        title={
            "text": f"Distribution of {get_display_label(metric)}",
            "subtitle": [subtitle],
        },
    )


def create_density_chart(
    density_df: pd.DataFrame,
    metric: str,
    use_log: bool,
    group_var: str,
) -> alt.Chart:
    """
    Create density chart.
    """
    x_title = (
        f"log10({get_display_label(metric)})"
        if use_log
        else get_display_label(metric)
    )

    grouped = group_var != "None"

    if grouped:
        chart = (
            alt.Chart(density_df)
            .mark_line(strokeWidth=2.5)
            .encode(
                x=alt.X("x:Q", title=x_title),
                y=alt.Y("density:Q", title="Density"),
                color=alt.Color("group:N", title=get_display_label(group_var)),
                tooltip=[
                    alt.Tooltip("group:N", title=get_display_label(group_var)),
                    alt.Tooltip("x:Q", title=x_title, format=",.3f"),
                    alt.Tooltip("density:Q", title="Density", format=".4f"),
                ],
            )
        )
    else:
        chart = (
            alt.Chart(density_df)
            .mark_line(strokeWidth=2.5)
            .encode(
                x=alt.X("x:Q", title=x_title),
                y=alt.Y("density:Q", title="Density"),
                tooltip=[
                    alt.Tooltip("x:Q", title=x_title, format=",.3f"),
                    alt.Tooltip("density:Q", title="Density", format=".4f"),
                ],
            )
        )

    subtitle = "Smoothed distribution curve"
    if use_log:
        subtitle += " on log10 scale"
    if grouped:
        subtitle += f" | Grouped by {get_display_label(group_var)}"

    return chart.properties(
        width=750,
        height=450,
        title={
            "text": f"Density of {get_display_label(metric)}",
            "subtitle": [subtitle],
        },
    )


def create_cdf_chart(
    cdf_df: pd.DataFrame,
    metric: str,
    use_log: bool,
    group_var: str,
) -> alt.Chart:
    """
    Create empirical CDF chart.
    """
    x_title = (
        f"log10({get_display_label(metric)})"
        if use_log
        else get_display_label(metric)
    )

    grouped = group_var != "None"

    if grouped:
        chart = (
            alt.Chart(cdf_df)
            .mark_line(strokeWidth=2.5)
            .encode(
                x=alt.X("x:Q", title=x_title),
                y=alt.Y(
                    "cdf:Q",
                    title="Cumulative Share",
                    axis=alt.Axis(format="%"),
                ),
                color=alt.Color("group:N", title=get_display_label(group_var)),
                tooltip=[
                    alt.Tooltip("group:N", title=get_display_label(group_var)),
                    alt.Tooltip("x:Q", title=x_title, format=",.3f"),
                    alt.Tooltip("cdf:Q", title="Cumulative Share", format=".1%"),
                ],
            )
        )
    else:
        chart = (
            alt.Chart(cdf_df)
            .mark_line(strokeWidth=2.5)
            .encode(
                x=alt.X("x:Q", title=x_title),
                y=alt.Y(
                    "cdf:Q",
                    title="Cumulative Share",
                    axis=alt.Axis(format="%"),
                ),
                tooltip=[
                    alt.Tooltip("x:Q", title=x_title, format=",.3f"),
                    alt.Tooltip("cdf:Q", title="Cumulative Share", format=".1%"),
                ],
            )
        )

    subtitle = "Empirical cumulative distribution"
    if use_log:
        subtitle += " on log10 scale"
    if grouped:
        subtitle += f" | Grouped by {get_display_label(group_var)}"

    return chart.properties(
        width=750,
        height=450,
        title={
            "text": f"CDF of {get_display_label(metric)}",
            "subtitle": [subtitle],
        },
    )


def build_source_table(
    plot_df: pd.DataFrame,
    metric: str,
    group_var: str,
) -> pd.DataFrame:
    """
    Build source table for display.
    """
    cols = [metric, "value"]
    if group_var != "None" and "group" in plot_df.columns:
        cols.append("group")

    table_df = plot_df[cols].copy()
    table_df = table_df.rename(columns={"value": "plot_value"})
    return rename_columns_for_display(table_df)


# ADD THESE NEW FUNCTIONS (place above main())

def get_view_type_explainer(view_type, metric, use_log, group_var):
    metric_label = get_display_label(metric)
    scale_text = " using log10 scale" if use_log else ""
    group_text = (
        f", grouped by {get_display_label(group_var)}"
        if group_var != "None"
        else ""
    )

    explainers = {
        "Histogram": f"Histogram = count of albums per value range for {metric_label}{scale_text}{group_text}.",
        "Density": f"Density = smoothed distribution of {metric_label}{scale_text}{group_text}. Useful for shape + overlap.",
        "CDF": f"CDF = cumulative share of albums ≤ value for {metric_label}{scale_text}{group_text}.",
    }
    return explainers.get(view_type, "")


def build_distribution_context_caption(
    metric: str,
    use_log: bool,
    group_var: str,
    selected_groups: list[str],
    top_n: int | None,
) -> str:
    """
    Build a natural-language caption describing the current distribution scope.
    """
    metric_label = get_display_label(metric)

    if group_var == "None":
        if use_log:
            return f"Showing the distribution of {metric_label} on the log10 scale across all albums in view."
        return f"Showing the distribution of {metric_label} across all albums in view."

    group_label = get_display_label(group_var).lower()

    if selected_groups:
        if len(selected_groups) <= 5:
            group_scope = f"the selected {group_label} values ({', '.join(selected_groups)})"
        else:
            shown = ", ".join(selected_groups[:5])
            group_scope = f"the selected {group_label} values ({shown}, ...)"
    else:
        group_scope = f"the top {top_n} {group_label} groups by album count"

    if use_log:
        return f"Comparing {metric_label} on the log10 scale across {group_scope}."
    return f"Comparing {metric_label} across {group_scope}."


def build_distribution_insight_summary(
    plot_df: pd.DataFrame,
    metric: str,
    group_var: str,
    use_log: bool,
) -> list[tuple[str, str, str]] | None:
    """
    Build top-line insight cards for the current distribution view.
    """
    if plot_df.empty:
        return None

    label_suffix = " (log10)" if use_log else ""

    if group_var == "None":
        median = float(plot_df["value"].median())
        p10 = float(plot_df["value"].quantile(0.10))
        p90 = float(plot_df["value"].quantile(0.90))
        spread = p90 - p10
        tail = (p90 / median) if median else 0.0

        return [
            ("Typical Value", f"{median:,.2f}", f"Median{label_suffix} across albums"),
            ("Spread", f"{spread:,.2f}", f"P90 - P10{label_suffix}"),
            ("Tail Strength", f"{tail:,.2f}×", f"Upper-tail stretch{label_suffix}"),
        ]

    summary = (
        plot_df.groupby("group")["value"]
        .agg(["count", "median"])
        .reset_index()
    )

    ranked = summary.sort_values(
        ["median", "count", "group"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    top = ranked.iloc[0]
    largest = summary.sort_values(
        ["count", "median", "group"],
        ascending=[False, False, True],
    ).iloc[0]

    if len(ranked) >= 2:
        median_gap = float(ranked.iloc[0]["median"] - ranked.iloc[1]["median"])
        gap_value = f"{median_gap:,.2f}"
        gap_caption = f"Gap between top two medians{label_suffix}"
    else:
        gap_value = "NA"
        gap_caption = "Only one visible group"

    return [
        ("Top Median Group", str(top["group"]), f"Median{label_suffix} = {top['median']:,.2f}"),
        ("Median Gap", gap_value, gap_caption),
        ("Largest Group", str(largest["group"]), f"{int(largest['count']):,} albums"),
    ]


def render_distribution_insight_cards(
    plot_df: pd.DataFrame,
    metric: str,
    group_var: str,
    use_log: bool,
) -> None:
    """
    Render top-line insight cards for the distribution view.
    """
    insights = build_distribution_insight_summary(
        plot_df=plot_df,
        metric=metric,
        group_var=group_var,
        use_log=use_log,
    )
    if not insights:
        return

    st.markdown("### 🧠 Key Insights")
    cols = st.columns(3)

    for i, (title, value, caption) in enumerate(insights):
        with cols[i]:
            st.metric(title, value)
            st.caption(caption)

def build_distribution_supporting_insight(
    plot_df: pd.DataFrame,
    metric: str,
    group_var: str,
    use_log: bool,
    view_type: str,
    selected_groups: list[str],
    top_n: int | None,
) -> str:
    """
    Build a short supporting insight for the current distribution view.

    The narrative adapts to:
        - raw vs log scale
        - grouped vs ungrouped view
        - histogram vs density vs CDF framing
        - selected groups vs top-N visible groups
    """
    if plot_df.empty:
        return "No visible pattern remains to summarize."

    metric_label = get_display_label(metric)
    grouped = group_var != "None"

    values = plot_df["value"].dropna().astype(float)

    if values.empty:
        return "No visible pattern remains to summarize."

    median = float(values.median())
    mean = float(values.mean())
    p10 = float(values.quantile(0.10))
    p90 = float(values.quantile(0.90))
    spread = p90 - p10

    # Build honest scope language.
    if grouped:
        if selected_groups:
            scope_text = "among the selected groups"
        elif top_n is not None:
            scope_text = f"among the top {top_n} visible groups by album count"
        else:
            scope_text = "across the visible groups"
    else:
        scope_text = "across albums in view"

    # Characterize shape differently on raw vs log scale.
    if use_log:
        mean_minus_median = mean - median
        p90_minus_median = p90 - median

        if mean_minus_median >= 0.30 or p90_minus_median >= 1.00:
            shape = "still meaningfully right-skewed even after log compression"
        elif mean_minus_median >= 0.15 or p90_minus_median >= 0.60:
            shape = "moderately right-skewed on the log scale"
        else:
            shape = "fairly compact on the log scale"

        scale_note = (
            " Because log scaling compresses large values, upper-tail differences "
            "look less extreme here than they do on the raw scale."
        )
    else:
        mean_to_median = (mean / median) if median != 0 else np.nan
        p90_to_median = (p90 / median) if median != 0 else np.nan

        if pd.notna(mean_to_median) and pd.notna(p90_to_median):
            if mean_to_median >= 1.5 or p90_to_median >= 3.0:
                shape = "strongly right-skewed"
            elif mean_to_median >= 1.2 or p90_to_median >= 2.0:
                shape = "moderately right-skewed"
            else:
                shape = "fairly compact"
        else:
            shape = "hard to characterize"

        scale_note = ""

    # Ungrouped insight
    if not grouped:
        if view_type == "Histogram":
            return (
                f"💡 {metric_label} appears {shape} {scope_text}. "
                f"The median plotted value is {median:,.2f}, while the middle-80% "
                f"spread runs from {p10:,.2f} to {p90:,.2f}, which gives a quick read "
                f"on how concentrated versus long-tailed the distribution is."
                f"{scale_note}"
            )

        if view_type == "Density":
            return (
                f"💡 The density curve suggests that {metric_label} is {shape} "
                f"{scope_text}. The center of the visible distribution sits around "
                f"{median:,.2f}, while the middle-80% spread extends from "
                f"{p10:,.2f} to {p90:,.2f}."
                f"{scale_note}"
            )

        # CDF
        return (
            f"💡 The CDF indicates that {metric_label} is {shape} {scope_text}. "
            f"Half of visible albums are at or below {median:,.2f}, and 90% are at "
            f"or below {p90:,.2f}, which shows how quickly the distribution accumulates."
            f"{scale_note}"
        )

    # Grouped insight
    group_summary = (
        plot_df.groupby("group")["value"]
        .agg(["count", "median", "mean"])
        .reset_index()
    )

    group_quantiles = (
        plot_df.groupby("group")["value"]
        .quantile([0.10, 0.90])
        .unstack()
        .reset_index()
        .rename(columns={0.10: "p10", 0.90: "p90"})
    )

    group_summary = group_summary.merge(group_quantiles, on="group", how="left")
    group_summary["spread"] = group_summary["p90"] - group_summary["p10"]

    highest_median_row = group_summary.sort_values(
        ["median", "count", "group"],
        ascending=[False, False, True],
    ).iloc[0]

    widest_spread_row = group_summary.sort_values(
        ["spread", "count", "group"],
        ascending=[False, False, True],
    ).iloc[0]

    largest_group_row = group_summary.sort_values(
        ["count", "median", "group"],
        ascending=[False, False, True],
    ).iloc[0]

    highest_group = str(highest_median_row["group"])
    widest_group = str(widest_spread_row["group"])
    largest_group = str(largest_group_row["group"])

    if view_type == "Histogram":
        return (
            f"💡 {scope_text.capitalize()}, {metric_label} appears {shape}. "
            f"{highest_group} has the highest median plotted value "
            f"({highest_median_row['median']:,.2f}), while {widest_group} shows the "
            f"widest middle-80% spread ({widest_spread_row['spread']:,.2f}). "
            f"The largest visible group is {largest_group} with "
            f"{int(largest_group_row['count']):,} albums."
            f"{scale_note}"
        )

    if view_type == "Density":
        return (
            f"💡 The density comparison suggests {metric_label} is {shape} "
            f"{scope_text}. {highest_group} sits highest on the median "
            f"({highest_median_row['median']:,.2f}), while {widest_group} spans the "
            f"broadest middle range ({widest_spread_row['spread']:,.2f})."
            f"{scale_note}"
        )

    # CDF
    return (
        f"💡 The grouped CDF view suggests {metric_label} is {shape} {scope_text}. "
        f"{highest_group} has the highest median plotted value "
        f"({highest_median_row['median']:,.2f}), while {largest_group} is the biggest "
        f"visible group with {int(largest_group_row['count']):,} albums."
        f"{scale_note}"
    )


def build_group_summary_table(plot_df: pd.DataFrame) -> pd.DataFrame | None:
    """
    Build a grouped summary table for the current distribution view.
    """
    if "group" not in plot_df.columns:
        return None

    summary = (
        plot_df.groupby("group")["value"]
        .agg(["count", "median", "mean"])
        .reset_index()
    )

    summary["p10"] = plot_df.groupby("group")["value"].quantile(0.10).values
    summary["p90"] = plot_df.groupby("group")["value"].quantile(0.90).values
    summary["spread"] = summary["p90"] - summary["p10"]

    return summary.sort_values(
        ["median", "count", "group"],
        ascending=[False, False, True],
    )

def main() -> None:
    """Render the Distribution Explorer page."""
    st.set_page_config(
        page_title="Distribution Explorer",
        layout="wide",
    )
    apply_app_styles()

    st.title("Distribution Explorer")
    st.write(
        """
        Explore the distribution of a single album-level variable.

        Each row represents one album.
        """
    )

    explorer_df = load_explorer_data()
    distribution_df = build_distribution_explorer_df(explorer_df)

    filter_inputs = get_global_filter_inputs(distribution_df)

    global_controls = get_global_filter_controls(
        min_year=filter_inputs["min_year"],
        max_year=filter_inputs["max_year"],
        film_genre_options=filter_inputs["film_genre_options"],
        album_genre_options=filter_inputs["album_genre_options"],
    )

    filtered_distribution_df = filter_dataset(
        distribution_df,
        global_controls,
    ).copy()

    if filtered_distribution_df.empty:
        st.warning("No albums remain under the current global filters.")
        return

    numeric_options = [
        col for col in PREFERRED_NUMERIC_COLS
        if col in filtered_distribution_df.columns
    ]
    group_options = get_distribution_group_options(filtered_distribution_df)
    group_value_options_map = build_group_value_options_map(
        filtered_distribution_df,
        group_options,
    )

    controls = get_distribution_controls(
        numeric_options=numeric_options,
        group_options=group_options,
        group_value_options_map=group_value_options_map,
    )

    metric = controls["metric"]
    view_type = controls["view_type"]
    use_log = controls["use_log"]
    group_var = controls["group_var"]
    selected_groups = controls["selected_groups"]
    top_n = controls["top_n"]

    plot_df = prepare_distribution_data(
        df=filtered_distribution_df,
        metric=metric,
        use_log=use_log,
        group_var=group_var,
        selected_groups=selected_groups,
        top_n=top_n,
    )

    if plot_df.empty:
        st.warning(
            "No valid rows remain for this metric after applying the current settings."
        )
        return

    st.subheader(get_display_label(metric))

    st.caption(
        get_view_type_explainer(
            view_type=view_type,
            metric=metric,
            use_log=use_log,
            group_var=group_var,
        )
    )

    st.markdown("**Filter Context**")
    st.caption(
        f"{len(filtered_distribution_df):,} albums remain after shared year/genre filtering."
    )
    st.markdown("**View Context**")
    st.caption(
        build_distribution_context_caption(
            metric=metric,
            use_log=use_log,
            group_var=group_var,
            selected_groups=selected_groups,
            top_n=top_n,
        )
    )

    render_distribution_insight_cards(
        plot_df=plot_df,
        metric=metric,
        group_var=group_var,
        use_log=use_log,
    )

    col1, col2, col3, col4 = st.columns(4)
    median_val = plot_df["value"].median()
    mean_val = plot_df["value"].mean()
    p90_val = plot_df["value"].quantile(0.9)

    label_suffix = " (log10)" if use_log else ""

    col1.metric("Albums", f"{len(plot_df):,}")
    col2.metric(f"Median{label_suffix}", f"{median_val:.2f}")
    col3.metric(f"Mean{label_suffix}", f"{mean_val:.2f}")
    col4.metric(f"P90{label_suffix}", f"{p90_val:.2f}")

    if use_log:
        st.caption(
            "Values are shown on the log10 scale. Differences reflect orders of magnitude rather than raw units."
        )

    if group_var != "None":
        if selected_groups:
            st.caption(
                f"Showing selected {get_display_label(group_var)} values: "
                f"{', '.join(selected_groups[:5])}"
                + (" ..." if len(selected_groups) > 5 else "")
            )
        elif top_n is not None:
            st.caption(
                f"Showing top {top_n} {get_display_label(group_var)} groups by album count."
            )

    if view_type == "Histogram":
        chart = create_histogram_chart(
            plot_df=plot_df,
            metric=metric,
            use_log=use_log,
            bins=controls["bins"],
            group_var=group_var,
        )
    elif view_type == "Density":
        density_df = build_density_curve_df(
            plot_df=plot_df,
            grouped=(group_var != "None"),
        )

        if density_df.empty:
            st.warning(
                "Not enough variation to compute a density curve for the current selection."
            )
            return

        chart = create_density_chart(
            density_df=density_df,
            metric=metric,
            use_log=use_log,
            group_var=group_var,
        )
    else:
        cdf_df = build_cdf_df(
            plot_df=plot_df,
            grouped=(group_var != "None"),
        )

        if cdf_df.empty:
            st.warning(
                "Not enough valid rows to compute a CDF for the current selection."
            )
            return

        chart = create_cdf_chart(
            cdf_df=cdf_df,
            metric=metric,
            use_log=use_log,
            group_var=group_var,
        )

    st.altair_chart(chart, width="stretch")

    st.caption(
        build_distribution_supporting_insight(
            plot_df=plot_df,
            metric=metric,
            group_var=group_var,
            use_log=use_log,
            view_type=view_type,
            selected_groups=selected_groups,
            top_n=top_n,
        )
    )

    if group_var != "None":
        summary_df = build_group_summary_table(plot_df)
        if summary_df is not None:
            st.subheader("Group Summary")
            st.dataframe(
                rename_columns_for_display(summary_df),
                width="stretch",
                hide_index=True,
            )

    if controls["show_table"]:
        st.subheader("Source Data")
        table_df = build_source_table(
            plot_df=plot_df,
            metric=metric,
            group_var=group_var,
        )
        st.dataframe(
            table_df,
            width="stretch",
            hide_index=True,
        )


if __name__ == "__main__":
    main()
from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from app.app_controls import get_distribution_controls
from app.app_data import load_explorer_data
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

    numeric_options = [
        col for col in PREFERRED_NUMERIC_COLS
        if col in distribution_df.columns
    ]
    group_options = get_distribution_group_options(distribution_df)
    group_value_options_map = build_group_value_options_map(
        distribution_df,
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
        df=distribution_df,
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

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Albums", f"{len(plot_df):,}")
    col2.metric("Median", f"{plot_df['value'].median():.2f}")
    col3.metric("Mean", f"{plot_df['value'].mean():.2f}")
    col4.metric("P90", f"{plot_df['value'].quantile(0.9):.2f}")

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
from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from app.app_controls import (
    get_global_filter_controls,
    get_track_distribution_controls,
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
)
from app.ui import apply_app_styles, get_display_label

def add_display_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Add grouped display fields used by the Track Distribution Explorer."""
    df = add_standard_multivalue_groups(df)
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


def build_group_value_options_map(
    df: pd.DataFrame,
    group_options: list[str],
) -> dict[str, list[str]]:
    """Build selectable value lists for each grouping field."""
    value_map: dict[str, list[str]] = {}

    for col in group_options:
        values = (
            df[col]
            .dropna()
            .astype(str)
            .str.strip()
            .replace("", pd.NA)
            .dropna()
            .unique()
            .tolist()
        )
        value_map[col] = sorted(values)

    return value_map


def prepare_distribution_data(
    df: pd.DataFrame,
    metric: str,
    group_var: str,
    selected_groups: list[str],
    top_n_groups: int | None,
    use_log: bool,
) -> pd.DataFrame:
    """Prepare the row-level dataframe used for the distribution charts."""
    required_cols = [metric]
    if group_var != "None":
        required_cols.append(group_var)

    plot_df = df[[col for col in required_cols if col in df.columns]].copy()
    plot_df = plot_df.dropna(subset=[metric]).copy()

    if group_var != "None":
        plot_df = plot_df.dropna(subset=[group_var]).copy()
        plot_df[group_var] = plot_df[group_var].astype(str)

        if selected_groups:
            plot_df = plot_df[plot_df[group_var].isin(selected_groups)].copy()
        elif top_n_groups is not None:
            top_groups = (
                plot_df[group_var]
                .value_counts()
                .head(top_n_groups)
                .index
                .tolist()
            )
            plot_df = plot_df[plot_df[group_var].isin(top_groups)].copy()

    if use_log:
        plot_df = plot_df[plot_df[metric] > 0].copy()
        plot_df["plot_value"] = np.log10(plot_df[metric])
    else:
        plot_df["plot_value"] = plot_df[metric]

    plot_df = plot_df.dropna(subset=["plot_value"]).copy()
    return plot_df


def build_distribution_context_caption(
    metric: str,
    view_type: str,
    use_log: bool,
    group_var: str,
    selected_groups: list[str],
    top_n_groups: int | None,
) -> str:
    """Build a short caption describing the current distribution view."""
    metric_label = get_display_label(metric)

    if group_var == "None":
        return (
            f"Showing the {view_type.lower()} of {metric_label.lower()} "
            f"across visible tracks"
            + (" on the log10 scale." if use_log else ".")
        )

    group_label = get_display_label(group_var).lower()

    if selected_groups:
        if len(selected_groups) <= 5:
            group_scope = f"the selected {group_label} values ({', '.join(selected_groups)})"
        else:
            shown = ", ".join(selected_groups[:5])
            group_scope = f"the selected {group_label} values ({shown}, ...)"
    else:
        group_scope = f"the top {top_n_groups} visible {group_label} groups by track count"

    return (
        f"Showing the {view_type.lower()} of {metric_label.lower()} "
        f"across {group_scope}"
        + (" on the log10 scale." if use_log else ".")
    )

def render_distribution_insight_cards(
    plot_df: pd.DataFrame,
    group_var: str,
) -> None:
    """Render top-row insight cards."""
    if plot_df.empty:
        st.warning("No data available.")
        return

    values = plot_df["plot_value"].dropna().astype(float)

    p10 = float(values.quantile(0.10))
    p90 = float(values.quantile(0.90))
    median = float(values.median())
    mean = float(values.mean())

    if group_var == "None":
        third_title = "Shape"
        third_value = "Right-skewed" if mean > median else "Balanced"
        third_caption = (
            "Mean sits above median, indicating a long upper tail."
            if mean > median
            else "Mean and median are relatively close."
        )
    else:
        medians = (
            plot_df.groupby(group_var)["plot_value"]
            .median()
            .sort_values(ascending=False)
        )
        top_group = str(medians.index[0])
        third_title = "Top Median Group"
        third_value = top_group
        third_caption = (
            f"{top_group} has the highest median displayed value in the current grouped view."
        )

    st.markdown("### 🧠 Key Insights")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Tracks in View", f"{len(plot_df):,}")
        st.caption("Rows contributing to the visible distribution.")

    with col2:
        st.metric("Middle 80%", f"{p10:,.3f} → {p90:,.3f}")
        st.caption("Most visible tracks fall inside this displayed-value range.")

    with col3:
        st.metric(third_title, third_value)
        st.caption(third_caption)

def histogram(plot_df: pd.DataFrame, bins: int) -> alt.Chart:
    """Create an ungrouped histogram."""
    return (
        alt.Chart(plot_df)
        .mark_bar(opacity=0.85)
        .encode(
            x=alt.X("plot_value:Q", bin=alt.Bin(maxbins=bins), title="Displayed Value"),
            y=alt.Y("count():Q", title="Track Count"),
            tooltip=[alt.Tooltip("count():Q", title="Tracks")],
        )
        .properties(height=420)
    )


def grouped_histogram(plot_df: pd.DataFrame, group_var: str, bins: int) -> alt.Chart:
    """Create a grouped histogram."""
    return (
        alt.Chart(plot_df)
        .mark_bar(opacity=0.45)
        .encode(
            x=alt.X("plot_value:Q", bin=alt.Bin(maxbins=bins), title="Displayed Value"),
            y=alt.Y("count():Q", title="Track Count"),
            color=alt.Color(f"{group_var}:N", title=get_display_label(group_var)),
            tooltip=[
                alt.Tooltip(f"{group_var}:N", title=get_display_label(group_var)),
                alt.Tooltip("count():Q", title="Tracks"),
            ],
        )
        .properties(height=420)
    )


def pdf_line(plot_df: pd.DataFrame) -> alt.Chart:
    """Create an ungrouped density line."""
    return (
        alt.Chart(plot_df)
        .transform_density(
            "plot_value",
            as_=["plot_value", "density"],
        )
        .mark_line()
        .encode(
            x=alt.X("plot_value:Q", title="Displayed Value"),
            y=alt.Y("density:Q", title="Density"),
        )
        .properties(height=420)
    )


def grouped_pdf_line(plot_df: pd.DataFrame, group_var: str) -> alt.Chart:
    """Create grouped density lines."""
    return (
        alt.Chart(plot_df)
        .transform_density(
            "plot_value",
            as_=["plot_value", "density"],
            groupby=[group_var],
        )
        .mark_line()
        .encode(
            x=alt.X("plot_value:Q", title="Displayed Value"),
            y=alt.Y("density:Q", title="Density"),
            color=alt.Color(f"{group_var}:N", title=get_display_label(group_var)),
            tooltip=[
                alt.Tooltip(f"{group_var}:N", title=get_display_label(group_var)),
            ],
        )
        .properties(height=420)
    )


def cdf_line(plot_df: pd.DataFrame) -> alt.Chart:
    """Create an ungrouped CDF line."""
    sorted_df = plot_df[["plot_value"]].dropna().sort_values("plot_value").reset_index(drop=True)
    sorted_df["cdf"] = (np.arange(len(sorted_df)) + 1) / len(sorted_df)

    return (
        alt.Chart(sorted_df)
        .mark_line()
        .encode(
            x=alt.X("plot_value:Q", title="Displayed Value"),
            y=alt.Y("cdf:Q", title="Cumulative Probability"),
        )
        .properties(height=420)
    )


def grouped_cdf_line(plot_df: pd.DataFrame, group_var: str) -> alt.Chart:
    """Create grouped CDF lines."""
    parts = []

    for group_name, group_df in plot_df.groupby(group_var):
        sub = group_df[["plot_value"]].dropna().sort_values("plot_value").reset_index(drop=True)
        if sub.empty:
            continue
        sub["cdf"] = (np.arange(len(sub)) + 1) / len(sub)
        sub[group_var] = str(group_name)
        parts.append(sub)

    if not parts:
        return alt.Chart(pd.DataFrame({"plot_value": [], "cdf": []})).mark_line()

    cdf_df = pd.concat(parts, ignore_index=True)

    return (
        alt.Chart(cdf_df)
        .mark_line()
        .encode(
            x=alt.X("plot_value:Q", title="Displayed Value"),
            y=alt.Y("cdf:Q", title="Cumulative Probability"),
            color=alt.Color(f"{group_var}:N", title=get_display_label(group_var)),
            tooltip=[
                alt.Tooltip(f"{group_var}:N", title=get_display_label(group_var)),
            ],
        )
        .properties(height=420)
    )


def build_distribution_supporting_insight(
    plot_df: pd.DataFrame,
    group_var: str,
    view_type: str,
) -> str:
    """Build a reader-facing interpretation for the current distribution view."""
    if plot_df.empty:
        return "No visible pattern remains."

    values = plot_df["plot_value"].dropna().astype(float)

    p10 = float(values.quantile(0.10))
    p90 = float(values.quantile(0.90))
    median = float(values.median())
    mean = float(values.mean())

    if group_var == "None":
        skew = "right-skewed" if mean > median else "fairly symmetric"

        if view_type == "CDF":
            return (
                f"💡 The CDF shows how quickly the visible track population accumulates across the metric scale. "
                f"The middle 80% of displayed values runs from {p10:,.3f} to {p90:,.3f}, "
                f"and the overall distribution is {skew}."
            )

        if view_type == "PDF":
            return (
                f"💡 The density curve highlights where tracks are most concentrated. "
                f"The peak of the visible distribution sits around the center of the middle 80% range "
                f"({p10:,.3f} to {p90:,.3f}), and the overall shape is {skew}."
            )

        return (
            f"💡 The histogram shows a {skew} distribution. "
            f"Most visible tracks fall between {p10:,.3f} and {p90:,.3f}, "
            f"with a median displayed value of {median:,.3f}."
        )

    medians = (
        plot_df.groupby(group_var)["plot_value"]
        .median()
        .sort_values(ascending=False)
    )
    top_group = str(medians.index[0])

    if view_type == "CDF":
        return (
            f"💡 In the grouped CDF view, {top_group} reaches higher cumulative share at larger displayed values, "
            f"which suggests that its distribution is shifted upward relative to other groups."
        )

    if view_type == "PDF":
        return (
            f"💡 The grouped density curves show where each group is most concentrated. "
            f"{top_group} has the highest median displayed value, while overlap between curves reveals "
            f"how distinct or similar the groups really are."
        )

    return (
        f"💡 In the grouped histogram, {top_group} has the highest median displayed value. "
        f"Use the amount of overlap to judge whether group differences are large and consistent or only partial."
    )


def main() -> None:
    st.set_page_config(page_title="Track Distribution Explorer", layout="wide")
    apply_app_styles()

    st.title("Track Distribution Explorer")
    st.write(
        """
        Explore how track-level metrics are distributed across the visible dataset.
        """
    )

    df = load_track_data_explorer_data()
    df = add_display_fields(df)

    filter_inputs = get_global_filter_inputs(df)
    global_controls = get_global_filter_controls(
        filter_inputs["min_year"],
        filter_inputs["max_year"],
        filter_inputs["film_genre_options"],
        filter_inputs["album_genre_options"],
    )

    df = filter_dataset(df, global_controls)

    if df.empty:
        st.warning("No data after filtering.")
        return

    metric_options = get_track_numeric_options(df)
    group_options = get_track_group_options(df)
    group_value_options_map = build_group_value_options_map(df, group_options)

    controls = get_track_distribution_controls(
        metric_options=metric_options,
        group_options=group_options,
        group_value_options_map=group_value_options_map,
    )

    metric = controls["metric"]
    view_type = controls["view_type"]
    use_log = controls["use_log"]
    bins = controls["bins"]
    group_var = controls["group_var"]
    selected_groups = controls["selected_groups"]
    top_n_groups = controls["top_n_groups"]

    plot_df = prepare_distribution_data(
        df=df,
        metric=metric,
        group_var=group_var,
        selected_groups=selected_groups,
        top_n_groups=top_n_groups,
        use_log=use_log,
    )

    if plot_df.empty:
        st.warning("No valid values remain for the selected metric.")
        return

    st.markdown("**View Context**")
    st.caption(
        build_distribution_context_caption(
            metric=metric,
            view_type=view_type,
            use_log=use_log,
            group_var=group_var,
            selected_groups=selected_groups,
            top_n_groups=top_n_groups,
        )
    )

    render_distribution_insight_cards(
        plot_df=plot_df,
        group_var=group_var,
    )

    st.markdown("### Distribution View")

    if group_var == "None":
        if view_type == "Histogram":
            chart = histogram(plot_df, bins)
        elif view_type == "PDF":
            chart = pdf_line(plot_df)
        else:
            chart = cdf_line(plot_df)
    else:
        if view_type == "Histogram":
            chart = grouped_histogram(plot_df, group_var, bins)
        elif view_type == "PDF":
            chart = grouped_pdf_line(plot_df, group_var)
        else:
            chart = grouped_cdf_line(plot_df, group_var)

    chart = chart.properties(
        title={
            "text": f"{view_type} of {get_display_label(metric)}",
            "subtitle": [
                "Displayed Value is log10-transformed." if use_log else "Displayed Value is shown on the raw scale."
            ],
        }
    )

    st.altair_chart(chart, width="stretch")
    st.caption(
        build_distribution_supporting_insight(
            plot_df=plot_df,
            group_var=group_var,
            view_type=view_type,
        )
    )

    st.markdown("### Summary Statistics")
    col1, col2, col3 = st.columns(3)

    values = plot_df["plot_value"].dropna().astype(float)

    col1.metric("Mean", f"{values.mean():,.3f}")
    col2.metric("Std Dev", f"{values.std():,.3f}")
    col3.metric("P90", f"{values.quantile(0.90):,.3f}")


if __name__ == "__main__":
    main()